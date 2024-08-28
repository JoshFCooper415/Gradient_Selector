import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from collections import deque
import random
import os
import time
import math

# Define the CNN for MNIST
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# QNetwork and GradientSelector classes (same as before)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class GradientSelector:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, load_path=None):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.memory = deque(maxlen=10000)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.steps_done = 0
        
        if load_path and os.path.exists(load_path):
            self.load_model(load_path)
        
        self.target_network.load_state_dict(self.q_network.state_dict())

    def load_model(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")

    def select_action(self, state):
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.q_network.fc3.out_features - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state).to(self.q_network.device))
                return q_values.argmax().item()

    # Other methods remain the same

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_network.fc3.out_features)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state).cuda())
                return q_values.argmax().item()

# Helper functions
def create_state(model, gradients, loss, accuracy):
    grad_mean = gradients.mean().item()
    grad_std = gradients.std().item()
    grad_max = gradients.max().item()
    grad_min = gradients.min().item()
    
    layer_norms = []
    for param in model.parameters():
        if param.grad is not None:
            layer_norms.append(param.grad.norm().item())
    
    layer_norms += [0] * (10 - len(layer_norms))  # Ensure we always have 10 layer norms
    
    state = [grad_mean, grad_std, grad_max, grad_min, loss, accuracy] + layer_norms
    return np.array(state)

def apply_selected_gradients(model, action, num_groups=10):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    flattened_grads = torch.cat([g.view(-1) for g in grads])
    
    grad_groups = torch.chunk(flattened_grads, num_groups)
    
    mask = torch.zeros_like(flattened_grads)
    start_idx = action * len(grad_groups[0])
    end_idx = (action + 1) * len(grad_groups[0])
    mask[start_idx:end_idx] = 1.0
    
    idx = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_size = p.grad.numel()
            p.grad.copy_(
                (flattened_grads[idx:idx+grad_size] * mask[idx:idx+grad_size]).view_as(p.grad)
            )
            idx += grad_size

def train_rl(model, device, train_loader, optimizer, q_selector, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()

        gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        accuracy = (output.argmax(dim=1) == target).float().mean().item()
        state = create_state(model, gradients, loss.item(), accuracy)

        action = q_selector.select_action(state)
        apply_selected_gradients(model, action)

        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'RL Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def train_adam(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Adam Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    # Training settings
    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    lr = 0.01
    seed = 1
    
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    # RL Optimizer model
    model_rl = MNISTCNN().to(device)
    optimizer_rl = optim.Adadelta(model_rl.parameters(), lr=lr)

    # Adam model
    model_adam = MNISTCNN().to(device)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=lr)

    # Initialize Q-learning components
    state_dim = 16  # 4 gradient stats + loss + accuracy + 10 layer norms
    action_dim = 10  # Number of gradient groups
    load_path = './saved_models/q_network_final.pth'  # Path to the pre-trained RL model
    q_selector = GradientSelector(state_dim, action_dim, load_path=load_path)
    q_selector.q_network.to(device)
    q_selector.target_network.to(device)

    # Training loop
    rl_accuracies = []
    adam_accuracies = []
    rl_times = []
    adam_times = []

    for epoch in range(1, epochs + 1):
        # RL Optimizer
        start_time = time.time()
        train_rl(model_rl, device, train_loader, optimizer_rl, q_selector, epoch)
        rl_accuracy = test(model_rl, device, test_loader)
        rl_time = time.time() - start_time
        rl_accuracies.append(rl_accuracy)
        rl_times.append(rl_time)
        print(f"RL Epoch {epoch} time: {rl_time:.2f} seconds")

        # Adam
        start_time = time.time()
        train_adam(model_adam, device, train_loader, optimizer_adam, epoch)
        adam_accuracy = test(model_adam, device, test_loader)
        adam_time = time.time() - start_time
        adam_accuracies.append(adam_accuracy)
        adam_times.append(adam_time)
        print(f"Adam Epoch {epoch} time: {adam_time:.2f} seconds")

    # Print final results
    print("\nFinal Results:")
    print(f"RL Optimizer - Final Accuracy: {rl_accuracies[-1]:.2f}%, Average Time per Epoch: {np.mean(rl_times):.2f} seconds")
    print(f"Adam Optimizer - Final Accuracy: {adam_accuracies[-1]:.2f}%, Average Time per Epoch: {np.mean(adam_times):.2f} seconds")

if __name__ == '__main__':
    main()