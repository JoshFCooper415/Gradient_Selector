import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import deque
import random
import multiprocessing
import os
# Define the CNN model for CIFAR-10
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=0.1, load_path=None):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        
        if load_path and os.path.exists(load_path):
            self.load_model(load_path)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=10000)

    def save_model(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_network.fc3.out_features)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state))
                return q_values.argmax().item()

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def create_state(model, gradients, loss, accuracy):
    # Gradient statistics
    grad_mean = gradients.mean().item()
    grad_std = gradients.std().item()
    grad_max = gradients.max().item()
    grad_min = gradients.min().item()
    
    # Layer-wise gradient norms
    layer_norms = []
    for param in model.parameters():
        if param.grad is not None:
            layer_norms.append(param.grad.norm().item())
    
    # Ensure we always have the same number of layer norms
    layer_norms += [0] * (10 - len(layer_norms))  # Assuming max 10 layers
    
    # Combine all state components
    state = [grad_mean, grad_std, grad_max, grad_min, loss, accuracy] + layer_norms
    return np.array(state)

def apply_selected_gradients(model, action, num_groups=10):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    flattened_grads = torch.cat([g.view(-1) for g in grads])
    
    # Split gradients into groups
    grad_groups = torch.chunk(flattened_grads, num_groups)
    
    # Apply gradients only for the selected group
    mask = torch.zeros_like(flattened_grads)
    start_idx = action * len(grad_groups[0])
    end_idx = (action + 1) * len(grad_groups[0])
    mask[start_idx:end_idx] = 1.0
    
    # Apply mask to gradients
    idx = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_size = p.grad.numel()
            p.grad.copy_(
                (flattened_grads[idx:idx+grad_size] * mask[idx:idx+grad_size]).view_as(p.grad)
            )
            idx += grad_size

def calculate_reward(prev_loss, current_loss, prev_accuracy, current_accuracy):
    loss_improvement = max(prev_loss - current_loss, 0)
    accuracy_improvement = max(current_accuracy - prev_accuracy, 0)
    return loss_improvement + accuracy_improvement * 100  # Weigh accuracy improvement more

def train_model_with_q_learning(model, optimizer, loss_fn, q_selector, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        prev_loss = float('inf')
        prev_accuracy = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()

            # Get gradients and create state representation
            gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            accuracy = (output.argmax(dim=1) == target).float().mean().item()
            state = create_state(model, gradients, loss.item(), accuracy)

            # Select action (which gradients to use)
            action = q_selector.select_action(state)

            # Apply selected gradients
            apply_selected_gradients(model, action)

            optimizer.step()

            # Calculate reward
            current_loss = loss.item()
            current_accuracy = accuracy
            reward = calculate_reward(prev_loss, current_loss, prev_accuracy, current_accuracy)

            # Get next state
            with torch.no_grad():
                output = model(data)
                new_loss = loss_fn(output, target).item()
                new_accuracy = (output.argmax(dim=1) == target).float().mean().item()
            next_state = create_state(model, gradients, new_loss, new_accuracy)

            # Store transition in memory
            q_selector.memory.append((state, action, reward, next_state, False))

            # Update Q-network
            q_selector.update(batch_size=32)

            prev_loss = current_loss
            prev_accuracy = current_accuracy

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%")

        # Validate the model
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += loss_fn(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs} completed. Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {100.*val_correct/val_total:.2f}%")
        
        # Update target network periodically
        if epoch % 10 == 0:
            q_selector.update_target_network()

## CIFAR-10 Dataset setup
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_data():
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # Use num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

    return train_loader, val_loader
def main():
    # Load data
    train_loader, val_loader = load_data()

    # Initialize model, optimizer, and Q-learning components
    model = CIFAR10CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    state_dim = 16 # 4 gradient stats + loss + accuracy + 10 layer norms
    action_dim = 10  # Number of gradient groups
    
    # Optionally load a pre-trained Q-network
    load_path = './saved_models/q_network_latest.pth'
    q_selector = GradientSelector(state_dim, action_dim, load_path=load_path)

    # Train the model
    num_epochs = 20
    train_model_with_q_learning(model, optimizer, loss_fn, q_selector, train_loader, val_loader, num_epochs)

    # Save the final model
    final_save_path = './saved_models/q_network_final.pth'
    q_selector.save_model(final_save_path)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()