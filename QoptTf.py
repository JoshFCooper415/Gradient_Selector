import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import time
from tqdm import tqdm
import numpy as np
from collections import deque
import random

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256
VOCAB_SIZE = 30000
EMBEDDING_DIM = 128
NUM_HEADS = 4
FFN_HID_DIM = 256
NUM_LAYERS = 2
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# Load dataset
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.model_max_length = MAX_SEQ_LENGTH

# Dataset class
class IMDBDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item['text'], item['label']

# Transformer model
class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, ffn_hid_dim, num_layers):
        super(SentimentTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=ffn_hid_dim),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embedding_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Global average pooling
        return torch.sigmoid(self.fc(x)).squeeze()

# QNetwork for RL Gradient Selector
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

# GradientSelector class
class GradientSelector:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.memory = deque(maxlen=10000)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_network.fc3.out_features - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state).to(device))
                return q_values.argmax().item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

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

# Data processing functions
def collate_batch(batch):
    texts, labels = zip(*batch)
    encoded = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    return encoded['input_ids'].to(device), torch.tensor(labels, dtype=torch.float).to(device)

# Create data loaders
train_loader = DataLoader(IMDBDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(IMDBDataset(test_dataset), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# Initialize the models
model_transformer = SentimentTransformer(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, FFN_HID_DIM, NUM_LAYERS).to(device)
model_rl = SentimentTransformer(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, FFN_HID_DIM, NUM_LAYERS).to(device)

criterion = nn.BCELoss()
optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=LEARNING_RATE)
optimizer_rl = optim.Adam(model_rl.parameters(), lr=LEARNING_RATE)

# Initialize Q-learning components
state_dim = 33  # 4 gradient stats + loss + accuracy + 10 layer norms
action_dim = 10  # Number of gradient groups
q_selector = GradientSelector(state_dim, action_dim)

# Training loop for transformer
def train_transformer(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for texts, labels in tqdm(loader, desc="Training Transformer"):
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Training loop for RL
def train_rl(model, loader, criterion, optimizer, q_selector):
    model.train()
    total_loss = 0
    for texts, labels in tqdm(loader, desc="Training RL"):
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()

        gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        accuracy = (outputs.round() == labels).float().mean().item()
        state = create_state(model, gradients, loss.item(), accuracy)

        action = q_selector.select_action(state)
        apply_selected_gradients(model, action)

        optimizer.step()
        q_selector.update_epsilon()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation loop
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in tqdm(loader, desc="Evaluating"):
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return total_loss / len(loader), accuracy

# Training
transformer_accuracies = []
rl_accuracies = []

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    # Train and evaluate transformer
    start_time = time.time()
    train_loss_transformer = train_transformer(model_transformer, train_loader, criterion, optimizer_transformer)
    test_loss_transformer, test_accuracy_transformer = evaluate(model_transformer, test_loader, criterion)
    transformer_time = time.time() - start_time
    transformer_accuracies.append(test_accuracy_transformer)
    
    print(f"Transformer - Train Loss: {train_loss_transformer:.4f}, Test Loss: {test_loss_transformer:.4f}, "
          f"Test Accuracy: {test_accuracy_transformer:.4f}, Time: {transformer_time:.2f}s")
    
    # Train and evaluate RL
    start_time = time.time()
    train_loss_rl = train_rl(model_rl, train_loader, criterion, optimizer_rl, q_selector)
    test_loss_rl, test_accuracy_rl = evaluate(model_rl, test_loader, criterion)
    rl_time = time.time() - start_time
    rl_accuracies.append(test_accuracy_rl)
    
    print(f"RL Gradient Selector - Train Loss: {train_loss_rl:.4f}, Test Loss: {test_loss_rl:.4f}, "
          f"Test Accuracy: {test_accuracy_rl:.4f}, Time: {rl_time:.2f}s")
    
    print()

print("Training completed!")

# Final comparison
print("\nFinal Results:")
print(f"Transformer - Final Accuracy: {transformer_accuracies[-1]:.4f}")
print(f"RL Gradient Selector - Final Accuracy: {rl_accuracies[-1]:.4f}")

# Plot learning curves
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_EPOCHS + 1), transformer_accuracies, label='Transformer')
plt.plot(range(1, NUM_EPOCHS + 1), rl_accuracies, label='RL Gradient Selector')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Learning Curves: Transformer vs RL Gradient Selector')
plt.legend()
plt.grid(True)
plt.show()