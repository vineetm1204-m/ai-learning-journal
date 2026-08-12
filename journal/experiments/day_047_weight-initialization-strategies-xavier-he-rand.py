import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9
SEED = 42
DATA_ROOT = "./data"
PLOT_SAVE_PATH = "day47_init_comparison.png"

torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# --- Dataset (MNIST) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --- Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

# --- Initialization Functions ---
def init_weights_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_xavier_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_he_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_he_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_random_normal(m, std=0.01):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_random_uniform(m, bound=0.1):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-bound, b=bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

INIT_STRATEGIES = {
    "Xavier Uniform": init_weights_xavier_uniform,
    "Xavier Normal": init_weights_xavier_normal,
    "He Uniform (Kaiming)": init_weights_he_uniform,
    "He Normal (Kaiming)": init_weights_he_normal,
    "Random Normal (std=0.01)": init_weights_random_normal,
    "Random Uniform (bound=0.1)": init_weights_random_uniform,
}

# --- Training & Evaluation ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / total, 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / total, 100. * correct / total

# --- Main Experiment Loop ---
results = {}
criterion = nn.CrossEntropyLoss()

print(f"Device: {DEVICE}")
print(f"{'Strategy':<30} | {'Final Train Acc':>15} | {'Final Test Acc':>14} | {'Time (s)':>8}")
print("-" * 75)

for name, init_fn in INIT_STRATEGIES.items():
    # Fresh model per strategy
    model = SimpleMLP().to(DEVICE)
    model.apply(init_fn)
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
    
    elapsed = time.time() - start_time
    results[name] = history
    
    print(f"{name:<30} | {history['train_acc'][-1]:>15.2f} | {history['test_acc'][-1]:>14.2f} | {elapsed:>8.1f}")

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Day 47: Weight Initialization Strategy Comparison (MNIST MLP)', fontsize=16)

epochs_range = range(1, EPOCHS + 1)
colors = plt.cm.tab10(np.linspace(0, 1, len(INIT_STRATEGIES)))

for idx, (name, history) in enumerate(results.items()):
    c = colors[idx]
    # Train Loss
    axes[0, 0].plot(epochs_range, history['train_loss'], label=name, color=c, marker='o', markersize=3)
    # Test Loss
    axes[0, 1].plot(epochs_range, history['test_loss'], label=name, color=c, marker='o', markersize=3)
    # Train Acc
    axes[1, 0].plot(epochs_range, history['train_acc'], label=name, color=c, marker='o', markersize=3)
    # Test Acc
    axes[1, 1].plot(epochs_range, history['test_acc'], label=name, color=c, marker='o', markersize=3)

axes[0, 0].set_title('Training Loss')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_yscale('log')
axes[0, 0].legend(fontsize='small', loc='upper right')

axes[0, 1].set_title('Test Loss')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_yscale('log')
axes[0, 1].legend(fontsize='small', loc='upper right')

axes[1, 0].set_title('Training Accuracy')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend(fontsize='small', loc='lower right')

axes[1, 1].set_title('Test Accuracy')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend(fontsize='small', loc='lower right')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(PLOT_SAVE_PATH, dpi=150)
print(f"\nPlot saved to {PLOT_SAVE_PATH}")

# Print summary table
print("\n--- Summary Statistics (Final Epoch) ---")
print(f"{'Strategy':<30} | {'Train Loss':>10} | {'Test Loss':>10} | {'Train Acc':>10} | {'Test Acc':>10}")
print("-" * 80)
for name, h in results.items():
    print(f"{name:<30} | {h['train_loss'][-1]:>10.4f} | {h['test_loss'][-1]:>10.4f} | {h['train_acc'][-1]:>9.2f}% | {h['test_acc'][-1]:>9.2f}%")