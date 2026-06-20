import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.01
MOMENTUM = 0.9
SEED = 42
HIDDEN_LAYERS = [512, 256, 128] # Deep enough to show init differences
DROPOUT_RATE = 0.0 # Keep off to isolate init effects

torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# --- Data Loading (MNIST) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --- Model Definition ---
class DeepMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=None, output_dim=10, init_fn=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_LAYERS
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            if init_fn:
                init_fn(linear)
            layers.append(linear)
            layers.append(nn.ReLU(inplace=True))
            if DROPOUT_RATE > 0:
                layers.append(nn.Dropout(DROPOUT_RATE))
            prev_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, output_dim)
        if init_fn:
            init_fn(self.classifier)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# --- Initialization Strategies ---
def init_xavier_uniform(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_xavier_normal(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_he_uniform(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_he_normal(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_standard_normal(module, std=0.01):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_standard_uniform(module, bound=0.1):
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, a=-bound, b=bound)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

INIT_STRATEGIES = {
    "Xavier Uniform": init_xavier_uniform,
    "Xavier Normal": init_xavier_normal,
    "He Uniform (Kaiming)": init_he_uniform,
    "He Normal (Kaiming)": init_he_normal,
    "Std Normal (0.01)": lambda m: init_standard_normal(m, std=0.01),
    "Std Normal (0.1)": lambda m: init_standard_normal(m, std=0.1),
    "Uniform (-0.1, 0.1)": lambda m: init_standard_uniform(m, bound=0.1),
}

# --- Training & Evaluation ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    return running_loss / total, 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return running_loss / total, 100. * correct / total

# --- Experiment Runner ---
def run_experiment(init_name, init_fn):
    print(f"\n{'='*20} Running: {init_name} {'='*20}")
    model = DeepMLP(init_fn=init_fn).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
              
    elapsed = time.time() - start_time
    print(f"Finished {init_name} in {elapsed:.1f}s. Best Test Acc: {max(history['test_acc']):.2f}%")
    return history

# --- Main Execution ---
if __name__ == "__main__":
    all_histories = {}
    
    for name, fn in INIT_STRATEGIES.items():
        try:
            all_histories[name] = run_experiment(name, fn)
        except Exception as e:
            print(f"Error running {name}: {e}")
            all_histories[name] = None

    # --- Plotting Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Weight Initialization Comparison (MLP on MNIST, {EPOCHS} Epochs)', fontsize=16)
    
    epochs_range = range(1, EPOCHS + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(INIT_STRATEGIES)))
    
    for idx, (name, history) in enumerate(all_histories.items()):
        if history is None: continue
        c = colors[idx]
        axes[0, 0].plot(epochs_range, history['train_loss'], label=name, color=c, alpha=0.8)
        axes[0, 1].plot(epochs_range, history['test_loss'], label=name, color=c, alpha=0.8)
        axes[1, 0].plot(epochs_range, history['train_acc'], label=name, color=c, alpha=0.8)
        axes[1, 1].plot(epochs_range, history['test_acc'], label=name, color=c, alpha=0.8)

    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(fontsize='small', loc='upper right')
    
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_yscale('log')
    
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_xlabel('Epoch')
    
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylim(0, 100)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    output_path = "day6_init_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {os.path.abspath(output_path)}")
    
    # Print Summary Table
    print("\n\n--- FINAL SUMMARY (Best Test Accuracy) ---")
    print(f"{'Initialization Strategy':<30} | {'Best Test Acc':>12} | {'Final Train Acc':>14} | {'Convergence Speed (Epoch@95%)':>25}")
    print("-" * 90)
    for name, history in all_histories.items():
        if history is None: continue
        best_acc = max(history['test_acc'])
        final_train = history['train_acc'][-1]
        # Find epoch where test acc first exceeds 95% (or max if never)
        conv_epoch = "N/A"
        for i, acc in enumerate(history['test_acc']):
            if acc >= 95.0:
                conv_epoch = f"Epoch {i+1}"
                break
        print(f"{name:<30} | {best_acc:>11.2f}% | {final_train:>13.2f}% | {conv_epoch:>25}")