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
BATCH_SIZE = 256
EPOCHS = 10
LR = 0.01
MOMENTUM = 0.9
SEED = 42
HIDDEN_LAYERS = [512, 256, 128] # Deep enough to show initialization effects
INPUT_DIM = 28 * 28
OUTPUT_DIM = 10

torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(SEED)

# --- Data Loading ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self, init_fn_name='default'):
        super().__init__()
        self.init_fn_name = init_fn_name
        layers = []
        prev_dim = INPUT_DIM
        for h_dim in HIDDEN_LAYERS:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, OUTPUT_DIM))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_fn_name == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_fn_name == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif self.init_fn_name == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_fn_name == 'he_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.init_fn_name == 'random_normal':
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif self.init_fn_name == 'random_uniform':
                    nn.init.uniform_(m.weight, a=-0.01, b=0.01)
                else: # PyTorch Default (Kaiming Uniform for Linear)
                    pass 
                
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

# --- Training & Evaluation ---
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
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

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / total, 100. * correct / total

# --- Experiment Runner ---
def run_experiment(init_name, tag):
    print(f"\n{'='*20} {tag} ({init_name}) {'='*20}")
    model = MLP(init_fn_name=init_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.4f} | "
              f"Tr Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}% | "
              f"Te Loss: {te_loss:.4f} Acc: {te_acc:.2f}%")
              
    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.1f}s. Best Test Acc: {max(history['test_acc']):.2f}%")
    return history

# --- Main Execution ---
if __name__ == "__main__":
    strategies = {
        'default': 'PyTorch Default (Kaiming Uniform)',
        'xavier_uniform': 'Xavier (Glorot) Uniform',
        'xavier_normal': 'Xavier (Glorot) Normal',
        'he_uniform': 'He (Kaiming) Uniform',
        'he_normal': 'He (Kaiming) Normal',
        'random_normal': 'Random Normal (std=0.01)',
        'random_uniform': 'Random Uniform (-0.01, 0.01)'
    }
    
    all_histories = {}
    
    for key, label in strategies.items():
        all_histories[key] = run_experiment(key, label)
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Day 88: Weight Initialization Comparison (MLP on MNIST, {EPOCHS} Epochs)', fontsize=16)
    
    epochs_range = range(1, EPOCHS + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    for idx, (key, label) in enumerate(strategies.items()):
        h = all_histories[key]
        c = colors[idx]
        axes[0, 0].plot(epochs_range, h['train_loss'], label=label, color=c, alpha=0.8)
        axes[0, 1].plot(epochs_range, h['train_acc'], label=label, color=c, alpha=0.8)
        axes[1, 0].plot(epochs_range, h['test_loss'], label=label, color=c, alpha=0.8)
        axes[1, 1].plot(epochs_range, h['test_acc'], label=label, color=c, alpha=0.8)
    
    axes[0, 0].set_title('Training Loss'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].legend(fontsize='small', loc='upper right')
    axes[0, 1].set_title('Training Accuracy'); axes[0, 1].set_ylabel('Accuracy %'); axes[0, 1].legend(fontsize='small', loc='lower right')
    axes[1, 0].set_title('Test Loss'); axes[1, 0].set_ylabel('Loss'); axes[1, 0].set_xlabel('Epoch'); axes[1, 0].legend(fontsize='small', loc='upper right')
    axes[1, 1].set_title('Test Accuracy'); axes[1, 1].set_ylabel('Accuracy %'); axes[1, 1].set_xlabel('Epoch'); axes[1, 1].legend(fontsize='small', loc='lower right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = 'day88_init_comparison.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {os.path.abspath(plot_path)}")
    
    # Final Summary Table
    print("\n\n--- FINAL SUMMARY (Best Test Accuracy) ---")
    print(f"{'Strategy':<25} | {'Best Test Acc':>14} | {'Final Train Acc':>16} | {'Gap (Overfit)':>14}")
    print("-" * 75)
    for key, label in strategies.items():
        h = all_histories[key]
        best_te = max(h['test_acc'])
        final_tr = h['train_acc'][-1]
        gap = final_tr - best_te
        print(f"{label:<25} | {best_te:>13.2f}% | {final_tr:>15.2f}% | {gap:>13.2f}%")