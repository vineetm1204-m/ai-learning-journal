import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', init_scheme='xavier'):
        super().__init__()
        self.activation_name = activation
        self.init_scheme = init_scheme
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_scheme == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif self.init_scheme == 'he':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.init_scheme == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                elif self.init_scheme == 'uniform':
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

def generate_synthetic_data(n_samples=5000, input_dim=20, n_classes=5):
    X = torch.randn(n_samples, input_dim)
    true_weights = torch.randn(input_dim, n_classes)
    logits = X @ true_weights
    y = torch.argmax(logits, dim=1)
    return X, y

def train_and_track_gradients(model, train_loader, criterion, optimizer, n_epochs=10):
    model.train()
    gradient_history = defaultdict(list)
    
    for epoch in range(n_epochs):
        epoch_gradients = defaultdict(list)
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None and 'weight' in name:
                    grad_norm = param.grad.data.norm(2).item()
                    epoch_gradients[name].append(grad_norm)
            
            optimizer.step()
        
        for name, grads in epoch_gradients.items():
            gradient_history[name].append(np.mean(grads))
    
    return gradient_history

def analyze_gradient_flow(gradient_history, hidden_dims):
    layer_names = [name for name in gradient_history.keys() if 'weight' in name]
    layer_names.sort(key=lambda x: int(x.split('.')[1]) if '.' in x else 0)
    
    final_grads = {}
    for name in layer_names:
        if gradient_history[name]:
            final_grads[name] = gradient_history[name][-1]
    
    return final_grads, layer_names

def run_experiment():
    input_dim = 20
    hidden_dims = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
    output_dim = 5
    n_epochs = 15
    batch_size = 64
    lr = 0.01
    
    X_train, y_train = generate_synthetic_data(4000, input_dim, output_dim)
    X_val, y_val = generate_synthetic_data(1000, input_dim, output_dim)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    
    configs = [
        ('sigmoid', 'xavier'),
        ('tanh', 'xavier'),
        ('relu', 'xavier'),
        ('relu', 'he'),
        ('relu', 'normal'),
    ]
    
    all_results = {}
    
    for activation, init_scheme in configs:
        print(f"\nTraining: activation={activation}, init={init_scheme}")
        model = DeepMLP(input_dim, hidden_dims, output_dim, activation, init_scheme).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        grad_history = train_and_track_gradients(model, train_loader, criterion, optimizer, n_epochs)
        final_grads, layer_names = analyze_gradient_flow(grad_history, hidden_dims)
        
        all_results[f"{activation}_{init_scheme}"] = {
            'history': grad_history,
            'final_grads': final_grads,
            'layer_names': layer_names
        }
    
    return all_results, hidden_dims

def plot_gradient_analysis(all_results, hidden_dims):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(hidden_dims) + 1))
    
    for idx, (config_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        layer_names = results['layer_names']
        history = results['history']
        
        for i, name in enumerate(layer_names):
            if name in history and history[name]:
                epochs = range(1, len(history[name]) + 1)
                ax.plot(epochs, history[name], label=f'Layer {i+1}', color=colors[i], alpha=0.8, linewidth=1.5)
        
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm (log scale)')
        ax.set_title(f'{config_name}')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    
    ax = axes[-1]
    for config_name, results in all_results.items():
        final_grads = results['final_grads']
        layer_names = results['layer_names']
        grads = [final_grads.get(name, 0) for name in layer_names]
        layer_indices = range(1, len(layer_names) + 1)
        ax.semilogy(layer_indices, grads, 'o-', label=config_name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Layer Index (1 = closest to input)')
    ax.set_ylabel('Final Gradient Norm (log scale)')
    ax.set_title('Gradient Norm Across Layers (Final Epoch)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1e-3, color='red', linestyle='--', alpha=0.5, label='Vanishing threshold')
    ax.axhline(y=1e2, color='orange', linestyle='--', alpha=0.5, label='Exploding threshold')
    
    plt.tight_layout()
    plt.savefig('gradient_analysis_day93.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_summary(all_results):
    print("\n" + "="*80)
    print("GRADIENT ANALYSIS SUMMARY - DAY 93")
    print("="*80)
    
    for config_name, results in all_results.items():
        final_grads = results['final_grads']
        layer_names = results['layer_names']
        grads = [final_grads.get(name, 0) for name in layer_names]
        
        print(f"\n{config_name}:")
        print(f"  Input layer grad norm:  {grads[0]:.6f}")
        print(f"  Middle layer grad norm: {grads[len(grads)//2]:.6f}")
        print(f"  Output layer grad norm: {grads[-1]:.6f}")
        print(f"  Ratio (input/output):   {grads[0]/grads[-1]:.2f}" if grads[-1] > 0 else "  Ratio: inf")
        
        vanishing = sum(1 for g in grads if g < 1e-3)
        exploding = sum(1 for g in grads if g > 1e2)
        print(f"  Vanishing layers (<1e-3): {vanishing}/{len(grads)}")
        print(f"  Exploding layers (>1e2):  {exploding}/{len(grads)}")

def demonstrate_gradient_computation():
    print("\n" + "="*80)
    print("MANUAL GRADIENT FLOW DEMONSTRATION")
    print("="*80)
    
    x = torch.randn(1, 10, requires_grad=True)
    W1 = torch.randn(10, 20, requires_grad=True)
    W2 = torch.randn(20, 20, requires_grad=True)
    W3 = torch.randn(20, 5, requires_grad=True)
    
    def sigmoid(x): return torch.sigmoid(x)
    def relu(x): return F.relu(x)
    
    for act_name, act_fn in [('sigmoid', sigmoid), ('relu', relu)]:
        x.grad = None
        W1.grad = None
        W2.grad = None
        W3.grad = None
        
        h1 = act_fn(x @ W1)
        h2 = act_fn(h1 @ W2)
        out = h2 @ W3
        loss = out.sum()
        loss.backward()
        
        print(f"\n{act_name.upper()} activation:")
        print(f"  grad W1 (input layer):  {W1.grad.norm().item():.6f}")
        print(f"  grad W2 (middle layer): {W2.grad.norm().item():.6f}")
        print(f"  grad W3 (output layer): {W3.grad.norm().item():.6f}")

if __name__ == '__main__':
    print("Day 93: Vanishing and Exploding Gradients in Deep ANNs")
    print("="*60)
    
    demonstrate_gradient_computation()
    
    all_results, hidden_dims = run_experiment()
    
    print_summary(all_results)
    
    plot_gradient_analysis(all_results, hidden_dims)
    
    print("\nExperiment complete. Plot saved as 'gradient_analysis_day93.png'")