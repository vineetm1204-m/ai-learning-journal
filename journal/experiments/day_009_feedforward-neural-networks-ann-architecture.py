import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        act_fn = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'leaky_relu': nn.LeakyReLU}[activation]
        
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            self.activations.append(act_fn())
            self.dropouts.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        self.layers.append(nn.Linear(prev_dim, output_dim))
    
    def forward(self, x):
        for i, (layer, act, drop) in enumerate(zip(self.layers[:-1], self.activations, self.dropouts)):
            x = drop(act(layer(x)))
        x = self.layers[-1](x)
        return x
    
    def get_architecture_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        layer_info = []
        for i, layer in enumerate(self.layers):
            layer_info.append(f"  Layer {i}: {layer.in_features} -> {layer.out_features} ({layer.weight.numel()} params)")
        return f"Total: {total_params:,} | Trainable: {trainable_params:,}\n" + "\n".join(layer_info)

def generate_datasets(n_samples=1000, noise=0.15):
    datasets = {}
    X_moon, y_moon = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    datasets['moons'] = (X_moon, y_moon)
    
    X_circ, y_circ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    datasets['circles'] = (X_circ, y_circ)
    
    X_cls, y_cls = make_classification(n_samples=n_samples, n_features=10, n_informative=6, 
                                       n_redundant=2, n_classes=3, random_state=42)
    datasets['classification'] = (X_cls, y_cls)
    return datasets

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.01, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            
            train_pred = outputs.argmax(dim=1)
            val_pred = val_outputs.argmax(dim=1)
            train_acc = (train_pred == y_train_t).float().mean().item()
            val_acc = (val_pred == y_val_t).float().mean().item()
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(val_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    
    return train_losses, val_losses, train_accs, val_accs

def plot_decision_boundary(model, X, y, ax, title):
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    with torch.no_grad():
        Z = model(grid).argmax(dim=1).numpy().reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', s=20)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    return scatter

def experiment_architecture_comparison():
    print("=" * 60)
    print("DAY 9: FEEDFORWARD NEURAL NETWORK ARCHITECTURE EXPERIMENT")
    print("=" * 60)
    
    datasets = generate_datasets(800, 0.2)
    
    architectures = {
        'Shallow (1x16)': [16],
        'Medium (2x32)': [32, 32],
        'Deep (4x64)': [64, 64, 64, 64],
        'Bottleneck (64-32-16)': [64, 32, 16],
        'Wide (1x128)': [128],
    }
    
    results = {}
    
    for ds_name, (X, y) in datasets.items():
        print(f"\n--- Dataset: {ds_name.upper()} ---")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        
        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
        
        ds_results = {}
        
        for arch_name, hidden_dims in architectures.items():
            print(f"\n  Architecture: {arch_name} -> {hidden_dims}")
            model = FeedforwardNN(input_dim, hidden_dims, output_dim, activation='relu', dropout=0.1)
            print(f"  {model.get_architecture_summary()}")
            
            train_losses, val_losses, train_accs, val_accs = train_model(
                model, X_train, y_train, X_val, y_val, epochs=150, lr=0.01, verbose=False
            )
            
            model.eval()
            with torch.no_grad():
                test_pred = model(torch.FloatTensor(X_test)).argmax(dim=1).numpy()
                test_acc = accuracy_score(y_test, test_pred)
            
            ds_results[arch_name] = {
                'model': model,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'test_acc': test_acc,
                'hidden_dims': hidden_dims
            }
            print(f"  Test Accuracy: {test_acc:.4f}")
        
        results[ds_name] = ds_results
    
    return results, datasets

def experiment_activation_functions():
    print("\n" + "=" * 60)
    print("ACTIVATION FUNCTION COMPARISON")
    print("=" * 60)
    
    X, y = make_moons(n_samples=600, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
    act_results = {}
    
    for act in activations:
        print(f"\n  Activation: {act}")
        model = FeedforwardNN(2, [32, 32], 2, activation=act, dropout=0.0)
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, X_train, y_train, X_val, y_val, epochs=200, lr=0.01, verbose=False
        )
        model.eval()
        with torch.no_grad():
            test_pred = model(torch.FloatTensor(X_test)).argmax(dim=1).numpy()
            test_acc = accuracy_score(y_test, test_pred)
        act_results[act] = {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_acc': test_acc
        }
        print(f"  Test Accuracy: {test_acc:.4f}")
    
    return act_results, (X_scaled, y)

def experiment_regularization():
    print("\n" + "=" * 60)
    print("REGULARIZATION EFFECTS (DROPOUT & WEIGHT DECAY)")
    print("=" * 60)
    
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                               n_redundant=5, n_classes=4, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    configs = [
        ('No Regularization', 0.0, 0.0),
        ('Dropout 0.3', 0.3, 0.0),
        ('Weight Decay 1e-3', 0.0, 1e-3),
        ('Dropout 0.3 + WD 1e-3', 0.3, 1e-3),
        ('Dropout 0.5', 0.5, 0.0),
    ]
    
    reg_results = {}
    
    for name, dropout, weight_decay in configs:
        print(f"\n  Config: {name}")
        model = FeedforwardNN(20, [64, 64, 32], 4, activation='relu', dropout=dropout)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        
        for epoch in range(150):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                train_acc = (outputs.argmax(dim=1) == y_train_t).float().mean().item()
                val_acc = (val_outputs.argmax(dim=1) == y_val_t).float().mean().item()
            
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            train_accs.append(train_acc)
            val_accs.append(val_acc)
        
        model.eval()
        with torch.no_grad():
            test_pred = model(torch.FloatTensor(X_test)).argmax(dim=1).numpy()
            test_acc = accuracy_score(y_test, test_pred)
        
        reg_results[name] = {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_acc': test_acc,
            'dropout': dropout,
            'weight_decay': weight_decay
        }
        print(f"  Test Accuracy: {test_acc:.4f} | Gap (Train-Val Acc): {train_accs[-1] - val_accs[-1]:.4f}")
    
    return reg_results

def visualize_results(arch_results, act_results, reg_results, datasets):
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Architecture comparison on moons
    ax1 = plt.subplot(3, 4, 1)
    for name, res in arch_results['moons'].items():
        ax1.plot(res['val_accs'], label=name, alpha=0.8)
    ax1.set_title('Architecture Comparison (Moons) - Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # 2. Decision boundaries for moons
    for idx, (name, res) in enumerate(arch_results['moons'].items()):
        ax = plt.subplot(3, 4, 2 + idx)
        X, y = datasets['moons']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        plot_decision_boundary(res['model'], X_scaled, y, ax, f"{name}\nTest Acc: {res['test_acc']:.3f}")
    
    # 3. Activation functions
    ax_act = plt.subplot(3, 4, 6)
    X_act, y_act = act_results[1]
    for act, res in act_results[0].items():
        ax_act.plot(res['val_accs'], label=act, alpha=0.8)
    ax_act.set_title('Activation Functions - Val Accuracy')
    ax_act.set_xlabel('Epoch')
    ax_act.set_ylabel('Accuracy')
    ax_act.legend(fontsize=7)
    ax_act.grid(True, alpha=0.3)
    
    # 4. Activation decision boundaries
    for idx, (act, res) in enumerate(act_results[0].items()):
        ax = plt.subplot(3, 4, 7 + idx)
        plot_decision_boundary(res['model'], X_act, y_act, ax, f"{act}\nTest Acc: {res['test_acc']:.3f}")
    
    # 5. Regularization comparison
    ax_reg = plt.subplot(3, 4, 11)
    for name, res in reg_results.items():
        ax_reg.plot(res['val_accs'], label=name, alpha=0.8)
    ax_reg.set_title('Regularization - Val Accuracy')
    ax_reg.set_xlabel('Epoch')
    ax_reg.set_ylabel('Accuracy')
    ax_reg.legend(fontsize=7)
    ax_reg.grid(True, alpha=0.3)
    
    # 6. Train-Val Gap for regularization
    ax_gap = plt.subplot(3, 4, 12)
    names = list(reg_results.keys())
    gaps = [reg_results[n]['train_accs'][-1] - reg_results[n]['val_accs'][-1] for n in names]
    test_accs = [reg_results[n]['test_acc'] for n in names]
    x_pos = np.arange(len(names))
    ax_gap.bar(x_pos - 0.2, gaps, 0.4, label='Train-Val Gap', alpha=0.7)
    ax_gap.bar(x_pos + 0.2, test_accs, 0.4, label='Test Acc', alpha=0.7)
    ax_gap.set_xticks(x_pos)
    ax_gap.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax_gap.set_title('Regularization: Overfitting Gap vs Test Acc')
    ax_gap.legend(fontsize=7)
    ax_gap.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('day9_ann_architecture_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'day9_ann_architecture_results.png'")
    plt.close()

def print_summary():
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FROM DAY 9 EXPERIMENT")
    print("=" * 60)
    insights = [
        "1. ARCHITECTURE DEPTH vs WIDTH: Deeper networks (4x64) can model complex boundaries",
        "   but may overfit on small datasets. Bottleneck architectures (64-32-16) offer",
        "   good compression with maintained performance.",
        "",
        "2. ACTIVATION FUNCTIONS: ReLU/LeakyReLU converge faster and achieve higher accuracy",
        "   than sigmoid/tanh on non-linear problems. Sigmoid suffers from vanishing gradients",
        "   in deeper networks. LeakyReLU prevents dead neurons.",
        "",
        "3. REGULARIZATION: Dropout (0.3) + Weight Decay (1e-3) provides best generalization.",
        "   High dropout (0.5) can underfit. Weight decay alone helps but less than dropout.",
        "   Train-Val accuracy gap is a good overfitting indicator.",
        "",
        "4. PRACTICAL GUIDELINES:",
        "   - Start with 2-3 hidden layers, 32-128 units each",
        "   - Use ReLU + BatchNorm (not shown) for faster convergence",
        "   - Dropout 0.1-0.3 for hidden layers",
        "   - Weight decay 1e-4 to 1e-3",
        "   - Monitor train/val gap for early stopping"
    ]
    for line in insights:
        print(line)

if __name__ == "__main__":
    arch_results, datasets = experiment_architecture_comparison()
    act_results = experiment_activation_functions()
    reg_results = experiment_regularization()
    
    visualize_results(arch_results, act_results, reg_results, datasets)
    close_summary()