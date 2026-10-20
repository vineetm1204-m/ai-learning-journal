import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import List, Dict, Tuple

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    n_samples = 5000
    n_features = 20
    n_classes = 4
    batch_size = 64
    epochs = 50
    lr = 1e-3
    weight_decay = 1e-4
    hidden_configs = [
        [64],
        [128, 64],
        [256, 128, 64],
        [512, 256, 128, 64],
        [128, 128, 128, 128],
    ]
    activations = ['relu', 'tanh', 'gelu', 'silu']
    dropout_rates = [0.0, 0.1, 0.3, 0.5]

def generate_data(n_samples: int, n_features: int, n_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(n_samples, n_features)
    true_weights = torch.randn(n_features, n_classes)
    logits = X @ true_weights
    y = torch.argmax(logits + 0.5 * torch.randn_like(logits), dim=1)
    return X, y

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'relu', dropout: float = 0.0, use_batchnorm: bool = True):
        super().__init__()
        self.activation_name = activation
        self.activation_fn = self._get_activation(activation)
        self.use_batchnorm = use_batchnorm
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_name in ['relu', 'leaky_relu', 'gelu', 'silu']:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.activation_name in ['tanh', 'sigmoid']:
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_layer_stats(self) -> Dict:
        stats = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'norm': param.data.norm().item(),
                    'shape': list(param.shape)
                }
        return stats

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, lr: float, weight_decay: float) -> Dict:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            train_correct += (outputs.argmax(1) == y_batch).sum().item()
            train_total += X_batch.size(0)
        
        scheduler.step()
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_correct += (outputs.argmax(1) == y_batch).sum().item()
                val_total += X_batch.size(0)
        
        history['train_loss'].append(train_loss / train_total)
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['val_acc'].append(val_correct / val_total)
        history['lr'].append(optimizer.param_groups[0]['lr'])
    
    return history

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model: nn.Module, input_shape: Tuple, n_runs: int = 100) -> float:
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
        start = time.time()
        for _ in range(n_runs):
            _ = model(dummy_input)
        end = time.time()
    return (end - start) / n_runs * 1000

def run_architecture_experiment():
    print("=" * 70)
    print("DAY 91: FEEDFORWARD NEURAL NETWORK ARCHITECTURE EXPERIMENT")
    print("=" * 70)
    
    X, y = generate_data(Config.n_samples, Config.n_features, Config.n_classes)
    n_train = int(0.8 * Config.n_samples)
    train_dataset = TensorDataset(X[:n_train], y[:n_train])
    val_dataset = TensorDataset(X[n_train:], y[n_train:])
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    
    results = {}
    
    print("\n[1] HIDDEN LAYER CONFIGURATION COMPARISON")
    print("-" * 50)
    for hidden_dims in Config.hidden_configs:
        model = FlexibleMLP(Config.n_features, hidden_dims, Config.n_classes,
                           activation='relu', dropout=0.1)
        n_params = count_parameters(model)
        history = train_model(model, train_loader, val_loader, Config.epochs, Config.lr, Config.weight_decay)
        inference_ms = measure_inference_time(model, (Config.batch_size, Config.n_features))
        
        key = f"hidden_{'-'.join(map(str, hidden_dims))}"
        results[key] = {
            'hidden_dims': hidden_dims,
            'n_params': n_params,
            'best_val_acc': max(history['val_acc']),
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'inference_ms': inference_ms,
            'history': history
        }
        print(f"  {key:25s} | Params: {n_params:7d} | Val Acc: {max(history['val_acc']):.4f} | Inference: {inference_ms:.2f}ms")
    
    print("\n[2] ACTIVATION FUNCTION COMPARISON")
    print("-" * 50)
    base_hidden = [256, 128, 64]
    for act in Config.activations:
        model = FlexibleMLP(Config.n_features, base_hidden, Config.n_classes,
                           activation=act, dropout=0.1)
        n_params = count_parameters(model)
        history = train_model(model, train_loader, val_loader, Config.epochs, Config.lr, Config.weight_decay)
        inference_ms = measure_inference_time(model, (Config.batch_size, Config.n_features))
        
        key = f"act_{act}"
        results[key] = {
            'activation': act,
            'n_params': n_params,
            'best_val_acc': max(history['val_acc']),
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'inference_ms': inference_ms,
            'history': history
        }
        print(f"  {act:10s} | Params: {n_params:7d} | Val Acc: {max(history['val_acc']):.4f} | Inference: {inference_ms:.2f}ms")
    
    print("\n[3] DROPOUT RATE COMPARISON")
    print("-" * 50)
    for dropout in Config.dropout_rates:
        model = FlexibleMLP(Config.n_features, base_hidden, Config.n_classes,
                           activation='relu', dropout=dropout)
        n_params = count_parameters(model)
        history = train_model(model, train_loader, val_loader, Config.epochs, Config.lr, Config.weight_decay)
        inference_ms = measure_inference_time(model, (Config.batch_size, Config.n_features))
        
        key = f"dropout_{dropout}"
        results[key] = {
            'dropout': dropout,
            'n_params': n_params,
            'best_val_acc': max(history['val_acc']),
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'inference_ms': inference_ms,
            'history': history
        }
        print(f"  Dropout={dropout:.1f} | Params: {n_params:7d} | Val Acc: {max(history['val_acc']):.4f} | Inference: {inference_ms:.2f}ms")
    
    print("\n[4] BATCH NORMALIZATION ABLATION")
    print("-" * 50)
    for use_bn in [True, False]:
        model = FlexibleMLP(Config.n_features, base_hidden, Config.n_classes,
                           activation='relu', dropout=0.1, use_batchnorm=use_bn)
        n_params = count_parameters(model)
        history = train_model(model, train_loader, val_loader, Config.epochs, Config.lr, Config.weight_decay)
        inference_ms = measure_inference_time(model, (Config.batch_size, Config.n_features))
        
        key = f"bn_{use_bn}"
        results[key] = {
            'use_batchnorm': use_bn,
            'n_params': n_params,
            'best_val_acc': max(history['val_acc']),
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'inference_ms': inference_ms,
            'history': history
        }
        print(f"  BatchNorm={use_bn} | Params: {n_params:7d} | Val Acc: {max(history['val_acc']):.4f} | Inference: {inference_ms:.2f}ms")
    
    return results, train_loader, val_loader

def analyze_weight_distributions():
    print("\n[5] WEIGHT DISTRIBUTION ANALYSIS")
    print("-" * 50)
    
    X, y = generate_data(1000, Config.n_features, Config.n_classes)
    model = FlexibleMLP(Config.n_features, [256, 128, 64], Config.n_classes,
                       activation='relu', dropout=0.1).to(device)
    
    init_stats = model.get_layer_stats()
    print("  Initial weight statistics:")
    for name, stats in init_stats.items():
        print(f"    {name:30s} mean={stats['mean']:7.4f} std={stats['std']:7.4f} norm={stats['norm']:7.4f}")
    
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.train()
    for epoch in range(10):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    trained_stats = model.get_layer_stats()
    print("\n  After 10 epochs training:")
    for name, stats in trained_stats.items():
        init = init_stats[name]
        print(f"    {name:30s} mean={stats['mean']:7.4f} std={stats['std']:7.4f} "
              f"Δmean={stats['mean']-init['mean']:+7.4f} Δstd={stats['std']-init['std']:+7.4f}")

def gradient_flow_analysis():
    print("\n[6] GRADIENT FLOW ANALYSIS")
    print("-" * 50)
    
    X, y = generate_data(500, Config.n_features, Config.n_classes)
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    for act in ['relu', 'tanh', 'gelu']:
        model = FlexibleMLP(Config.n_features, [256, 128, 64], Config.n_classes,
                           activation=act, dropout=0.0, use_batchnorm=False).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        model.train()
        X_batch, y_batch = next(iter(train_loader))
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        grad_norms = []
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                grad_norms.append((name, param.grad.norm().item()))
        
        print(f"  {act:6s} gradient norms:")
        for name, norm in grad_norms:
            layer_idx = name.split('.')[1] if '.' in name else '?'
            print(f"    Layer {layer_idx:>2s} ({name:30s}): {norm:.6f}")

def visualize_results(results: Dict):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Day 91: Feedforward Neural Network Architecture Analysis', fontsize=14)
    
    hidden_keys = [k for k in results if k.startswith('hidden_')]
    if hidden_keys:
        ax = axes[0, 0]
        names = [k.replace('hidden_', '') for k in hidden_keys]
        val_accs = [results[k]['best_val_acc'] for k in hidden_keys]
        params = [results[k]['n_params'] for k in hidden_keys]
        ax.bar(range(len(names)), val_accs, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Best Val Accuracy')
        ax.set_title('Hidden Layer Configurations')
        ax.grid(axis='y', alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(range(len(names)), params, 'ro-', label='Parameters')
        ax2.set_ylabel('Parameters', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    act_keys = [k for k in results if k.startswith('act_')]
    if act_keys:
        ax = axes[0, 1]
        names = [k.replace('act_', '') for k in act_keys]
        val_accs = [results[k]['best_val_acc'] for k in act_keys]
        ax.bar(names, val_accs, color='coral', alpha=0.7)
        ax.set_ylabel('Best Val Accuracy')
        ax.set_title('Activation Functions')
        ax.grid(axis='y', alpha=0.3)
    
    dropout_keys = [k for k in results if k.startswith('dropout_')]
    if dropout_keys:
        ax = axes[0, 2]
        dropouts = [results[k]['dropout'] for k in dropout_keys]
        val_accs = [results[k]['best_val_acc'] for k in dropout_keys]
        train_accs = [results[k]['final_train_acc'] for k in dropout_keys]
        ax.plot(dropouts, val_accs, 'bo-', label='Val Acc')
        ax.plot(dropouts, train_accs, 'ro-', label='Train Acc')
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('Accuracy')
        ax.set_title('Dropout Rate Effect')
        ax.legend()
        ax.grid(alpha=0.3)
    
    bn_keys = [k for k in results if k.startswith('bn_')]
    if bn_keys:
        ax = axes[1, 0]
        names = ['With BN' if results[k]['use_batchnorm'] else 'Without BN' for k in bn_keys]
        val_accs = [results[k]['best_val_acc'] for k in bn_keys]
        ax.bar(names, val_accs, color=['green', 'orange'], alpha=0.7)
        ax.set_ylabel('Best Val Accuracy')
        ax.set_title('Batch Normalization Ablation')
        ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1, 1]
    for key in hidden_keys:
        history = results[key]['history']
        epochs = range(1, len(history['val_acc']) + 1)
        ax.plot(epochs, history['val_acc'], label=key.replace('hidden_', ''), alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Training Curves (Hidden Configs)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    ax = axes[1, 2]
    for key in act_keys:
        history = results[key]['history']
        epochs = range(1, len(history['val_acc']) + 1)
        ax.plot(epochs, history['val_acc'], label=key.replace('act_', ''), alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Training Curves (Activations)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day91_ann_architecture_results.png', dpi=150, bbox_inches='tight')
    print("\n[7] Visualization saved to 'day91_ann_architecture_results.png'")

def save_results(results: Dict):
    serializable = {}
    for key, val in results.items():
        serializable[key] = {k: v for k, v in val.items() if k != 'history'}
        if 'history' in val:
            serializable[key]['history'] = {k: [float(x) for x in v] for k, v in val['history'].items()}
    
    with open('day91_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print("Results saved to 'day91_results.json'")

def main():
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    results, train_loader, val_loader = run_architecture_experiment()
    analyze_weight_distributions()
    gradient_flow_analysis()
    visualize_results(results)
    save_results(results)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE - KEY FINDINGS:")
    print("=" * 70)
    
    best_hidden = max([k for k in results if k.startswith('hidden_')], 
                      key=lambda k: results[k]['best_val_acc'])
    best_act = max([k for k in results if k.startswith('act_')], 
                   key=lambda k: results[k]['best_val_acc'])
    best_dropout = max([k for k in results if k.startswith('dropout_')], 
                       key=lambda k: results[k]['best_val_acc'])
    
    print(f"  Best hidden config: {best_hidden} -> Val Acc: {results[best_hidden]['best_val_acc']:.4f}")
    print(f"  Best activation:    {best_act} -> Val Acc: {results[best_act]['best_val_acc']:.4f}")
    print(f"  Best dropout:       {best_dropout} -> Val Acc: {results[best_dropout]['best_val_acc']:.4f}")
    print(f"  BatchNorm benefit:  {results['bn_True']['best_val_acc'] - results['bn_False']['best_val_acc']:+.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()