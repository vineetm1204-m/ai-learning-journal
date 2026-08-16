import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        self.activation = activations.get(activation, nn.ReLU())
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = self.activation(x)
                x = self.dropout(x)
        x = self.output_layer(x)
        return x

def generate_data(n_samples=2000, n_features=20, n_classes=3, n_informative=10):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=1.5,
        random_state=42
    )
    return X, y

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    return history, best_val_acc

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train Acc', color='blue')
    axes[1].plot(history['val_acc'], label='Val Acc', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day50_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_architecture_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    architectures = list(results.keys())
    val_accs = [results[arch]['val_acc'] for arch in architectures]
    param_counts = [results[arch]['params'] for arch in architectures]
    
    axes[0].bar(range(len(architectures)), val_accs, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(len(architectures)))
    axes[0].set_xticklabels(architectures, rotation=45, ha='right')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Architecture Comparison: Validation Accuracy')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(range(len(architectures)), param_counts, color='coral', alpha=0.7)
    axes[1].set_xticks(range(len(architectures)))
    axes[1].set_xticklabels(architectures, rotation=45, ha='right')
    axes[1].set_ylabel('Parameter Count')
    axes[1].set_title('Architecture Comparison: Model Size')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('day50_architecture_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)
    print("DAY 50: Feedforward Neural Network Architecture Experiment")
    print("=" * 60)
    
    X, y = generate_data(n_samples=3000, n_features=30, n_classes=4, n_informative=15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test)
    )
    
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X.shape[1]
    output_dim = len(np.unique(y))
    
    architectures = {
        'Shallow (64)': [64],
        'Medium (128, 64)': [128, 64],
        'Deep (256, 128, 64)': [256, 128, 64],
        'Wide (512, 256)': [512, 256],
        'Bottleneck (256, 32, 256)': [256, 32, 256],
    }
    
    results = {}
    
    for name, hidden_dims in architectures.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")
        print(f"{'='*60}")
        
        model = FeedforwardNN(input_dim, hidden_dims, output_dim, dropout_rate=0.3, activation='relu').to(device)
        params = count_parameters(model)
        print(f"Trainable parameters: {params:,}")
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
        
        history, best_val_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            epochs=100, device=device
        )
        
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(targets).sum().item()
                test_total += targets.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_acc = test_correct / test_total
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(classification_report(all_targets, all_preds, digits=4))
        
        results[name] = {
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'params': params,
            'history': history
        }
        
        if name == 'Deep (256, 128, 64)':
            plot_history(history)
    
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Architecture':<25} {'Val Acc':>10} {'Test Acc':>10} {'Params':>12}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<25} {res['val_acc']:>10.4f} {res['test_acc']:>10.4f} {res['params']:>12,}")
    
    plot_architecture_comparison(results)
    
    print("\nExperiment complete. Plots saved:")
    print("  - day50_training_curves.png")
    print("  - day50_architecture_comparison.png")

if __name__ == '__main__':
    main()