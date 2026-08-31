import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# 1. Define Augmentation Pipelines
# ============================================================

# Basic normalization for CIFAR-10
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

# No augmentation (baseline)
transform_none = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# Light augmentation
transform_light = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normalize
])

# Strong augmentation (AutoAugment-style)
transform_strong = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    normalize
])

# Test transform (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# ============================================================
# 2. Simple CNN Model
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================
# 3. Training & Evaluation Functions
# ============================================================

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
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

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
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================================
# 4. Visualization Utilities
# ============================================================

def imshow_tensor(tensor, title=None):
    """Display a tensor image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    # Denormalize
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')

def visualize_augmentations(dataset, transform, num_samples=8, title="Augmentations"):
    """Show original vs augmented samples."""
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 4))
    fig.suptitle(title, fontsize=14)
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        
        # Original (using test transform)
        orig_img = transform_test(dataset.data[idx])
        imshow_tensor(orig_img)
        axes[0, i].set_title(f"Class: {dataset.classes[label]}")
        
        # Augmented
        aug_img = transform(dataset.data[idx])
        imshow_tensor(aug_img)
        axes[1, i].set_title("Augmented")
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {title.lower().replace(' ', '_')}.png")

# ============================================================
# 5. Main Experiment
# ============================================================

def run_experiment(name, train_transform, epochs=20, batch_size=128, lr=0.001):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Use subset for faster experimentation (optional: comment out for full dataset)
    # train_indices = list(range(10000))
    # train_dataset = Subset(train_dataset, train_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_test_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'best_model_{name.lower().replace(" ", "_")}.pth')
        
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
              f"Best: {best_test_acc:.2f}%")
    
    print(f"\nBest Test Accuracy for {name}: {best_test_acc:.2f}%")
    return history, best_test_acc

def plot_comparison(histories, names):
    """Plot training curves for comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Data Augmentation Comparison on CIFAR-10', fontsize=14)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for (name, history), color in zip(zip(names, histories), colors):
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, history['train_loss'], label=name, color=color)
        axes[0, 1].plot(epochs, history['train_acc'], label=name, color=color)
        axes[1, 0].plot(epochs, history['test_loss'], label=name, color=color)
        axes[1, 1].plot(epochs, history['test_acc'], label=name, color=color)
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Test Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('augmentation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved comparison plot: augmentation_comparison.png")

# ============================================================
# 6. Run All Experiments
# ============================================================

if __name__ == "__main__":
    # Create data directory
    os.makedirs('./data', exist_ok=True)
    
    # Load base dataset for visualization
    base_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    
    # Visualize augmentations
    print("Generating augmentation visualizations...")
    visualize_augmentations(base_dataset, transform_light, title="Light Augmentation")
    visualize_augmentations(base_dataset, transform_strong, title="Strong Augmentation")
    
    # Run experiments
    print("\nStarting experiments...")
    
    # Experiment 1: No augmentation
    hist_none, best_none = run_experiment("No Augmentation", transform_none, epochs=20)
    
    # Experiment 2: Light augmentation
    hist_light, best_light = run_experiment("Light Augmentation", transform_light, epochs=20)
    
    # Experiment 3: Strong augmentation
    hist_strong, best_strong = run_experiment("Strong Augmentation", transform_strong, epochs=20)
    
    # Plot comparison
    plot_comparison(
        [hist_none, hist_light, hist_strong],
        ["No Augmentation", "Light Augmentation", "Strong Augmentation"]
    )
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<25} {'Best Test Acc':<15} {'Improvement'}")
    print("-"*60)
    print(f"{'No Augmentation':<25} {best_none:<15.2f} {'--'}")
    print(f"{'Light Augmentation':<25} {best_light:<15.2f} {best_light - best_none:+.2f}%")
    print(f"{'Strong Augmentation':<25} {best_strong:<15.2f} {best_strong - best_none:+.2f}%")
    print("="*60)
    
    # Save results
    results = {
        'no_aug': {'history': hist_none, 'best_acc': best_none},
        'light_aug': {'history': hist_light, 'best_acc': best_light},
        'strong_aug': {'history': hist_strong, 'best_acc': best_strong}
    }
    torch.save(results, 'augmentation_results.pth')
    print("\nResults saved to augmentation_results.pth")