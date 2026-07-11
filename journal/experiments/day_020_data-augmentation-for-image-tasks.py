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
import time

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 1. Data Augmentation Pipelines
def get_transforms(augmentation_level="none"):
    """Return train/test transforms based on augmentation level."""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    if augmentation_level == "none":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    elif augmentation_level == "basic":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif augmentation_level == "strong":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform

# 2. Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3. Training & Evaluation
def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            rand_index = torch.randperm(inputs.size(0)).to(device)
            target_a, target_b = targets, targets[rand_index]
            inputs = lam * inputs + (1 - lam) * inputs[rand_index]
            outputs = model(inputs)
            loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / total, 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
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

# 4. Visualization
def imshow_tensor(img_tensor, title=None):
    """Denormalize and show a tensor image."""
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title: plt.title(title)
    plt.axis('off')

def visualize_augmentations(dataset, indices, transform, n_rows=2, n_cols=5):
    """Show original vs augmented versions."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        if i >= n_rows * n_cols: break
        img, label = dataset[idx]
        # Apply transform (assuming it's a Compose ending with ToTensor)
        # We need to handle the fact that dataset might already return tensor
        if isinstance(img, torch.Tensor):
            # If dataset returns tensor, we can't easily re-apply PIL transforms.
            # For visualization, we assume dataset returns PIL or we use a raw dataset.
            pass 
    plt.tight_layout()
    plt.show()

def show_augmentation_examples():
    """Load raw CIFAR10 (PIL) and show augmentation effects."""
    raw_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    classes = raw_train.classes
    
    _, basic_tf = get_transforms("basic")
    _, strong_tf = get_transforms("strong")
    # Remove Normalize and ToTensor for visualization
    basic_vis = transforms.Compose([t for t in basic_tf.transforms if not isinstance(t, (transforms.ToTensor, transforms.Normalize))])
    strong_vis = transforms.Compose([t for t in strong_tf.transforms if not isinstance(t, (transforms.ToTensor, transforms.Normalize, transforms.RandomErasing))])

    indices = np.random.choice(len(raw_train), 5, replace=False)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for col, idx in enumerate(indices):
        img, label = raw_train[idx]
        # Original
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"Original: {classes[label]}")
        axes[0, col].axis('off')
        # Basic
        axes[1, col].imshow(basic_vis(img))
        axes[1, col].set_title("Basic Aug")
        axes[1, col].axis('off')
        # Strong
        axes[2, col].imshow(strong_vis(img))
        axes[2, col].set_title("Strong Aug")
        axes[2, col].axis('off')
    plt.suptitle("Data Augmentation Comparison (CIFAR-10)", fontsize=16)
    plt.tight_layout()
    plt.savefig("augmentation_examples.png", dpi=150)
    print("Saved augmentation_examples.png")
    plt.close()

# 5. Main Experiment Runner
def run_experiment(aug_level, epochs=10, batch_size=128, lr=0.001, mixup=0.0, tag=""):
    print(f"\n{'='*20} Running: {aug_level.upper()} {tag} {'='*20}")
    train_tf, test_tf = get_transforms(aug_level)
    
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
    
    # Use subset for speed in demo (optional: comment out for full)
    # train_set = Subset(train_set, range(10000))
    # test_set = Subset(test_set, range(2000))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, mixup)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"best_model_{aug_level}{tag}.pth")
        
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
              f"Best: {best_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"Finished in {total_time:.1f}s. Best Test Acc: {best_acc:.2f}%")
    return history, best_acc

def plot_histories(histories, labels, save_path="results_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for hist, label in zip(histories, labels):
        axes[0].plot(hist['train_acc'], label=f'{label} Train', linestyle='--')
        axes[0].plot(hist['test_acc'], label=f'{label} Test', linestyle='-')
        axes[1].plot(hist['train_loss'], label=f'{label} Train', linestyle='--')
        axes[1].plot(hist['test_loss'], label=f'{label} Test', linestyle='-')
    
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Acc (%)"); axes[0].legend(); axes[0].grid(True)
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend(); axes[1].grid(True)
    plt.suptitle("Data Augmentation Impact on CIFAR-10 (SimpleCNN)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.close()

# 6. Entry Point
if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    
    # 1. Visualize Augmentations
    print("Generating augmentation visualization...")
    show_augmentation_examples()
    
    # 2. Run Experiments
    configs = [
        ("none", "No Aug"),
        ("basic", "Basic Aug"),
        ("strong", "Strong Aug"),
        ("strong", "Strong Aug + Mixup", {"mixup": 1.0}),
    ]
    
    histories = []
    labels = []
    
    for config in configs:
        if len(config) == 2:
            level, label = config
            extra = {}
        else:
            level, label, extra = config
        
        hist, best = run_experiment(level, epochs=15, **extra, tag=f"_{label.replace(' ', '_')}")
        histories.append(hist)
        labels.append(label)
    
    # 3. Plot & Summary
    plot_histories(histories, labels)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY (Best Test Accuracy)")
    print("="*60)
    for label, hist in zip(labels, histories):
        print(f"{label:<25}: {max(hist['test_acc']):.2f}%")
    print("="*60)