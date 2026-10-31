import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
DATA_ROOT = "./data"
PLOT_DIR = "./augmentation_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# AUGMENTATION PIPELINES
# ============================================================
# Base normalization for CIFAR-10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

# 1. No augmentation (baseline)
transform_none = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# 2. Standard augmentations
transform_standard = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# 3. Strong augmentations (AutoAugment + Cutout)
transform_strong = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

# 4. MixUp / CutMix will be applied in the training loop

# ============================================================
# DATA LOADING
# ============================================================
def get_dataloaders(transform_train, transform_test, batch_size=BATCH_SIZE):
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader

# ============================================================
# MODEL: Simple CNN for CIFAR-10
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================
# MIXUP / CUTMIX IMPLEMENTATIONS
# ============================================================
def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    y_a, y_b = y, y[index]
    
    # Generate random box
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y_a, y_b, lam

# ============================================================
# TRAINING / EVALUATION
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, aug_mode="none"):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if aug_mode == "mixup":
            x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
            logits = model(x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        elif aug_mode == "cutmix":
            x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
            logits = model(x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(x)
            loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        _, pred = logits.max(1)
        if aug_mode in ("mixup", "cutmix"):
            # For mixed targets, compute accuracy against y_a (approximate)
            correct += pred.eq(y_a).sum().item()
        else:
            correct += pred.eq(y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total

# ============================================================
# VISUALIZATION UTILITIES
# ============================================================
def denormalize(tensor, mean=CIFAR_MEAN, std=CIFAR_STD):
    """Denormalize a tensor image for visualization"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def show_augmentation_examples():
    """Visualize different augmentation pipelines on the same images"""
    # Get a few samples from test set (no augmentation)
    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transforms.ToTensor())
    indices = list(range(8))
    subset = Subset(test_set, indices)
    loader = DataLoader(subset, batch_size=8, shuffle=False)
    images, labels = next(iter(loader))
    
    class_names = test_set.classes
    
    # Apply each transform pipeline
    pipelines = {
        "Original": transforms.Compose([transforms.ToTensor()]),
        "Standard": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]),
        "Strong (AutoAugment)": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
        ]),
        "Strong + Erasing": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ]),
    }
    
    fig, axes = plt.subplots(len(pipelines), 8, figsize=(16, 4 * len(pipelines)))
    for row_idx, (name, pipeline) in enumerate(pipelines.items()):
        for i in range(8):
            img = images[i]
            # Convert to PIL for torchvision transforms
            img_pil = transforms.ToPILImage()(img)
            aug_img = pipeline(img_pil)
            aug_img = denormalize(aug_img).clamp(0, 1)
            aug_img = aug_img.permute(1, 2, 0).numpy()
            
            ax = axes[row_idx, i]
            ax.imshow(aug_img)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(name, fontsize=12, rotation=90, labelpad=20)
            if row_idx == 0:
                ax.set_title(f"Class: {class_names[labels[i]]}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "augmentation_examples.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved augmentation examples to {PLOT_DIR}/augmentation_examples.png")

def plot_training_curves(histories, labels):
    """Plot training curves for multiple experiments"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for history, label in zip(histories, labels):
        epochs = range(1, len(history["train_acc"]) + 1)
        axes[0].plot(epochs, history["train_acc"], label=f"{label} (train)")
        axes[0].plot(epochs, history["test_acc"], label=f"{label} (test)", linestyle="--")
        axes[1].plot(epochs, history["train_loss"], label=f"{label} (train)")
        axes[1].plot(epochs, history["test_loss"], label=f"{label} (test)", linestyle="--")
    
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {PLOT_DIR}/training_curves.png")

# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================
def run_experiment(name, transform_train, aug_mode="none", epochs=EPOCHS):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")
    
    train_loader, test_loader = get_dataloaders(transform_train, transform_none)
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if aug_mode in ("mixup", "cutmix") else 0.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, aug_mode)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    return history, model

def main():
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    
    # 1. Visualize augmentations
    print("\nGenerating augmentation visualization...")
    show_augmentation_examples()
    
    # 2. Run experiments
    experiments = [
        ("No Augmentation", transform_none, "none"),
        ("Standard Aug", transform_standard, "none"),
        ("Strong Aug (AutoAugment)", transform_strong, "none"),
        ("Standard + MixUp", transform_standard, "mixup"),
        ("Standard + CutMix", transform_standard, "cutmix"),
    ]
    
    histories = []
    labels = []
    
    for name, transform, mode in experiments:
        history, _ = run_experiment(name, transform, mode)
        histories.append(history)
        labels.append(name)
    
    # 3. Plot results
    plot_training_curves(histories, labels)
    
    # 4. Summary table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<30} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8}")
    print("-"*80)
    for label, history in zip(labels, histories):
        train_acc = history["train_acc"][-1]
        test_acc = history["test_acc"][-1]
        gap = train_acc - test_acc
        print(f"{label:<30} {train_acc:>10.4f} {test_acc:>10.4f} {gap:>8.4f}")
    
    # Best model
    best_idx = np.argmax([h["test_acc"][-1] for h in histories])
    print(f"\nBest: {labels[best_idx]} with Test Acc = {histories[best_idx]['test_acc'][-1]:.4f}")

if __name__ == "__main__":
    main()