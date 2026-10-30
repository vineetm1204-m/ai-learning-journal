import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import numpy as np

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Config
BATCH_SIZE = 64
EPOCHS_FE = 5      # Feature extraction epochs
EPOCHS_FT = 5      # Fine-tuning epochs
LR_FE = 1e-3       # LR for classifier only
LR_FT = 1e-4       # LR for fine-tuning (lower)
NUM_CLASSES = 10   # CIFAR-10
NUM_WORKERS = 2
DATA_ROOT = "./data"

# Data transforms - resize to 224 for ResNet
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-10 (download if needed)
print("Loading CIFAR-10...")
train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_transform)

# Use subset for speed (5000 train, 1000 test)
train_indices = torch.randperm(len(train_dataset))[:5000]
test_indices = torch.randperm(len(test_dataset))[:1000]
train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train samples: {len(train_subset)}, Test samples: {len(test_subset)}")

def build_model(pretrained=True, freeze_backbone=False):
    """Build ResNet18 with custom classifier head."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    
    return model.to(DEVICE)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

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
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def run_experiment(name, model, train_loader, test_loader, epochs, lr, freeze_backbone=False):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Trainable params: {count_trainable_params(model):,}")
    print(f"Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train: {train_loss:.4f} ({train_acc:.2f}%) | "
              f"Test: {test_loss:.4f} ({test_acc:.2f}%) | "
              f"Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    best_test_acc = max(history["test_acc"])
    print(f"\n{name} Summary: Best Test Acc: {best_test_acc:.2f}% | Total Time: {total_time:.1f}s")
    return history, best_test_acc

# ============================================================
# EXPERIMENT 1: Feature Extraction (Frozen Backbone)
# ============================================================
print("\n" + "#"*60)
print("# EXPERIMENT 1: FEATURE EXTRACTION (FROZEN BACKBONE)")
print("#"*60)

model_fe = build_model(pretrained=True, freeze_backbone=True)
history_fe, best_fe = run_experiment(
    "Feature Extraction", model_fe, train_loader, test_loader,
    epochs=EPOCHS_FE, lr=LR_FE, freeze_backbone=True
)

# ============================================================
# EXPERIMENT 2: Fine-Tuning (Unfrozen Backbone)
# ============================================================
print("\n" + "#"*60)
print("# EXPERIMENT 2: FINE-TUNING (UNFROZEN BACKBONE)")
print("#"*60)

model_ft = build_model(pretrained=True, freeze_backbone=False)
history_ft, best_ft = run_experiment(
    "Fine-Tuning", model_ft, train_loader, test_loader,
    epochs=EPOCHS_FT, lr=LR_FT, freeze_backbone=False
)

# ============================================================
# EXPERIMENT 3: Training from Scratch (Baseline)
# ============================================================
print("\n" + "#"*60)
print("# EXPERIMENT 3: TRAINING FROM SCRATCH (NO PRETRAINING)")
print("#"*60)

model_scratch = build_model(pretrained=False, freeze_backbone=False)
history_scratch, best_scratch = run_experiment(
    "From Scratch", model_scratch, train_loader, test_loader,
    epochs=EPOCHS_FT, lr=LR_FT, freeze_backbone=False
)

# ============================================================
# FINAL COMPARISON
# ============================================================
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"{'Method':<25} {'Best Test Acc':>15} {'Trainable Params':>18}")
print("-"*60)
print(f"{'Feature Extraction':<25} {best_fe:>14.2f}% {count_trainable_params(model_fe):>18,}")
print(f"{'Fine-Tuning':<25} {best_ft:>14.2f}% {count_trainable_params(model_ft):>18,}")
print(f"{'From Scratch':<25} {best_scratch:>14.2f}% {count_trainable_params(model_scratch):>18,}")
print("-"*60)

improvement_ft_vs_fe = best_ft - best_fe
improvement_ft_vs_scratch = best_ft - best_scratch
print(f"\nFine-tuning vs Feature Extraction: {improvement_ft_vs_fe:+.2f}%")
print(f"Fine-tuning vs From Scratch:       {improvement_ft_vs_scratch:+.2f}%")

# Key takeaways
print("\n" + "="*60)
print("KEY TAKEAWAYS")
print("="*60)
print("1. Feature extraction: Fast, fewer params, good baseline")
print("2. Fine-tuning: Higher capacity, adapts features to target domain")
print("3. From scratch: Needs more data/epochs, no prior knowledge")
print("4. Use lower LR for fine-tuning to preserve pretrained features")
print("5. Freeze early layers, unfreeze later layers for large domain shift")