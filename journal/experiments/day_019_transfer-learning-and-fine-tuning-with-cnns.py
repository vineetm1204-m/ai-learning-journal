import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import time
import copy
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR_FEATURE_EXTRACT = 1e-3
LR_FINE_TUNE = 1e-4
LR_SCRATCH = 1e-3
NUM_WORKERS = 2
SEED = 42
DATA_ROOT = "./data"

torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DATA PREPARATION (CIFAR-10, resized to 224x224 for ResNet)
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_train = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
full_test = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_transform)

# Use subset for speed (5000 train, 1000 test)
train_indices = torch.randperm(len(full_train))[:5000]
test_indices = torch.randperm(len(full_test))[:1000]
train_set = Subset(full_train, train_indices)
test_set = Subset(full_test, test_indices)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

CLASSES = full_train.classes
NUM_CLASSES = len(CLASSES)

# ============================================================
# MODEL DEFINITIONS
# ============================================================
class SmallCNN(nn.Module):
    """Train-from-scratch baseline."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_resnet18_feature_extractor(num_classes):
    """ResNet18 with frozen backbone, new head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet18_fine_tune(num_classes, unfreeze_blocks=2):
    """ResNet18 with last N residual blocks unfrozen."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze all first
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last `unfreeze_blocks` layer groups (layer4, layer3, ...)
    layer_names = ['layer4', 'layer3', 'layer2', 'layer1']
    for name in layer_names[:unfreeze_blocks]:
        for param in getattr(model, name).parameters():
            param.requires_grad = True
    # Always train the new head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================================================
# TRAINING / EVALUATION UTILITIES
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
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(name, model, train_loader, test_loader, lr, epochs, device):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"Trainable params: {count_trainable_params(model):,}")
    print(f"{'='*60}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train: {tr_loss:.4f} / {tr_acc:.4f} | "
              f"Test: {te_loss:.4f} / {te_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

    print(f"Best test accuracy: {best_acc:.4f}")
    return history, best_acc, best_state


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")
    print(f"Classes: {CLASSES}")

    results = {}

    # 1. Train from scratch (Small CNN)
    print("\n\n" + "#"*60)
    print("# 1. TRAIN FROM SCRATCH (Small CNN)")
    print("#"*60)
    model_scratch = SmallCNN(NUM_CLASSES)
    hist_scratch, best_scratch, _ = run_experiment(
        "Scratch (Small CNN)", model_scratch, train_loader, test_loader,
        LR_SCRATCH, NUM_EPOCHS, DEVICE
    )
    results["Scratch"] = best_scratch

    # 2. Feature Extraction (Frozen ResNet18)
    print("\n\n" + "#"*60)
    print("# 2. FEATURE EXTRACTION (Frozen ResNet18)")
    print("#"*60)
    model_fe = get_resnet18_feature_extractor(NUM_CLASSES)
    hist_fe, best_fe, _ = run_experiment(
        "Feature Extraction", model_fe, train_loader, test_loader,
        LR_FEATURE_EXTRACT, NUM_EPOCHS, DEVICE
    )
    results["Feature Extraction"] = best_fe

    # 3. Fine-tuning (Last 2 blocks unfrozen)
    print("\n\n" + "#"*60)
    print("# 3. FINE-TUNING (ResNet18, last 2 blocks unfrozen)")
    print("#"*60)
    model_ft = get_resnet18_fine_tune(NUM_CLASSES, unfreeze_blocks=2)
    hist_ft, best_ft, _ = run_experiment(
        "Fine-tuning (2 blocks)", model_ft, train_loader, test_loader,
        LR_FINE_TUNE, NUM_EPOCHS, DEVICE
    )
    results["Fine-tuning (2 blocks)"] = best_ft

    # 4. Fine-tuning (Last 3 blocks unfrozen)
    print("\n\n" + "#"*60)
    print("# 4. FINE-TUNING (ResNet18, last 3 blocks unfrozen)")
    print("#"*60)
    model_ft3 = get_resnet18_fine_tune(NUM_CLASSES, unfreeze_blocks=3)
    hist_ft3, best_ft3, _ = run_experiment(
        "Fine-tuning (3 blocks)", model_ft3, train_loader, test_loader,
        LR_FINE_TUNE, NUM_EPOCHS, DEVICE
    )
    results["Fine-tuning (3 blocks)"] = best_ft3

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for name, acc in results.items():
        print(f"{name:30s} : {acc:.4f}")

    print("\nKey Takeaways:")
    print("- Feature extraction leverages pre-trained features without updating backbone.")
    print("- Fine-tuning adapts higher-level features to the target domain.")
    print("- More unfrozen blocks = more capacity but risk of overfitting on small data.")
    print("- Pre-trained models significantly outperform training from scratch on small datasets.")