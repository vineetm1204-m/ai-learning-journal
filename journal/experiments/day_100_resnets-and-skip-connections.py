import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# ============================================================
# Day 100: ResNets & Skip Connections - Mini Experiment
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Day 100] Device: {DEVICE}")

# ---- Data ----
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ---- Building Blocks ----
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip:
            out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_skip=True):
        super().__init__()
        self.in_planes = 64
        self.use_skip = use_skip
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, self.use_skip))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18(use_skip=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], use_skip=use_skip)

# ---- Training / Evaluation ----
def train_epoch(model, loader, criterion, optimizer):
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

# ---- Experiment: With vs Without Skip Connections ----
def run_experiment(use_skip, label, epochs=20):
    print(f"\n{'='*60}")
    print(f"[Day 100] Training: {label} (skip={use_skip})")
    print(f"{'='*60}")
    
    model = ResNet18(use_skip=use_skip).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, testloader, criterion)
        scheduler.step()
        epoch_time = time.time() - start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
              f"Test: {test_loss:.4f} / {test_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"[Day 100] {label} Best Test Acc: {best_acc:.2f}%")
    return history, best_acc

# ---- Gradient Flow Analysis ----
def analyze_gradient_flow(model, label):
    model.eval()
    inputs, _ = next(iter(trainloader))
    inputs = inputs[:4].to(DEVICE)
    inputs.requires_grad_(True)
    
    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name and 'bn' not in name:
            grad_norms.append((name, param.grad.norm().item()))
    
    print(f"\n[Day 100] Gradient Norms ({label}):")
    for name, norm in grad_norms[:10]:
        print(f"  {name:40s} : {norm:.6f}")
    if len(grad_norms) > 10:
        print(f"  ... and {len(grad_norms)-10} more layers")
    return grad_norms

# ---- Visualization ----
def plot_results(history_skip, history_no_skip):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(history_skip['train_loss']) + 1)
    
    axes[0,0].plot(epochs, history_skip['train_loss'], 'b-', label='ResNet-18 (skip)')
    axes[0,0].plot(epochs, history_no_skip['train_loss'], 'r-', label='PlainNet-18 (no skip)')
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(epochs, history_skip['test_loss'], 'b-', label='ResNet-18 (skip)')
    axes[0,1].plot(epochs, history_no_skip['test_loss'], 'r-', label='PlainNet-18 (no skip)')
    axes[0,1].set_title('Test Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(epochs, history_skip['train_acc'], 'b-', label='ResNet-18 (skip)')
    axes[1,0].plot(epochs, history_no_skip['train_acc'], 'r-', label='PlainNet-18 (no skip)')
    axes[1,0].set_title('Training Accuracy')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy (%)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(epochs, history_skip['test_acc'], 'b-', label='ResNet-18 (skip)')
    axes[1,1].plot(epochs, history_no_skip['test_acc'], 'r-', label='PlainNet-18 (no skip)')
    axes[1,1].set_title('Test Accuracy')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Accuracy (%)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('Day 100: ResNets vs PlainNets on CIFAR-10', fontsize=14)
    plt.tight_layout()
    plt.savefig('day100_resnet_comparison.png', dpi=150)
    print("\n[Day 100] Plot saved to day100_resnet_comparison.png")
    plt.close()

# ---- Main ----
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DAY 100: RESNETS & SKIP CONNECTIONS - MINI EXPERIMENT")
    print("="*60)
    print("Hypothesis: Skip connections enable training of deeper networks")
    print("            by mitigating vanishing gradients.")
    print("="*60 + "\n")
    
    # Train both models
    history_skip, best_skip = run_experiment(use_skip=True, label="ResNet-18 (with skip)", epochs=20)
    history_no_skip, best_no_skip = run_experiment(use_skip=False, label="PlainNet-18 (no skip)", epochs=20)
    
    # Gradient flow analysis
    print("\n" + "="*60)
    print("[Day 100] GRADIENT FLOW ANALYSIS")
    print("="*60)
    model_skip = ResNet18(use_skip=True).to(DEVICE)
    model_no_skip = ResNet18(use_skip=False).to(DEVICE)
    
    # Quick forward/backward for gradient analysis
    _ = analyze_gradient_flow(model_skip, "ResNet-18 (skip)")
    _ = analyze_gradient_flow(model_no_skip, "PlainNet-18 (no skip)")
    
    # Plot results
    plot_results(history_skip, history_no_skip)
    
    # Summary
    print("\n" + "="*60)
    print("  DAY 100: EXPERIMENT SUMMARY")
    print("="*60)
    print(f"  ResNet-18 (with skip)   Best Test Acc: {best_skip:.2f}%")
    print(f"  PlainNet-18 (no skip)   Best Test Acc: {best_no_skip:.2f}%")
    print(f"  Improvement: {best_skip - best_no_skip:.2f}%")
    print("\n  Key Insight: Skip connections create gradient highways,")
    print("  allowing gradients to flow directly through the network.")
    print("  This enables training of much deeper architectures.")
    print("="*60)