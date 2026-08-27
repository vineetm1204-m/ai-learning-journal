import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def create_synthetic_data(n_samples=5000, img_size=32, num_classes=10):
    X = torch.randn(n_samples, 3, img_size, img_size)
    y = torch.randint(0, num_classes, (n_samples,))
    return X, y

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100. * correct / total)
        
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(100. * correct / total)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} Acc: {train_accs[-1]:.2f}% | Test Loss: {test_losses[-1]:.4f} Acc: {test_accs[-1]:.2f}%")
    
    return train_losses, train_accs, test_losses, test_accs

def plot_results(resnet_history, plain_history, title="ResNet vs Plain Network"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(resnet_history[0]) + 1)
    
    axes[0, 0].plot(epochs, resnet_history[0], 'b-', label='ResNet Train')
    axes[0, 0].plot(epochs, plain_history[0], 'r-', label='Plain Train')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, resnet_history[2], 'b-', label='ResNet Test')
    axes[0, 1].plot(epochs, plain_history[2], 'r-', label='Plain Test')
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, resnet_history[1], 'b-', label='ResNet Train')
    axes[1, 0].plot(epochs, plain_history[1], 'r-', label='Plain Train')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, resnet_history[3], 'b-', label='ResNet Test')
    axes[1, 1].plot(epochs, plain_history[3], 'r-', label='Plain Test')
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('resnet_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_gradient_flow(model, input_tensor):
    model.eval()
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    target = torch.zeros_like(output)
    target[0, 0] = 1.0
    loss = F.mse_loss(output, target)
    loss.backward()
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            grad_norms.append((name, param.grad.norm().item()))
    return grad_norms

def main():
    print(f"Device: {device}")
    print("=" * 60)
    print("Day 59: ResNets and Skip Connections Mini-Experiment")
    print("=" * 60)
    
    X_train, y_train = create_synthetic_data(4000)
    X_test, y_test = create_synthetic_data(1000)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("\n1. Training ResNet-20 (with skip connections)...")
    resnet = ResNet(ResidualBlock, [3, 3, 3])
    resnet_history = train_model(resnet, train_loader, test_loader, epochs=15)
    
    print("\n2. Training Plain-20 (without skip connections)...")
    plain_net = ResNet(PlainBlock, [3, 3, 3])
    plain_history = train_model(plain_net, train_loader, test_loader, epochs=15)
    
    print("\n3. Analyzing gradient flow...")
    sample_input = torch.randn(1, 3, 32, 32).to(device)
    resnet_grads = analyze_gradient_flow(resnet, sample_input.clone())
    plain_grads = analyze_gradient_flow(plain_net, sample_input.clone())
    
    print("\nGradient norms (ResNet):")
    for name, norm in resnet_grads[:5]:
        print(f"  {name}: {norm:.6f}")
    print("\nGradient norms (Plain):")
    for name, norm in plain_grads[:5]:
        print(f"  {name}: {norm:.6f}")
    
    print("\n4. Plotting results...")
    plot_results(resnet_history, plain_history)
    
    print("\n5. Summary:")
    print(f"  ResNet Final Test Acc: {resnet_history[3][-1]:.2f}%")
    print(f"  PlainNet Final Test Acc: {plain_history[3][-1]:.2f}%")
    print(f"  Improvement: {resnet_history[3][-1] - plain_history[3][-1]:.2f}%")
    print("\nKey Insight: Skip connections enable gradient flow through identity mapping,")
    print("mitigating vanishing gradients and allowing deeper networks to train effectively.")

if __name__ == "__main__":
    main()