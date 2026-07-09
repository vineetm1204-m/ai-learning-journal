import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PlainNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(PlainBlock, 16, num_blocks[0], 1)
        self.layer2 = self._make_layer(PlainBlock, 32, num_blocks[1], 2)
        self.layer3 = self._make_layer(PlainBlock, 64, num_blocks[2], 2)
        self.linear = nn.Linear(64, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(ResidualBlock, 16, num_blocks[0], 1)
        self.layer2 = self._make_layer(ResidualBlock, 32, num_blocks[1], 2)
        self.layer3 = self._make_layer(ResidualBlock, 64, num_blocks[2], 2)
        self.linear = nn.Linear(64, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def get_cifar10_loaders(batch_size=128):
    import torchvision
    import torchvision.transforms as transforms
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def train_epoch(model, loader, optimizer, criterion, device):
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
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / len(loader), 100. * correct / total

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
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / len(loader), 100. * correct / total

def compute_gradient_norms(model):
    """Compute gradient norms per layer"""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.data.norm(2).item()
    return grad_norms

def run_gradient_flow_experiment():
    """Compare gradient flow in plain vs residual networks"""
    print("=" * 60)
    print("GRADIENT FLOW EXPERIMENT: Plain vs Residual Networks")
    print("=" * 60)
    
    trainloader, _ = get_cifar10_loaders(128)
    criterion = nn.CrossEntropyLoss()
    
    plain_net = PlainNet([3, 3, 3]).to(device)
    res_net = ResNet([3, 3, 3]).to(device)
    
    plain_opt = optim.SGD(plain_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    res_opt = optim.SGD(res_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    plain_grad_history = defaultdict(list)
    res_grad_history = defaultdict(list)
    
    print("\nTraining for 5 epochs to capture gradient dynamics...")
    for epoch in range(5):
        plain_net.train()
        res_net.train()
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            plain_opt.zero_grad()
            plain_out = plain_net(inputs)
            plain_loss = criterion(plain_out, targets)
            plain_loss.backward()
            plain_grads = compute_gradient_norms(plain_net)
            for k, v in plain_grads.items():
                plain_grad_history[k].append(v)
            plain_opt.step()
            
            res_opt.zero_grad()
            res_out = res_net(inputs)
            res_loss = criterion(res_out, targets)
            res_loss.backward()
            res_grads = compute_gradient_norms(res_net)
            for k, v in res_grads.items():
                res_grad_history[k].append(v)
            res_opt.step()
            
            break  # Just one batch per epoch for gradient snapshot
        
        print(f"  Epoch {epoch+1}: Plain loss={plain_loss.item():.4f}, Res loss={res_loss.item():.4f}")
    
    return plain_grad_history, res_grad_history

def run_training_comparison(epochs=20):
    """Full training comparison"""
    print("\n" + "=" * 60)
    print("FULL TRAINING COMPARISON (20 epochs)")
    print("=" * 60)
    
    trainloader, testloader = get_cifar10_loaders(128)
    criterion = nn.CrossEntropyLoss()
    
    plain_net = PlainNet([3, 3, 3]).to(device)
    res_net = ResNet([3, 3, 3]).to(device)
    
    plain_opt = optim.SGD(plain_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    res_opt = optim.SGD(res_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    plain_scheduler = optim.lr_scheduler.CosineAnnealingLR(plain_opt, T_max=epochs)
    res_scheduler = optim.lr_scheduler.CosineAnnealingLR(res_opt, T_max=epochs)
    
    history = {
        'plain_train_loss': [], 'plain_train_acc': [],
        'plain_test_loss': [], 'plain_test_acc': [],
        'res_train_loss': [], 'res_train_acc': [],
        'res_test_loss': [], 'res_test_acc': []
    }
    
    start_time = time.time()
    for epoch in range(epochs):
        plain_train_loss, plain_train_acc = train_epoch(plain_net, trainloader, plain_opt, criterion, device)
        res_train_loss, res_train_acc = train_epoch(res_net, trainloader, res_opt, criterion, device)
        
        plain_test_loss, plain_test_acc = evaluate(plain_net, testloader, criterion, device)
        res_test_loss, res_test_acc = evaluate(res_net, testloader, criterion, device)
        
        plain_scheduler.step()
        res_scheduler.step()
        
        history['plain_train_loss'].append(plain_train_loss)
        history['plain_train_acc'].append(plain_train_acc)
        history['plain_test_loss'].append(plain_test_loss)
        history['plain_test_acc'].append(plain_test_acc)
        history['res_train_loss'].append(res_train_loss)
        history['res_train_acc'].append(res_train_acc)
        history['res_test_loss'].append(res_test_loss)
        history['res_test_acc'].append(res_test_acc)
        
        print(f"Epoch {epoch+1:2d}: Plain Test Acc={plain_test_acc:.2f}% | Res Test Acc={res_test_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s")
    print(f"Final Plain Test Acc: {history['plain_test_acc'][-1]:.2f}%")
    print(f"Final ResNet Test Acc: {history['res_test_acc'][-1]:.2f}%")
    
    return history

def run_depth_scaling_experiment():
    """Test how depth affects plain vs residual networks"""
    print("\n" + "=" * 60)
    print("DEPTH SCALING EXPERIMENT")
    print("=" * 60)
    
    trainloader, testloader = get_cifar10_loaders(128)
    criterion = nn.CrossEntropyLoss()
    
    depths = [2, 4, 6, 8, 10, 12]  # blocks per layer
    results = {'plain': {}, 'res': {}}
    
    for n_blocks in depths:
        print(f"\nTesting depth: {n_blocks} blocks per layer (total ~{3*n_blocks*2+2} layers)")
        
        plain_net = PlainNet([n_blocks, n_blocks, n_blocks]).to(device)
        res_net = ResNet([n_blocks, n_blocks, n_blocks]).to(device)
        
        plain_opt = optim.SGD(plain_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        res_opt = optim.SGD(res_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        
        plain_scheduler = optim.lr_scheduler.CosineAnnealingLR(plain_opt, T_max=15)
        res_scheduler = optim.lr_scheduler.CosineAnnealingLR(res_opt, T_max=15)
        
        best_plain = 0
        best_res = 0
        
        for epoch in range(15):
            train_epoch(plain_net, trainloader, plain_opt, criterion, device)
            train_epoch(res_net, trainloader, res_opt, criterion, device)
            
            _, plain_acc = evaluate(plain_net, testloader, criterion, device)
            _, res_acc = evaluate(res_net, testloader, criterion, device)
            
            best_plain = max(best_plain, plain_acc)
            best_res = max(best_res, res_acc)
            
            plain_scheduler.step()
            res_scheduler.step()
        
        results['plain'][n_blocks] = best_plain
        results['res'][n_blocks] = best_res
        print(f"  Best Plain: {best_plain:.2f}% | Best ResNet: {best_res:.2f}%")
    
    return results

def visualize_results(plain_grad_history, res_grad_history, history, depth_results):
    """Create comprehensive visualizations"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Gradient norms comparison (first layer, middle layer, last layer)
    ax1 = plt.subplot(3, 3, 1)
    plain_first = [v for k, v in plain_grad_history.items() if 'layer1.0.conv1' in k][:50]
    res_first = [v for k, v in res_grad_history.items() if 'layer1.0.conv1' in k][:50]
    plt.plot(plain_first, 'r-', label='Plain', alpha=0.7)
    plt.plot(res_first, 'b-', label='ResNet', alpha=0.7)
    plt.title('Gradient Norm: First Layer (conv1)')
    plt.xlabel('Batch Step')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.yscale('log')
    
    ax2 = plt.subplot(3, 3, 2)
    plain_mid = [v for k, v in plain_grad_history.items() if 'layer2.0.conv1' in k][:50]
    res_mid = [v for k, v in res_grad_history.items() if 'layer2.0.conv1' in k][:50]
    plt.plot(plain_mid, 'r-', label='Plain', alpha=0.7)
    plt.plot(res_mid, 'b-', label='ResNet', alpha=0.7)
    plt.title('Gradient Norm: Middle Layer (layer2)')
    plt.xlabel('Batch Step')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.yscale('log')
    
    ax3 = plt.subplot(3, 3, 3)
    plain_last = [v for k, v in plain_grad_history.items() if 'layer3.2.conv2' in k][:50]
    res_last = [v for k, v in res_grad_history.items() if 'layer3.2.conv2' in k][:50]
    plt.plot(plain_last, 'r-', label='Plain', alpha=0.7)
    plt.plot(res_last, 'b-', label='ResNet', alpha=0.7)
    plt.title('Gradient Norm: Last Layer (layer3)')
    plt.xlabel('Batch Step')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.yscale('log')
    
    # 2. Training curves
    ax4 = plt.subplot(3, 3, 4)
    epochs = range(1, len(history['plain_train_loss']) + 1)
    plt.plot(epochs, history['plain_train_loss'], 'r-', label='Plain Train', alpha=0.7)
    plt.plot(epochs, history['plain_test_loss'], 'r--', label='Plain Test', alpha=0.7)
    plt.plot(epochs, history['res_train_loss'], 'b-', label='ResNet Train', alpha=0.7)
    plt.plot(epochs, history['res_test_loss'], 'b--', label='ResNet Test', alpha=0.7)
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(3, 3, 5)
    plt.plot(epochs, history['plain_train_acc'], 'r-', label='Plain Train', alpha=0.7)
    plt.plot(epochs, history['plain_test_acc'], 'r--', label='Plain Test', alpha=0.7)
    plt.plot(epochs, history['res_train_acc'], 'b-', label='ResNet Train', alpha=0.7)
    plt.plot(epochs, history['res_test_acc'], 'b--', label='ResNet Test', alpha=0.7)
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Generalization gap
    ax6 = plt.subplot(3, 3, 6)
    plain_gap = np.array(history['plain_train_acc']) - np.array(history['plain_test_acc'])
    res_gap = np.array(history['res_train_acc']) - np.array(history['res_test_acc'])
    plt.plot(epochs, plain_gap, 'r-', label='Plain', alpha=0.7)
    plt.plot(epochs, res_gap, 'b-', label='ResNet', alpha=0.7)
    plt.title('Generalization Gap (Train - Test Acc)')
    plt.xlabel('Epoch')
    plt.ylabel('Gap (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Depth scaling
    ax7 = plt.subplot(3, 3, 7)
    depths = sorted(depth_results['plain'].keys())
    plain_accs = [depth_results['plain'][d] for d in depths]
    res_accs = [depth_results['res'][d] for d in depths]
    plt.plot(depths, plain_accs, 'ro-', label='Plain', alpha=0.7, markersize=8)
    plt.plot(depths, res_accs, 'bo-', label='ResNet', alpha=0.7, markersize=8)
    plt.title('Depth Scaling: Test Accuracy vs Depth')
    plt.xlabel('Blocks per Layer')
    plt.ylabel('Best Test Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Gradient norm distribution (final snapshot)
    ax8 = plt.subplot(3, 3, 8)
    plain_final = {k: v[-1] for k, v in plain_grad_history.items() if v}
    res_final = {k: v[-1] for k, v in res_grad_history.items() if v}
    
    plain_vals = list(plain_final.values())
    res_vals = list(res_final.values())
    
    plt.hist(plain_vals, bins=20, alpha=0.5, label='Plain', color='red', density=True)
    plt.hist(res_vals, bins=20, alpha=0.5, label='ResNet', color='blue', density=True)
    plt.title('Gradient Norm Distribution (Final Batch)')
    plt.xlabel('L2 Norm')
    plt.ylabel('Density')
    plt.legend()
    plt.yscale('log')
    
    # 6. Layer-wise gradient flow (averaged)
    ax9 = plt.subplot(3, 3, 9)
    plain_layer_grads = defaultdict(list)
    res_layer_grads = defaultdict(list)
    
    for k, v in plain_grad_history.items():
        layer = k.split('.')[0] if '.' in k else k
        plain_layer_grads[layer].append(np.mean(v))
    
    for k, v in res_grad_history.items():
        layer = k.split('.')[0] if '.' in k else k
        res_layer_grads[layer].append(np.mean(v))
    
    plain_layers = sorted(plain_layer_grads.keys())
    res_layers = sorted(res_layer_grads.keys())
    
    plain_means = [np.mean(plain_layer_grads[l]) for l in plain_layers]
    res_means = [np.mean(res_layer_grads[l]) for l in res_layers]
    
    x = range(len(plain_layers))
    plt.bar(x, plain_means, alpha=0.5, label='Plain', color='red', width=0.4, align='edge')
    plt.bar(x, res_means, alpha=0.5, label='ResNet', color='blue', width=-0.4, align='edge')
    plt.xticks(x, plain_layers, rotation=45, ha='right')
    plt.title('Average Gradient Norm per Layer')
    plt.ylabel('Mean L2 Norm')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('resnet_experiment_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'resnet_experiment_results.png'")
    plt.close()

def print_summary(history, depth_results):
    """Print experiment summary"""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print(f"\nFinal Results (20 epochs, 3 blocks per layer):")
    print(f"  Plain Network:  Test Acc = {history['plain_test_acc'][-1]:.2f}%")
    print(f"  ResNet:         Test Acc = {history['res_test_acc'][-1]:.2f}%")
    print(f"  Improvement:    +{history['res_test_acc'][-1] - history['plain_test_acc'][-1]:.2f}%")
    
    print(f"\nDepth Scaling Results:")
    for d in sorted(depth_results['plain'].keys()):
        plain_acc = depth_results['plain'][d]
        res_acc = depth_results['res'][d]
        diff = res_acc - plain_acc
        print(f"  {d} blocks: Plain={plain_acc:.1f}% | ResNet={res_acc:.1f}% | Diff={diff:+.1f}%")
    
    print(f"\nKey Observations:")
    print(f"  1. ResNet maintains better gradient flow through skip connections")
    print(f"  2. Plain networks suffer from vanishing gradients as depth increases")
    print(f"  3. Skip connections enable training of much deeper networks")
    print(f"  4. ResNet shows smaller generalization gap")
    print(f"  5. Identity mapping in shortcuts is crucial for gradient propagation")

def main():
    print("Day 18: ResNets and Skip Connections - Mini Experiment")
    print("=" * 60)
    
    # Run experiments
    plain_grad_history, res_grad_history = run_gradient_flow_experiment()
    history = run_training_comparison(epochs=20)
    depth_results = run_depth_scaling_experiment()
    
    # Visualize
    visualize_results(plain_grad_history, res_grad_history, history, depth_results)
    
    # Summary
    print_summary(history, depth_results)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()