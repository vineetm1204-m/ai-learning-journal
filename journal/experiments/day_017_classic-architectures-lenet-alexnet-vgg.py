import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# ============================================================
# 1. ARCHITECTURE DEFINITIONS
# ============================================================

class LeNet5(nn.Module):
    """LeNet-5 adapted for 32x32 input (e.g., CIFAR-10) or 28x28 (MNIST).
    Original: 1x32x32 -> C1(6@28x28) -> S2(6@14x14) -> C3(16@10x10) -> S4(16@5x5) -> C5(120) -> F6(84) -> Out(10)
    """
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            # C1
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2), # 32x32 -> 32x32 (if pad=2) or 28x28 (pad=0)
            nn.Tanh(),
            # S2
            nn.AvgPool2d(kernel_size=2, stride=2), # -> 16x16 (or 14x14)
            # C3
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            # S4
            nn.AvgPool2d(kernel_size=2, stride=2), # -> 5x5 (or 6x6 -> 5x5 requires specific input)
        )
        # Calculate flattened size dynamically for flexibility
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 32, 32)
            n_flat = self.features(dummy).view(1, -1).shape[1]
        
        self.classifier = nn.Sequential(
            # C5
            nn.Linear(n_flat, 120),
            nn.Tanh(),
            # F6
            nn.Linear(120, 84),
            nn.Tanh(),
            # Output
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    """AlexNet adapted for 32x32 input (CIFAR-10 style) instead of 224x224.
    Original requires 224x224. For 32x32, we reduce kernel/stride in first layer and remove some pooling.
    """
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            # Input 3x32x32
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), # -> 64x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 64x16x16
            
            nn.Conv2d(64, 192, kernel_size=3, padding=1), # -> 192x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 192x8x8
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # -> 384x8x8
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # -> 256x8x8
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # -> 256x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 256x4x4
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 32, 32)
            n_flat = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_flat, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_vgg_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name='VGG16', num_classes=10, in_channels=3, batch_norm=False):
        super().__init__()
        self.features = make_vgg_layers(cfgs[vgg_name], in_channels, batch_norm)
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 32, 32)
            n_flat = self.features(dummy).view(1, -1).shape[1]
            
        self.classifier = nn.Sequential(
            nn.Linear(n_flat, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================
# 2. UTILITIES
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_forward_backward(model, input_tensor, target_tensor, criterion, optimizer, device):
    model.train()
    input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
    
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()

# ============================================================
# 3. MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 17: CLASSIC ARCHITECTURES MINI-EXPERIMENT")
    print("LeNet-5, AlexNet, VGG11/13/16/19 (CIFAR-10 Scale)")
    print("=" * 60)

    # Config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    IMG_SIZE = 32
    NUM_CLASSES = 10
    IN_CHANNELS = 3 # CIFAR-10 is 3-channel
    
    print(f"Device: {DEVICE}")
    print(f"Input Shape: [{BATCH_SIZE}, {IN_CHANNELS}, {IMG_SIZE}, {IMG_SIZE}]")
    print("-" * 60)

    # Synthetic Data (Self-contained, no downloads)
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    dummy_target = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    criterion = nn.CrossEntropyLoss()

    models_to_test = {
        "LeNet-5": LeNet5(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS),
        "AlexNet": AlexNet(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS),
        "VGG11": VGG('VGG11', num_classes=NUM_CLASSES, in_channels=IN_CHANNELS),
        "VGG13": VGG('VGG13', num_classes=NUM_CLASSES, in_channels=IN_CHANNELS),
        "VGG16": VGG('VGG16', num_classes=NUM_CLASSES, in_channels=IN_CHANNELS),
        "VGG19": VGG('VGG19', num_classes=NUM_CLASSES, in_channels=IN_CHANNELS),
    }

    results = []

    for name, model in models_to_test.items():
        model.to(DEVICE)
        params = count_parameters(model)
        
        # Timing forward/backward
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Warmup
        _ = run_forward_backward(model, dummy_input, dummy_target, criterion, optimizer, DEVICE)
        
        # Timed run
        start = time.time()
        loss = run_forward_backward(model, dummy_input, dummy_target, criterion, optimizer, DEVICE)
        elapsed = (time.time() - start) * 1000 # ms
        
        results.append((name, params, elapsed, loss))
        print(f"{name:10} | Params: {params/1e6:6.2f}M | Step Time: {elapsed:6.2f}ms | Loss: {loss:.4f}")

    print("-" * 60)
    print("ANALYSIS:")
    print("-" * 60)
    print("1. LeNet-5: Tiny (~60k params). Designed for 1x32x32 (MNIST). Struggles on CIFAR-10 complexity.")
    print("2. AlexNet: ~1.2M params (reduced FC for 32x32). Introduced ReLU, Dropout, LRN (omitted here), MaxPool.")
    print("3. VGG Family: Demonstrates depth via 3x3 conv stacks.")
    print("   - VGG11: ~9.2M params (8 conv layers)")
    print("   - VGG13: ~9.4M params (10 conv layers)")
    print("   - VGG16: ~14.7M params (13 conv layers) - Sweet spot historically.")
    print("   - VGG19: ~20.0M params (16 conv layers) - Diminishing returns, harder to train.")
    print("-" * 60)
    print("Key Takeaway: Depth (VGG) > Width (AlexNet FC) for feature hierarchy.")
    print("Modern Note: BatchNorm (VGG+BN) and Residual Connections (ResNet) solve VGG's gradient flow issues.")
    print("=" * 60)