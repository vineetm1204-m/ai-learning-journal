import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

# ============================================================
# 1. Architecture Definitions
# ============================================================

class LeNet5(nn.Module):
    """LeNet-5 adapted for 32x32 RGB input (CIFAR-10 style)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    """AlexNet adapted for 32x32 input (CIFAR-10). Original expects 227x227."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Adaptive pool allows handling variable input sizes (here 4x4 -> 6x6)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_vgg_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
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
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # Standard VGG expects 7x7 before FC
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def vgg11(num_classes=10):
    return VGG(make_vgg_layers(cfgs['VGG11'], batch_norm=True), num_classes)

def vgg16(num_classes=10):
    return VGG(make_vgg_layers(cfgs['VGG16'], batch_norm=True), num_classes)

# ============================================================
# 2. Experiment Utilities
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

def benchmark_step(model, device, input_shape=(32, 3, 32, 32), steps=5):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Warmup
    dummy_input = torch.randn(input_shape, device=device)
    dummy_target = torch.randint(0, 10, (input_shape[0],), device=device)
    for _ in range(3):
        optimizer.zero_grad()
        out = model(dummy_input)
        loss = criterion(out, dummy_target)
        loss.backward()
        optimizer.step()

    # Timed run
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(dummy_input)
        loss = criterion(out, dummy_target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()
    
    return (end - start) / steps * 1000 # ms per step

# ============================================================
# 3. Main Execution
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Day 99: Classic Architectures Comparison ===")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print("-" * 60)

    models = {
        "LeNet-5": LeNet5(),
        "AlexNet (CIFAR)": AlexNet(),
        "VGG11 (BN)": vgg11(),
        "VGG16 (BN)": vgg16(),
    }

    results = []
    input_shape = (32, 3, 32, 32) # Batch, Channels, H, W

    for name, model in models.items():
        print(f"\n--- Analyzing {name} ---")
        try:
            params = count_parameters(model)
            size_mb = get_model_size_mb(model)
            
            # Forward pass shape check
            model.eval()
            with torch.no_grad():
                test_in = torch.randn(1, 3, 32, 32).to(device)
                test_out = model.to(device)(test_in)
                out_shape = tuple(test_out.shape)
            
            # Benchmark
            time_ms = benchmark_step(model, device, input_shape)
            
            results.append({
                "name": name,
                "params": params,
                "size_mb": size_mb,
                "out_shape": out_shape,
                "step_ms": time_ms
            })
            
            print(f"  Params: {params:,} ({params/1e6:.2f}M)")
            print(f"  Size:   {size_mb:.2f} MB")
            print(f"  Output: {out_shape}")
            print(f"  Step:   {time_ms:.2f} ms")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": name, "error": str(e)})

    # Summary Table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    header = f"{'Model':<20} | {'Params (M)':>10} | {'Size (MB)':>9} | {'Step (ms)':>9} | Output Shape"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" not in r:
            print(f"{r['name']:<20} | {r['params']/1e6:>10.2f} | {r['size_mb']:>9.2f} | {r['step_ms']:>9.2f} | {r['out_shape']}")
        else:
            print(f"{r['name']:<20} | {'FAILED':>10} | {'-':>9} | {'-':>9} | {r['error']}")

    print("\nExperiment Complete.")

if __name__ == "__main__":
    main()