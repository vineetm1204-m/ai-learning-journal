import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import time
import copy
import os
import sys

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_dataloaders(batch_size=128, use_synthetic=False):
    if use_synthetic:
        print("Using synthetic data...")
        train_data = TensorDataset(
            torch.randn(5000, 3, 32, 32),
            torch.randint(0, 10, (5000,))
        )
        test_data = TensorDataset(
            torch.randn(1000, 3, 32, 32),
            torch.randint(0, 10, (1000,))
        )
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        try:
            train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        except Exception as e:
            print(f"Dataset download failed: {e}. Falling back to synthetic data.")
            return get_dataloaders(batch_size, use_synthetic=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, 100. * correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return 100. * correct / total

@torch.no_grad()
def benchmark_inference(model, input_shape=(1, 3, 32, 32), warmup=10, runs=100):
    model.eval()
    dummy = torch.randn(input_shape).to(DEVICE)
    for _ in range(warmup):
        _ = model(dummy)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        _ = model(dummy)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000

def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

def apply_magnitude_pruning(model, amount=0.5):
    model_pruned = copy.deepcopy(model)
    parameters_to_prune = []
    for name, module in model_pruned.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    return model_pruned

def apply_quantization_post_training(model, calibration_loader):
    model_q = copy.deepcopy(model)
    model_q.eval()
    model_q.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model_q, inplace=True)
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            model_q(inputs.to(DEVICE))
            break
    torch.quantization.convert(model_q, inplace=True)
    return model_q

def apply_quantization_aware_training(model, train_loader, epochs=3):
    model_qat = copy.deepcopy(model)
    model_qat.train()
    model_qat.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model_qat, inplace=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_qat.parameters(), lr=1e-4)
    for epoch in range(epochs):
        loss, acc = train_one_epoch(model_qat, train_loader, criterion, optimizer)
        print(f"  QAT Epoch {epoch+1}/{epochs}: Loss={loss:.4f}, Acc={acc:.2f}%")
    model_qat.eval()
    torch.quantization.convert(model_qat, inplace=True)
    return model_qat

def run_experiment():
    print("=" * 60)
    print("Day 122: Quantization & Pruning for Deployment")
    print("=" * 60)

    train_loader, test_loader = get_dataloaders(batch_size=128, use_synthetic=False)
    calib_loader, _ = get_dataloaders(batch_size=32, use_synthetic=False)

    print("\n[1] Training baseline model...")
    baseline = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    for epoch in range(5):
        loss, acc = train_one_epoch(baseline, train_loader, criterion, optimizer)
        test_acc = evaluate(baseline, test_loader)
        print(f"  Epoch {epoch+1}/5: Train Loss={loss:.4f}, Train Acc={acc:.2f}%, Test Acc={test_acc:.2f}%")

    baseline_size = get_model_size(baseline)
    baseline_latency = benchmark_inference(baseline)
    baseline_acc = evaluate(baseline, test_loader)
    print(f"\nBaseline: Size={baseline_size:.2f} MB, Latency={baseline_latency:.2f} ms, Acc={baseline_acc:.2f}%")

    print("\n[2] Magnitude Pruning (50%)...")
    pruned = apply_magnitude_pruning(baseline, amount=0.5)
    pruned_size = get_model_size(pruned)
    pruned_latency = benchmark_inference(pruned)
    pruned_acc = evaluate(pruned, test_loader)
    print(f"Pruned:   Size={pruned_size:.2f} MB, Latency={pruned_latency:.2f} ms, Acc={pruned_acc:.2f}%")

    print("\n[3] Post-Training Quantization (INT8)...")
    quantized_ptq = apply_quantization_post_training(baseline, calib_loader)
    quantized_ptq_size = get_model_size(quantized_ptq)
    quantized_ptq_latency = benchmark_inference(quantized_ptq)
    quantized_ptq_acc = evaluate(quantized_ptq, test_loader)
    print(f"PTQ:      Size={quantized_ptq_size:.2f} MB, Latency={quantized_ptq_latency:.2f} ms, Acc={quantized_ptq_acc:.2f}%")

    print("\n[4] Quantization-Aware Training (3 epochs)...")
    quantized_qat = apply_quantization_aware_training(baseline, train_loader, epochs=3)
    quantized_qat_size = get_model_size(quantized_qat)
    quantized_qat_latency = benchmark_inference(quantized_qat)
    quantized_qat_acc = evaluate(quantized_qat, test_loader)
    print(f"QAT:      Size={quantized_qat_size:.2f} MB, Latency={quantized_qat_latency:.2f} ms, Acc={quantized_qat_acc:.2f}%")

    print("\n[5] Pruning + Quantization (PTQ)...")
    pruned_quantized = apply_quantization_post_training(pruned, calib_loader)
    pq_size = get_model_size(pruned_quantized)
    pq_latency = benchmark_inference(pruned_quantized)
    pq_acc = evaluate(pruned_quantized, test_loader)
    print(f"Prune+PTQ: Size={pq_size:.2f} MB, Latency={pq_latency:.2f} ms, Acc={pq_acc:.2f}%")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<15} {'Size (MB)':>10} {'Latency (ms)':>12} {'Accuracy (%)':>12} {'Size ↓':>8} {'Speed ↑':>8}")
    print("-" * 60)
    rows = [
        ("Baseline", baseline_size, baseline_latency, baseline_acc, 1.0, 1.0),
        ("Pruned 50%", pruned_size, pruned_latency, pruned_acc, baseline_size/pruned_size, baseline_latency/pruned_latency),
        ("PTQ INT8", quantized_ptq_size, quantized_ptq_latency, quantized_ptq_acc, baseline_size/quantized_ptq_size, baseline_latency/quantized_ptq_latency),
        ("QAT INT8", quantized_qat_size, quantized_qat_latency, quantized_qat_acc, baseline_size/quantized_qat_size, baseline_latency/quantized_qat_latency),
        ("Prune+PTQ", pq_size, pq_latency, pq_acc, baseline_size/pq_size, baseline_latency/pq_latency),
    ]
    for name, size, lat, acc, sz_ratio, sp_ratio in rows:
        print(f"{name:<15} {size:>10.2f} {lat:>12.2f} {acc:>12.2f} {sz_ratio:>7.2f}x {sp_ratio:>7.2f}x")

    print("\nKey Takeaways:")
    print("- Pruning reduces model size but may not speed up inference on dense hardware")
    print("- PTQ gives ~4x size reduction with minimal accuracy drop")
    print("- QAT recovers accuracy lost in PTQ, especially at lower bit-widths")
    print("- Combining pruning + quantization yields maximum compression")
    print("- For real deployment: consider structured pruning, hardware-specific kernels")

if __name__ == "__main__":
    run_experiment()