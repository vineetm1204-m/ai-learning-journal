import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
import io
import copy

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 3  # Keep low for demo speed
LR = 1e-3
PRUNE_AMOUNT = 0.3  # Prune 30% of weights
SEED = 42
torch.manual_seed(SEED)

# --- 1. Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- 2. Data Loading (MNIST) ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --- 3. Utilities ---
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

def get_model_size(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1024  # Size in KB

def measure_inference_time(model, loader, device, runs=10):
    model.eval()
    # Warmup
    with torch.no_grad():
        for data, _ in loader:
            model(data.to(device))
            break
    # Timed runs
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            for data, _ in loader:
                model(data.to(device))
    end = time.time()
    total_samples = len(loader.dataset) * runs
    return (end - start) / total_samples * 1000  # ms per sample

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# --- 4. Baseline Training ---
print(f"--- Day 40: Quantization & Pruning Experiment ---")
print(f"Device: {DEVICE}")
model = SimpleCNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("\n[Phase 1] Training Baseline Model...")
for epoch in range(EPOCHS):
    train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    acc = evaluate(model, test_loader, DEVICE)
    print(f"  Epoch {epoch+1}/{EPOCHS} | Test Acc: {acc:.2f}%")

baseline_acc = evaluate(model, test_loader, DEVICE)
baseline_size = get_model_size(model)
baseline_speed = measure_inference_time(model, test_loader, DEVICE)
print(f"  Baseline -> Acc: {baseline_acc:.2f}% | Size: {baseline_size:.1f} KB | Latency: {baseline_speed:.4f} ms/sample")

# --- 5. Experiment A: Magnitude Pruning ---
print("\n[Phase 2] Applying Global Magnitude Pruning (30%)...")
pruned_model = copy.deepcopy(model)
parameters_to_prune = []
for module in pruned_model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=PRUNE_AMOUNT,
)

# Make pruning permanent (remove masks, zero out weights)
for module, name in parameters_to_prune:
    prune.remove(module, name)

# Fine-tune pruned model (critical for recovery)
print("  Fine-tuning pruned model...")
ft_optimizer = optim.Adam(pruned_model.parameters(), lr=LR * 0.1)
for epoch in range(2): # Few epochs fine-tune
    train_one_epoch(pruned_model, train_loader, ft_optimizer, criterion, DEVICE)

pruned_acc = evaluate(pruned_model, test_loader, DEVICE)
pruned_size = get_model_size(pruned_model) # Note: Standard save doesn't compress sparse matrices automatically.
                                           # Real deployment needs sparse tensor format. 
                                           # Here we measure dense size to show "structural" sparsity isn't saved by default.
pruned_speed = measure_inference_time(pruned_model, test_loader, DEVICE)
print(f"  Pruned   -> Acc: {pruned_acc:.2f}% | Size (Dense): {pruned_size:.1f} KB | Latency: {pruned_speed:.4f} ms/sample")
print(f"  Sparsity: {100 * sum((p==0).sum().item() for p in pruned_model.parameters()) / sum(p.numel() for p in pruned_model.parameters()):.1f}%")

# --- 6. Experiment B: Post-Training Dynamic Quantization (INT8) ---
print("\n[Phase 3] Applying Post-Training Dynamic Quantization (INT8)...")
# Quantizes Linear/Conv weights to INT8 on the fly. Activations remain FP32.
quantized_model = torch.quantization.quantize_dynamic(
    copy.deepcopy(model),  # Quantize the original dense model
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)

quant_acc = evaluate(quantized_model, test_loader, DEVICE)
quant_size = get_model_size(quantized_model)
quant_speed = measure_inference_time(quantized_model, test_loader, DEVICE)
print(f"  Quantized-> Acc: {quant_acc:.2f}% | Size: {quant_size:.1f} KB | Latency: {quant_speed:.4f} ms/sample")

# --- 7. Experiment C: Pruning + Quantization Combo ---
print("\n[Phase 4] Combining Pruning + Quantization...")
combo_model = torch.quantization.quantize_dynamic(
    pruned_model, # Quantize the already pruned model
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)
combo_acc = evaluate(combo_model, test_loader, DEVICE)
combo_size = get_model_size(combo_model)
combo_speed = measure_inference_time(combo_model, test_loader, DEVICE)
print(f"  Combo    -> Acc: {combo_acc:.2f}% | Size: {combo_size:.1f} KB | Latency: {combo_speed:.4f} ms/sample")

# --- 8. Summary Report ---
print("\n" + "="*60)
print(f"{'MODEL VARIANT':<20} | {'ACCURACY':>8} | {'SIZE (KB)':>10} | {'LATENCY (ms)':>12} | {'SIZE RED':>8} | {'SPEEDUP':>7}")
print("-"*60)
print(f"{'Baseline (FP32)':<20} | {baseline_acc:>7.2f}% | {baseline_size:>9.1f} | {baseline_speed:>11.4f} | {'1.00x':>8} | {'1.00x':>7}")
print(f"{'Pruned 30% (FP32)':<20} | {pruned_acc:>7.2f}% | {pruned_size:>9.1f} | {pruned_speed:>11.4f} | {baseline_size/pruned_size:>7.2f}x | {baseline_speed/pruned_speed:>6.2f}x")
print(f"{'Quantized (INT8)':<20} | {quant_acc:>7.2f}% | {quant_size:>9.1f} | {quant_speed:>11.4f} | {baseline_size/quant_size:>7.2f}x | {baseline_speed/quant_speed:>6.2f}x")
print(f"{'Pruned + Quantized':<20} | {combo_acc:>7.2f}% | {combo_size:>9.1f} | {combo_speed:>11.4f} | {baseline_size/combo_size:>7.2f}x | {baseline_speed/combo_speed:>6.2f}x")
print("="*60)
print("\nNote: Pruned model size shown is dense storage size. Real deployment requires sparse kernels/formats (e.g., ONNX sparse, TensorRT) for actual disk/memory reduction.")