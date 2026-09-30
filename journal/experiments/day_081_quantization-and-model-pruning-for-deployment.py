import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.quantization
import torch.nn.utils.prune as prune
import time
import copy
import os

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy, test_loss

def measure_inference_time(model, device, test_loader, num_batches=10):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            data = data.to(device)
            start = time.time()
            _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    avg_time = sum(times) / len(times) * 1000
    return avg_time

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

def apply_magnitude_pruning(model, amount=0.3):
    model_pruned = copy.deepcopy(model)
    parameters_to_prune = []
    for name, module in model_pruned.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    return model_pruned

def apply_dynamic_quantization(model):
    model_dq = copy.deepcopy(model)
    model_dq.eval()
    model_dq = torch.quantization.quantize_dynamic(
        model_dq, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    return model_dq

def apply_static_quantization(model, train_loader, device):
    model_sq = copy.deepcopy(model)
    model_sq.eval()
    model_sq.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model_sq)
    print("Calibrating static quantization...")
    with torch.no_grad():
        for i, (data, _) in enumerate(train_loader):
            if i >= 10:
                break
            model_prepared(data.to(device))
    model_quantized = torch.quantization.convert(model_prepared)
    return model_quantized

def main():
    train_loader, test_loader = get_data_loaders(64)
    
    print("=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, device, train_loader, optimizer, 1)
    baseline_acc, _ = test(model, device, test_loader)
    baseline_size = get_model_size(model)
    baseline_time = measure_inference_time(model, device, test_loader)
    print(f"Baseline - Size: {baseline_size:.2f} MB, Inference: {baseline_time:.2f} ms/batch")
    
    print("\n" + "=" * 60)
    print("MAGNITUDE PRUNING (30%)")
    print("=" * 60)
    pruned_model = apply_magnitude_pruning(model, amount=0.3)
    pruned_acc, _ = test(pruned_model, device, test_loader)
    pruned_size = get_model_size(pruned_model)
    pruned_time = measure_inference_time(pruned_model, device, test_loader)
    print(f"Pruned - Size: {pruned_size:.2f} MB, Inference: {pruned_time:.2f} ms/batch")
    print(f"Accuracy drop: {baseline_acc - pruned_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("DYNAMIC QUANTIZATION")
    print("=" * 60)
    dyn_quant_model = apply_dynamic_quantization(model)
    dyn_acc, _ = test(dyn_quant_model, device, test_loader)
    dyn_size = get_model_size(dyn_quant_model)
    dyn_time = measure_inference_time(dyn_quant_model, device, test_loader)
    print(f"Dynamic Quant - Size: {dyn_size:.2f} MB, Inference: {dyn_time:.2f} ms/batch")
    print(f"Accuracy drop: {baseline_acc - dyn_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("STATIC QUANTIZATION (INT8)")
    print("=" * 60)
    static_quant_model = apply_static_quantization(model, train_loader, device)
    static_acc, _ = test(static_quant_model, device, test_loader)
    static_size = get_model_size(static_quant_model)
    static_time = measure_inference_time(static_quant_model, device, test_loader)
    print(f"Static Quant - Size: {static_size:.2f} MB, Inference: {static_time:.2f} ms/batch")
    print(f"Accuracy drop: {baseline_acc - static_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("PRUNING + DYNAMIC QUANTIZATION")
    print("=" * 60)
    pruned_quant_model = apply_dynamic_quantization(pruned_model)
    pq_acc, _ = test(pruned_quant_model, device, test_loader)
    pq_size = get_model_size(pruned_quant_model)
    pq_time = measure_inference_time(pruned_quant_model, device, test_loader)
    print(f"Pruned+Quant - Size: {pq_size:.2f} MB, Inference: {pq_time:.2f} ms/batch")
    print(f"Accuracy drop: {baseline_acc - pq_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Size (MB)':<12} {'Inference (ms)':<15} {'Accuracy (%)':<12} {'Size Reduction':<15} {'Speedup'}")
    print("-" * 90)
    print(f"{'Baseline':<25} {baseline_size:<12.2f} {baseline_time:<15.2f} {baseline_acc:<12.2f} {'1.00x':<15} {'1.00x'}")
    print(f"{'Pruned (30%)':<25} {pruned_size:<12.2f} {pruned_time:<15.2f} {pruned_acc:<12.2f} {baseline_size/pruned_size:<15.2f}x {baseline_time/pruned_time:<15.2f}x")
    print(f"{'Dynamic Quant':<25} {dyn_size:<12.2f} {dyn_time:<15.2f} {dyn_acc:<12.2f} {baseline_size/dyn_size:<15.2f}x {baseline_time/dyn_time:<15.2f}x")
    print(f"{'Static Quant':<25} {static_size:<12.2f} {static_time:<15.2f} {static_acc:<12.2f} {baseline_size/static_size:<15.2f}x {baseline_time/static_time:<15.2f}x")
    print(f"{'Pruned + Dyn Quant':<25} {pq_size:<12.2f} {pq_time:<15.2f} {pq_acc:<12.2f} {baseline_size/pq_size:<15.2f}x {baseline_time/pq_time:<15.2f}x")

if __name__ == "__main__":
    main()