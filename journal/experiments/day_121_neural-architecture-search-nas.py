import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import time
import copy

# ============================================================
# Configuration
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 121
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BATCH_SIZE = 64
SEARCH_EPOCHS = 3
EVAL_EPOCHS = 5
LR = 0.025
MOMENTUM = 0.9
WEIGHT_DECAY = 3e-4
NUM_CLASSES = 10
SEARCH_SPACE = [
    'skip_connect',
    'max_pool_3x3',
    'avg_pool_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]
NUM_NODES = 4  # intermediate nodes in cell
NUM_EDGES = NUM_NODES * (NUM_NODES + 1) // 2  # fully connected DAG

# ============================================================
# Operations
# ============================================================
class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=True)
        )
    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_in, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, 1, padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out, affine=True)
        )
    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out, affine=True)
        )
    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

def get_op(name, C, stride):
    if name == 'skip_connect':
        return Identity() if stride == 1 else nn.Sequential(
            nn.AvgPool2d(3, stride, 1, count_include_pad=False),
            nn.Conv2d(C, C, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C, affine=True)
        )
    elif name == 'max_pool_3x3':
        return nn.MaxPool2d(3, stride, 1)
    elif name == 'avg_pool_3x3':
        return nn.AvgPool2d(3, stride, 1, count_include_pad=False)
    elif name == 'sep_conv_3x3':
        return SepConv(C, C, 3, stride, 1)
    elif name == 'sep_conv_5x5':
        return SepConv(C, C, 5, stride, 2)
    elif name == 'dil_conv_3x3':
        return DilConv(C, C, 3, stride, 2, 2)
    elif name == 'dil_conv_5x5':
        return DilConv(C, C, 5, stride, 4, 2)
    else:
        raise ValueError(f"Unknown op: {name}")

# ============================================================
# Cell (DAG of mixed operations)
# ============================================================
class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList([get_op(name, C, stride) for name in SEARCH_SPACE])
        self.alphas = nn.Parameter(torch.zeros(len(SEARCH_SPACE)))

    def forward(self, x, weights=None):
        if weights is None:
            weights = F.softmax(self.alphas, dim=-1)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class Cell(nn.Module):
    def __init__(self, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0) if not reduction_prev else nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_prev_prev, C, 1, 2, 0, bias=False),
            nn.BatchNorm2d(C, affine=True)
        )
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self.steps = NUM_NODES
        self.mixed_ops = nn.ModuleList()
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                self.mixed_ops.append(MixedOp(C, stride))

    def forward(self, s0, s1, weights=None):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            s = sum(self.mixed_ops[offset + j](h, weights[offset + j] if weights is not None else None)
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self.steps:], dim=1)

# ============================================================
# SuperNet
# ============================================================
class SuperNet(nn.Module):
    def __init__(self, C=16, layers=4, num_classes=10):
        super().__init__()
        self.C = C
        self.layers = layers
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, C_curr * NUM_NODES
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()

    def _initialize_alphas(self):
        num_ops = len(SEARCH_SPACE)
        num_edges = sum(2 + i for i in range(NUM_NODES)) * self.layers
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(num_edges // 2, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(num_edges // 2, num_ops))
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self):
        return self._arch_parameters

    def forward(self, x, weights_normal=None, weights_reduce=None):
        s0 = s1 = self.stem(x)
        edge_idx = 0
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                w = weights_reduce[edge_idx] if weights_reduce is not None else None
                weights = F.softmax(self.alphas_reduce[edge_idx], dim=-1) if w is None else w
            else:
                w = weights_normal[edge_idx] if weights_normal is not None else None
                weights = F.softmax(self.alphas_normal[edge_idx], dim=-1) if w is None else w
            s0, s1 = s1, cell(s0, s1, weights)
            edge_idx += 1
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def genotype(self):
        def parse(alphas):
            gene = []
            start = 0
            for i in range(NUM_NODES):
                end = start + 2 + i
                W = F.softmax(alphas[start:end], dim=-1).detach().cpu().numpy()
                edges = sorted(range(2 + i), key=lambda x: -max(W[x][k] for k in range(len(SEARCH_SPACE)) if SEARCH_SPACE[k] != 'skip_connect'))[:2]
                for j in edges:
                    k_best = np.argmax(W[j])
                    gene.append((SEARCH_SPACE[k_best], j))
                start = end
            return gene
        return {
            'normal': parse(self.alphas_normal),
            'reduce': parse(self.alphas_reduce)
        }

# ============================================================
# Discrete Evaluation Network (from genotype)
# ============================================================
class Network(nn.Module):
    def __init__(self, C, num_classes, layers, genotype):
        super().__init__()
        self.C = C
        self.layers = layers
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = DiscreteCell(C_prev_prev, C_prev, C_curr, reduction, reduction_prev, genotype)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, C_curr * NUM_NODES
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        return self.classifier(out.view(out.size(0), -1))

class DiscreteCell(nn.Module):
    def __init__(self, C_prev_prev, C_prev, C, reduction, reduction_prev, genotype):
        super().__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_prev_prev, C, 1, 2, 0, bias=False),
                nn.BatchNorm2d(C, affine=True)
            )
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype['reduce'])
        else:
            op_names, indices = zip(*genotype['normal'])
        self.ops = nn.ModuleList()
        self.indices = indices
        for name, idx in zip(op_names, indices):
            stride = 2 if reduction and idx < 2 else 1
            self.ops.append(get_op(name, C, stride))

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i, (op, idx) in enumerate(zip(self.ops, self.indices)):
            states.append(op(states[idx]))
        return torch.cat(states[-NUM_NODES:], dim=1)

# ============================================================
# Data
# ============================================================
def get_data():
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
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # Split train into train/valid for search
    indices = list(range(len(train_data)))
    split = len(indices) // 2
    train_idx, valid_idx = indices[:split], indices[split:]
    train_loader = DataLoader(Subset(train_data, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(Subset(train_data, valid_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader

# ============================================================
# Training / Evaluation
# ============================================================
def train(model, loader, optimizer, criterion, device, arch_optimizer=None):
    model.train()
    total_loss, total_acc, total_num = 0., 0., 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if arch_optimizer:
            arch_optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if arch_optimizer:
            arch_optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(1) == y).sum().item()
        total_num += x.size(0)
    return total_loss / total_num, total_acc / total_num

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, total_num = 0., 0., 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_acc += (logits.argmax(1) == y).sum().item()
            total_num += x.size(0)
    return total_loss / total_num, total_acc / total_num

# ============================================================
# Search (Simplified DARTS: first-order approximation)
# ============================================================
def search():
    print("=" * 60)
    print("Day 121: Neural Architecture Search (DARTS-style)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Search space: {SEARCH_SPACE}")
    print(f"Nodes per cell: {NUM_NODES}, Layers: 4")
    print()

    train_loader, valid_loader, _ = get_data()

    model = SuperNet(C=16, layers=4, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Weight optimizer
    w_optimizer = optim.SGD(model.parameters(), LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # Architecture optimizer (only alphas)
    arch_optimizer = optim.Adam(model.arch_parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)

    print("Starting architecture search...")
    start_time = time.time()
    for epoch in range(SEARCH_EPOCHS):
        # Train weights (on train set)
        train_loss, train_acc = train(model, train_loader, w_optimizer, criterion, DEVICE)
        # Train architecture (on valid set) - first order: no second-order gradient
        model.eval()  # freeze weights for arch step
        valid_loss, valid_acc = train(model, valid_loader, None, criterion, DEVICE, arch_optimizer)
        model.train()

        # Logging
        genotype = model.genotype()
        print(f"Epoch {epoch+1}/{SEARCH_EPOCHS} | "
              f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Valid: Loss={valid_loss:.4f} Acc={valid_acc:.4f}")
        print(f"  Normal: {genotype['normal']}")
        print(f"  Reduce: {genotype['reduce']}")

    search_time = time.time() - start_time
    print(f"\nSearch completed in {search_time:.1f}s")
    return model.genotype()

# ============================================================
# Evaluation of Discovered Architecture
# ============================================================
def evaluate_architecture(genotype):
    print("\n" + "=" * 60)
    print("Evaluating Discovered Architecture")
    print("=" * 60)

    _, _, test_loader = get_data()
    # Use full train set for final training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = Network(C=16, num_classes=NUM_CLASSES, layers=4, genotype=genotype).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EVAL_EPOCHS)

    print(f"Training discovered architecture for {EVAL_EPOCHS} epochs...")
    best_acc = 0.
    for epoch in range(EVAL_EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()
        best_acc = max(best_acc, test_acc)
        print(f"Epoch {epoch+1}/{EVAL_EPOCHS} | "
              f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Test: Loss={test_loss:.4f} Acc={test_acc:.4f} (Best: {best_acc:.4f})")

    print(f"\nFinal Test Accuracy: {best_acc:.4f}")
    return best_acc

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    genotype = search()
    evaluate_architecture(genotype)
    print("\nDay 121 NAS experiment complete.")