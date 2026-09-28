import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Day 80 NAS] Using device: {DEVICE}")

# ============================================================
# 1. Search Space Primitives
# ============================================================
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine),
    'conv_3x3': lambda C, stride, affine: ConvBNReLU(C, C, 3, stride, 1, affine=affine),
    'conv_5x5': lambda C, stride, affine: ConvBNReLU(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
}

PRIMITIVES = list(OPS.keys())

class ConvBNReLU(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False)
        )
    def forward(self, x): return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False)
        )
    def forward(self, x): return self.op(x)

class Identity(nn.Module):
    def forward(self, x): return x

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    def forward(self, x):
        if self.stride == 1: return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)

# ============================================================
# 2. Mixed Operation (Continuous Relaxation)
# ============================================================
class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for prim in PRIMITIVES:
            op = OPS[prim](C, stride, affine=False)
            self._ops.append(op)
        # Architecture parameters (logits)
        self.alpha = nn.Parameter(torch.zeros(len(PRIMITIVES)))

    def forward(self, x, weights=None):
        if weights is None:
            weights = F.softmax(self.alpha, dim=-1)
        return sum(w * op(x) for w, op in zip(weights, self._ops))

# ============================================================
# 3. Cell (DAG of MixedOps)
# ============================================================
class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        self.steps = steps
        self.multiplier = multiplier

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ConvBNReLU(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ConvBNReLU(C_prev, C, 1, 1, 0, affine=False)

        self._ops = nn.ModuleList()
        self._compile(C, reduction)

    def _compile(self, C, reduction):
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights=None):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            s = sum(self._ops[offset + j](h, weights[offset + j] if weights is not None else None)
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self.multiplier:], dim=1)

# ============================================================
# 4. Network with Architecture Parameters
# ============================================================
class Network(nn.Module):
    def __init__(self, C=16, num_classes=10, layers=8, steps=4, multiplier=4, stem_multiplier=3):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
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
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for _ in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self):
        return self._arch_parameters

    def forward(self, input):
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)

        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
            else:
                weights = weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(n), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = max(range(len(W[j])), key=lambda k: W[j][k] if k != PRIMITIVES.index('none') else -1)
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        return {
            'normal': gene_normal,
            'normal_concat': list(concat),
            'reduce': gene_reduce,
            'reduce_concat': list(concat)
        }

# ============================================================
# 5. Data (CIFAR-10 subset for speed)
# ============================================================
def get_data(batch_size=64, subset_frac=0.1):
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

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Subset for fast experimentation
    n_train = int(len(train_data) * subset_frac)
    n_test = int(len(test_data) * subset_frac)
    train_idx = torch.randperm(len(train_data))[:n_train]
    test_idx = torch.randperm(len(test_data))[:n_test]

    train_loader = DataLoader(Subset(train_data, train_idx), batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(Subset(train_data, torch.randperm(len(train_data))[:n_train//4]), batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(Subset(test_data, test_idx), batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader

# ============================================================
# 6. Training Loop (Bilevel Optimization)
# ============================================================
def train_search(model, train_loader, valid_loader, epochs=10, lr_w=0.025, lr_a=3e-4, w_decay=3e-4):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Weight optimizer (inner loop)
    optimizer_w = optim.SGD(model.parameters(), lr=lr_w, momentum=0.9, weight_decay=w_decay)
    scheduler_w = optim.lr_scheduler.CosineAnnealingLR(optimizer_w, T_max=epochs)

    # Architecture optimizer (outer loop)
    optimizer_a = optim.Adam(model.arch_parameters(), lr=lr_a, betas=(0.5, 0.999), weight_decay=1e-3)

    history = {'train_loss': [], 'valid_acc': [], 'arch_normal': [], 'arch_reduce': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for step, (input, target) in enumerate(train_loader):
            input, target = input.to(DEVICE), target.to(DEVICE)

            # ---- Phase 1: Update architecture parameters (outer loop) ----
            # Use validation batch
            try:
                input_search, target_search = next(valid_iter)
            except:
                valid_iter = iter(valid_loader)
                input_search, target_search = next(valid_iter)
            input_search, target_search = input_search.to(DEVICE), target_search.to(DEVICE)

            optimizer_a.zero_grad()
            logits_search = model(input_search)
            loss_a = criterion(logits_search, target_search)
            loss_a.backward()
            optimizer_a.step()

            # ---- Phase 2: Update network weights (inner loop) ----
            optimizer_w.zero_grad()
            logits = model(input)
            loss_w = criterion(logits, target)
            loss_w.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer_w.step()

            train_loss += loss_w.item()

        scheduler_w.step()

        # Validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for input, target in valid_loader:
                input, target = input.to(DEVICE), target.to(DEVICE)
                logits = model(input)
                _, pred = logits.topk(1, 1, True, True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        valid_acc = 100. * correct / total

        # Log architecture weights
        w_normal = F.softmax(model.alphas_normal, dim=-1).detach().cpu().numpy()
        w_reduce = F.softmax(model.alphas_reduce, dim=-1).detach().cpu().numpy()
        history['train_loss'].append(train_loss / len(train_loader))
        history['valid_acc'].append(valid_acc)
        history['arch_normal'].append(w_normal.copy())
        history['arch_reduce'].append(w_reduce.copy())

        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {valid_acc:.2f}%")

    return history

# ============================================================
# 7. Visualization
# ============================================================
def plot_architecture_evolution(history, primitives=PRIMITIVES):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = len(history['train_loss'])

    # Loss & Accuracy
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', color='tab:blue')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(history['valid_acc'], label='Val Acc', color='tab:orange')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(loc='lower right')
    ax.set_title('Training Dynamics')

    # Normal cell architecture weights (final epoch)
    ax = axes[0, 1]
    w_norm = history['arch_normal'][-1]
    im = ax.imshow(w_norm.T, aspect='auto', cmap='viridis')
    ax.set_yticks(range(len(primitives)))
    ax.set_yticklabels(primitives, fontsize=8)
    ax.set_xlabel('Edge Index')
    ax.set_title('Normal Cell: Final Softmax Weights')
    plt.colorbar(im, ax=ax)

    # Reduce cell architecture weights (final epoch)
    ax = axes[1, 0]
    w_red = history['arch_reduce'][-1]
    im = ax.imshow(w_red.T, aspect='auto', cmap='viridis')
    ax.set_yticks(range(len(primitives)))
    ax.set_yticklabels(primitives, fontsize=8)
    ax.set_xlabel('Edge Index')
    ax.set_title('Reduce Cell: Final Softmax Weights')
    plt.colorbar(im, ax=ax)

    # Evolution of top-1 op per edge (normal cell)
    ax = axes[1, 1]
    for edge in range(w_norm.shape[0]):
        top1_ops = [np.argmax(epoch_w[edge]) for epoch_w in history['arch_normal']]
        ax.plot(top1_ops, alpha=0.7, linewidth=1)
    ax.set_yticks(range(len(primitives)))
    ax.set_yticklabels(primitives, fontsize=8)
    ax.set_xlabel('Epoch')
    ax.set_title('Normal Cell: Top-1 Op per Edge Over Time')
    ax.set_ylim(-0.5, len(primitives)-0.5)

    plt.tight_layout()
    plt.savefig('nas_day80_architecture_evolution.png', dpi=150)
    print("Saved architecture evolution plot to nas_day80_architecture_evolution.png")
    plt.close()

def print_genotype(genotype):
    print("\n=== Discovered Genotype ===")
    print("Normal cell:")
    for i, (op, idx) in enumerate(genotype['normal']):
        print(f"  Edge {i}: {op} <- node {idx}")
    print(f"  Concat: {genotype['normal_concat']}")
    print("Reduce cell:")
    for i, (op, idx) in enumerate(genotype['reduce']):
        print(f"  Edge {i}: {op} <- node {idx}")
    print(f"  Concat: {genotype['reduce_concat']}")

# ============================================================
# 8. Main Experiment
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Day 80: Neural Architecture Search (DARTS-style) Mini-Experiment")
    print("=" * 60)

    # Data
    train_loader, valid_loader, test_loader = get_data(batch_size=64, subset_frac=0.1)
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # Model
    model = Network(C=16, num_classes=10, layers=8, steps=4, multiplier=4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Architecture parameters: {sum(p.numel() for p in model.arch_parameters())}")

    # Search
    history = train_search(model, train_loader, valid_loader, epochs=15, lr_w=0.025, lr_a=3e-4)

    # Final genotype
    genotype = model.genotype()
    print_genotype(genotype)

    # Plot
    plot_architecture_evolution(history)

    # Final test evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(DEVICE), target.to(DEVICE)
            logits = model(input)
            _, pred = logits.topk(1, 1, True, True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    test_acc = 100. * correct / total
    print(f"\nFinal Test Accuracy (on subset): {test_acc:.2f}%")

    print("\nExperiment complete. Check nas_day80_architecture_evolution.png for visualization.")