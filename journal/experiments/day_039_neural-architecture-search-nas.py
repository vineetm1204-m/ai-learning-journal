import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# 1. Search Space: Primitive Operations
# ------------------------------------------------------------
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine),
}

PRIMITIVES = list(OPS.keys())

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    def forward(self, x): return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    def forward(self, x): return self.op(x)

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
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
        self.conv1 = nn.Conv2d(C_in, C_out//2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out//2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)

# ------------------------------------------------------------
# 2. Mixed Operation (Continuous Relaxation)
# ------------------------------------------------------------
class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for prim in PRIMITIVES:
            op = OPS[prim](C, stride, False)
            self._ops.append(op)
        # Architecture parameters (logits)
        self.alpha = nn.Parameter(torch.randn(len(PRIMITIVES)) * 1e-3)

    def forward(self, x, weights=None):
        if weights is None:
            weights = F.softmax(self.alpha, dim=-1)
        return sum(w * op(x) for w, op in zip(weights, self._ops))

# ------------------------------------------------------------
# 3. Cell (DAG of MixedOps)
# ------------------------------------------------------------
class Cell(nn.Module):
    def __init__(self, steps, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        self.steps = steps
        self.C = C

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._ops = nn.ModuleList()
        self._compile()

    def _compile(self):
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if self.reduction and j < 2 else 1
                op = MixedOp(self.C, stride)
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
        return torch.cat(states[-self.steps:], dim=1)

# ------------------------------------------------------------
# 4. Network (Stack of Cells)
# ------------------------------------------------------------
class Network(nn.Module):
    def __init__(self, C=16, num_classes=10, layers=4, steps=4, stem_multiplier=3):
        super().__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self.steps = steps

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, steps * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # Architecture parameters: one per edge per cell
        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 for i in range(self.steps) for n in range(2 + i))
        num_edges = k
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(num_edges, len(PRIMITIVES)))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(num_edges, len(PRIMITIVES)))
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self):
        return self._arch_parameters

    def forward(self, x):
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)

        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self.steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = max(range(len(W[j])), key=lambda k: W[j][k] if k != PRIMITIVES.index('none') else -1)
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        return {'normal': gene_normal, 'reduce': gene_reduce}

# ------------------------------------------------------------
# 5. Synthetic Dataset (for self-contained speed)
# ------------------------------------------------------------
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, img_size=32, num_classes=10):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        # Generate structured data: class = dominant frequency pattern
        self.data = torch.randn(num_samples, 3, img_size, img_size)
        self.targets = torch.randint(0, num_classes, (num_samples,))
        # Add class-specific pattern
        for i in range(num_samples):
            c = self.targets[i].item()
            freq = (c + 1) * 2
            x = torch.linspace(0, 4*np.pi, img_size)
            y = torch.linspace(0, 4*np.pi, img_size)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            pattern = torch.sin(freq * xx) * torch.cos(freq * yy)
            self.data[i, 0] += 0.5 * pattern
            self.data[i, 1] += 0.5 * pattern.roll(shifts=c, dims=0)
            self.data[i, 2] += 0.5 * pattern.roll(shifts=c, dims=1)

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

# ------------------------------------------------------------
# 6. Training Loop (Bi-level Optimization)
# ------------------------------------------------------------
def train_search(model, train_loader, valid_loader, epochs=10, lr=0.025, lr_arch=3e-4, wd=3e-4, wd_arch=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Weight optimizer (model weights)
    optimizer_w = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    # Architecture optimizer (alphas)
    optimizer_a = optim.Adam(model.arch_parameters(), lr=lr_arch, betas=(0.5, 0.999), weight_decay=wd_arch)

    scheduler_w = optim.lr_scheduler.CosineAnnealingLR(optimizer_w, epochs)

    print(f"Search started on {device}...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, train_n = 0, 0, 0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # --- Phase 1: Update architecture parameters (alpha) ---
            # Use validation batch for architecture gradient
            x_val, y_val = next(iter(valid_loader))
            x_val, y_val = x_val.to(device), y_val.to(device)

            optimizer_a.zero_grad()
            logits_val = model(x_val)
            loss_val = criterion(logits_val, y_val)
            loss_val.backward()
            optimizer_a.step()

            # --- Phase 2: Update network weights (w) ---
            optimizer_w.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer_w.step()

            train_loss += loss.item() * x.size(0)
            train_acc += (logits.argmax(1) == y).sum().item()
            train_n += x.size(0)

        scheduler_w.step()

        # Validation accuracy
        model.eval()
        val_acc, val_n = 0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_acc += (logits.argmax(1) == y).sum().item()
                val_n += x.size(0)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss/train_n:.4f} | "
              f"Train Acc: {train_acc/train_n:.4f} | "
              f"Val Acc: {val_acc/val_n:.4f} | "
              f"LR: {scheduler_w.get_last_lr()[0]:.6f}")

    print(f"Search finished in {time.time()-start_time:.1f}s")
    return model

# ------------------------------------------------------------
# 7. Evaluation of Derived Architecture
# ------------------------------------------------------------
class FixedNetwork(nn.Module):
    """Network with fixed architecture (discretized)"""
    def __init__(self, genotype, C=16, num_classes=10, layers=4, steps=4, stem_multiplier=3):
        super().__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self.steps = steps

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = FixedCell(steps, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, genotype)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, steps * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        return self.classifier(out.view(out.size(0), -1))

class FixedCell(nn.Module):
    def __init__(self, steps, C_prev_prev, C_prev, C, reduction, reduction_prev, genotype):
        super().__init__()
        self.reduction = reduction
        self.steps = steps

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._ops = nn.ModuleList()
        self._indices = []
        self._compile(C, reduction, genotype)

    def _compile(self, C, reduction, genotype):
        gene = genotype['reduce'] if reduction else genotype['normal']
        for i, (prim, idx) in enumerate(gene):
            stride = 2 if reduction and idx < 2 else 1
            op = OPS[prim](C, stride, True)
            self._ops.append(op)
            self._indices.append(idx)

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for op, idx in zip(self._ops, self._indices):
            s = op(states[idx])
            states.append(s)
        return torch.cat(states[-self.steps:], dim=1)

def evaluate_fixed(genotype, train_loader, valid_loader, epochs=20, lr=0.025):
    model = FixedNetwork(genotype).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    print("\nEvaluating derived architecture...")
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        acc, n = 0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                acc += (logits.argmax(1) == y).sum().item()
                n += x.size(0)
        print(f"  Epoch {epoch+1:2d} | Val Acc: {acc/n:.4f}")
    return acc/n

# ------------------------------------------------------------
# 8. Main Experiment
# ------------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameters (small for speed)
    BATCH_SIZE = 64
    SEARCH_EPOCHS = 8
    EVAL_EPOCHS = 15
    LAYERS = 4
    STEPS = 3
    CHANNELS = 16

    # Data
    train_data = SyntheticDataset(2000)
    valid_data = SyntheticDataset(500)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Search
    model = Network(C=CHANNELS, layers=LAYERS, steps=STEPS)
    model = train_search(model, train_loader, valid_loader, epochs=SEARCH_EPOCHS)

    # Genotype
    genotype = model.genotype()
    print("\nDiscovered Genotype:")
    for k, v in genotype.items():
        print(f"  {k}: {v}")

    # Evaluate fixed architecture
    final_acc = evaluate_fixed(genotype, train_loader, valid_loader, epochs=EVAL_EPOCHS)
    print(f"\nFinal Test Accuracy (Fixed Arch): {final_acc:.4f}")

    # Parameter count
    fixed_model = FixedNetwork(genotype, C=CHANNELS, layers=LAYERS, steps=STEPS)
    param_count = sum(p.numel() for p in fixed_model.parameters())
    print(f"Model Parameters: {param_count:,}")