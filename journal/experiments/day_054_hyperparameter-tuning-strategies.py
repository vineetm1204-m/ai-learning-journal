import os
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Callable, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Hyperparameters:
    lr: float
    batch_size: int
    hidden_dim: int
    num_layers: int
    dropout: float
    weight_decay: float
    optimizer: str
    activation: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Hyperparameters":
        return cls(**d)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 dropout: float, activation: str, output_dim: int = 2):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def create_dataset(n_samples: int = 5000, n_features: int = 20, n_classes: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=15,
                               n_redundant=3, n_classes=n_classes, random_state=SEED)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=SEED, stratify=y_train)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    return train_ds, val_ds, test_ds


def get_optimizer(model: nn.Module, hp: Hyperparameters) -> optim.Optimizer:
    if hp.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    elif hp.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    elif hp.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=hp.lr, momentum=0.9, weight_decay=hp.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {hp.optimizer}")


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


def run_training(hp: Hyperparameters, train_ds: TensorDataset, val_ds: TensorDataset, 
                 epochs: int = 30, patience: int = 5) -> Dict[str, Any]:
    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False)

    input_dim = train_ds.tensors[0].shape[1]
    model = MLP(input_dim, hp.hidden_dim, hp.num_layers, hp.dropout, hp.activation).to(DEVICE)
    optimizer = get_optimizer(model, hp)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    return {"best_val_acc": best_val_acc, "history": history, "epochs_run": epoch + 1}


def grid_search(param_grid: Dict[str, List], train_ds: TensorDataset, val_ds: TensorDataset, 
                epochs: int = 30) -> List[Dict]:
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results = []

    def recurse(idx: int, current: Dict):
        if idx == len(keys):
            hp = Hyperparameters(**current)
            print(f"  Grid: {current}")
            res = run_training(hp, train_ds, val_ds, epochs=epochs)
            results.append({"params": current, "val_acc": res["best_val_acc"], "history": res["history"]})
            return
        for v in values[idx]:
            current[keys[idx]] = v
            recurse(idx + 1, current)

    recurse(0, {})
    return results


def random_search(param_distributions: Dict[str, Callable], train_ds: TensorDataset, val_ds: TensorDataset,
                  n_trials: int = 20, epochs: int = 30) -> List[Dict]:
    results = []
    for i in range(n_trials):
        params = {k: v() for k, v in param_distributions.items()}
        hp = Hyperparameters(**params)
        print(f"  Random [{i+1}/{n_trials}]: {params}")
        res = run_training(hp, train_ds, val_ds, epochs=epochs)
        results.append({"params": params, "val_acc": res["best_val_acc"], "history": res["history"]})
    return results


class SimpleBayesianOptimizer:
    def __init__(self, param_space: Dict[str, Tuple], n_initial: int = 5):
        self.param_space = param_space
        self.n_initial = n_initial
        self.observed_params = []
        self.observed_scores = []

    def suggest(self) -> Dict:
        if len(self.observed_params) < self.n_initial:
            return {k: self._sample(v) for k, v in self.param_space.items()}
        
        best_idx = np.argmax(self.observed_scores)
        best_params = self.observed_params[best_idx]
        
        candidate = {}
        for k, v in self.param_space.items():
            if isinstance(v[0], int):
                candidate[k] = int(np.clip(
                    best_params[k] + np.random.normal(0, (v[1] - v[0]) * 0.1), v[0], v[1]))
            else:
                candidate[k] = np.clip(
                    best_params[k] + np.random.normal(0, (v[1] - v[0]) * 0.1), v[0], v[1])
        return candidate

    def _sample(self, space: Tuple) -> Any:
        low, high = space
        if isinstance(low, int):
            return random.randint(low, high)
        return random.uniform(low, high)

    def update(self, params: Dict, score: float):
        self.observed_params.append(params)
        self.observed_scores.append(score)


def bayesian_optimization(param_space: Dict[str, Tuple], train_ds: TensorDataset, val_ds: TensorDataset,
                          n_trials: int = 20, epochs: int = 30) -> List[Dict]:
    optimizer = SimpleBayesianOptimizer(param_space, n_initial=5)
    results = []
    for i in range(n_trials):
        params = optimizer.suggest()
        hp = Hyperparameters(**params)
        print(f"  Bayes [{i+1}/{n_trials}]: {params}")
        res = run_training(hp, train_ds, val_ds, epochs=epochs)
        score = res["best_val_acc"]
        optimizer.update(params, score)
        results.append({"params": params, "val_acc": score, "history": res["history"]})
    return results


def plot_results(grid_res: List[Dict], random_res: List[Dict], bayes_res: List[Dict], save_path: str = "day54_hpo_results.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Day 54: Hyperparameter Tuning Strategies Comparison", fontsize=14)

    methods = [("Grid Search", grid_res), ("Random Search", random_res), ("Bayesian Opt", bayes_res)]

    for idx, (name, results) in enumerate(methods):
        if not results:
            continue
        scores = [r["val_acc"] for r in results]
        axes[0, idx].plot(scores, 'o-', alpha=0.7)
        axes[0, idx].set_title(f"{name}: Validation Accuracy per Trial")
        axes[0, idx].set_xlabel("Trial")
        axes[0, idx].set_ylabel("Val Accuracy")
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].axhline(max(scores), color='r', linestyle='--', label=f'Best: {max(scores):.4f}')
        axes[0, idx].legend()

        best = max(results, key=lambda x: x["val_acc"])
        hist = best["history"]
        axes[1, idx].plot(hist["train_acc"], label="Train Acc")
        axes[1, idx].plot(hist["val_acc"], label="Val Acc")
        axes[1, idx].set_title(f"{name}: Best Run Learning Curves")
        axes[1, idx].set_xlabel("Epoch")
        axes[1, idx].set_ylabel("Accuracy")
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.close()


def print_summary(grid_res: List[Dict], random_res: List[Dict], bayes_res: List[Dict]):
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*60)
    for name, results in [("Grid Search", grid_res), ("Random Search", random_res), ("Bayesian Opt", bayes_res)]:
        if not results:
            print(f"\n{name}: No results")
            continue
        best = max(results, key=lambda x: x["val_acc"])
        print(f"\n{name}:")
        print(f"  Best Val Acc: {best['val_acc']:.4f}")
        print(f"  Best Params:  {best['params']}")
        print(f"  Trials:       {len(results)}")


def main():
    print(f"Device: {DEVICE}")
    print("Creating dataset...")
    train_ds, val_ds, test_ds = create_dataset(n_samples=5000, n_features=20)

    param_grid = {
        "lr": [1e-3, 3e-3, 1e-2],
        "batch_size": [32, 64, 128],
        "hidden_dim": [64, 128],
        "num_layers": [2, 3],
        "dropout": [0.1, 0.3],
        "weight_decay": [1e-4, 1e-3],
        "optimizer": ["adam", "adamw"],
        "activation": ["relu", "gelu"],
    }

    param_distributions = {
        "lr": lambda: 10**np.random.uniform(-4, -1),
        "batch_size": lambda: random.choice([32, 64, 128, 256]),
        "hidden_dim": lambda: random.choice([32, 64, 128, 256]),
        "num_layers": lambda: random.randint(1, 4),
        "dropout": lambda: np.random.uniform(0.0, 0.5),
        "weight_decay": lambda: 10**np.random.uniform(-5, -2),
        "optimizer": lambda: random.choice(["adam", "adamw", "sgd"]),
        "activation": lambda: random.choice(["relu", "tanh", "gelu"]),
    }

    param_space = {
        "lr": (1e-4, 1e-1),
        "batch_size": (32, 256),
        "hidden_dim": (32, 256),
        "num_layers": (1, 4),
        "dropout": (0.0, 0.5),
        "weight_decay": (1e-5, 1e-2),
        "optimizer": (0, 2),
        "activation": (0, 2),
    }

    optimizer_names = ["adam", "adamw", "sgd"]
    activation_names = ["relu", "tanh", "gelu"]

    def decode_params(p: Dict) -> Dict:
        p = p.copy()
        p["optimizer"] = optimizer_names[int(np.clip(p["optimizer"], 0, 2))]
        p["activation"] = activation_names[int(np.clip(p["activation"], 0, 2))]
        p["batch_size"] = int(p["batch_size"])
        p["hidden_dim"] = int(p["hidden_dim"])
        p["num_layers"] = int(p["num_layers"])
        return p

    print("\n" + "="*60)
    print("GRID SEARCH (subset for speed)")
    print("="*60)
    small_grid = {
        "lr": [1e-3, 1e-2],
        "batch_size": [64, 128],
        "hidden_dim": [64, 128],
        "num_layers": [2],
        "dropout": [0.1, 0.3],
        "weight_decay": [1e-4],
        "optimizer": ["adam"],
        "activation": ["relu"],
    }
    grid_results = grid_search(small_grid, train_ds, val_ds, epochs=20)

    print("\n" + "="*60)
    print("RANDOM SEARCH")
    print("="*60)
    random_results = random_search(param_distributions, train_ds, val_ds, n_trials=15, epochs=20)

    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION (simple)")
    print("="*60)
    bayes_results = bayesian_optimization(param_space, train_ds, val_ds, n_trials=15, epochs=20)
    bayes_results = [{**r, "params": decode_params(r["params"])} for r in bayes_results]

    print_summary(grid_results, random_results, bayes_results)
    plot_results(grid_results, random_results, bayes_results)

    all_results = {
        "grid_search": grid_results,
        "random_search": random_results,
        "bayesian_optimization": bayes_results,
    }
    with open("day54_hpo_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to day54_hpo_results.json")


if __name__ == "__main__":
    main()