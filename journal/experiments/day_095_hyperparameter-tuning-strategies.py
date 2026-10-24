import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, ParameterGrid, ParameterSampler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# ─── Synthetic Dataset ───
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.LongTensor(y_val))

# ─── Model Definition ───
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ─── Training / Evaluation ───
def train_eval(params, epochs=30, batch_size=64, verbose=False):
    model = MLP(20, params['hidden_dims'], params['dropout'])
    opt = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_val))
        preds = logits.argmax(1).numpy()
        acc = (preds == y_val).mean()
    return acc

# ─── Search Spaces ───
param_grid = {
    'lr': [1e-3, 3e-3, 1e-2],
    'wd': [0.0, 1e-4, 1e-3],
    'dropout': [0.1, 0.3, 0.5],
    'hidden_dims': [[64], [128], [64, 32], [128, 64]]
}

param_dist = {
    'lr': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    'wd': [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'hidden_dims': [[32], [64], [128], [256], [64,32], [128,64], [256,128], [128,64,32]]
}

# ─── 1. Grid Search ───
def grid_search(param_grid, max_combos=20):
    print("\n=== Grid Search ===")
    best_acc, best_params = 0, None
    combos = list(ParameterGrid(param_grid))[:max_combos]
    for i, params in enumerate(combos):
        acc = train_eval(params, epochs=20, verbose=False)
        print(f"  [{i+1}/{len(combos)}] {params} → val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_params = acc, params
    return best_acc, best_params

# ─── 2. Random Search ───
def random_search(param_dist, n_iter=20):
    print("\n=== Random Search ===")
    best_acc, best_params = 0, None
    sampler = ParameterSampler(param_dist, n_iter=n_iter, random_state=42)
    for i, params in enumerate(sampler):
        acc = train_eval(params, epochs=20, verbose=False)
        print(f"  [{i+1}/{n_iter}] {params} → val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_params = acc, params
    return best_acc, best_params

# ─── 3. Bayesian Optimization (Simple GP-based) ───
def encode_params(params):
    """Flatten params to fixed-length vector for GP."""
    hid_map = {tuple([32]):0, tuple([64]):1, tuple([128]):2, tuple([256]):3,
               tuple([64,32]):4, tuple([128,64]):5, tuple([256,128]):6, tuple([128,64,32]):7}
    return [
        np.log10(params['lr']),
        np.log10(params['wd'] + 1e-10),
        params['dropout'],
        hid_map.get(tuple(params['hidden_dims']), 0)
    ]

def decode_params(vec):
    hid_options = [[32],[64],[128],[256],[64,32],[128,64],[256,128],[128,64,32]]
    return {
        'lr': 10**vec[0],
        'wd': 10**vec[1],
        'dropout': np.clip(vec[2], 0, 0.5),
        'hidden_dims': hid_options[int(np.clip(vec[3], 0, 7))]
    }

def bayesian_opt(param_dist, n_init=5, n_iter=15):
    print("\n=== Bayesian Optimization ===")
    # Initial random points
    X_obs, y_obs = [], []
    sampler = ParameterSampler(param_dist, n_iter=n_init, random_state=123)
    for params in sampler:
        acc = train_eval(params, epochs=20, verbose=False)
        X_obs.append(encode_params(params))
        y_obs.append(acc)
        print(f"  [init] {params} → val_acc={acc:.4f}")

    X_obs = np.array(X_obs)
    y_obs = np.array(y_obs)
    best_idx = y_obs.argmax()
    best_acc, best_params = y_obs[best_idx], decode_params(X_obs[best_idx])

    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, random_state=42)

    for it in range(n_iter):
        gp.fit(X_obs, y_obs)

        # Expected Improvement acquisition (simple version)
        def ei(x):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = sigma + 1e-9
            best = y_obs.max()
            z = (mu - best) / sigma
            from scipy.stats import norm
            return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)

        # Random candidates + optimize best
        candidates = np.random.uniform(
            low=[-4, -5, 0, 0], high=[-2, -2, 0.5, 7], size=(200, 4)
        )
        ei_vals = np.array([ei(c) for c in candidates])
        next_x = candidates[ei_vals.argmax()]

        next_params = decode_params(next_x)
        acc = train_eval(next_params, epochs=20, verbose=False)
        print(f"  [{it+1}/{n_iter}] {next_params} → val_acc={acc:.4f}")

        X_obs = np.vstack([X_obs, next_x])
        y_obs = np.append(y_obs, acc)

        if acc > best_acc:
            best_acc, best_params = acc, next_params

    return best_acc, best_params

# ─── 4. Successive Halving (Hyperband-style) ───
def successive_halving(param_dist, max_epochs=27, eta=3, n_configs=27):
    print("\n=== Successive Halving ===")
    # Sample initial configurations
    sampler = ParameterSampler(param_dist, n_iter=n_configs, random_state=999)
    configs = list(sampler)

    # Rungs: (n_configs, epochs_per_config)
    rungs = []
    n = n_configs
    r = 1
    while n >= 1:
        rungs.append((n, r))
        n = n // eta
        r = r * eta
        if r > max_epochs:
            r = max_epochs
            break

    print(f"  Rungs: {rungs}")

    for rung_idx, (n_keep, epochs) in enumerate(rungs):
        print(f"  Rung {rung_idx+1}: evaluate {len(configs)} configs for {epochs} epochs")
        results = []
        for params in configs:
            acc = train_eval(params, epochs=epochs, verbose=False)
            results.append((acc, params))
            print(f"    {params} → val_acc={acc:.4f}")

        results.sort(key=lambda x: x[0], reverse=True)
        configs = [p for _, p in results[:n_keep]]
        if len(configs) == 1:
            break

    best_params = configs[0]
    best_acc = train_eval(best_params, epochs=max_epochs, verbose=False)
    print(f"  Final best: {best_params} → val_acc={best_acc:.4f}")
    return best_acc, best_params

# ─── Run All Strategies ───
if __name__ == "__main__":
    print("=" * 60)
    print("Day 95: Hyperparameter Tuning Strategies Mini-Experiment")
    print("=" * 60)

    results = {}

    # Grid Search
    t0 = time.time()
    gs_acc, gs_params = grid_search(param_grid, max_combos=16)
    results['Grid Search'] = (gs_acc, gs_params, time.time() - t0)

    # Random Search
    t0 = time.time()
    rs_acc, rs_params = random_search(param_dist, n_iter=20)
    results['Random Search'] = (rs_acc, rs_params, time.time() - t0)

    # Bayesian Optimization
    t0 = time.time()
    bo_acc, bo_params = bayesian_opt(param_dist, n_init=5, n_iter=10)
    results['Bayesian Opt'] = (bo_acc, bo_params, time.time() - t0)

    # Successive Halving
    t0 = time.time()
    sh_acc, sh_params = successive_halving(param_dist, max_epochs=27, eta=3, n_configs=27)
    results['Successive Halving'] = (sh_acc, sh_params, time.time() - t0)

    # ─── Summary ───
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Best Val Acc':<15} {'Time (s)':<10} Best Params")
    print("-" * 60)
    for name, (acc, params, t) in results.items():
        param_str = f"lr={params['lr']:.1e}, wd={params['wd']:.1e}, do={params['dropout']:.1f}, hid={params['hidden_dims']}"
        print(f"{name:<20} {acc:.4f}           {t:.1f}        {param_str}")

    # Best overall
    best_name = max(results, key=lambda k: results[k][0])
    print(f"\n🏆 Best strategy: {best_name} (val_acc={results[best_name][0]:.4f})")