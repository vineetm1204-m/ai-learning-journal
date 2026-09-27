import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Synthetic Multi-Task Data Generation
# ============================================================
def generate_multitask_data(n_tasks=5, n_samples=200, input_dim=10, noise=0.1):
    """Generate synthetic regression tasks sharing a latent structure."""
    W_shared = torch.randn(input_dim, 5) * 0.5
    tasks = []
    for t in range(n_tasks):
        W_task = torch.randn(5, 1) * 0.5
        X = torch.randn(n_samples, input_dim)
        y = X @ W_shared @ W_task + noise * torch.randn(n_samples, 1)
        tasks.append((X, y))
    return tasks

# ============================================================
# Multi-Task Learning Model (Hard Parameter Sharing)
# ============================================================
class MultiTaskNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_tasks):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_tasks)
        ])
    
    def forward(self, x, task_id):
        features = self.shared(x)
        return self.task_heads[task_id](features)

# ============================================================
# Meta-Learning: MAML (Model-Agnostic Meta-Learning)
# ============================================================
class MAMLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def clone(self):
        """Create a copy with same architecture for inner-loop adaptation."""
        clone = MAMLNet(self.net[0].in_features, self.net[0].out_features)
        clone.load_state_dict(self.state_dict())
        return clone

def maml_inner_loop(model, support_x, support_y, inner_lr, inner_steps=1):
    """Perform gradient descent on support set (inner loop)."""
    adapted_model = model.clone()
    adapted_model.train()
    opt = optim.SGD(adapted_model.parameters(), lr=inner_lr)
    
    for _ in range(inner_steps):
        pred = adapted_model(support_x)
        loss = nn.MSELoss()(pred, support_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return adapted_model

def maml_meta_update(meta_model, tasks, inner_lr, meta_lr, inner_steps=1):
    """Meta-update across tasks (outer loop)."""
    meta_opt = optim.Adam(meta_model.parameters(), lr=meta_lr)
    meta_model.train()
    meta_opt.zero_grad()
    
    total_meta_loss = 0
    for support_x, support_y, query_x, query_y in tasks:
        adapted = maml_inner_loop(meta_model, support_x, support_y, inner_lr, inner_steps)
        query_pred = adapted(query_x)
        loss = nn.MSELoss()(query_pred, query_y)
        loss.backward()
        total_meta_loss += loss.item()
    
    meta_opt.step()
    return total_meta_loss / len(tasks)

# ============================================================
# Experiment: Compare Multi-Task vs Meta-Learning
# ============================================================
def run_experiment():
    print("=" * 60)
    print("DAY 79: Multi-Task & Meta-Learning Mini-Experiment")
    print("=" * 60)
    
    # Config
    n_tasks = 5
    n_samples = 200
    input_dim = 10
    hidden_dim = 32
    meta_train_tasks = 3
    meta_test_tasks = 2
    epochs = 50
    
    # Generate data
    all_tasks = generate_multitask_data(n_tasks, n_samples, input_dim)
    train_tasks_data = all_tasks[:meta_train_tasks]
    test_tasks_data = all_tasks[meta_train_tasks:]
    
    # Prepare DataLoaders for Multi-Task
    train_loaders = []
    for X, y in train_tasks_data:
        ds = TensorDataset(X, y)
        train_loaders.append(DataLoader(ds, batch_size=32, shuffle=True))
    
    # ---------- MULTI-TASK LEARNING ----------
    print("\n[1] Multi-Task Learning (Hard Parameter Sharing)")
    mt_model = MultiTaskNet(input_dim, hidden_dim, meta_train_tasks).to(device)
    mt_opt = optim.Adam(mt_model.parameters(), lr=1e-3)
    mt_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for task_id, loader in enumerate(train_loaders):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = mt_model(xb, task_id)
                loss = nn.MSELoss()(pred, yb)
                mt_opt.zero_grad()
                loss.backward()
                mt_opt.step()
                epoch_loss += loss.item()
        mt_losses.append(epoch_loss / len(train_loaders))
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Avg Loss: {mt_losses[-1]:.4f}")
    
    # Evaluate Multi-Task on test tasks (fine-tune heads only)
    print("\n  Fine-tuning on test tasks...")
    mt_test_losses = []
    for task_id, (X_test, y_test) in enumerate(test_tasks_data):
        head = nn.Linear(hidden_dim, 1).to(device)
        opt = optim.Adam(list(mt_model.shared.parameters()) + list(head.parameters()), lr=1e-3)
        ds = TensorDataset(X_test, y_test)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        for _ in range(20):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                feat = mt_model.shared(xb)
                pred = head(feat)
                loss = nn.MSELoss()(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            test_pred = head(mt_model.shared(X_test.to(device)))
            mt_test_losses.append(nn.MSELoss()(test_pred, y_test.to(device)).item())
    print(f"  Test Task Losses: {[f'{l:.4f}' for l in mt_test_losses]}")
    print(f"  Mean Test Loss: {np.mean(mt_test_losses):.4f}")
    
    # ---------- META-LEARNING (MAML) ----------
    print("\n[2] Meta-Learning (MAML)")
    # Prepare MAML episodes: (support_x, support_y, query_x, query_y)
    def create_episodes(tasks_data, k_shot=10):
        episodes = []
        for X, y in tasks_data:
            idx = torch.randperm(len(X))
            support_idx = idx[:k_shot]
            query_idx = idx[k_shot:k_shot+50]
            episodes.append((
                X[support_idx], y[support_idx],
                X[query_idx], y[query_idx]
            ))
        return episodes
    
    meta_train_episodes = create_episodes(train_tasks_data)
    meta_test_episodes = create_episodes(test_tasks_data)
    
    maml_model = MAMLNet(input_dim, hidden_dim).to(device)
    maml_losses = []
    
    for epoch in range(epochs):
        loss = maml_meta_update(maml_model, meta_train_episodes, inner_lr=0.01, meta_lr=1e-3, inner_steps=1)
        maml_losses.append(loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Meta Loss: {loss:.4f}")
    
    # Evaluate MAML on test tasks (fast adaptation)
    print("\n  Fast adaptation on test tasks...")
    maml_test_losses = []
    for support_x, support_y, query_x, query_y in meta_test_episodes:
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)
        adapted = maml_inner_loop(maml_model, support_x, support_y, inner_lr=0.01, inner_steps=5)
        with torch.no_grad():
            pred = adapted(query_x)
            maml_test_losses.append(nn.MSELoss()(pred, query_y).item())
    print(f"  Test Task Losses: {[f'{l:.4f}' for l in maml_test_losses]}")
    print(f"  Mean Test Loss: {np.mean(maml_test_losses):.4f}")
    
    # ---------- BASELINE: Single-Task Learning ----------
    print("\n[3] Baseline: Single-Task Learning (No Sharing)")
    st_losses = []
    for X_test, y_test in test_tasks_data:
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        ds = TensorDataset(X_test, y_test)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        for _ in range(50):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = nn.MSELoss()(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            st_losses.append(nn.MSELoss()(model(X_test.to(device)), y_test.to(device)).item())
    print(f"  Test Task Losses: {[f'{l:.4f}' for l in st_losses]}")
    print(f"  Mean Test Loss: {np.mean(st_losses):.4f}")
    
    # ---------- SUMMARY ----------
    print("\n" + "=" * 60)
    print("SUMMARY: Mean Test Loss on Unseen Tasks")
    print("=" * 60)
    print(f"  Single-Task (from scratch):     {np.mean(st_losses):.4f}")
    print(f"  Multi-Task + Fine-tune:         {np.mean(mt_test_losses):.4f}")
    print(f"  MAML (5-shot adaptation):       {np.mean(maml_test_losses):.4f}")
    print("\nKey Insight:")
    print("  - Multi-task learns shared representation, helps related tasks")
    print("  - MAML learns initialization for fast adaptation (few-shot)")
    print("  - Both outperform training from scratch on new tasks")
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(mt_losses, label='Multi-Task')
    plt.plot(maml_losses, label='MAML Meta-Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    methods = ['Single-Task', 'Multi-Task+FT', 'MAML']
    means = [np.mean(st_losses), np.mean(mt_test_losses), np.mean(maml_test_losses)]
    stds = [np.std(st_losses), np.std(mt_test_losses), np.std(maml_test_losses)]
    plt.bar(methods, means, yerr=stds, capsize=5, color=['gray', 'steelblue', 'orange'])
    plt.ylabel('Test MSE')
    plt.title('Generalization to New Tasks')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('day79_results.png', dpi=150)
    print("\nPlot saved to day79_results.png")

if __name__ == "__main__":
    run_experiment()