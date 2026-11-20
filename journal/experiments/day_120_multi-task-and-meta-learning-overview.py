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
# 1. SYNTHETIC MULTI-TASK DATA GENERATION
# ============================================================
def generate_multitask_data(n_samples=2000, n_tasks=3, input_dim=10):
    """Generate synthetic data for multiple regression tasks sharing a latent structure."""
    W_shared = torch.randn(input_dim, 5) * 0.5
    task_heads = [torch.randn(5, 1) * 0.5 for _ in range(n_tasks)]
    
    X = torch.randn(n_samples, input_dim)
    ys = []
    for t in range(n_tasks):
        h = X @ W_shared
        y = h @ task_heads[t] + torch.randn(n_samples, 1) * 0.1
        ys.append(y)
    return X, ys, W_shared, task_heads

# ============================================================
# 2. MULTI-TASK MODEL: SHARED BACKBONE + TASK-SPECIFIC HEADS
# ============================================================
class MultiTaskNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, n_tasks=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(16, 1) for _ in range(n_tasks)
        ])
    
    def forward(self, x, task_id=None):
        feat = self.shared(x)
        if task_id is not None:
            return self.heads[task_id](feat)
        return [head(feat) for head in self.heads]

def train_multitask(model, X, ys, epochs=200, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    losses_history = {t: [] for t in range(len(ys))}
    
    for epoch in range(epochs):
        total_loss = 0
        for t, y in enumerate(ys):
            pred = model(X, task_id=t)
            loss = nn.MSELoss()(pred, y)
            losses_history[t].append(loss.item())
            total_loss += loss
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  MTL Epoch {epoch+1}: Total Loss = {total_loss.item():.4f}")
    return losses_history

# ============================================================
# 3. META-LEARNING (MAML) ON FEW-SHOT REGRESSION TASKS
# ============================================================
class MetaLearner(nn.Module):
    """Simple 2-layer net for MAML inner-loop adaptation."""
    def __init__(self, input_dim=1, hidden_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def clone(self):
        """Create a copy with same architecture for inner-loop updates."""
        clone = MetaLearner()
        clone.load_state_dict(self.state_dict())
        return clone

def sample_task(k_shot=5, q_query=10, noise=0.1):
    """Sample a sine regression task: y = a * sin(x + b)"""
    a = np.random.uniform(0.5, 2.0)
    b = np.random.uniform(0, np.pi)
    x_all = np.random.uniform(-5, 5, k_shot + q_query)
    y_all = a * np.sin(x_all + b) + np.random.normal(0, noise, k_shot + q_query)
    
    idx = np.random.permutation(k_shot + q_query)
    x_spt, y_spt = x_all[idx[:k_shot]], y_all[idx[:k_shot]]
    x_qry, y_qry = x_all[idx[k_shot:]], y_all[idx[k_shot:]]
    
    return (torch.FloatTensor(x_spt).unsqueeze(1), torch.FloatTensor(y_spt).unsqueeze(1),
            torch.FloatTensor(x_qry).unsqueeze(1), torch.FloatTensor(y_qry).unsqueeze(1),
            a, b)

def maml_inner_loop(model, x_spt, y_spt, inner_lr=0.01, inner_steps=5):
    """Perform gradient steps on support set (inner loop)."""
    adapted = model.clone()
    adapted.train()
    opt = optim.SGD(adapted.parameters(), lr=inner_lr)
    
    for _ in range(inner_steps):
        pred = adapted(x_spt)
        loss = nn.MSELoss()(pred, y_spt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return adapted

def train_maml(meta_model, n_iterations=2000, meta_batch=4, inner_lr=0.01, meta_lr=1e-3):
    meta_opt = optim.Adam(meta_model.parameters(), lr=meta_lr)
    meta_losses = []
    
    for it in range(n_iterations):
        meta_loss = 0.0
        for _ in range(meta_batch):
            x_spt, y_spt, x_qry, y_qry, _, _ = sample_task()
            x_spt, y_spt = x_spt.to(device), y_spt.to(device)
            x_qry, y_qry = x_qry.to(device), y_qry.to(device)
            
            adapted = maml_inner_loop(meta_model, x_spt, y_spt, inner_lr=inner_lr)
            qry_pred = adapted(x_qry)
            loss = nn.MSELoss()(qry_pred, y_qry)
            meta_loss += loss
        
        meta_loss /= meta_batch
        meta_opt.zero_grad()
        meta_loss.backward()
        meta_opt.step()
        meta_losses.append(meta_loss.item())
        
        if (it + 1) % 500 == 0:
            print(f"  MAML Iter {it+1}: Meta Loss = {meta_loss.item():.4f}")
    return meta_losses

def evaluate_maml(meta_model, n_test_tasks=100, inner_lr=0.01, inner_steps=10):
    """Evaluate meta-model on new tasks with varying adaptation steps."""
    meta_model.eval()
    losses_pre = []
    losses_post = []
    
    for _ in range(n_test_tasks):
        x_spt, y_spt, x_qry, y_qry, _, _ = sample_task(k_shot=5, q_query=20)
        x_spt, y_spt = x_spt.to(device), y_spt.to(device)
        x_qry, y_qry = x_qry.to(device), y_qry.to(device)
        
        # Pre-adaptation loss
        with torch.no_grad():
            pre_pred = meta_model(x_qry)
            losses_pre.append(nn.MSELoss()(pre_pred, y_qry).item())
        
        # Post-adaptation loss
        adapted = maml_inner_loop(meta_model, x_spt, y_spt, inner_lr=inner_lr, inner_steps=inner_steps)
        with torch.no_grad():
            post_pred = adapted(x_qry)
            losses_post.append(nn.MSELoss()(post_pred, y_qry).item())
    
    return np.mean(losses_pre), np.mean(losses_post)

# ============================================================
# 4. COMPARISON: MULTI-TASK vs SINGLE-TASK BASELINES
# ============================================================
class SingleTaskNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_single_task(X, y, epochs=200, lr=1e-3):
    model = SingleTaskNet().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        pred = model(X)
        loss = nn.MSELoss()(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

# ============================================================
# 5. MAIN EXPERIMENT RUNNER
# ============================================================
def run_experiment():
    print("=" * 60)
    print("DAY 120: MULTI-TASK & META-LEARNING MINI-EXPERIMENT")
    print("=" * 60)
    
    # --- Multi-Task Learning ---
    print("\n[1] MULTI-TASK LEARNING (Shared Backbone)")
    print("-" * 40)
    X, ys, W_true, heads_true = generate_multitask_data(n_samples=2000, n_tasks=3)
    X, ys = X.to(device), [y.to(device) for y in ys]
    
    mtl_model = MultiTaskNet(n_tasks=3).to(device)
    mtl_losses = train_multitask(mtl_model, X, ys, epochs=200)
    
    # Compare with single-task baselines
    print("\n  Training single-task baselines...")
    stl_losses = []
    for t in range(3):
        model = train_single_task(X, ys[t], epochs=200)
        with torch.no_grad():
            pred = model(X)
            loss = nn.MSELoss()(pred, ys[t]).item()
            stl_losses.append(loss)
            print(f"    Task {t} STL Loss: {loss:.4f}")
    
    with torch.no_grad():
        for t in range(3):
            pred = mtl_model(X, task_id=t)
            loss = nn.MSELoss()(pred, ys[t]).item()
            print(f"    Task {t} MTL Loss: {loss:.4f} (STL: {stl_losses[t]:.4f})")
    
    # --- Meta-Learning (MAML) ---
    print("\n[2] META-LEARNING (MAML on Sine Regression)")
    print("-" * 40)
    meta_model = MetaLearner().to(device)
    maml_losses = train_maml(meta_model, n_iterations=2000, meta_batch=4)
    
    print("\n  Evaluating on unseen tasks...")
    pre_loss, post_loss = evaluate_maml(meta_model, n_test_tasks=200)
    print(f"  Pre-adaptation MSE:  {pre_loss:.4f}")
    print(f"  Post-adaptation MSE: {post_loss:.4f}")
    print(f"  Improvement: {(pre_loss - post_loss) / pre_loss * 100:.1f}%")
    
    # --- Visualization ---
    print("\n[3] GENERATING PLOTS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # MTL training curves
    ax = axes[0, 0]
    for t in range(3):
        ax.plot(mtl_losses[t], label=f'Task {t}', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Multi-Task Training Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # MAML meta-training curve
    ax = axes[0, 1]
    ax.plot(maml_losses, color='orange', alpha=0.8)
    ax.set_xlabel('Meta-Iteration')
    ax.set_ylabel('Meta-Loss (Query MSE)')
    ax.set_title('MAML Meta-Training Curve')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # MAML adaptation visualization (single task)
    ax = axes[1, 0]
    x_spt, y_spt, x_qry, y_qry, a_true, b_true = sample_task(k_shot=5, q_query=30)
    x_spt, y_spt = x_spt.to(device), y_spt.to(device)
    x_qry, y_qry = x_qry.to(device), y_qry.to(device)
    
    adapted = maml_inner_loop(meta_model, x_spt, y_spt, inner_lr=0.01, inner_steps=10)
    
    with torch.no_grad():
        x_plot = torch.linspace(-5, 5, 100).unsqueeze(1).to(device)
        y_true = a_true * torch.sin(x_plot + b_true)
        y_meta = meta_model(x_plot).cpu()
        y_adapted = adapted(x_plot).cpu()
    
    ax.scatter(x_spt.cpu(), y_spt.cpu(), c='red', s=60, label='Support (K=5)', zorder=5)
    ax.scatter(x_qry.cpu(), y_qry.cpu(), c='gray', s=20, alpha=0.5, label='Query')
    ax.plot(x_plot.cpu(), y_true.cpu(), 'k--', label=f'True: {a_true:.2f}sin(x+{b_true:.2f})', alpha=0.7)
    ax.plot(x_plot.cpu(), y_meta, 'b-', label='Meta-Init', alpha=0.7)
    ax.plot(x_plot.cpu(), y_adapted, 'r-', label='Adapted (10 steps)', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('MAML Adaptation on Unseen Task')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # MTL vs STL comparison bar chart
    ax = axes[1, 1]
    tasks = ['Task 0', 'Task 1', 'Task 2']
    mtl_final = [mtl_losses[t][-1] for t in range(3)]
    x_pos = np.arange(len(tasks))
    width = 0.35
    ax.bar(x_pos - width/2, stl_losses, width, label='Single-Task', alpha=0.7, color='skyblue')
    ax.bar(x_pos + width/2, mtl_final, width, label='Multi-Task', alpha=0.7, color='lightcoral')
    ax.set_ylabel('Final MSE Loss')
    ax.set_title('MTL vs STL Final Performance')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('day120_multitask_metalearning.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to day120_multitask_metalearning.png")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  • Multi-Task Learning shares representations across related tasks,")
    print("    often outperforming single-task baselines when tasks share structure.")
    print("  • MAML learns an initialization that enables fast adaptation (few-shot)")
    print("    to new tasks via gradient-based inner-loop optimization.")
    print("  • Both approaches leverage task distribution structure: MTL for")
    print("    simultaneous multi-task performance, MAML for rapid adaptation.")

if __name__ == "__main__":
    run_experiment()