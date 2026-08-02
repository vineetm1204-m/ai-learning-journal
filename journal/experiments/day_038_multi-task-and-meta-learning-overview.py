import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. MULTI-TASK LEARNING: Shared backbone, task-specific heads
# ============================================================
class MultiTaskNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_tasks=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x):
        shared_feat = self.shared(x)
        return [head(shared_feat) for head in self.task_heads]

def generate_multitask_data(n_samples=1000, input_dim=10):
    X = torch.randn(n_samples, input_dim)
    # Task 1: y1 = sum of first 5 features + noise
    y1 = X[:, :5].sum(dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)
    # Task 2: y2 = product of features 5-9 + noise
    y2 = X[:, 5:].prod(dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)
    return X, y1, y2

def train_multitask():
    print("=" * 60)
    print("MULTI-TASK LEARNING EXPERIMENT")
    print("=" * 60)
    
    X, y1, y2 = generate_multitask_data(2000)
    train_size = 1600
    X_train, X_test = X[:train_size], X[train_size:]
    y1_train, y1_test = y1[:train_size], y1[train_size:]
    y2_train, y2_test = y2[:train_size], y2[train_size:]

    model = MultiTaskNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X_train, y1_train, y2_train = X_train.to(device), y1_train.to(device), y2_train.to(device)
    X_test, y1_test, y2_test = X_test.to(device), y1_test.to(device), y2_test.to(device)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        pred1, pred2 = model(X_train)
        loss = criterion(pred1, y1_train) + criterion(pred2, y2_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred1, test_pred2 = model(X_test)
                test_loss = criterion(test_pred1, y1_test) + criterion(test_pred2, y2_test)
                print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

    # Compare with single-task baselines
    print("\n--- Single-Task Baselines ---")
    for task_idx, (y_train, y_test, name) in enumerate([(y1_train, y1_test, "Task 1"), (y2_train, y2_test, "Task 2")]):
        single_model = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        opt = optim.Adam(single_model.parameters(), lr=1e-3)
        for _ in range(50):
            single_model.train()
            opt.zero_grad()
            pred = single_model(X_train)
            l = criterion(pred, y_train)
            l.backward()
            opt.step()
        single_model.eval()
        with torch.no_grad():
            test_pred = single_model(X_test)
            print(f"{name} Single-Task Test Loss: {criterion(test_pred, y_test).item():.4f}")

    print()

# ============================================================
# 2. META-LEARNING: MAML on few-shot sine regression
# ============================================================
class MAMLModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=40, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, params=None):
        if params is None:
            return self.net(x)
        # Manual forward with adapted params
        x = torch.relu(torch.linear(x, params[0], params[1]))
        x = torch.relu(torch.linear(x, params[2], params[3]))
        x = torch.linear(x, params[4], params[5])
        return x

def sample_sine_task():
    """Sample a sine wave: y = A * sin(x - phi)"""
    amplitude = np.random.uniform(0.1, 5.0)
    phase = np.random.uniform(0, np.pi)
    x = np.random.uniform(-5, 5, 20)
    y = amplitude * np.sin(x - phase)
    return torch.FloatTensor(x).unsqueeze(1), torch.FloatTensor(y).unsqueeze(1), amplitude, phase

def maml_inner_update(model, support_x, support_y, inner_lr=0.01):
    """Compute adapted parameters via 1 gradient step"""
    preds = model(support_x)
    loss = nn.MSELoss()(preds, support_y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    adapted_params = []
    for param, grad in zip(model.parameters(), grads):
        adapted_params.append(param - inner_lr * grad)
    return adapted_params

def train_maml():
    print("=" * 60)
    print("META-LEARNING (MAML) EXPERIMENT")
    print("=" * 60)
    
    model = MAMLModel().to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=1e-3)
    inner_lr = 0.01
    meta_batch_size = 10
    num_inner_steps = 1

    for meta_iter in range(200):
        meta_optimizer.zero_grad()
        meta_loss = 0.0

        for _ in range(meta_batch_size):
            # Sample task
            sx, sy, _, _ = sample_sine_task()
            qx, qy, _, _ = sample_sine_task()  # Same task, different points
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

            # Inner loop adaptation
            adapted_params = maml_inner_update(model, sx, sy, inner_lr)

            # Query loss with adapted params
            q_pred = model(qx, adapted_params)
            task_loss = nn.MSELoss()(q_pred, qy)
            meta_loss += task_loss

        meta_loss /= meta_batch_size
        meta_loss.backward()
        meta_optimizer.step()

        if meta_iter % 40 == 0:
            print(f"Meta-Iter {meta_iter:3d} | Meta Loss: {meta_loss.item():.4f}")

    # Evaluation: few-shot adaptation on new tasks
    print("\n--- Few-Shot Adaptation Evaluation ---")
    model.eval()
    adaptation_losses = []
    for _ in range(20):
        sx, sy, amp, phase = sample_sine_task()
        qx, qy, _, _ = sample_sine_task()
        sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

        # Adapt
        adapted_params = maml_inner_update(model, sx, sy, inner_lr)
        with torch.no_grad():
            q_pred = model(qx, adapted_params)
            loss = nn.MSELoss()(q_pred, qy).item()
            adaptation_losses.append(loss)

    print(f"Mean Adapted Query Loss: {np.mean(adaptation_losses):.4f} ± {np.std(adaptation_losses):.4f}")

    # Compare with random initialization (no meta-learning)
    print("\n--- Random Init Baseline (No Meta-Learning) ---")
    random_model = MAMLModel().to(device)
    random_losses = []
    for _ in range(20):
        sx, sy, _, _ = sample_sine_task()
        qx, qy, _, _ = sample_sine_task()
        sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
        adapted_params = maml_inner_update(random_model, sx, sy, inner_lr)
        with torch.no_grad():
            q_pred = random_model(qx, adapted_params)
            random_losses.append(nn.MSELoss()(q_pred, qy).item())
    print(f"Mean Random Init Query Loss: {np.mean(random_losses):.4f} ± {np.std(random_losses):.4f}")
    print()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train_multitask()
    train_maml()
    print("=" * 60)
    print("DAY 38 COMPLETE: Multi-task & Meta-learning overview")
    print("=" * 60)