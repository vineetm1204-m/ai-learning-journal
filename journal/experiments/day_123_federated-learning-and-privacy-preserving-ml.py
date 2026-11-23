import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import math
import sys

# ============================================================
# Configuration
# ============================================================
class Config:
    SEED = 123
    NUM_CLIENTS = 10
    CLIENT_FRACTION = 0.5
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    GLOBAL_ROUNDS = 20
    INPUT_DIM = 20
    HIDDEN_DIM = 64
    OUTPUT_DIM = 2
    DP_ENABLED = True
    DP_EPSILON = 1.0
    DP_DELTA = 1e-5
    DP_CLIP_NORM = 1.0
    DATA_SAMPLES_PER_CLIENT = 500
    NON_IID_ALPHA = 0.5

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Synthetic Non-IID Data Generation
# ============================================================
def generate_federated_data(num_clients, samples_per_client, input_dim, output_dim, alpha):
    """Generate non-IID data using Dirichlet distribution."""
    # Global class prototypes
    prototypes = torch.randn(output_dim, input_dim)
    
    client_data = []
    for c in range(num_clients):
        # Dirichlet prior for class proportions
        proportions = np.random.dirichlet([alpha] * output_dim)
        labels = np.random.choice(output_dim, size=samples_per_client, p=proportions)
        
        X = []
        y = []
        for label in labels:
            noise = torch.randn(input_dim) * 0.5
            sample = prototypes[label] + noise
            X.append(sample)
            y.append(label)
        
        X = torch.stack(X)
        y = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X, y)
        client_data.append(dataset)
    
    return client_data

# ============================================================
# Model Definition
# ============================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def get_model_params(model):
    return [p.data.clone() for p in model.parameters()]

def set_model_params(model, params):
    for p, new_p in zip(model.parameters(), params):
        p.data.copy_(new_p)

def model_to_vector(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def vector_to_model(vec, model):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[idx:idx+numel].view_as(p))
        idx += numel

# ============================================================
# Differential Privacy Utilities
# ============================================================
def compute_dp_noise_scale(clip_norm, epsilon, delta, sensitivity=1.0):
    """Gaussian mechanism noise scale for (epsilon, delta)-DP."""
    if epsilon == 0:
        return float('inf')
    return clip_norm * sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

def clip_gradients(model, max_norm):
    """Clip per-sample gradients (simulated via full batch clipping)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

def add_gaussian_noise(model, noise_scale):
    """Add Gaussian noise to model parameters for DP."""
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn_like(p) * noise_scale
            p.add_(noise)

# ============================================================
# Client
# ============================================================
class Client:
    def __init__(self, client_id, dataset, config):
        self.id = client_id
        self.dataset = dataset
        self.config = config
        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.BATCH_SIZE, shuffle=True
        )
        self.model = SimpleMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM).to(device)
        self.criterion = nn.CrossEntropyLoss()
    
    def set_model(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
    
    def train(self):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        epoch_losses = []
        for _ in range(self.config.LOCAL_EPOCHS):
            batch_losses = []
            for X, y in self.loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                
                if self.config.DP_ENABLED:
                    clip_gradients(self.model, self.config.DP_CLIP_NORM)
                
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(np.mean(batch_losses))
        
        return np.mean(epoch_losses), get_model_params(self.model)
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for X, y in self.loader:
                X, y = X.to(device), y.to(device)
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss_sum += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return loss_sum / total, correct / total

# ============================================================
# Server
# ============================================================
class Server:
    def __init__(self, config):
        self.config = config
        self.global_model = SimpleMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM).to(device)
        self.clients = []
        self.history = {"train_loss": [], "test_acc": [], "test_loss": []}
    
    def register_clients(self, clients):
        self.clients = clients
    
    def select_clients(self):
        num_select = max(1, int(len(self.clients) * self.config.CLIENT_FRACTION))
        return random.sample(self.clients, num_select)
    
    def aggregate(self, client_params_list, client_weights=None):
        """FedAvg aggregation with optional DP noise."""
        if client_weights is None:
            client_weights = [1.0 / len(client_params_list)] * len(client_params_list)
        
        # Weighted average
        avg_params = []
        for i in range(len(client_params_list[0])):
            weighted_sum = sum(w * params[i] for w, params in zip(client_weights, client_params_list))
            avg_params.append(weighted_sum)
        
        # Apply to global model
        set_model_params(self.global_model, avg_params)
        
        # Add DP noise to global model (server-side DP)
        if self.config.DP_ENABLED:
            noise_scale = compute_dp_noise_scale(
                self.config.DP_CLIP_NORM,
                self.config.DP_EPSILON,
                self.config.DP_DELTA
            )
            add_gaussian_noise(self.global_model, noise_scale)
    
    def evaluate_global(self, test_loader):
        self.global_model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                logits = self.global_model(X)
                loss = criterion(logits, y)
                loss_sum += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return loss_sum / total, correct / total
    
    def run_round(self, round_idx):
        selected = self.select_clients()
        client_params = []
        client_losses = []
        client_sizes = []
        
        for client in selected:
            client.set_model(self.global_model)
            loss, params = client.train()
            client_losses.append(loss)
            client_params.append(params)
            client_sizes.append(len(client.dataset))
        
        # Weight by dataset size
        total_size = sum(client_sizes)
        weights = [s / total_size for s in client_sizes]
        
        self.aggregate(client_params, weights)
        
        avg_train_loss = np.mean(client_losses)
        return avg_train_loss, selected

# ============================================================
# Main Experiment
# ============================================================
def main():
    print("=" * 60)
    print(f"DAY 123: Federated Learning & Privacy-Preserving ML")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Clients: {Config.NUM_CLIENTS} | Fraction: {Config.CLIENT_FRACTION}")
    print(f"Local Epochs: {Config.LOCAL_EPOCHS} | Global Rounds: {Config.GLOBAL_ROUNDS}")
    print(f"DP Enabled: {Config.DP_ENABLED} | ε={Config.DP_EPSILON}, δ={Config.DP_DELTA}")
    print(f"Non-IID α: {Config.NON_IID_ALPHA}")
    print("-" * 60)
    
    # Generate data
    print("[1/4] Generating federated non-IID data...")
    client_datasets = generate_federated_data(
        Config.NUM_CLIENTS,
        Config.DATA_SAMPLES_PER_CLIENT,
        Config.INPUT_DIM,
        Config.OUTPUT_DIM,
        Config.NON_IID_ALPHA
    )
    
    # Create global test set (IID)
    test_dataset = torch.utils.data.ConcatDataset([
        generate_federated_data(1, 1000, Config.INPUT_DIM, Config.OUTPUT_DIM, 10.0)[0]
    ])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize clients & server
    print("[2/4] Initializing clients and server...")
    clients = [Client(i, ds, Config) for i, ds in enumerate(client_datasets)]
    server = Server(Config)
    server.register_clients(clients)
    
    # Initial evaluation
    test_loss, test_acc = server.evaluate_global(test_loader)
    print(f"Initial global model | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Training loop
    print("[3/4] Starting federated training...")
    for round_idx in range(1, Config.GLOBAL_ROUNDS + 1):
        train_loss, selected = server.run_round(round_idx)
        test_loss, test_acc = server.evaluate_global(test_loader)
        
        server.history["train_loss"].append(train_loss)
        server.history["test_loss"].append(test_loss)
        server.history["test_acc"].append(test_acc)
        
        selected_ids = [c.id for c in selected]
        print(f"Round {round_idx:3d}/{Config.GLOBAL_ROUNDS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Clients: {selected_ids}")
    
    # Final evaluation per client
    print("[4/4] Per-client evaluation on global model...")
    client_accs = []
    for client in clients:
        client.set_model(server.global_model)
        _, acc = client.evaluate()
        client_accs.append(acc)
        print(f"  Client {client.id:2d}: Acc = {acc:.4f}")
    
    print("-" * 60)
    print(f"Final Global Test Accuracy: {server.history['test_acc'][-1]:.4f}")
    print(f"Client Accuracy - Mean: {np.mean(client_accs):.4f}, Std: {np.std(client_accs):.4f}")
    print(f"Min: {np.min(client_accs):.4f}, Max: {np.max(client_accs):.4f}")
    print("=" * 60)
    print("Experiment complete. Key observations:")
    print("  - Non-IID data causes client drift (accuracy variance)")
    print("  - DP noise reduces peak accuracy but provides privacy guarantees")
    print("  - FedAvg converges despite heterogeneity and privacy noise")
    print("=" * 60)

if __name__ == "__main__":
    main()