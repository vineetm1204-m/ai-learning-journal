import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from collections import OrderedDict

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 10
CLIENTS_PER_ROUND = 5
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_ROUNDS = 15
DP_NOISE_MULTIPLIER = 0.1
DP_MAX_GRAD_NORM = 1.0

class SimpleNN(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def generate_synthetic_data(num_samples=1000, input_dim=20, num_classes=3, num_clients=10, non_iid=True):
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    true_weights = np.random.randn(input_dim, num_classes).astype(np.float32)
    logits = X @ true_weights
    y = np.argmax(logits + np.random.randn(*logits.shape) * 0.5, axis=1)
    
    client_data = []
    indices = np.arange(num_samples)
    
    if non_iid:
        class_indices = [np.where(y == c)[0] for c in range(num_classes)]
        client_class_prefs = np.random.dirichlet([0.5]*num_classes, num_clients)
        
        for c in range(num_clients):
            client_indices = []
            for cls in range(num_classes):
                n_cls = int(len(class_indices[cls]) * client_class_prefs[c, cls] / num_clients * 2)
                n_cls = min(n_cls, len(class_indices[cls]))
                if n_cls > 0:
                    chosen = np.random.choice(class_indices[cls], n_cls, replace=False)
                    client_indices.extend(chosen)
                    class_indices[cls] = np.setdiff1d(class_indices[cls], chosen)
            if len(client_indices) == 0:
                client_indices = np.random.choice(indices, num_samples // num_clients, replace=False)
            client_data.append((X[client_indices], y[client_indices]))
    else:
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        for idx in splits:
            client_data.append((X[idx], y[idx]))
    
    return client_data

class Client:
    def __init__(self, client_id, X, y, model_template):
        self.client_id = client_id
        self.dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.LongTensor(y)
        )
        self.loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        self.model = copy.deepcopy(model_template).to(DEVICE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
    
    def set_parameters(self, global_params):
        self.model.load_state_dict(global_params)
    
    def train_local(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for _ in range(LOCAL_EPOCHS):
            for X_batch, y_batch in self.loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                
                if DP_NOISE_MULTIPLIER > 0:
                    self._clip_and_add_noise()
                
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        
        avg_loss = total_loss / (len(self.loader) * LOCAL_EPOCHS)
        accuracy = 100. * correct / total
        return self.model.state_dict(), avg_loss, accuracy
    
    def _clip_and_add_noise(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = DP_MAX_GRAD_NORM / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.normal(0, DP_NOISE_MULTIPLIER * DP_MAX_GRAD_NORM, 
                                   size=p.grad.shape, device=DEVICE)
                p.grad.data.add_(noise)
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = self.model(X_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        return 100. * correct / total

class Server:
    def __init__(self, model_template):
        self.global_model = copy.deepcopy(model_template).to(DEVICE)
        self.global_params = self.global_model.state_dict()
    
    def aggregate(self, client_updates, client_weights):
        new_params = OrderedDict()
        total_weight = sum(client_weights)
        
        for key in self.global_params.keys():
            weighted_sum = torch.zeros_like(self.global_params[key], dtype=torch.float32)
            for update, weight in zip(client_updates, client_weights):
                weighted_sum += update[key].float() * weight
            new_params[key] = (weighted_sum / total_weight).to(self.global_params[key].dtype)
        
        self.global_params = new_params
        self.global_model.load_state_dict(self.global_params)
    
    def get_global_params(self):
        return copy.deepcopy(self.global_params)
    
    def evaluate(self, test_loader):
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = self.global_model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        return 100. * correct / total, total_loss / len(test_loader)

def run_federated_experiment():
    print("=" * 60)
    print("DAY 41: FEDERATED LEARNING & PRIVACY-PRESERVING ML")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Clients: {NUM_CLIENTS}, Clients/Round: {CLIENTS_PER_ROUND}")
    print(f"Local Epochs: {LOCAL_EPOCHS}, Rounds: {NUM_ROUNDS}")
    print(f"DP Noise Multiplier: {DP_NOISE_MULTIPLIER}, Max Grad Norm: {DP_MAX_GRAD_NORM}")
    print("-" * 60)
    
    print("\n[1/5] Generating synthetic non-IID data...")
    client_data = generate_synthetic_data(
        num_samples=5000, input_dim=20, num_classes=3, 
        num_clients=NUM_CLIENTS, non_iid=True
    )
    
    test_X = np.random.randn(1000, 20).astype(np.float32)
    test_y = np.random.randint(0, 3, 1000).astype(np.int64)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(test_X), torch.LongTensor(test_y)
        ), batch_size=64, shuffle=False
    )
    
    print(f"    Train samples per client: {[len(y) for _, y in client_data]}")
    print(f"    Test samples: {len(test_y)}")
    
    print("\n[2/5] Initializing clients and server...")
    model_template = SimpleNN(input_dim=20, hidden_dim=64, num_classes=3)
    clients = [
        Client(i, X, y, model_template) 
        for i, (X, y) in enumerate(client_data)
    ]
    server = Server(model_template)
    
    client_weights = [len(y) for _, y in client_data]
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]
    
    print("\n[3/5] Starting federated training rounds...")
    history = {"round": [], "global_acc": [], "global_loss": [], "avg_local_acc": []}
    
    for round_idx in range(NUM_ROUNDS):
        selected_clients = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
        client_updates = []
        local_accuracies = []
        
        global_params = server.get_global_params()
        
        for cid in selected_clients:
            clients[cid].set_parameters(global_params)
            update, loss, acc = clients[cid].train_local()
            client_updates.append(update)
            local_accuracies.append(acc)
        
        server.aggregate(client_updates, [client_weights[cid] for cid in selected_clients])
        
        global_acc, global_loss = server.evaluate(test_loader)
        avg_local_acc = np.mean(local_accuracies)
        
        history["round"].append(round_idx + 1)
        history["global_acc"].append(global_acc)
        history["global_loss"].append(global_loss)
        history["avg_local_acc"].append(avg_local_acc)
        
        print(f"    Round {round_idx+1:2d}/{NUM_ROUNDS} | "
              f"Global Acc: {global_acc:5.2f}% | "
              f"Global Loss: {global_loss:.4f} | "
              f"Avg Local Acc: {avg_local_acc:5.2f}%")
    
    print("\n[4/5] Evaluating final model...")
    final_acc, final_loss = server.evaluate(test_loader)
    print(f"    Final Test Accuracy: {final_acc:.2f}%")
    print(f"    Final Test Loss: {final_loss:.4f}")
    
    print("\n[5/5] Privacy Analysis...")
    print(f"    Differential Privacy: {'ENABLED' if DP_NOISE_MULTIPLIER > 0 else 'DISABLED'}")
    if DP_NOISE_MULTIPLIER > 0:
        epsilon_estimate = DP_NOISE_MULTIPLIER * np.sqrt(2 * NUM_ROUNDS * np.log(1.25 / 1e-5))
        print(f"    Estimated ε (approx): {epsilon_estimate:.2f}")
        print(f"    Gradient Clipping Norm: {DP_MAX_GRAD_NORM}")
        print(f"    Noise Scale: {DP_NOISE_MULTIPLIER * DP_MAX_GRAD_NORM:.4f}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return history, server, clients

def demonstrate_secure_aggregation_concept():
    print("\n" + "=" * 60)
    print("BONUS: SECURE AGGREGATION CONCEPT DEMONSTRATION")
    print("=" * 60)
    print("In production FL, secure aggregation ensures the server")
    print("only sees the SUM of client updates, not individual ones.")
    print("This is typically done via:")
    print("  1. Pairwise masking (Bonawitz et al., 2017)")
    print("  2. Threshold Paillier encryption")
    print("  3. Shamir secret sharing")
    print()
    print("Simplified pairwise masking example:")
    
    num_clients = 5
    model_dim = 10
    updates = [np.random.randn(model_dim) for _ in range(num_clients)]
    
    masks = {}
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            mask = np.random.randn(model_dim)
            masks[(i, j)] = mask
            masks[(j, i)] = -mask
    
    masked_updates = []
    for i in range(num_clients):
        masked = updates[i].copy()
        for j in range(num_clients):
            if i != j:
                masked += masks[(i, j)]
        masked_updates.append(masked)
    
    server_sum = sum(masked_updates)
    true_sum = sum(updates)
    
    print(f"  True sum of updates: {true_sum[:3]}...")
    print(f"  Server computed sum: {server_sum[:3]}...")
    print(f"  Match: {np.allclose(true_sum, server_sum)}")
    print("  (Masks cancel out when summed!)")

if __name__ == "__main__":
    history, server, clients = run_federated_experiment()
    demonstrate_secure_aggregation_concept()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS FOR DAY 41:")
    print("=" * 60)
    print("1. Federated Learning keeps data LOCAL, shares MODEL UPDATES")
    print("2. FedAvg: Weighted average of client model parameters")
    print("3. Non-IID data is the main challenge (client drift)")
    print("4. Differential Privacy adds noise to gradients for privacy")
    print("5. Secure Aggregation prevents server from seeing individual updates")
    print("6. Trade-off: Privacy (DP noise) vs Utility (model accuracy)")
    print("7. Communication efficiency: Compression, quantization, sparsification")
    print("=" * 60)