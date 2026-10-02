import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import copy
import random

# ==========================================
# Configuration
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 10
CLIENT_FRACTION = 1.0  # Fraction of clients participating per round
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LR = 0.01
ROUNDS = 15
DP_NOISE_MULTIPLIER = 0.1  # Gaussian noise std dev for DP (0.0 to disable)
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ==========================================
# Model Definition (Simple CNN for MNIST)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# Data Preparation (Non-IID Dirichlet Split)
# ==========================================
def get_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return train_ds, test_ds

def dirichlet_split(dataset, num_clients, alpha=0.5, min_size=10):
    """Splits dataset indices non-IID using Dirichlet distribution."""
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Balance proportions so clients with fewer samples get more
        proportions = np.array([p * (len(idx_j) < len(targets) / num_clients) for p, idx_j in zip(proportions, client_indices)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        split = np.split(idx_k, proportions)
        for i, idx in enumerate(split):
            client_indices[i].extend(idx)
            
    # Ensure minimum size
    for i in range(num_clients):
        if len(client_indices[i]) < min_size:
             # Pad from random other clients (simplification)
             pass 
    return [Subset(dataset, indices) for indices in client_indices]

# ==========================================
# Federated Learning Components
# ==========================================
def add_dp_noise(state_dict, noise_multiplier, device):
    """Adds Gaussian noise to model parameters for Differential Privacy."""
    if noise_multiplier == 0:
        return state_dict
    noisy_state = {}
    for key, param in state_dict.items():
        noise = torch.normal(0, noise_multiplier, size=param.shape, device=device)
        noisy_state[key] = param + noise
    return noisy_state

def train_local(model, dataloader, epochs, lr, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    total_loss = 0.0
    for _ in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / (len(dataloader) * epochs)

def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    acc = 100. * correct / len(dataloader.dataset)
    return avg_loss, acc

def fed_avg(global_model, client_models, client_weights):
    """Federated Averaging aggregation."""
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i][key].float() * client_weights[i] 
                                        for i in range(len(client_models))], 0).sum(0)
    global_model.load_state_dict(global_dict)
    return global_model

# ==========================================
# Main Experiment Loop
# ==========================================
def run_experiment():
    print(f"--- Day 82: Federated Learning & Privacy-Preserving ML ---")
    print(f"Device: {DEVICE} | Clients: {NUM_CLIENTS} | Rounds: {ROUNDS} | DP Noise: {DP_NOISE_MULTIPLIER}")
    
    # 1. Data
    train_ds, test_ds = get_mnist_data()
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    client_datasets = dirichlet_split(train_ds, NUM_CLIENTS, alpha=0.5)
    client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]
    client_sizes = [len(ds) for ds in client_datasets]
    total_size = sum(client_sizes)
    client_weights = [s / total_size for s in client_sizes]
    
    print(f"Client data distribution (sizes): {client_sizes}")
    
    # 2. Global Model
    global_model = SimpleCNN().to(DEVICE)
    
    # 3. Training Rounds
    history = {'round': [], 'global_acc': [], 'global_loss': [], 'avg_local_loss': []}
    
    for r in range(1, ROUNDS + 1):
        # Select clients
        m = max(int(CLIENT_FRACTION * NUM_CLIENTS), 1)
        selected_indices = np.random.choice(NUM_CLIENTS, m, replace=False)
        
        client_models = []
        local_losses = []
        
        # Client Update
        for idx in selected_indices:
            local_model = copy.deepcopy(global_model)
            loss = train_local(local_model, client_loaders[idx], LOCAL_EPOCHS, LR, DEVICE)
            
            # Privacy: Add noise to local model weights before sending to server
            local_state = add_dp_noise(local_model.state_dict(), DP_NOISE_MULTIPLIER, DEVICE)
            client_models.append(local_state)
            local_losses.append(loss)
        
        # Server Aggregation
        # Re-normalize weights for selected clients
        sel_weights = [client_weights[i] for i in selected_indices]
        sel_weights = np.array(sel_weights) / np.sum(sel_weights)
        
        global_model = fed_avg(global_model, client_models, sel_weights)
        
        # Global Evaluation
        g_loss, g_acc = evaluate(global_model, test_loader, DEVICE)
        avg_local_loss = np.mean(local_losses)
        
        history['round'].append(r)
        history['global_acc'].append(g_acc)
        history['global_loss'].append(g_loss)
        history['avg_local_loss'].append(avg_local_loss)
        
        print(f"Round {r:02d}/{ROUNDS} | Global Acc: {g_acc:.2f}% | Global Loss: {g_loss:.4f} | Avg Local Loss: {avg_local_loss:.4f}")

    print("\n--- Experiment Complete ---")
    print(f"Final Global Accuracy: {history['global_acc'][-1]:.2f}%")
    
    # Simple Privacy Accounting Note
    if DP_NOISE_MULTIPLIER > 0:
        print(f"DP Mechanism: Gaussian Noise (sigma={DP_NOISE_MULTIPLIER}) applied to client updates.")
        print("Note: Formal (epsilon, delta) accounting requires RDP/Moments Accountant (omitted for brevity).")

if __name__ == "__main__":
    run_experiment()