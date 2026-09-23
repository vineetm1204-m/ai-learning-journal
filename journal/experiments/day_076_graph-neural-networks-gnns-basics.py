import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Synthetic Graph Generation (Cora-like)
# ==========================================
def generate_synthetic_graph(num_nodes=2708, num_features=1433, num_classes=7, edge_prob=0.005, seed=42):
    """
    Generates a synthetic citation graph similar to Cora.
    Nodes: Papers. Edges: Citations. Features: Bag-of-words. Labels: Topics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Node Features (Sparse Bag-of-Words simulation)
    # Each node has ~1% non-zero features
    x = torch.zeros(num_nodes, num_features)
    for i in range(num_nodes):
        num_words = np.random.randint(10, 50)
        indices = np.random.choice(num_features, num_words, replace=False)
        x[i, indices] = torch.rand(num_words)

    # 2. Labels (Community structure)
    # Assign labels to create clusters
    y = torch.randint(0, num_classes, (num_nodes,))

    # 3. Edges (Preferential attachment within class + noise)
    edge_list = []
    # High probability within class
    for c in range(num_classes):
        class_nodes = (y == c).nonzero(as_tuple=True)[0]
        n_class = len(class_nodes)
        for i in range(n_class):
            for j in range(i + 1, n_class):
                if np.random.rand() < 0.05: # 5% connection density within class
                    edge_list.append([class_nodes[i].item(), class_nodes[j].item()])
                    edge_list.append([class_nodes[j].item(), class_nodes[i].item()]) # Undirected

    # Low probability cross-class (noise)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if y[i] != y[j] and np.random.rand() < 0.0005:
                edge_list.append([i, j])
                edge_list.append([j, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 4. Masks (Standard Planetoid split: 140 train, 500 val, 1000 test)
    indices = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:140]] = True
    val_mask[indices[140:640]] = True
    test_mask[indices[640:1640]] = True

    data = Data(x=x, edge_index=edge_index, y=y, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

# ==========================================
# 2. GCN Model Definition
# ==========================================
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index)
        return x

# ==========================================
# 3. Training & Evaluation Loop
# ==========================================
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    acc = (pred == data.y[mask]).sum().item() / mask.sum().item()
    return acc

# ==========================================
# 4. Visualization Helper
# ==========================================
def visualize_graph(data, preds=None, title="Graph Structure"):
    """Plots a subgraph for visualization (first 200 nodes)."""
    # Convert to NetworkX for layout
    G = to_networkx(data, to_undirected=True)
    # Subgraph for visibility
    sub_nodes = list(range(min(200, data.num_nodes)))
    G_sub = G.subgraph(sub_nodes)
    
    pos = nx.spring_layout(G_sub, seed=42, k=0.5)
    
    plt.figure(figsize=(10, 8))
    if preds is not None:
        colors = preds[sub_nodes].cpu().numpy()
    else:
        colors = data.y[sub_nodes].cpu().numpy()
    
    nx.draw_networkx_nodes(G_sub, pos, node_color=colors, cmap=plt.cm.Set1, node_size=50, alpha=0.8)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.1, width=0.5)
    plt.title(title)
    plt.axis('off')
    plt.show()

# ==========================================
# 5. Main Execution Block
# ==========================================
if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    HIDDEN_CHANNELS = 16
    LR = 0.01
    WEIGHT_DECAY = 5e-4
    EPOCHS = 200
    PATIENCE = 20

    # 1. Load Data
    print("\n[1/5] Generating Synthetic Graph Data...")
    dataset = generate_synthetic_graph()
    dataset = dataset.to(device)
    print(f"Nodes: {dataset.num_nodes}, Edges: {dataset.num_edges}, Features: {dataset.num_node_features}, Classes: {dataset.y.max().item()+1}")
    print(f"Train/Val/Test nodes: {dataset.train_mask.sum().item()}/{dataset.val_mask.sum().item()}/{dataset.test_mask.sum().item()}")

    # 2. Init Model
    print("\n[2/5] Initializing 2-Layer GCN...")
    model = GCN(dataset.num_node_features, HIDDEN_CHANNELS, dataset.y.max().item()+1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    # 3. Training
    print("\n[3/5] Starting Training...")
    best_val_acc = 0
    best_test_acc = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, dataset, optimizer, criterion)
        val_acc = evaluate(model, dataset, dataset.val_mask)
        test_acc = evaluate(model, dataset, dataset.test_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            epochs_no_improve = 0
            # Save best model state
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # 4. Final Evaluation
    print("\n[4/5] Final Evaluation...")
    model.load_state_dict(best_model_state)
    final_test_acc = evaluate(model, dataset, dataset.test_mask)
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Corresponding Test Accuracy: {final_test_acc:.4f}")

    # 5. Visualization (Optional - requires matplotlib)
    print("\n[5/5] Generating Visualization (First 200 nodes)...")
    try:
        model.eval()
        with torch.no_grad():
            logits = model(dataset.x, dataset.edge_index)
            preds = logits.argmax(dim=1)
        
        visualize_graph(dataset.cpu(), preds.cpu(), title="GCN Predictions (First 200 Nodes)")
        visualize_graph(dataset.cpu(), title="Ground Truth Labels (First 200 Nodes)")
    except Exception as e:
        print(f"Visualization skipped: {e}")