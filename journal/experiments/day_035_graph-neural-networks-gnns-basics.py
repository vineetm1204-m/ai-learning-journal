import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# 1. Synthetic Graph Generation (Cora-like)
# ==========================================
def generate_synthetic_graph(num_nodes=2708, num_features=1433, num_classes=7, edge_prob=0.005, seed=42):
    """
    Generates a synthetic citation graph resembling Cora.
    Nodes: Papers. Edges: Citations. Features: Bag-of-words. Labels: Topics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Features: Sparse binary vectors (Bag of Words)
    x = torch.zeros(num_nodes, num_features)
    for i in range(num_nodes):
        # Each paper has ~10-20 active words
        num_words = np.random.randint(10, 20)
        indices = np.random.choice(num_features, num_words, replace=False)
        x[i, indices] = 1.0

    # 2. Edges: Random connections with community structure (homophily)
    # Assign communities first to drive homophily
    y = torch.randint(0, num_classes, (num_nodes,))
    
    edge_list = []
    # High probability within class, low probability across classes
    p_intra = 0.02
    p_inter = 0.0005
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if y[i] == y[j]:
                if np.random.rand() < p_intra:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
            else:
                if np.random.rand() < p_inter:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
    
    # Ensure graph is not empty
    if not edge_list:
        edge_list = [[0, 1], [1, 0]]
        
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # 3. Masks (Standard Planetoid split ratios roughly)
    num_train = int(num_nodes * 0.05)
    num_val = int(num_nodes * 0.15)
    
    indices = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train+num_val]] = True
    test_mask[indices[num_train+num_val:]] = True
    
    data = Data(x=x, edge_index=edge_index, y=y, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

# ==========================================
# 2. GCN Model Definition
# ==========================================
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        """Returns embeddings from the first layer for visualization."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x

# ==========================================
# 3. Training & Evaluation
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
def visualize_embeddings(model, data, epoch, max_nodes=500):
    """Plots 2D t-SNE of embeddings colored by ground truth."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("sklearn not installed, skipping t-SNE visualization.")
        return

    model.eval()
    # Get embeddings from hidden layer
    embeddings = model.get_embeddings(data.x, data.edge_index).cpu().numpy()
    labels = data.y.cpu().numpy()
    
    # Subsample for speed
    if embeddings.shape[0] > max_nodes:
        idx = np.random.choice(embeddings.shape[0], max_nodes, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        
    z = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500).fit_transform(embeddings)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8)
    plt.colorbar(scatter, label='Class')
    plt.title(f'GCN Hidden Embeddings (t-SNE) - Epoch {epoch}')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tight_layout()
    plt.savefig(f'gnn_embeddings_epoch_{epoch}.png')
    plt.close()
    print(f"  -> Saved embedding visualization to gnn_embeddings_epoch_{epoch}.png")

def visualize_graph_structure(data, max_nodes=200):
    """Plots the raw graph topology."""
    try:
        G = to_networkx(data, to_undirected=True)
        # Subsample
        if G.number_of_nodes() > max_nodes:
            nodes = list(G.nodes())[:max_nodes]
            G = G.subgraph(nodes)
        
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, seed=42, k=0.15, iterations=20)
        nx.draw_networkx_nodes(G, pos, node_size=10, node_color=data.y[:max_nodes].numpy(), cmap='tab10', alpha=0.6)
        nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.2)
        plt.title("Graph Topology (Subsampled)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('gnn_graph_structure.png')
        plt.close()
        print("  -> Saved graph structure to gnn_graph_structure.png")
    except Exception as e:
        print(f"  -> Graph visualization failed: {e}")

# ==========================================
# 5. Main Experiment Runner
# ==========================================
def main():
    print("=" * 60)
    print("Day 35: Graph Neural Networks (GNN) Basics - Mini Experiment")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hyperparameters
    HIDDEN_CHANNELS = 16
    LR = 0.01
    WEIGHT_DECAY = 5e-4
    EPOCHS = 200
    PATIENCE = 20

    # 1. Data
    print("\n[1/5] Generating Synthetic Citation Graph...")
    data = generate_synthetic_graph()
    data = data.to(device)
    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_node_features}, Classes: {data.y.max().item()+1}")
    print(f"  Train/Val/Test: {data.train_mask.sum().item()}/{data.val_mask.sum().item()}/{data.test_mask.sum().item()}")

    # 2. Model
    print("\n[2/5] Initializing 2-Layer GCN...")
    model = GCN(data.num_node_features, HIDDEN_CHANNELS, data.y.max().item()+1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"  Model:\n{model}")

    # 3. Visualize Raw Graph
    print("\n[3/5] Visualizing Graph Structure...")
    visualize_graph_structure(data.cpu())

    # 4. Training Loop
    print("\n[4/5] Starting Training...")
    best_val_acc = 0
    best_test_acc = 0
    epochs_no_improve = 0
    
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, data, optimizer, criterion)
        val_acc = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            epochs_no_improve = 0
            # Save best model state
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
            
        # Visualize embeddings at specific intervals
        if epoch in [1, 50, 100, EPOCHS]:
            visualize_embeddings(model, data.cpu(), epoch)
            
        if epochs_no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    # 5. Final Evaluation
    print("\n[5/5] Final Results")
    print("-" * 30)
    model.load_state_dict(best_model_state)
    final_test_acc = evaluate(model, data, data.test_mask)
    print(f"Best Val Accuracy:  {best_val_acc:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    print("-" * 30)
    print("Experiment Complete.")

if __name__ == "__main__":
    main()