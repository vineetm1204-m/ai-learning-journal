import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# ============================================================
# Day 117: Graph Neural Networks (GNNs) Basics
# Self-contained mini-experiment: From-scratch GCN on synthetic graph
# ============================================================

# ---- Reproducibility ----
SEED = 117
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- 1. Synthetic Graph Generation ----
def generate_synthetic_graph(num_nodes=100, num_features=16, num_classes=4, edge_prob=0.05):
    """
    Creates a synthetic graph with community structure.
    Nodes in same community have similar features and dense connections.
    """
    # Community assignment
    community = torch.randint(0, num_classes, (num_nodes,))
    
    # Node features: community-specific mean + noise
    community_means = torch.randn(num_classes, num_features) * 2.0
    x = community_means[community] + torch.randn(num_nodes, num_features) * 0.5
    
    # Adjacency: higher probability within community
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            p = 0.3 if community[i] == community[j] else edge_prob
            if random.random() < p:
                adj[i, j] = adj[j, i] = 1.0
    
    # Ensure no isolated nodes
    for i in range(num_nodes):
        if adj[i].sum() == 0:
            j = random.randint(0, num_nodes-1)
            if j != i:
                adj[i, j] = adj[j, i] = 1.0
    
    # Labels = community
    y = community
    
    # Train/val/test split
    indices = list(range(num_nodes))
    random.shuffle(indices)
    train_idx = indices[:60]
    val_idx = indices[60:80]
    test_idx = indices[80:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return x, adj, y, train_mask, val_mask, test_mask

# ---- 2. Graph Utilities ----
def normalize_adjacency(adj, add_self_loops=True):
    """
    Symmetric normalization: D^{-1/2} (A + I) D^{-1/2}
    """
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0))
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

# ---- 3. GCN Layer (From Scratch) ----
class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer: H^{(l+1)} = sigma( D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)} )
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj_norm):
        # x: [N, in_features], adj_norm: [N, N]
        support = x @ self.weight          # [N, out_features]
        out = adj_norm @ support           # [N, out_features]
        if self.bias is not None:
            out = out + self.bias
        return out

# ---- 4. Two-Layer GCN Model ----
class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.5):
        super().__init__()
        self.conv1 = GCNLayer(in_features, hidden_features)
        self.conv2 = GCNLayer(hidden_features, out_features)
        self.dropout = dropout
    
    def forward(self, x, adj_norm):
        x = self.conv1(x, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj_norm)
        return x

# ---- 5. Training & Evaluation ----
def train_step(model, x, adj_norm, y, mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    logits = model(x, adj_norm)
    loss = criterion(logits[mask], y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, x, adj_norm, y, mask):
    model.eval()
    logits = model(x, adj_norm)
    preds = logits[mask].argmax(dim=1)
    acc = (preds == y[mask]).float().mean().item()
    return acc

# ---- 6. Main Experiment ----
def main():
    print("=" * 60)
    print("Day 117: GNN Basics - From-Scratch GCN Experiment")
    print("=" * 60)
    
    # Hyperparameters
    NUM_NODES = 200
    NUM_FEATURES = 32
    NUM_CLASSES = 5
    HIDDEN_DIM = 64
    LR = 0.01
    WEIGHT_DECAY = 5e-4
    EPOCHS = 200
    PATIENCE = 20
    
    # Generate graph
    print("\n[1/5] Generating synthetic graph with community structure...")
    x, adj, y, train_mask, val_mask, test_mask = generate_synthetic_graph(
        num_nodes=NUM_NODES,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        edge_prob=0.02
    )
    print(f"    Nodes: {NUM_NODES}, Features: {NUM_FEATURES}, Classes: {NUM_CLASSES}")
    print(f"    Edges: {int(adj.sum().item() // 2)} (undirected)")
    print(f"    Train/Val/Test: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")
    
    # Normalize adjacency
    print("\n[2/5] Computing normalized adjacency (D^{-1/2} A D^{-1/2})...")
    adj_norm = normalize_adjacency(adj)
    print(f"    Adjacency shape: {adj_norm.shape}")
    print(f"    Row sums (should be ~1): {adj_norm.sum(dim=1)[:5]}")
    
    # Model
    print("\n[3/5] Building 2-layer GCN model...")
    model = GCN(NUM_FEATURES, HIDDEN_DIM, NUM_CLASSES, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("\n[4/5] Training...")
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        loss = train_step(model, x, adj_norm, y, train_mask, optimizer, criterion)
        val_acc = evaluate(model, x, adj_norm, y, val_mask)
        test_acc = evaluate(model, x, adj_norm, y, test_mask)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
            # Save best model state
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | Loss: {loss:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")
        
        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    # Restore best model
    model.load_state_dict(best_state)
    
    # Final evaluation
    print("\n[5/5] Final Evaluation...")
    final_train_acc = evaluate(model, x, adj_norm, y, train_mask)
    final_val_acc = evaluate(model, x, adj_norm, y, val_mask)
    final_test_acc = evaluate(model, x, adj_norm, y, test_mask)
    
    print(f"    Train Accuracy: {final_train_acc:.4f}")
    print(f"    Val Accuracy:   {final_val_acc:.4f}")
    print(f"    Test Accuracy:  {final_test_acc:.4f}")
    
    # ---- 7. Inspect Learned Representations ----
    print("\n[Bonus] Analyzing learned node embeddings...")
    model.eval()
    with torch.no_grad():
        # Get hidden layer activations (after first GCN + ReLU)
        h1 = model.conv1(x, adj_norm)
        h1 = F.relu(h1)
        logits = model(x, adj_norm)
        preds = logits.argmax(dim=1)
    
    # Compute homophily: fraction of edges connecting same-class nodes
    edge_mask = adj > 0
    same_class_edges = (y.unsqueeze(1) == y.unsqueeze(0)) & edge_mask
    homophily = same_class_edges.sum().item() / edge_mask.sum().item()
    print(f"    Graph homophily (label): {homophily:.4f}")
    
    # Prediction homophily
    same_pred_edges = (preds.unsqueeze(1) == preds.unsqueeze(0)) & edge_mask
    pred_homophily = same_pred_edges.sum().item() / edge_mask.sum().item()
    print(f"    Graph homophily (pred):  {pred_homophily:.4f}")
    
    # Embedding visualization (PCA-like: project to 2D via SVD)
    U, S, V = torch.svd_lowrank(h1, q=2)
    emb_2d = U * S  # [N, 2]
    
    # Compute class separation in embedding space
    class_centers = []
    for c in range(NUM_CLASSES):
        mask = (y == c)
        if mask.any():
            class_centers.append(emb_2d[mask].mean(dim=0))
    class_centers = torch.stack(class_centers)
    center_dists = torch.cdist(class_centers, class_centers)
    avg_center_dist = center_dists[center_dists > 0].mean().item()
    print(f"    Avg inter-class center distance (2D proj): {avg_center_dist:.4f}")
    
    print("\n" + "=" * 60)
    print("Experiment complete. Key takeaways:")
    print("  - GCN propagates features via normalized adjacency")
    print("  - Message passing = linear transform + neighborhood aggregation")
    print("  - Symmetric normalization prevents degree explosion")
    print("  - Two layers = 2-hop receptive field")
    print("=" * 60)

if __name__ == "__main__":
    main()