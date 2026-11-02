import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Day 104: RNN Unrolled Through Time - Mini Experiment
# ------------------------------------------------------------
# This script manually implements a simple Vanilla RNN forward pass
# and explicitly visualizes the "unrolled" computational graph
# structure across time steps.
# ------------------------------------------------------------

np.random.seed(42)

# --- 1. Configuration ---
INPUT_DIM = 3      # Dimension of input vector x_t
HIDDEN_DIM = 4     # Dimension of hidden state h_t
OUTPUT_DIM = 2     # Dimension of output vector y_t
SEQ_LEN = 5        # Number of time steps to unroll (T)
BATCH_SIZE = 1     # Keep simple for visualization

# --- 2. Initialize Weights (Shared across all time steps) ---
# In a real RNN, W_xh, W_hh, W_hy are shared. Unrolling just copies the *structure*.
W_xh = np.random.randn(HIDDEN_DIM, INPUT_DIM) * 0.1  # Input -> Hidden
W_hh = np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1  # Hidden -> Hidden (Recurrent)
W_hy = np.random.randn(OUTPUT_DIM, HIDDEN_DIM) * 0.1  # Hidden -> Output

b_h = np.zeros((HIDDEN_DIM, 1))
b_y = np.zeros((OUTPUT_DIM, 1))

# --- 3. Generate Synthetic Input Sequence ---
# Shape: (SEQ_LEN, INPUT_DIM, BATCH_SIZE)
X_seq = np.random.randn(SEQ_LEN, INPUT_DIM, BATCH_SIZE)

# Target sequence (dummy)
Y_true_seq = np.random.randn(SEQ_LEN, OUTPUT_DIM, BATCH_SIZE)

# --- 4. Manual Unrolled Forward Pass ---
# We store the state of every layer at every time step to visualize the graph.
# This mimics the computational graph created by frameworks like PyTorch/TensorFlow
# when they "unroll" the loop for Backpropagation Through Time (BPTT).

print(f"{'='*60}")
print(f"RNN UNROLLED THROUGH TIME (T={SEQ_LEN})")
print(f"{'='*60}")
print(f"Shared Weights: W_xh{W_xh.shape}, W_hh{W_hh.shape}, W_hy{W_hy.shape}")
print(f"Input Sequence Shape: {X_seq.shape} (Time, Features, Batch)")
print("-" * 60)

# Storage for visualization/analysis
hidden_states = []  # h_0, h_1, ..., h_T
outputs = []        # y_1, ..., y_T
pre_activations = [] # a_t = W_xh x_t + W_hh h_{t-1} + b_h

# Initial hidden state h_0
h_prev = np.zeros((HIDDEN_DIM, BATCH_SIZE))
hidden_states.append(h_prev.copy())

print(f"{'Time':<6} | {'Input x_t':<25} | {'Hidden h_{t-1}':<25} | {'Pre-act a_t':<25} | {'Hidden h_t':<25} | {'Output y_t'}")
print("-" * 160)

for t in range(SEQ_LEN):
    x_t = X_seq[t] # (INPUT_DIM, BATCH)
    
    # 1. Compute Pre-activation (Affine transform)
    # a_t = W_xh * x_t + W_hh * h_{t-1} + b_h
    a_t = W_xh @ x_t + W_hh @ h_prev + b_h
    
    # 2. Activation (tanh)
    h_t = np.tanh(a_t)
    
    # 3. Output Layer
    # y_t = W_hy * h_t + b_y
    y_t = W_hy @ h_t + b_y
    
    # Store for graph
    pre_activations.append(a_t.copy())
    hidden_states.append(h_t.copy())
    outputs.append(y_t.copy())
    
    # Print step detail (first batch element only for readability)
    print(f"t={t:<2} | {str(x_t[:,0].round(2)):<25} | {str(h_prev[:,0].round(2)):<25} | {str(a_t[:,0].round(2)):<25} | {str(h_t[:,0].round(2)):<25} | {str(y_t[:,0].round(2))}")
    
    # Update for next step
    h_prev = h_t

print("-" * 160)
print(f"Final Hidden State h_{SEQ_LEN}: {hidden_states[-1][:,0].round(3)}")
print(f"Output Sequence Shape: {np.array(outputs).shape} -> (Time, Output_Dim, Batch)")

# --- 5. Visualizing the Unrolled Graph Structure ---
# We create a diagram showing the data dependencies.
# Nodes: x_t, h_t, y_t. Edges: Weight matrices.

def plot_unrolled_graph(seq_len, hidden_dim, input_dim, output_dim):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-1, seq_len * 2.8)
    ax.set_ylim(-1.5, 3.5)
    ax.axis('off')
    ax.set_title("RNN Computational Graph Unrolled Through Time (Forward Pass)", fontsize=14, weight='bold')

    # Vertical positions for layers
    Y_X = 0.0
    Y_H = 1.5
    Y_Y = 3.0

    # Colors
    COL_X = '#3498db'  # Blue
    COL_H = '#e74c3c'  # Red
    COL_Y = '#2ecc71'  # Green
    COL_W = '#95a5a6'  # Gray

    # Draw Nodes and Edges for each time step
    for t in range(seq_len):
        x_center = t * 2.8 + 0.5
        
        # --- Input x_t ---
        ax.add_patch(plt.Circle((x_center, Y_X), 0.35, color=COL_X, alpha=0.8, zorder=3, ec='k'))
        ax.text(x_center, Y_X, f"$x_{t}$", ha='center', va='center', fontsize=10, color='white', weight='bold')
        ax.text(x_center, Y_X - 0.7, f"Input\n$\\in \\mathbb{R}^{input_dim}$", ha='center', va='top', fontsize=8, color=COL_X)

        # --- Hidden h_t ---
        h_center_x = x_center + 1.0
        ax.add_patch(plt.Circle((h_center_x, Y_H), 0.35, color=COL_H, alpha=0.8, zorder=3, ec='k'))
        ax.text(h_center_x, Y_H, f"$h_{t}$", ha='center', va='center', fontsize=10, color='white', weight='bold')
        ax.text(h_center_x, Y_H + 0.7, f"Hidden\n$\\in \\mathbb{R}^{hidden_dim}$", ha='center', va='bottom', fontsize=8, color=COL_H)

        # --- Output y_t ---
        ax.add_patch(plt.Circle((h_center_x, Y_Y), 0.35, color=COL_Y, alpha=0.8, zorder=3, ec='k'))
        ax.text(h_center_x, Y_Y, f"$y_{t}$", ha='center', va='center', fontsize=10, color='white', weight='bold')
        ax.text(h_center_x, Y_Y + 0.7, f"Output\n$\\in \\mathbb{R}^{output_dim}$", ha='center', va='bottom', fontsize=8, color=COL_Y)

        # --- Edges (Weight Matrices) ---
        # 1. Input -> Hidden (W_xh)
        ax.annotate("", xy=(h_center_x - 0.25, Y_H - 0.2), xytext=(x_center + 0.25, Y_X + 0.2),
                    arrowprops=dict(arrowstyle="->", color=COL_W, lw=1.5, linestyle='-'))
        ax.text((x_center + h_center_x)/2, (Y_X + Y_H)/2 - 0.1, "$W_{xh}$", fontsize=9, color=COL_W, rotation=30, ha='center')

        # 2. Hidden -> Output (W_hy)
        ax.annotate("", xy=(h_center_x, Y_Y - 0.35), xytext=(h_center_x, Y_H + 0.35),
                    arrowprops=dict(arrowstyle="->", color=COL_W, lw=1.5))
        ax.text(h_center_x + 0.4, (Y_H + Y_Y)/2, "$W_{hy}$", fontsize=9, color=COL_W, ha='left')

        # 3. Recurrent Hidden -> Hidden (W_hh) - Connects h_{t-1} to h_t
        if t > 0:
            prev_h_center_x = (t-1) * 2.8 + 1.5
            # Curve the arrow slightly to distinguish from vertical
            ax.annotate("", xy=(h_center_x - 0.35, Y_H), xytext=(prev_h_center_x + 0.35, Y_H),
                        arrowprops=dict(arrowstyle="->", color=COL_H, lw=2, linestyle='--', connectionstyle="arc3,rad=-0.2"))
            mid_x = (prev_h_center_x + h_center_x) / 2
            ax.text(mid_x, Y_H + 0.5, "$W_{hh}$ (Shared)", fontsize=9, color=COL_H, ha='center', weight='bold', style='italic')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Input $x_t$', markerfacecolor=COL_X, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Hidden $h_t$', markerfacecolor=COL_H, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Output $y_t$', markerfacecolor=COL_Y, markersize=10),
        plt.Line2D([0], [0], color=COL_W, lw=2, label='Feedforward Weights ($W_{xh}, W_{hy}$)'),
        plt.Line2D([0], [0], color=COL_H, lw=2, linestyle='--', label='Recurrent Weight ($W_{hh}$ Shared)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig("rnn_unrolled_graph.png", dpi=150, bbox_inches='tight')
    print("\n[Info] Graph visualization saved to 'rnn_unrolled_graph.png'")
    # plt.show() # Commented out for non-interactive environments

plot_unrolled_graph(SEQ_LEN, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM)

# --- 6. Gradient Flow Sanity Check (Conceptual) ---
# In BPTT, gradients flow backwards through the unrolled graph.
# dL/dW_hh = Sum_t (dL/dh_t * dh_t/da_t * h_{t-1}^T)
# This sums gradients across all time steps because W_hh is SHARED.

print("\n" + "="*60)
print("BPTT CONCEPT: Gradient Accumulation for Shared Weights")
print("="*60)
print("For shared weight W_hh:")
print("  dL/dW_hh = Σ_{t=1}^T (δ_t ⊙ (1 - h_t^2)) @ h_{t-1}^T")
print("  Where δ_t = dL/dh_t (backpropagated error from future steps + current output)")
print("\nThis summation is why we 'unroll': we create a distinct node h_t for every t")
print("so the chain rule can be applied explicitly across time.")
print("="*60)