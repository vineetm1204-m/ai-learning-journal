import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DAY 1: PERCEPTRONS & THE BIOLOGICAL NEURON ANALOGY
# ============================================================
# This script implements a Perceptron from scratch to solve a
# linearly separable problem (AND gate), visualizing the
# decision boundary evolution and mapping components to biology.
# ============================================================

# ------------------------------------------------------------
# 1. BIOLOGICAL ANALOGY MAPPING (Printed for the Journal)
# ------------------------------------------------------------
def print_biological_analogy():
    print("=" * 60)
    print("DAY 1 JOURNAL ENTRY: THE PERCEPTRON")
    print("=" * 60)
    print("\n[ BIOLOGICAL NEURON ]          -> [ ARTIFICIAL PERCEPTRON ]")
    print("----------------------------------------------------------------")
    print("Dendrites (Receive signals)    -> Input Vector (x)             ")
    print("Synapses (Weights/Strength)    -> Weight Vector (w)            ")
    print("Soma (Summation)               -> Weighted Sum (z = w·x + b)   ")
    print("Axon Hillock (Threshold)       -> Activation Function (f(z))   ")
    print("Axon (Output signal)           -> Output (y_hat)               ")
    print("----------------------------------------------------------------")
    print("\nKey Insight: The Perceptron models the 'Integrate-and-Fire'")
    print("mechanism. If weighted input exceeds threshold -> Fire (1).")
    print("Else -> Silence (0).")
    print("=" * 60 + "\n")

# ------------------------------------------------------------
# 2. DATA GENERATION: AND Gate (Linearly Separable)
# ------------------------------------------------------------
def get_and_data():
    # Inputs: [Input1, Input2]
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    # Labels: AND logic (Only 1,1 -> 1)
    y = np.array([0, 0, 0, 1])
    return X, y

# ------------------------------------------------------------
# 3. PERCEPTRON IMPLEMENTATION (From Scratch)
# ------------------------------------------------------------
class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        # Weights (Synaptic strengths) initialized small random
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0          # Threshold offset
        self.lr = lr             # Learning rate (Plasticity)
        self.epochs = epochs
        self.history = []        # Track boundary for visualization

    def weighted_sum(self, x):
        """Soma: Linear combination z = w·x + b"""
        return np.dot(x, self.weights) + self.bias

    def activation(self, z):
        """Axon Hillock: Heaviside Step Function (Threshold)"""
        return 1 if z > 0 else 0

    def predict(self, x):
        z = self.weighted_sum(x)
        return self.activation(z)

    def train(self, X, y):
        print(f"Training started (LR={self.lr}, Epochs={self.epochs})...\n")
        converged = False

        for epoch in range(self.epochs):
            total_error = 0
            # Store weights for decision boundary plotting
            self.history.append((self.weights.copy(), self.bias))

            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction

                # Hebbian-style Update Rule (Delta Rule):
                # Δw = η * (target - output) * input
                # "Neurons that fire together, wire together"
                if error != 0:
                    self.weights += error * xi * self.lr
                    self.bias += error * self.lr
                    total_error += abs(error)

            if total_error == 0:
                print(f"Converged perfectly at Epoch {epoch + 1}!")
                converged = True
                self.history.append((self.weights.copy(), self.bias))
                break
        
        if not converged:
            print(f"Stopped after {self.epochs} epochs (Data may not be linearly separable).")
        
        return converged

# ------------------------------------------------------------
# 4. VISUALIZATION: Decision Boundary Evolution
# ------------------------------------------------------------
def plot_decision_boundary(X, y, model):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Plot 1: Final Decision Boundary ---
    ax = axes[0]
    # Plot data points
    ax.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Class 0 (False)', s=100, edgecolors='k')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Class 1 (True)', s=100, edgecolors='k')
    
    # Plot boundary: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
    w1, w2 = model.weights
    b = model.bias
    
    if abs(w2) > 1e-6:
        x_vals = np.array([-0.5, 1.5])
        y_vals = -(w1 * x_vals + b) / w2
        ax.plot(x_vals, y_vals, 'k-', linewidth=2, label='Decision Boundary')
        # Shade regions
        ax.fill_between(x_vals, y_vals, 2, alpha=0.1, color='blue')
        ax.fill_between(x_vals, -1, y_vals, alpha=0.1, color='red')
    else:
        # Vertical line case
        x_line = -b / w1
        ax.axvline(x=x_line, color='k', linewidth=2)
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Input 1 (Dendrite 1)")
    ax.set_ylabel("Input 2 (Dendrite 2)")
    ax.set_title("Final Learned Decision Boundary")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

    # --- Plot 2: Learning Dynamics (Weight Trajectory) ---
    ax = axes[1]
    history = model.history
    w1_hist = [h[0][0] for h in history]
    w2_hist = [h[0][1] for h in history]
    
    ax.plot(w1_hist, w2_hist, 'o-', color='purple', markersize=4)
    ax.scatter(w1_hist[0], w2_hist[0], c='green', s=100, label='Init', zorder=5, edgecolors='k')
    ax.scatter(w1_hist[-1], w2_hist[-1], c='red', s=100, label='Final', zorder=5, edgecolors='k')
    
    ax.set_xlabel("Weight 1 (Synapse 1 Strength)")
    ax.set_ylabel("Weight 2 (Synapse 2 Strength)")
    ax.set_title("Weight Space Trajectory (Learning Path)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 5. MAIN EXECUTION BLOCK
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1. Print Theory
    print_biological_analogy()

    # 2. Prepare Data
    X, y = get_and_data()
    print(f"Training Data (AND Gate):\n{X}\nLabels: {y}\n")

    # 3. Initialize Model
    p = Perceptron(input_size=2, lr=0.1, epochs=50)

    # 4. Train
    p.train(X, y)

    # 5. Final Evaluation
    print("\n--- Final Evaluation ---")
    print(f"Learned Weights (w): {p.weights.round(3)}")
    print(f"Learned Bias (b):    {p.bias.round(3)}")
    print("\nTruth Table Verification:")
    print("Input  | Target | Predicted | Match")
    print("-------|--------|-----------|------")
    for xi, target in zip(X, y):
        pred = p.predict(xi)
        match = "✓" if pred == target else "✗"
        print(f"  {xi[0]} {xi[1]}   |   {target}    |     {pred}      |  {match}")

    # 6. Visualize
    plot_decision_boundary(X, y, p)