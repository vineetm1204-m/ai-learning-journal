import numpy as np

# ------------------------------------------------------------
# Day 24: LSTM Gates Explained - Mini Experiment
# ------------------------------------------------------------
# This script implements a minimal LSTM cell from scratch using NumPy
# to visualize the internal gate activations (input, forget, output)
# and cell state dynamics on a simple sequence task.
# ------------------------------------------------------------

np.random.seed(42)

# -------------------------
# 1. LSTM Cell Implementation
# -------------------------
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weights: [W_i, W_f, W_o, W_c] for input, forget, output, candidate
        # Shapes: (hidden_size, input_size + hidden_size)
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        
        # Biases
        self.b_i = np.zeros((hidden_size, 1))
        self.b_f = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))
        self.b_c = np.zeros((hidden_size, 1))
        
        # For visualization: store last forward pass details
        self.last_gates = {}
    
    def forward(self, x_t, h_prev, c_prev):
        """
        x_t: (input_size, 1)
        h_prev: (hidden_size, 1)
        c_prev: (hidden_size, 1)
        Returns: h_t, c_t
        """
        # Concatenate input and previous hidden state
        combined = np.vstack([x_t, h_prev])  # (input_size + hidden_size, 1)
        
        # Gate computations
        i_t = self._sigmoid(self.W_i @ combined + self.b_i)      # Input gate
        f_t = self._sigmoid(self.W_f @ combined + self.b_f)      # Forget gate
        o_t = self._sigmoid(self.W_o @ combined + self.b_o)      # Output gate
        c_tilde = np.tanh(self.W_c @ combined + self.b_c)        # Candidate cell state
        
        # Cell state update
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Hidden state output
        h_t = o_t * np.tanh(c_t)
        
        # Store for inspection
        self.last_gates = {
            'i_t': i_t.copy(),
            'f_t': f_t.copy(),
            'o_t': o_t.copy(),
            'c_tilde': c_tilde.copy(),
            'c_t': c_t.copy(),
            'h_t': h_t.copy()
        }
        
        return h_t, c_t
    
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

# -------------------------
# 2. Simple Sequence Task: Learn to Delay Input by 1 Step
# -------------------------
# We'll train the LSTM to output the previous input (identity with lag=1).
# This requires the forget gate to preserve information and input gate to write new info.

def generate_data(seq_len=10, input_size=1):
    """Generate random sequence and target (shifted by 1)."""
    X = np.random.randn(seq_len, input_size, 1)  # (seq_len, input_size, 1)
    Y = np.zeros_like(X)
    Y[1:] = X[:-1]  # Target is previous input
    Y[0] = 0        # First target is zero
    return X, Y

# -------------------------
# 3. Training Loop (Simple SGD)
# -------------------------
def train_lstm(cell, X, Y, epochs=200, lr=0.01):
    seq_len = X.shape[0]
    losses = []
    
    for epoch in range(epochs):
        # Initialize hidden and cell states
        h = np.zeros((cell.hidden_size, 1))
        c = np.zeros((cell.hidden_size, 1))
        
        total_loss = 0.0
        
        # Forward pass through sequence
        for t in range(seq_len):
            h, c = cell.forward(X[t], h, c)
            # Simple MSE loss on hidden state (using first hidden unit as output)
            pred = h[0, 0]
            target = Y[t, 0, 0]
            total_loss += (pred - target) ** 2
        
        avg_loss = total_loss / seq_len
        losses.append(avg_loss)
        
        # Backpropagation Through Time (BPTT) - simplified for demonstration
        # We'll compute gradients numerically for clarity (not efficient but educational)
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")
    
    return losses

# Numerical gradient check (for verification only)
def numerical_gradient(cell, X, Y, param_name, eps=1e-5):
    """Compute numerical gradient for a parameter."""
    original = getattr(cell, param_name).copy()
    grad = np.zeros_like(original)
    it = np.nditer(original, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        # Perturb +
        original[idx] += eps
        setattr(cell, param_name, original)
        loss_plus = compute_loss(cell, X, Y)
        
        # Perturb -
        original[idx] -= 2 * eps
        setattr(cell, param_name, original)
        loss_minus = compute_loss(cell, X, Y)
        
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        original[idx] += eps  # restore
        setattr(cell, param_name, original)
        it.iternext()
    
    return grad

def compute_loss(cell, X, Y):
    h = np.zeros((cell.hidden_size, 1))
    c = np.zeros((cell.hidden_size, 1))
    loss = 0.0
    for t in range(X.shape[0]):
        h, c = cell.forward(X[t], h, c)
        loss += (h[0, 0] - Y[t, 0, 0]) ** 2
    return loss / X.shape[0]

# -------------------------
# 4. Visualization of Gate Dynamics
# -------------------------
def visualize_gates(cell, X):
    """Run forward pass and print gate activations for each timestep."""
    h = np.zeros((cell.hidden_size, 1))
    c = np.zeros((cell.hidden_size, 1))
    
    print("\n=== GATE ACTIVATIONS PER TIMESTEP ===")
    print(f"{'t':>3} | {'Input':>8} | {'Forget':>8} | {'Output':>8} | {'Candidate':>8} | {'Cell':>8} | {'Hidden':>8}")
    print("-" * 85)
    
    for t in range(X.shape[0]):
        h, c = cell.forward(X[t], h, c)
        g = cell.last_gates
        # Show first hidden unit for readability
        print(f"{t:3d} | {g['i_t'][0,0]:8.4f} | {g['f_t'][0,0]:8.4f} | {g['o_t'][0,0]:8.4f} | "
              f"{g['c_tilde'][0,0]:8.4f} | {g['c_t'][0,0]:8.4f} | {g['h_t'][0,0]:8.4f}")

# -------------------------
# 5. Main Experiment
# -------------------------
if __name__ == "__main__":
    # Hyperparameters
    INPUT_SIZE = 1
    HIDDEN_SIZE = 4  # Small hidden size for clear visualization
    SEQ_LEN = 8
    EPOCHS = 300
    LR = 0.05
    
    # Generate data
    X, Y = generate_data(SEQ_LEN, INPUT_SIZE)
    
    # Initialize LSTM
    lstm = LSTMCell(INPUT_SIZE, HIDDEN_SIZE)
    
    print("=" * 60)
    print("DAY 24: LSTM GATES EXPLAINED - MINI EXPERIMENT")
    print("=" * 60)
    print(f"Task: Learn to output previous input (delay by 1)")
    print(f"Sequence length: {SEQ_LEN}, Hidden size: {HIDDEN_SIZE}")
    print(f"Input sequence (first 5): {X[:5,0,0]}")
    print(f"Target sequence (first 5): {Y[:5,0,0]}")
    
    # Show initial gate behavior (random weights)
    print("\n--- INITIAL GATE BEHAVIOR (Random Weights) ---")
    visualize_gates(lstm, X)
    
    # Train
    print("\n--- TRAINING ---")
    losses = train_lstm(lstm, X, Y, epochs=EPOCHS, lr=LR)
    
    # Show final gate behavior
    print("\n--- FINAL GATE BEHAVIOR (After Training) ---")
    visualize_gates(lstm, X)
    
    # Final predictions vs targets
    print("\n--- PREDICTIONS vs TARGETS ---")
    h = np.zeros((HIDDEN_SIZE, 1))
    c = np.zeros((HIDDEN_SIZE, 1))
    print(f"{'t':>3} | {'Input':>8} | {'Target':>8} | {'Prediction':>10} | {'Error':>8}")
    print("-" * 50)
    for t in range(SEQ_LEN):
        h, c = lstm.forward(X[t], h, c)
        pred = h[0, 0]
        target = Y[t, 0, 0]
        print(f"{t:3d} | {X[t,0,0]:8.4f} | {target:8.4f} | {pred:10.4f} | {abs(pred-target):8.4f}")
    
    print("\n--- KEY INSIGHTS ---")
    print("1. INPUT GATE (i_t): Controls how much NEW information enters the cell.")
    print("   - Should be HIGH when current input is relevant for future.")
    print("2. FORGET GATE (f_t): Controls how much PREVIOUS cell state is retained.")
    print("   - Should be HIGH to remember long-term dependencies.")
    print("3. OUTPUT GATE (o_t): Controls how much cell state is EXPOSED as hidden state.")
    print("   - Modulates the readout from cell memory.")
    print("4. CELL STATE (c_t): The 'conveyor belt' - additive updates allow gradient flow.")
    print("   - c_t = f_t * c_{t-1} + i_t * c_tilde")
    print("5. After training, notice how gates adapt to the delay task:")
    print("   - Forget gate ~1.0 (preserve input in cell)")
    print("   - Input gate ~1.0 (write new input)")
    print("   - Output gate modulates readout for prediction")
    print("\nExperiment complete.")