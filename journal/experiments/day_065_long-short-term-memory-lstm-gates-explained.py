import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class LSTMCell:
    """Educational LSTM cell exposing all gate activations."""
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weights: [W_f, W_i, W_c, W_o] for [forget, input, candidate, output]
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * 0.1
        self.b = np.zeros((4 * hidden_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x, h_prev, c_prev):
        """
        x: (input_size, 1)
        h_prev: (hidden_size, 1)
        c_prev: (hidden_size, 1)
        Returns: h_next, c_next, gates_dict
        """
        combined = np.vstack([h_prev, x])  # (hidden+input, 1)
        gates = self.W @ combined + self.b  # (4*hidden, 1)
        
        # Split gates
        f = self.sigmoid(gates[0:self.hidden_size])           # Forget gate
        i = self.sigmoid(gates[self.hidden_size:2*self.hidden_size])  # Input gate
        c_tilde = np.tanh(gates[2*self.hidden_size:3*self.hidden_size])  # Candidate
        o = self.sigmoid(gates[3*self.hidden_size:4*self.hidden_size])   # Output gate
        
        # Cell state update
        c_next = f * c_prev + i * c_tilde
        h_next = o * np.tanh(c_next)
        
        gates_dict = {
            'forget': f.flatten(),
            'input': i.flatten(),
            'candidate': c_tilde.flatten(),
            'output': o.flatten(),
            'cell_state': c_next.flatten(),
            'hidden_state': h_next.flatten()
        }
        return h_next, c_next, gates_dict


def run_experiment():
    print("=" * 60)
    print("DAY 65: LSTM GATES MINI-EXPERIMENT")
    print("=" * 60)
    
    # Configuration
    input_size = 1
    hidden_size = 4
    seq_len = 20
    
    # Create a simple sequence: step function (0->1 at t=10)
    sequence = np.zeros((seq_len, 1))
    sequence[10:] = 1.0
    
    # Initialize LSTM
    lstm = LSTMCell(input_size, hidden_size)
    
    # Initial states
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))
    
    # Storage for visualization
    history = {key: [] for key in ['forget', 'input', 'candidate', 'output', 'cell_state', 'hidden_state']}
    history['input_signal'] = []
    
    print("\nRunning forward pass through sequence...")
    print("Sequence: 0 for t<10, 1 for t>=10 (step function)")
    print("-" * 60)
    
    for t in range(seq_len):
        x = sequence[t:t+1].T  # (1, 1)
        h, c, gates = lstm.forward(x, h, c)
        
        for key in history:
            if key != 'input_signal':
                history[key].append(gates[key].copy())
        history['input_signal'].append(x[0, 0])
        
        # Print gate values for first hidden unit at key timesteps
        if t in [0, 9, 10, 11, 19]:
            print(f"t={t:2d} | x={x[0,0]:.1f} | "
                  f"f={gates['forget'][0]:.3f} | "
                  f"i={gates['input'][0]:.3f} | "
                  f"c~={gates['candidate'][0]:.3f} | "
                  f"o={gates['output'][0]:.3f} | "
                  f"c={gates['cell_state'][0]:.3f} | "
                  f"h={gates['hidden_state'][0]:.3f}")
    
    # Convert to arrays
    for key in history:
        history[key] = np.array(history[key])  # (seq_len, hidden_size) or (seq_len,)
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('LSTM Gate Dynamics on Step Input (Hidden Unit 0)', fontsize=14)
    
    t = np.arange(seq_len)
    unit = 0  # Visualize first hidden unit
    
    # Input signal
    axes[0, 0].plot(t, history['input_signal'], 'ko-', markersize=4)
    axes[0, 0].set_title('Input Signal')
    axes[0, 0].set_ylabel('x_t')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=9.5, color='r', linestyle='--', alpha=0.5, label='Step change')
    axes[0, 0].legend()
    
    # Forget gate
    axes[0, 1].plot(t, history['forget'][:, unit], 'b-o', markersize=3, label='Forget gate (f)')
    axes[0, 1].set_title('Forget Gate: "What to discard from past?"')
    axes[0, 1].set_ylabel('Activation')
    axes[0, 1].set_ylim(-0.1, 1.1)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Input gate & Candidate
    axes[1, 0].plot(t, history['input'][:, unit], 'g-o', markersize=3, label='Input gate (i)')
    axes[1, 0].plot(t, history['candidate'][:, unit], 'm-s', markersize=3, label='Candidate (c~)')
    axes[1, 0].set_title('Input Gate & Candidate: "What new info to store?"')
    axes[1, 0].set_ylabel('Activation')
    axes[1, 0].set_ylim(-1.1, 1.1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Output gate
    axes[1, 1].plot(t, history['output'][:, unit], 'r-o', markersize=3, label='Output gate (o)')
    axes[1, 1].set_title('Output Gate: "What to expose as hidden state?"')
    axes[1, 1].set_ylabel('Activation')
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Cell state
    axes[2, 0].plot(t, history['cell_state'][:, unit], 'c-o', markersize=3, label='Cell state (c)')
    axes[2, 0].set_title('Cell State: Long-term Memory')
    axes[2, 0].set_ylabel('Value')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Hidden state
    axes[2, 1].plot(t, history['hidden_state'][:, unit], 'k-o', markersize=3, label='Hidden state (h)')
    axes[2, 1].set_title('Hidden State: Output to Next Layer')
    axes[2, 1].set_ylabel('Value')
    axes[2, 1].set_xlabel('Time Step')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig('lstm_gates_day65.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'lstm_gates_day65.png'")
    
    # Explanation
    print("\n" + "=" * 60)
    print("GATE INTERPRETATION")
    print("=" * 60)
    print("""
FORGET GATE (f): 
  - Sigmoid → values in [0, 1]
  - f ≈ 1: Keep previous cell state (remember)
  - f ≈ 0: Discard previous cell state (forget)
  - On step input: Should open (→1) to retain new pattern

INPUT GATE (i) & CANDIDATE (c~):
  - i: Sigmoid → how much new info to add
  - c~: Tanh → new candidate values [-1, 1]
  - i * c~: Gated addition to cell state
  - On step input: i opens, c~ encodes new value

OUTPUT GATE (o):
  - Sigmoid → how much cell state to expose
  - h = o * tanh(c): Filtered readout
  - Controls what next layer sees

CELL STATE (c):
  - c_t = f * c_{t-1} + i * c~
  - Additive update → gradient flows unchanged (highway)
  - Long-term memory pathway
""")
    
    # Show gradient flow property
    print("=" * 60)
    print("GRADIENT FLOW DEMONSTRATION")
    print("=" * 60)
    print("Cell state update: c_t = f_t * c_{t-1} + i_t * c~_t")
    print("∂c_t / ∂c_{t-1} = f_t (diagonal Jacobian)")
    print("If f_t ≈ 1: Gradient flows unchanged across many steps!")
    print("This solves the vanishing gradient problem of vanilla RNNs.\n")
    
    # Numerical gradient check
    print("Numerical check: ∂c_t/∂c_{t-1} ≈ forget gate value")
    print(f"At t=15: forget gate = {history['forget'][15, 0]:.4f}")
    print(f"Cell state change ratio c_15/c_14 = {history['cell_state'][15, 0]/max(history['cell_state'][14, 0], 1e-8):.4f}")
    print("(Ratio ≈ forget gate when input gate contribution is small)")


if __name__ == "__main__":
    run_experiment()