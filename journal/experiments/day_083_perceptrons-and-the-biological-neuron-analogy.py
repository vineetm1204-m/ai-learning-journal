import random
import math

# =============================================================================
# DAY 83: PERCEPTRONS & THE BIOLOGICAL NEURON ANALOGY
# A self-contained mini-experiment demonstrating the mapping between
# biological neurons and the mathematical perceptron model.
# =============================================================================

class Perceptron:
    """
    The Perceptron: A mathematical abstraction of a biological neuron.
    
    BIOLOGICAL MAPPING:
    -------------------
    Dendrites (Receiving inputs)      -> Input Vector (x)
    Synapses (Connection strengths)   -> Weights Vector (w)
    Soma (Summation & Integration)    -> Weighted Sum (z = w·x + b)
    Axon Hillock (Threshold check)    -> Activation Function (f(z))
    Axon Terminal (Output signal)     -> Output (y_hat)
    """
    
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize synaptic weights with small random values (Hebbian initialization)
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.history = [] # For tracking decision boundary evolution

    def weighted_sum(self, inputs):
        """Soma: Spatial and temporal summation of post-synaptic potentials."""
        return sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

    def activation(self, weighted_sum):
        """Axon Hillock: All-or-nothing firing (Heaviside Step Function)."""
        return 1 if weighted_sum > 0 else 0

    def forward(self, inputs):
        """Full forward pass: Stimulus -> Response."""
        z = self.weighted_sum(inputs)
        return self.activation(z)

    def train(self, training_inputs, labels):
        """
        Perceptron Learning Rule (Rosenblatt, 1957):
        'Neurons that fire together, wire together.' (Hebbian Theory approximation)
        
        Weight Update: w_i = w_i + η * (target - output) * input_i
        Bias Update:   b   = b   + η * (target - output)
        """
        print(f"\n{'='*60}")
        print(f"TRAINING STARTED (Learning Rate η = {self.learning_rate})")
        print(f"{'='*60}")
        
        epoch = 0
        converged = False
        
        while not converged and epoch < 100:
            total_error = 0
            epoch += 1
            
            for inputs, label in zip(training_inputs, labels):
                prediction = self.forward(inputs)
                error = label - prediction
                
                # Synaptic Plasticity: Update weights based on error
                if error != 0:
                    for i in range(len(self.weights)):
                        self.weights[i] += error * inputs[i] * self.learning_rate
                    self.bias += error * self.learning_rate
                    total_error += abs(error)
            
            # Store decision boundary for visualization (2D only)
            if len(self.weights) == 2:
                w1, w2 = self.weights
                b = self.bias
                # Boundary: w1*x + w2*y + b = 0 => y = -(w1/w2)x - b/w2
                if abs(w2) > 1e-5:
                    self.history.append((-w1/w2, -b/w2))
            
            if total_error == 0:
                converged = True
                print(f"\n✅ CONVERGED perfectly at Epoch {epoch}!")
            elif epoch % 10 == 0:
                print(f"Epoch {epoch}: Total Error = {total_error}")

        if not converged:
            print(f"\n⚠️ Stopped at max epochs ({epoch}). Data may not be linearly separable.")
        
        return converged

def visualize_analogy():
    """Prints the core biological-to-mathematical mapping."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           BIOLOGICAL NEURON          ⟷          PERCEPTRON MODEL             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Dendrites (Input receivers)       ⟷   Input Vector x = [x₁, x₂, ..., xₙ]    ║
║  Synapses (Variable conductivity)  ⟷   Weights Vector w = [w₁, w₂, ..., wₙ]  ║
║  Soma (Summation: Σ wᵢxᵢ)          ⟷   Net Input z = w·x + b                ║
║  Axon Hillock (Threshold θ)        ⟷   Activation Function f(z) = 1 if z>0  ║
║  Action Potential (Spike/No Spike) ⟷   Output ŷ ∈ {0, 1}                    ║
║  Neurotransmitter Release          ⟷   Forward Pass to next layer            ║
║  Synaptic Plasticity (LTP/LTD)     ⟷   Weight Update: w ← w + η(y-ŷ)x       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

def run_logic_gate_experiment(gate_name, training_inputs, labels):
    """Runs a complete experiment on a specific logic gate."""
    print(f"\n>>> EXPERIMENT: Learning the '{gate_name}' Gate")
    print(f"    Truth Table: {list(zip(training_inputs, labels))}")
    
    p = Perceptron(input_size=2, learning_rate=0.1)
    success = p.train(training_inputs, labels)
    
    print(f"\n--- Final Synaptic Weights (w) ---")
    print(f"    w1 (Input 1 weight): {p.weights[0]:.4f}")
    print(f"    w2 (Input 2 weight): {p.weights[1]:.4f}")
    print(f"    Bias (Threshold θ):  {p.bias:.4f}")
    
    print(f"\n--- Inference Test (Forward Pass) ---")
    for x, y_true in zip(training_inputs, labels):
        y_pred = p.forward(x)
        status = "✓" if y_pred == y_true else "✗"
        print(f"    Input: {x}  ->  Weighted Sum: {p.weighted_sum(x):.4f}  ->  Output: {y_pred}  (Target: {y_true}) {status}")
    
    return p

def visualize_decision_boundary(perceptron, gate_name):
    """ASCII visualization of the learned decision boundary."""
    if len(perceptron.weights) != 2:
        return

    w1, w2 = perceptron.weights
    b = perceptron.bias
    
    print(f"\n--- Decision Boundary Geometry ({gate_name}) ---")
    print(f"    Equation: {w1:.2f}x₁ + {w2:.2f}x₂ + {b:.2f} = 0")
    
    if abs(w2) > 1e-5:
        slope = -w1 / w2
        intercept = -b / w2
        print(f"    Slope-Intercept: x₂ = {slope:.2f}x₁ + {intercept:.2f}")
    
    # ASCII Grid
    print("\n    Input Space (x1, x2) with Decision Boundary:")
    print("    x2 ^")
    for y in [1.5, 1.0, 0.5, 0.0, -0.5]:
        line = f"  {y:4.1f} |"
        for x in [-0.5, 0.0, 0.5, 1.0, 1.5]:
            # Calculate which side of boundary point lies
            val = w1 * x + w2 * y + b
            pred = 1 if val > 0 else 0
            
            # Mark training points
            marker = " "
            if (x, y) in [(0,0), (0,1), (1,0), (1,1)]:
                marker = "●" if pred == 1 else "○"
            else:
                marker = "▓" if pred == 1 else "░"
            line += f" {marker} "
        print(line)
    print("       +--+--+--+--+--> x1")
    print("       -0.5 0  0.5 1  1.5")

def demonstrate_limitation_xor():
    """Shows why a single Perceptron fails on XOR (Non-linear separability)."""
    print("\n" + "="*70)
    print("THEORETICAL LIMIT: THE XOR PROBLEM (Minsky & Papert, 1969)")
    print("="*70)
    print("""
    A single Perceptron implements a LINEAR decision boundary (Hyperplane).
    
    AND / OR Gates: Linearly Separable (One straight line splits classes).
    XOR Gate:       NOT Linearly Separable (Requires two lines / curve).
    
    Biological Implication: 
    A single biological neuron (or perceptron) cannot compute XOR.
    The brain solves this via MULTI-LAYER NETWORKS (Hidden Layers).
    This limitation sparked the first 'AI Winter' until Backpropagation (1986).
    """)
    
    # Attempt to train on XOR
    xor_inputs = [(0,0), (0,1), (1,0), (1,1)]
    xor_labels = [0, 1, 1, 0]
    
    p = Perceptron(input_size=2, learning_rate=0.1)
    p.train(xor_inputs, xor_labels)
    
    print("\n--- XOR Training Result ---")
    print("    Notice the oscillation / failure to converge to 0 error.")
    for x, y in zip(xor_inputs, xor_labels):
        pred = p.forward(x)
        print(f"    Input: {x} -> Pred: {pred} (Target: {y}) {'✓' if pred==y else '✗ FAIL'}")

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # 1. Display the Core Analogy
    visualize_analogy()
    
    # 2. Define Linearly Separable Problems (AND, OR)
    # Inputs: (x1, x2) with implicit bias handled by Perceptron class
    logic_inputs = [(0,0), (0,1), (1,0), (1,1)]
    
    # AND Gate: Only fires if BOTH inputs active (Coincidence Detection)
    and_labels = [0, 0, 0, 1]
    
    # OR Gate: Fires if ANY input active (Feature Detection)
    or_labels = [0, 1, 1, 1]
    
    # 3. Run Experiments
    p_and = run_logic_gate_experiment("AND", logic_inputs, and_labels)
    visualize_decision_boundary(p_and, "AND")
    
    p_or = run_logic_gate_experiment("OR", logic_inputs, or_labels)
    visualize_decision_boundary(p_or, "OR")
    
    # 4. Demonstrate Fundamental Limitation
    demonstrate_limitation_xor()
    
    print("\n" + "="*70)
    print("DAY 83 CONCLUSION:")
    print("  The Perceptron is a faithful *computational* model of a single neuron.")
    print("  It captures: Weighted Summation, Thresholding, and Hebbian Learning.")
    print("  It fails on: Non-linear problems (XOR) -> Necessitates Deep Networks.")
    print("="*70)