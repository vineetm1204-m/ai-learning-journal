import random
import math

# ============================================================
# DAY 42: PERCEPTRONS & THE BIOLOGICAL NEURON ANALOGY
# ============================================================
# This script implements a Perceptron from scratch to demonstrate
# the direct mapping between the mathematical model and the
# biological neuron.
#
# BIOLOGICAL MAPPING:
# -------------------
# Dendrites (Inputs)      -> x_i (Input features)
# Synapses (Weights)      -> w_i (Connection strengths)
# Soma (Summation)        -> Σ(w_i * x_i) + b (Weighted sum + Bias)
# Axon Hillock (Threshold)-> Activation Function (Step/Heaviside)
# Axon Terminal (Output)  -> y_hat (Firing decision: 0 or 1)
# ============================================================

class BiologicalPerceptron:
    """
    A Perceptron modeled explicitly with biological terminology.
    """
    def __init__(self, num_inputs, learning_rate=0.1):
        # SYNAPSES: Connection strengths (plastic, change with learning)
        self.synaptic_weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        
        # RESTING POTENTIAL / THRESHOLD OFFSET: 
        # Bias acts as the intrinsic excitability of the neuron.
        # High bias = easier to fire (lower threshold).
        self.threshold_offset = random.uniform(-1, 1) 
        
        self.learning_rate = learning_rate # Neuroplasticity rate (Hebbian-like)

    def _summation(self, dendrite_signals):
        """
        SOMA INTEGRATION: Spatial summation of weighted inputs.
        Σ (w_i * x_i) + b
        """
        weighted_sum = 0.0
        for i, signal in enumerate(dendrite_signals):
            weighted_sum += signal * self.synaptic_weights[i]
        weighted_sum += self.threshold_offset # Add bias
        return weighted_sum

    def _activation(self, membrane_potential):
        """
        AXON HILLOCK: All-or-None Law (Heaviside Step Function).
        Neuron fires (1) only if potential exceeds threshold (0).
        """
        return 1 if membrane_potential > 0 else 0

    def forward(self, dendrite_signals):
        """ Full forward pass: Integration -> Firing Decision """
        potential = self._summation(dendrite_signals)
        spike = self._activation(potential)
        return spike, potential

    def hebbian_update(self, dendrite_signals, target, prediction):
        """
        SYNAPTIC PLASTICITY (Delta Rule / Perceptron Learning Rule):
        "Neurons that fire together, wire together."
        Δw_i = η * (target - prediction) * x_i
        Δb   = η * (target - prediction)
        """
        error = target - prediction
        if error != 0:
            for i in range(len(self.synaptic_weights)):
                # Long-Term Potentiation (LTP) if error > 0
                # Long-Term Depression (LTD) if error < 0
                self.synaptic_weights[i] += error * dendrite_signals[i] * self.learning_rate
            
            self.threshold_offset += error * self.learning_rate

def run_experiment():
    print("="*60)
    print("DAY 42: PERCEPTRON - THE ARTIFICIAL BIOLOGICAL NEURON")
    print("="*60)
    
    # ---------------------------------------------------------
    # EXPERIMENT 1: LOGIC GATES (Linearly Separable Problems)
    # ---------------------------------------------------------
    print("\n[EXPERIMENT 1] Learning Linearly Separable Logic (AND / OR)")
    print("-" * 50)
    
    # Training Data: [Input1, Input2, Bias_Input(Always 1)]
    # Note: We treat bias as a constant input '1' connected to 'threshold_offset' weight
    # for cleaner biological analogy (a tonic input signal).
    training_data = [
        ([0, 0, 1], 0), # AND: 0, OR: 0
        ([0, 1, 1], 0), # AND: 0, OR: 1
        ([1, 0, 1], 0), # AND: 0, OR: 1
        ([1, 1, 1], 1), # AND: 1, OR: 1
    ]
    
    # Target: AND Gate (Linearly Separable)
    targets_and = [0, 0, 0, 1]
    # Target: OR Gate (Linearly Separable)
    targets_or  = [0, 1, 1, 1]

    for gate_name, targets in [("AND", targets_and), ("OR", targets_or)]:
        print(f"\n--- Training {gate_name} Gate ---")
        # 3 inputs: x1, x2, bias(1)
        neuron = BiologicalPerceptron(num_inputs=3, learning_rate=0.1)
        
        epochs = 0
        max_epochs = 100
        converged = False
        
        while not converged and epochs < max_epochs:
            total_error = 0
            epochs += 1
            for i, (inputs, _) in enumerate(training_data):
                target = targets[i]
                prediction, potential = neuron.forward(inputs)
                neuron.hebbian_update(inputs, target, prediction)
                total_error += abs(target - prediction)
            
            if total_error == 0:
                converged = True
        
        status = "CONVERGED" if converged else "FAILED"
        print(f"  Status: {status} in {epochs} epochs")
        print(f"  Final Synaptic Weights (w1, w2): [{neuron.synaptic_weights[0]:.3f}, {neuron.synaptic_weights[1]:.3f}]")
        print(f"  Threshold Offset (Bias Weight)   :  {neuron.synaptic_weights[2]:.3f}")
        
        # Test
        print("  Truth Table Verification:")
        for inputs, _ in training_data:
            pred, pot = neuron.forward(inputs)
            print(f"    Input: [{int(inputs[0])}, {int(inputs[1])}] -> Potential: {pot:.3f} -> Spike: {pred}")

    # ---------------------------------------------------------
    # EXPERIMENT 2: THE XOR PROBLEM (Non-Linear Separability)
    # ---------------------------------------------------------
    print("\n\n[EXPERIMENT 2] The XOR Problem (Non-Linearly Separable)")
    print("-" * 50)
    print("Biological Context: A single neuron (Perceptron) cannot solve XOR.")
    print("This mirrors the biological limitation: a single linear threshold unit")
    print("cannot implement non-linear decision boundaries. The brain solves this")
    print("via MULTI-LAYER NETWORKS (Hidden Layers / Interneurons).")
    
    targets_xor = [0, 1, 1, 0]
    neuron_xor = BiologicalPerceptron(num_inputs=3, learning_rate=0.1)
    
    epochs = 0
    max_epochs = 200
    for epoch in range(max_epochs):
        total_error = 0
        for i, (inputs, _) in enumerate(training_data):
            target = targets_xor[i]
            pred, _ = neuron_xor.forward(inputs)
            neuron_xor.hebbian_update(inputs, target, pred)
            total_error += abs(target - pred)
        if total_error == 0: break
        epochs += 1
    
    print(f"\n  Training finished after {epochs} epochs.")
    print("  Final Weights:", [f"{w:.3f}" for w in neuron_xor.synaptic_weights])
    print("  XOR Truth Table Results:")
    for i, (inputs, _) in enumerate(training_data):
        pred, pot = neuron_xor.forward(inputs)
        target = targets_xor[i]
        match = "✓" if pred == target else "✗ (FAIL)"
        print(f"    Input: [{int(inputs[0])}, {int(inputs[1])}] Target: {target} Pred: {pred} {match}")

    # ---------------------------------------------------------
    # EXPERIMENT 3: VISUALIZING THE DECISION BOUNDARY
    # ---------------------------------------------------------
    print("\n\n[EXPERIMENT 3] Decision Boundary Geometry (AND Gate)")
    print("-" * 50)
    print("Equation: w1*x1 + w2*x2 + b = 0  ->  x2 = (-w1/w2)*x1 - (b/w2)")
    print("This is a straight line (Hyperplane) dividing the input space.")
    
    # Re-train AND quickly for clean weights
    neuron_viz = BiologicalPerceptron(num_inputs=3, learning_rate=0.1)
    for _ in range(20):
        for i, (inputs, _) in enumerate(training_data):
            pred, _ = neuron_viz.forward(inputs)
            neuron_viz.hebbian_update(inputs, targets_and[i], pred)

    w1, w2, b = neuron_viz.synaptic_weights
    if abs(w2) > 1e-5:
        slope = -w1 / w2
        intercept = -b / w2
        print(f"  Learned Boundary: x2 = {slope:.2f} * x1 + {intercept:.2f}")
        print("  Region w1*x1 + w2*x2 + b > 0  ->  CLASS 1 (Fire)")
        print("  Region w1*x1 + w2*x2 + b <= 0 ->  CLASS 0 (Silent)")
    else:
        print("  Vertical boundary (w2 approx 0).")

    # ---------------------------------------------------------
    # SUMMARY: THE ANALOGY TABLE
    # ---------------------------------------------------------
    print("\n\n" + "="*60)
    print("SUMMARY: PERCEPTRON <-> BIOLOGICAL NEURON MAPPING")
    print("="*60)
    analogy = [
        ("Component", "Perceptron (Math)", "Biological Neuron (Wetware)"),
        ("---------", "------------------", "---------------------------"),
        ("Input", "Vector **x**", "Dendritic Spikes (Neurotransmitters)"),
        ("Weight", "Vector **w**", "Synaptic Efficacy (AMPA/NMDA Receptors)"),
        ("Bias", "Scalar **b**", "Resting Potential / Leak Conductance"),
        ("Summation", "Dot Product **w·x + b**", "Spatial/Temporal Summation at Soma"),
        ("Activation", "Step Function H(z)", "Action Potential Generation (All-or-None)"),
        ("Output", "Scalar **y** (0/1)", "Axon Spike Train / Neurotransmitter Release"),
        ("Learning", "Δw = η(y_target - y)x", "Hebbian Plasticity (LTP/LTD / STDP)"),
        ("Limitation", "Linear Separability", "Single Neuron Compute Capacity"),
        ("Solution", "Multi-Layer Perceptron", "Cortical Columns / Deep Circuits"),
    ]
    for row in analogy:
        print(f"  {row[0]:<12} | {row[1]:<22} | {row[2]}")

if __name__ == "__main__":
    run_experiment()