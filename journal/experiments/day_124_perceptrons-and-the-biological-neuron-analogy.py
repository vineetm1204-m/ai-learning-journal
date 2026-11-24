import random
import math

# =============================================================================
# DAY 124: PERCEPTRONS & THE BIOLOGICAL NEURON ANALOGY
# =============================================================================
#
# BIOLOGICAL MAPPING:
# -------------------
# 1. Dendrites (Input Receivers)  -> Input Vector (x)
# 2. Synapses (Connection Strengths) -> Weights Vector (w) + Bias (b)
# 3. Soma / Cell Body (Summation) -> Weighted Sum: z = w·x + b
# 4. Axon Hillock (Threshold)     -> Activation Function: f(z)
# 5. Axon (Output Signal)         -> Output y_hat (0 or 1)
#
# THE PERCEPTRON LEARNING RULE (Rosenblatt, 1957):
# ------------------------------------------------
# If prediction is wrong: w_i <- w_i + η * (target - prediction) * x_i
# This is Hebbian Learning ("Fire together, wire together") with error correction.
#
# LIMITATION: Single perceptrons can ONLY solve linearly separable problems.
# (e.g., AND, OR work; XOR fails). This mirrors a single biological neuron's
# inability to compute non-linear feature combinations without a network (layers).
# =============================================================================

class BiologicalPerceptron:
    """
    A Perceptron implemented with explicit biological terminology.
    No external dependencies (pure Python stdlib).
    """
    def __init__(self, input_size, learning_rate=0.1):
        # Synaptic Weights: Random initialization (synaptogenesis)
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        # Bias: Threshold potential (resting membrane potential offset)
        self.bias = random.uniform(-1, 1)
        self.learning_rate = learning_rate  # η (Neuromodulator influence e.g. Dopamine)

    def _summation(self, inputs):
        """Soma: Spatial & Temporal Summation (Weighted Sum)"""
        # z = Σ (w_i * x_i) + b
        total = 0.0
        for w, x in zip(self.weights, inputs):
            total += w * x
        return total + self.bias

    def _activation(self, weighted_sum):
        """Axon Hillock: All-or-None Firing (Heaviside Step Function)"""
        # Biological neuron fires (1) if threshold crossed, else silent (0)
        return 1 if weighted_sum > 0 else 0

    def forward(self, inputs):
        """Signal Propagation: Dendrites -> Soma -> Axon"""
        self.last_inputs = inputs
        self.last_weighted_sum = self._summation(inputs)
        self.output = self._activation(self.last_weighted_sum)
        return self.output

    def backward(self, target):
        """
        Synaptic Plasticity (Error-Corrected Hebbian Learning).
        Δw = η * (target - output) * input
        """
        error = target - self.output
        if error != 0:
            # Long-Term Potentiation (LTP) if error > 0
            # Long-Term Depression (LTD) if error < 0
            for i in range(len(self.weights)):
                self.weights[i] += error * self.last_inputs[i] * self.learning_rate
            self.bias += error * self.learning_rate
        return error

    def train(self, training_inputs, labels, epochs):
        print(f"\n{'='*60}")
        print(f"TRAINING START (Epochs: {epochs}, LR: {self.learning_rate})")
        print(f"{'='*60}")
        print(f"{'Epoch':<6} | {'Total Error':<12} | {'Weights (w)':<25} | {'Bias (b)'}")
        print(f"{'-'*60}")

        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_inputs, labels):
                self.forward(inputs)
                err = self.backward(label)
                total_error += abs(err)

            w_str = ", ".join([f"{w:.3f}" for w in self.weights])
            print(f"{epoch+1:<6} | {total_error:<12} | [{w_str:<20}] | {self.bias:.3f}")

            if total_error == 0:
                print(f"\n>>> CONVERGENCE ACHIEVED at Epoch {epoch+1} (Zero Error)")
                break
        else:
            print(f"\n>>> TRAINING FINISHED (Non-zero error remains - likely non-linear problem)")

def run_experiment(gate_name, inputs, labels, epochs=50):
    print(f"\n\n{'#'*70}")
    print(f"## EXPERIMENT: {gate_name} GATE (Linearly Separable: {gate_name != 'XOR'})")
    print(f"{'#'*70}")

    p = BiologicalPerceptron(input_size=2, learning_rate=0.1)
    p.train(inputs, labels, epochs)

    # Final Evaluation (Inference Phase)
    print(f"\n--- POST-TRAINING INFERENCE (Action Potentials) ---")
    print(f"{'Inputs (Dendrites)':<20} | {'Sum (Soma)':<12} | {'Output (Axon)':<15} | {'Target'}")
    print(f"{'-'*65}")
    correct = 0
    for x, y in zip(inputs, labels):
        y_hat = p.forward(x)
        status = "✓" if y_hat == y else "✗"
        if y_hat == y: correct += 1
        print(f"{str(x):<20} | {p.last_weighted_sum:<12.3f} | {y_hat:<15} | {y} {status}")

    acc = correct / len(inputs) * 100
    print(f"\nAccuracy: {acc:.1f}%")
    return acc

def visualize_decision_boundary(perceptron, gate_name):
    """ASCII Visualization of the learned hyperplane (Decision Boundary)"""
    print(f"\n--- DECISION BOUNDARY VISUALIZATION ({gate_name}) ---")
    print("Region '1' = Firing (Action Potential), Region '0' = Silent")
    print("Axes: x1 (Horizontal), x2 (Vertical)")

    w1, w2 = perceptron.weights[0], perceptron.weights[1]
    b = perceptron.bias

    # Equation: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b)/w2
    # Handle vertical lines (w2 approx 0)
    grid = []
    for x2_val in [1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5]:
        row = f"x2={x2_val:>4}: "
        for x1_val in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
            # Add small noise to inputs to see boundary clearly
            val = perceptron.forward([x1_val, x2_val])
            row += " █ " if val == 1 else " · "
        grid.append(row)

    # Print Y-axis top-down
    for row in grid:
        print(row)
    print("        " + "   ".join([f"{x:>3}" for x in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]]))
    print("        x1 (Input 1) ->")

    # Print Equation
    if abs(w2) > 1e-3:
        eq = f"x2 = {-w1/w2:.2f} * x1 + {-b/w2:.2f}"
    else:
        eq = f"x1 = {-b/w1:.2f} (Vertical Line)"
    print(f"\nLearned Hyperplane Equation: {eq}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    random.seed(42) # Reproducible "Genetic Initialization"

    # Training Data: [Bias_Input, Input1, Input2] -> Bias is handled internally in class
    # Inputs: (x1, x2)
    training_inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    # 1. AND Gate (Linearly Separable)
    labels_and = [0, 0, 0, 1]
    p_and = BiologicalPerceptron(input_size=2, learning_rate=0.1)
    p_and.train(training_inputs, labels_and, epochs=20)
    visualize_decision_boundary(p_and, "AND")

    # 2. OR Gate (Linearly Separable)
    labels_or = [0, 1, 1, 1]
    p_or = BiologicalPerceptron(input_size=2, learning_rate=0.1)
    p_or.train(training_inputs, labels_or, epochs=20)
    visualize_decision_boundary(p_or, "OR")

    # 3. XOR Gate (NOT Linearly Separable - Single Neuron Failure)
    labels_xor = [0, 1, 1, 0]
    p_xor = BiologicalPerceptron(input_size=2, learning_rate=0.1)
    p_xor.train(training_inputs, labels_xor, epochs=50)
    visualize_decision_boundary(p_xor, "XOR (FAILURE EXPECTED)")

    print(f"\n{'='*70}")
    print("CONCLUSION:")
    print("1. AND/OR: Single 'Neuron' (Perceptron) solves linearly separable logic.")
    print("2. XOR:  Single Neuron FAILS. Requires Multi-Layer Perceptron (MLP) /")
    print("         Hidden Layers (Cortical Columns) to create non-linear boundaries.")
    print("3. This mirrors biology: Single neurons are linear classifiers;")
    print("   Intelligence emerges from NETWORKS (Deep Learning).")
    print(f"{'='*70}")