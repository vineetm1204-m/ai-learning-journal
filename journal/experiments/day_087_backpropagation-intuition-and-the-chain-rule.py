import math
import random
from typing import List, Tuple, Optional

# ============================================================
# Day 87: Backpropagation Intuition & The Chain Rule
# A minimal, pure-Python autograd engine (Micrograd style)
# ============================================================

class Value:
    """
    A scalar value that supports automatic differentiation via backpropagation.
    Stores the value, gradient, and the operation graph for the backward pass.
    """
    __slots__ = ("data", "grad", "_prev", "_op", "_backward")

    def __init__(self, data: float, _children: Tuple["Value", ...] = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- Forward Pass Operations ---

    def __add__(self, other: "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d/dx (x + y) = 1 * dout
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d/dx (x * y) = y * dout
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other: float) -> "Value":
        assert isinstance(other, (int, float)), "Only scalar powers supported"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d/dx (x^n) = n * x^(n-1) * dout
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            # d/dx ReLU(x) = 1 if x > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d/dx tanh(x) = (1 - tanh^2(x)) * dout
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        return out

    # --- Reverse Pass (Backpropagation) ---

    def backward(self):
        """Topological sort followed by reverse-mode autodiff."""
        topo = []
        visited = set()

        def build_topo(v: "Value"):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0  # Seed: dL/dL = 1
        for v in reversed(topo):
            v._backward()

    # --- Operator Overloads for Convenience ---
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __neg__(self): return self * -1
    def __truediv__(self, other): return self * (other ** -1)
    def __rtruediv__(self, other): return other * (self ** -1)


# ============================================================
# Demo: A Tiny Neuron (w1*x1 + w2*x2 + b) -> tanh -> Loss
# ============================================================

def run_experiment():
    print("=" * 60)
    print("DAY 87: BACKPROPAGATION INTUITION & THE CHAIN RULE")
    print("=" * 60)

    # 1. Initialize Inputs & Parameters (Leaf Nodes)
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    b = Value(6.8813735870195432) # Bias chosen so output ~0.7 initially

    print("\n[1] FORWARD PASS: Building the Computation Graph")
    print("-" * 40)

    # Neuron: n = w1*x1 + w2*x2 + b
    n = w1 * x1 + w2 * x2 + b
    print(f"  n = w1*x1 + w2*x2 + b = {n.data:.4f}")

    # Activation: o = tanh(n)
    o = n.tanh()
    print(f"  o = tanh(n)             = {o.data:.4f}")

    # Target & Loss (MSE style: (target - output)^2)
    target = Value(0.0)
    loss = (target - o) ** 2
    print(f"  Loss = (target - o)^2   = {loss.data:.4f}")

    # 2. Backward Pass
    print("\n[2] BACKWARD PASS: Applying Chain Rule Recursively")
    print("-" * 40)
    loss.backward()

    # 3. Inspect Gradients (Intuition Check)
    print("\n[3] GRADIENT INSPECTION (dLoss/dVariable)")
    print("-" * 40)
    params = [("w1", w1), ("w2", w2), ("b", b), ("x1", x1), ("x2", x2)]
    for name, v in params:
        print(f"  dL/d{name:<2} = {v.grad:>10.6f}")

    # 4. Manual Chain Rule Verification (The "Why it works" section)
    print("\n[4] MANUAL CHAIN RULE VERIFICATION")
    print("-" * 40)
    print("  Loss = (0 - o)^2")
    print("  dL/do = 2 * (o - 0) = 2 * o")
    print(f"  dL/do (calc) = {2 * o.data:.6f} | dL/do (auto) = {o.grad:.6f}")

    print("\n  o = tanh(n) -> do/dn = 1 - tanh(n)^2")
    print(f"  do/dn (calc) = {1 - o.data**2:.6f} | (stored in n.grad contribution)")

    print("\n  n = w1*x1 + w2*x2 + b")
    print("  dn/dw1 = x1 = 2.0")
    print("  dn/dw2 = x2 = 0.0")
    print("  dn/db  = 1.0")
    print("  dn/dx1 = w1 = -3.0")
    print("  dn/dx2 = w2 = 1.0")

    print("\n  Chain Rule: dL/dw1 = dL/do * do/dn * dn/dw1")
    manual_w1_grad = (2 * o.data) * (1 - o.data**2) * x1.data
    print(f"  Manual Calc: {manual_w1_grad:.6f} | Auto Grad: {w1.grad:.6f} {'✅ MATCH' if abs(manual_w1_grad - w1.grad) < 1e-6 else '❌ MISMATCH'}")

    # 5. Gradient Descent Step
    print("\n[5] GRADIENT DESCENT STEP (lr=0.1)")
    print("-" * 40)
    lr = 0.1
    for name, v in params:
        if name in ("w1", "w2", "b"): # Only update parameters, not inputs
            v.data -= lr * v.grad
            print(f"  {name}: {v.data + lr * v.grad:.4f} -> {v.data:.4f}")

    # 6. Verify Loss Decreased
    print("\n[6] RE-EVALUATE LOSS AFTER UPDATE")
    print("-" * 40)
    # Re-run forward with new weights (new graph)
    n_new = w1 * x1 + w2 * x2 + b
    o_new = n_new.tanh()
    loss_new = (target - o_new) ** 2
    print(f"  Old Loss: {loss.data:.6f}")
    print(f"  New Loss: {loss_new.data:.6f} {'✅ DECREASED' if loss_new.data < loss.data else '❌ INCREASED'}")

    print("\n" + "=" * 60)
    print("INTUITION SUMMARY:")
    print("1. Forward pass builds a graph of Values (nodes) and Ops (edges).")
    print("2. Backward pass traverses graph in reverse topological order.")
    print("3. Every _backward() implements LOCAL derivative (Chain Rule link).")
    print("4. Gradients accumulate (+=) because a variable can fan-out.")
    print("5. This is EXACTLY what PyTorch/TensorFlow do, just in C++/CUDA.")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()