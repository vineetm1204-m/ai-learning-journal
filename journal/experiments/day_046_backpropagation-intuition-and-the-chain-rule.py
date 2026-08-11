import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Day 46: Backpropagation Intuition & The Chain Rule
# A self-contained numerical & visual experiment.
# ============================================================

np.random.seed(42)

# ------------------------------------------------------------
# 1. The Computational Graph: f(x) = (W2 * relu(W1 * x + b1) + b2)^2
#    We track gradients manually (autograd style) and via PyTorch-style tape.
# ------------------------------------------------------------

class Tensor:
    """Minimal autograd tensor supporting +, *, relu, square, sum."""
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, label='{self.label}')"

    # --- Ops ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+', f"{self.label}+{other.label}")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*', f"{self.label}*{other.label}")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU', f"relu({self.label})")
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def square(self):
        out = Tensor(self.data ** 2, (self,), '^2', f"{self.label}^2")
        def _backward():
            self.grad += 2 * self.data * out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), (self,), 'sum', f"sum({self.label})")
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    # --- Backprop ---
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

# ------------------------------------------------------------
# 2. Numerical Gradient Checker (Sanity Check)
# ------------------------------------------------------------
def numerical_grad(f, x, eps=1e-5):
    """Central difference gradient for scalar-output function f."""
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old = x[ix]
        x[ix] = old + eps
        fx_plus = f(x).copy()
        x[ix] = old - eps
        fx_minus = f(x).copy()
        x[ix] = old
        grad[ix] = np.sum((fx_plus - fx_minus) / (2 * eps))
        it.iternext()
    return grad

# ------------------------------------------------------------
# 3. Experiment: 2-Layer MLP Scalar Regression
# ------------------------------------------------------------
# Dimensions
D_in, H, D_out = 3, 4, 1
N = 5 # Batch size

# Data
X_np = np.random.randn(N, D_in)
y_np = np.random.randn(N, D_out)

# Parameters (Tensors)
W1 = Tensor(np.random.randn(D_in, H) * 0.1, label='W1')
b1 = Tensor(np.zeros(H), label='b1')
W2 = Tensor(np.random.randn(H, D_out) * 0.1, label='W2')
b2 = Tensor(np.zeros(D_out), label='b2')

params = [W1, b1, W2, b2]

def forward_manual(X_batch):
    """Manual forward pass returning loss Tensor."""
    # X_batch: (N, D_in) -> Tensor
    x = Tensor(X_batch, label='x')
    # Layer 1
    h = Tensor(x.data @ W1.data + b1.data, (x, W1, b1), '@+', 'h_pre')
    def _bw_h():
        x.grad += h.grad @ W1.data.T
        W1.grad += x.data.T @ h.grad
        b1.grad += np.sum(h.grad, axis=0)
    h._backward = _bw_h
    
    h_relu = h.relu()
    # Layer 2
    y_pred = Tensor(h_relu.data @ W2.data + b2.data, (h_relu, W2, b2), '@+', 'y_pred')
    def _bw_y():
        h_relu.grad += y_pred.grad @ W2.data.T
        W2.grad += h_relu.data.T @ y_pred.grad
        b2.grad += np.sum(y_pred.grad, axis=0)
    y_pred._backward = _bw_y
    
    # Loss: MSE
    diff = y_pred + Tensor(-y_np, label='-y')
    loss = diff.square().sum()
    return loss

# ------------------------------------------------------------
# 4. Run & Verify
# ------------------------------------------------------------
print("="*60)
print("DAY 46: BACKPROPAGATION INTUITION & CHAIN RULE")
print("="*60)

# 1. Forward & Backward (Autograd)
loss = forward_manual(X_np)
loss.backward()

print(f"\nLoss: {loss.data:.6f}")
print("\n--- Gradients (Autograd) ---")
for p in params:
    print(f"{p.label} grad norm: {np.linalg.norm(p.grad):.6f}")

# 2. Numerical Gradient Check
print("\n--- Numerical Gradient Check ---")
def loss_fn_flat(W1_f, b1_f, W2_f, b2_f):
    # Reconstruct forward pass using pure numpy for numerical grad
    h = X_np @ W1_f + b1_f
    h_relu = np.maximum(0, h)
    y_pred = h_relu @ W2_f + b2_f
    diff = y_pred - y_np
    return np.sum(diff ** 2)

# Check W1
num_grad_W1 = numerical_grad(lambda w: loss_fn_flat(w, b1.data, W2.data, b2.data), W1.data.copy())
ana_grad_W1 = W1.grad
err_W1 = np.linalg.norm(num_grad_W1 - ana_grad_W1) / (np.linalg.norm(num_grad_W1) + 1e-9)
print(f"W1 Rel Error: {err_W1:.2e} {'PASS' if err_W1 < 1e-7 else 'FAIL'}")

# Check W2
num_grad_W2 = numerical_grad(lambda w: loss_fn_flat(W1.data, b1.data, w, b2.data), W2.data.copy())
ana_grad_W2 = W2.grad
err_W2 = np.linalg.norm(num_grad_W2 - ana_grad_W2) / (np.linalg.norm(num_grad_W2) + 1e-9)
print(f"W2 Rel Error: {err_W2:.2e} {'PASS' if err_W2 < 1e-7 else 'FAIL'}")

# ------------------------------------------------------------
# 5. Chain Rule Visualization: Gradient Flow Magnitude
# ------------------------------------------------------------
print("\n--- Gradient Flow Analysis (Chain Rule Intuition) ---")
# Simulate deep linear network: y = W_L ... W_1 x
# Gradient w.r.t W_1 involves product of all subsequent weights.
depth = 10
dim = 5
Ws = [Tensor(np.random.randn(dim, dim) * 0.5, label=f'W{i}') for i in range(depth)]
x = Tensor(np.random.randn(1, dim), label='x')

# Forward
acts = [x]
for i, W in enumerate(Ws):
    nxt = Tensor(acts[-1].data @ W.data, (acts[-1], W), '@', f'h{i+1}')
    def make_bw(a, w):
        def _bw():
            a.grad += nxt.grad @ w.data.T
            w.grad += a.data.T @ nxt.grad
        return _bw
    nxt._backward = make_bw(acts[-1], W)
    acts.append(nxt)

loss_deep = acts[-1].square().sum()
loss_deep.backward()

print("Gradient norms per layer (Input -> Output):")
for i, W in enumerate(Ws):
    print(f"  dL/dW{i+1}: {np.linalg.norm(W.grad):.6f}  (Shape: {W.grad.shape})")

# ------------------------------------------------------------
# 6. Plotting: Gradient Vanishing/Exploding
# ------------------------------------------------------------
def simulate_grad_flow(init_scale, steps=50):
    """Track gradient norm of first layer vs depth."""
    dim = 10
    norms = []
    for d in range(1, steps+1):
        Ws = [np.random.randn(dim, dim) * init_scale for _ in range(d)]
        # Input
        x = np.random.randn(1, dim)
        # Forward
        h = x
        for W in Ws: h = h @ W
        # Loss grad (dL/dh = 2h)
        grad = 2 * h
        # Backward (Chain Rule Product)
        for W in reversed(Ws):
            grad = grad @ W.T
        norms.append(np.linalg.norm(grad))
    return norms

scales = [0.5, 1.0, 1.5]
plt.figure(figsize=(10, 6))
for s in scales:
    norms = simulate_grad_flow(s, 30)
    plt.plot(range(1, 31), norms, label=f'Init Scale={s}', marker='o', markersize=3)
plt.yscale('log')
plt.xlabel('Network Depth (Layers)')
plt.ylabel('|| Gradient at Input || (Log Scale)')
plt.title('Chain Rule Effect: Gradient Vanishing / Exploding')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('day46_grad_flow.png', dpi=150)
print("\nPlot saved to 'day46_grad_flow.png'")

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)