import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def target_function(x):
    return np.sin(2 * x) + 0.5 * np.cos(5 * x) + 0.3 * x

class ShallowNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
    
    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2
    
    def backward(self, x, y, lr):
        m = x.shape[0]
        dz2 = (self.z2 - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1**2)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
    
    def train(self, x, y, epochs, lr, verbose=False):
        losses = []
        for epoch in range(epochs):
            pred = self.forward(x)
            loss = np.mean((pred - y)**2)
            losses.append(loss)
            self.backward(x, y, lr)
            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
        return losses

def run_experiment(hidden_sizes=[5, 10, 20, 50], epochs=3000, lr=0.05):
    x_train = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_train = target_function(x_train)
    
    x_test = np.linspace(-3, 3, 500).reshape(-1, 1)
    y_test = target_function(x_test)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, hidden_dim in enumerate(hidden_sizes):
        net = ShallowNet(1, hidden_dim, 1)
        losses = net.train(x_train, y_train, epochs, lr, verbose=(idx==0))
        y_pred = net.forward(x_test)
        
        ax = axes[idx]
        ax.plot(x_test, y_test, 'k-', label='True function', linewidth=2, alpha=0.7)
        ax.plot(x_test, y_pred, 'r--', label=f'NN approx (h={hidden_dim})', linewidth=2)
        ax.scatter(x_train, y_train, s=10, alpha=0.3, color='gray', label='Training points')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Hidden units: {hidden_dim} | Final MSE: {losses[-1]:.6f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Universal Approximation Theorem: Single Hidden Layer Networks\n'
                 'Approximating f(x) = sin(2x) + 0.5cos(5x) + 0.3x', fontsize=14)
    plt.tight_layout()
    plt.savefig('universal_approximation_demo.png', dpi=150)
    plt.show()
    
    print("\n--- Convergence Comparison ---")
    for hidden_dim in hidden_sizes:
        net = ShallowNet(1, hidden_dim, 1)
        losses = net.train(x_train, y_train, epochs, lr)
        print(f"Hidden={hidden_dim:2d} | Final MSE: {losses[-1]:.6f} | Params: {hidden_dim*2 + 2}")

if __name__ == "__main__":
    run_experiment()