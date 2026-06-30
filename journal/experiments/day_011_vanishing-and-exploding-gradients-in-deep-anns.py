import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

class DeepMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=12, activation='tanh', init_scale=1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation_name = activation
        
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        
        self._initialize_weights(init_scale)
    
    def _initialize_weights(self, scale):
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=scale / np.sqrt(layer.in_features))
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x

def get_gradient_norms(model):
    norms = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            norms.append(param.grad.data.norm(2).item())
    return norms

def run_experiment(name, model, input_data, target):
    model.zero_grad()
    output = model(input_data)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    grad_norms = get_gradient_norms(model)
    return grad_norms, loss.item()

def print_gradient_table(name, grad_norms):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")
    print(f"{'Layer':<8} {'Grad Norm':<15} {'Log10 Norm':<15}")
    print("-" * 60)
    for i, norm in enumerate(grad_norms):
        log_norm = np.log10(norm) if norm > 0 else -np.inf
        print(f"{i:<8} {norm:<15.6e} {log_norm:<15.4f}")
    print("-" * 60)
    ratio = grad_norms[0] / grad_norms[-1] if grad_norms[-1] > 0 else np.inf
    print(f"First/Last layer gradient ratio: {ratio:.2e}")
    if ratio > 1e3:
        print(">>> VANISHING GRADIENTS detected (early layers >> later layers)")
    elif ratio < 1e-3:
        print(">>> EXPLODING GRADIENTS detected (later layers >> early layers)")
    else:
        print(">>> Gradients appear HEALTHY")

def main():
    batch_size = 32
    input_dim = 10
    hidden_dim = 64
    num_layers = 12
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, 1)
    
    print("=" * 60)
    print("DAY 11: Vanishing & Exploding Gradients in Deep ANNs")
    print("=" * 60)
    
    # Experiment 1: Vanishing gradients (sigmoid + small init)
    model_vanish = DeepMLP(input_dim, hidden_dim, num_layers, 
                           activation='sigmoid', init_scale=0.1)
    grads_vanish, loss_vanish = run_experiment("Vanishing (Sigmoid, small init)", 
                                                model_vanish, x, y)
    print_gradient_table("Vanishing (Sigmoid, small init)", grads_vanish)
    
    # Experiment 2: Exploding gradients (large init)
    model_explode = DeepMLP(input_dim, hidden_dim, num_layers,
                            activation='tanh', init_scale=10.0)
    grads_explode, loss_explode = run_experiment("Exploding (Tanh, large init)",
                                                  model_explode, x, y)
    print_gradient_table("Exploding (Tanh, large init)", grads_explode)
    
    # Experiment 3: Healthy gradients (Xavier/He init + ReLU)
    model_healthy = DeepMLP(input_dim, hidden_dim, num_layers,
                            activation='relu', init_scale=np.sqrt(2))  # He init
    grads_healthy, loss_healthy = run_experiment("Healthy (ReLU, He init)",
                                                  model_healthy, x, y)
    print_gradient_table("Healthy (ReLU, He init)", grads_healthy)
    
    # Experiment 4: Xavier init with Tanh (classic healthy)
    model_xavier = DeepMLP(input_dim, hidden_dim, num_layers,
                           activation='tanh', init_scale=1.0)  # Xavier ~ 1/sqrt(n)
    grads_xavier, loss_xavier = run_experiment("Healthy (Tanh, Xavier init)",
                                                model_xavier, x, y)
    print_gradient_table("Healthy (Tanh, Xavier init)", grads_xavier)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Vanishing gradients: Saturating activations (sigmoid/tanh) + deep nets")
    print("                     cause gradients to shrink exponentially backward.")
    print("Exploding gradients: Large weight initialization amplifies gradients")
    print("                     exponentially through layers.")
    print("Solutions:           Proper initialization (Xavier/He), ReLU variants,")
    print("                     BatchNorm, residual connections, gradient clipping.")
    print("=" * 60)

if __name__ == "__main__":
    main()