import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

def create_synthetic_feature_map():
    """Create a synthetic feature map with distinct patterns."""
    base = np.zeros((1, 1, 16, 16))
    base[0, 0, 2:6, 2:6] = 1.0
    base[0, 0, 2:6, 10:14] = 0.5
    base[0, 0, 10:14, 2:6] = 0.8
    base[0, 0, 10:14, 10:14] = 0.3
    base[0, 0, 7:9, 7:9] = 2.0
    noise = np.random.normal(0, 0.1, base.shape)
    return torch.tensor(base + noise, dtype=torch.float32)

def visualize_pooling(input_tensor, max_out, avg_out, kernel_size, stride):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Input Feature Map', f'Max Pool (k={kernel_size}, s={stride})', f'Avg Pool (k={kernel_size}, s={stride})']
    data = [input_tensor[0,0].numpy(), max_out[0,0].detach().numpy(), avg_out[0,0].detach().numpy()]
    vmin, vmax = 0, 2.2
    for ax, d, t in zip(axes, data, titles):
        im = ax.imshow(d, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=11)
        ax.axis('off')
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    plt.savefig('day56_pooling_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def analyze_pooling_properties():
    x = create_synthetic_feature_map()
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}], mean: {x.mean():.3f}")
    
    configs = [(2, 2), (3, 2), (4, 4)]
    results = {}
    
    for kernel_size, stride in configs:
        max_pool = nn.MaxPool2d(kernel_size, stride)
        avg_pool = nn.AvgPool2d(kernel_size, stride)
        
        max_out = max_pool(x)
        avg_out = avg_pool(x)
        
        print(f"\n--- Kernel={kernel_size}, Stride={stride} ---")
        print(f"Output shape: {max_out.shape}")
        print(f"MaxPool - range: [{max_out.min():.3f}, {max_out.max():.3f}], mean: {max_out.mean():.3f}")
        print(f"AvgPool - range: [{avg_out.min():.3f}, {avg_out.max():.3f}], mean: {avg_out.mean():.3f}")
        
        if kernel_size == 2 and stride == 2:
            visualize_pooling(x, max_out, avg_out, kernel_size, stride)
        
        results[(kernel_size, stride)] = {
            'max': max_out, 'avg': avg_out,
            'max_stats': (max_out.min().item(), max_out.max().item(), max_out.mean().item()),
            'avg_stats': (avg_out.min().item(), avg_out.max().item(), avg_out.mean().item())
        }
    return results

def demonstrate_gradient_flow():
    print("\n=== Gradient Flow Analysis ===")
    x = create_synthetic_feature_map().requires_grad_(True)
    
    max_pool = nn.MaxPool2d(2, 2, return_indices=True)
    avg_pool = nn.AvgPool2d(2, 2)
    
    max_out, indices = max_pool(x)
    avg_out = avg_pool(x)
    
    loss_max = max_out.sum()
    loss_avg = avg_out.sum()
    
    loss_max.backward(retain_graph=True)
    grad_max = x.grad.clone()
    x.grad.zero_()
    
    loss_avg.backward()
    grad_avg = x.grad.clone()
    
    print(f"MaxPool gradient: non-zero elements = {(grad_max != 0).sum().item()}/{grad_max.numel()}")
    print(f"AvgPool gradient: non-zero elements = {(grad_avg != 0).sum().item()}/{grad_avg.numel()}")
    print(f"MaxPool grad sparsity: {(grad_max == 0).float().mean():.3f}")
    print(f"AvgPool grad sparsity: {(grad_avg == 0).float().mean():.3f}")
    
    return grad_max, grad_avg

def pooling_in_cnn_context():
    print("\n=== Pooling in CNN Context ===")
    class TinyCNN(nn.Module):
        def __init__(self, pool_type='max'):
            super().__init__()
            self.conv = nn.Conv2d(1, 8, 3, padding=1)
            self.pool = nn.MaxPool2d(2) if pool_type == 'max' else nn.AvgPool2d(2)
            self.fc = nn.Linear(8 * 8 * 8, 10)
        
        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    x = torch.randn(4, 1, 16, 16)
    y = torch.randint(0, 10, (4,))
    
    for pool_type in ['max', 'avg']:
        model = TinyCNN(pool_type)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        for _ in range(20):
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        
        print(f"{pool_type.capitalize()}Pool - Final loss: {losses[-1]:.4f}, Reduction: {(losses[0]-losses[-1])/losses[0]*100:.1f}%")

def main():
    print("=" * 60)
    print("DAY 56: Pooling Layers - Max vs Average Pooling")
    print("=" * 60)
    
    analyze_pooling_properties()
    demonstrate_gradient_flow()
    pooling_in_cnn_context()
    
    print("\n=== Key Takeaways ===")
    print("1. MaxPool preserves strongest activations (sharp features)")
    print("2. AvgPool preserves average activation (smoother features)")
    print("3. MaxPool gradients are sparse (only max positions)")
    print("4. AvgPool gradients are dense (distributed evenly)")
    print("5. MaxPool typically better for classification tasks")
    print("6. AvgPool useful for global pooling before FC layers")
    print("\nVisualization saved to 'day56_pooling_comparison.png'")

if __name__ == "__main__":
    main()