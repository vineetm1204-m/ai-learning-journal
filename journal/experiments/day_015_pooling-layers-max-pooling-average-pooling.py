import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

def create_sample_feature_map():
    """Create a synthetic feature map with distinct patterns."""
    fm = torch.zeros(1, 1, 8, 8)
    # Vertical edge
    fm[0, 0, :, 3:5] = 1.0
    # Horizontal edge
    fm[0, 0, 2:4, :] = 0.8
    # Bright spot
    fm[0, 0, 6:8, 6:8] = 1.5
    # Gradient region
    for i in range(8):
        fm[0, 0, i, :] += i * 0.05
    return fm

def apply_pooling(fm, kernel_size, stride, mode='max'):
    """Apply pooling and return output + indices (for max)."""
    if mode == 'max':
        pool = nn.MaxPool2d(kernel_size, stride, return_indices=True)
        out, indices = pool(fm)
        return out, indices
    else:
        pool = nn.AvgPool2d(kernel_size, stride)
        out = pool(fm)
        return out, None

def visualize_pooling():
    fm = create_sample_feature_map()
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    
    # Original
    axes[0, 0].imshow(fm[0, 0].numpy(), cmap='viridis', vmin=0, vmax=1.5)
    axes[0, 0].set_title('Original Feature Map (8x8)')
    axes[0, 0].axis('off')
    plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0], fraction=0.046)
    
    configs = [
        (2, 2, 'max', 'MaxPool 2x2, stride=2'),
        (2, 2, 'avg', 'AvgPool 2x2, stride=2'),
        (3, 2, 'max', 'MaxPool 3x3, stride=2'),
        (3, 2, 'avg', 'AvgPool 3x3, stride=2'),
        (2, 1, 'max', 'MaxPool 2x2, stride=1'),
        (2, 1, 'avg', 'AvgPool 2x2, stride=1'),
        (4, 4, 'max', 'MaxPool 4x4, stride=4 (global-ish)'),
        (4, 4, 'avg', 'AvgPool 4x4, stride=4 (global-ish)'),
    ]
    
    for idx, (k, s, mode, title) in enumerate(configs):
        row = (idx // 3) + 1 if idx < 3 else (idx // 3) + 1
        col = idx % 3 + 1 if idx < 3 else (idx - 3) % 3
        if idx >= 6:
            row = 2
            col = idx - 5
        
        out, indices = apply_pooling(fm, k, s, mode)
        im = axes[row, col].imshow(out[0, 0].detach().numpy(), cmap='viridis', vmin=0, vmax=1.5)
        axes[row, col].set_title(f'{title}\nOutput: {out.shape[2]}x{out.shape[3]}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    
    # Hide unused subplot
    axes[2, 3].axis('off')
    
    plt.suptitle('Pooling Layer Comparison: Max vs Average Pooling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pooling_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def demonstrate_maxpool_indices():
    """Show how max pooling indices work for unpooling."""
    print("\n=== MaxPool Indices Demonstration ===")
    fm = create_sample_feature_map()
    print(f"Input shape: {fm.shape}")
    print(f"Input:\n{fm[0, 0].numpy().round(2)}")
    
    pool = nn.MaxPool2d(2, 2, return_indices=True)
    out, indices = pool(fm)
    print(f"\nOutput shape: {out.shape}")
    print(f"Output:\n{out[0, 0].detach().numpy().round(2)}")
    print(f"Indices (flattened):\n{indices[0, 0].numpy()}")
    
    # Unpooling
    unpool = nn.MaxUnpool2d(2, 2)
    reconstructed = unpool(out, indices, output_size=fm.shape)
    print(f"\nReconstructed shape: {reconstructed.shape}")
    print(f"Reconstructed:\n{reconstructed[0, 0].detach().numpy().round(2)}")
    print(f"Match original: {torch.allclose(fm, reconstructed)}")

def pooling_gradient_flow():
    """Compare gradient flow through max vs avg pooling."""
    print("\n=== Gradient Flow Comparison ===")
    fm = create_sample_feature_map().requires_grad_(True)
    
    # Max pooling
    max_pool = nn.MaxPool2d(2, 2)
    max_out = max_pool(fm)
    max_loss = max_out.sum()
    max_loss.backward()
    max_grad = fm.grad.clone()
    print(f"MaxPool gradient stats: mean={max_grad.abs().mean():.4f}, max={max_grad.abs().max():.4f}")
    print(f"Non-zero gradient positions: {(max_grad != 0).sum().item()} / {fm.numel()}")
    
    # Reset
    fm.grad.zero_()
    
    # Avg pooling
    avg_pool = nn.AvgPool2d(2, 2)
    avg_out = avg_pool(fm)
    avg_loss = avg_out.sum()
    avg_loss.backward()
    avg_grad = fm.grad.clone()
    print(f"AvgPool gradient stats: mean={avg_grad.abs().mean():.4f}, max={avg_grad.abs().max():.4f}")
    print(f"Non-zero gradient positions: {(avg_grad != 0).sum().item()} / {fm.numel()}")
    
    print("\nKey insight: MaxPool routes gradients only to max positions (sparse).")
    print("AvgPool distributes gradients evenly across all positions (dense).")

def adaptive_pooling_demo():
    """Show adaptive pooling for fixed output size."""
    print("\n=== Adaptive Pooling Demo ===")
    # Different input sizes
    inputs = [
        torch.randn(1, 3, 32, 32),
        torch.randn(1, 3, 17, 23),
        torch.randn(1, 3, 100, 50),
    ]
    
    adaptive_max = nn.AdaptiveMaxPool2d((7, 7))
    adaptive_avg = nn.AdaptiveAvgPool2d((7, 7))
    
    for i, x in enumerate(inputs):
        max_out = adaptive_max(x)
        avg_out = adaptive_avg(x)
        print(f"Input {i+1}: {x.shape[2:]} -> Max: {max_out.shape[2:]}, Avg: {avg_out.shape[2:]}")

def pooling_in_cnn_context():
    """Show pooling in a simple CNN feature extraction pipeline."""
    print("\n=== Pooling in CNN Context ===")
    
    class TinyCNN(nn.Module):
        def __init__(self, pool_type='max'):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2) if pool_type == 'max' else nn.AvgPool2d(2, 2)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    x = torch.randn(4, 1, 28, 28)
    
    for pool_type in ['max', 'avg']:
        model = TinyCNN(pool_type)
        params = sum(p.numel() for p in model.parameters())
        out = model(x)
        print(f"{pool_type.capitalize()}Pool CNN: {params} params, output shape: {out.shape}")

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 15: Pooling Layers - Max Pooling vs Average Pooling")
    print("=" * 60)
    
    visualize_pooling()
    demonstrate_maxpool_indices()
    pooling_gradient_flow()
    adaptive_pooling_demo()
    pooling_in_cnn_context()
    
    print("\n" + "=" * 60)
    print("Experiment complete. Check 'pooling_comparison.png' for visualization.")
    print("=" * 60)