import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def manual_max_pool2d(x, kernel_size=2, stride=2):
    """Manual implementation of 2D Max Pooling for educational purposes."""
    B, C, H, W = x.shape
    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sH, sW = (stride, stride) if isinstance(stride, int) else stride
    
    out_H = (H - kH) // sH + 1
    out_W = (W - kW) // sW + 1
    
    out = torch.zeros(B, C, out_H, out_W)
    
    for b in range(B):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    h_start = i * sH
                    h_end = h_start + kH
                    w_start = j * sW
                    w_end = w_start + kW
                    
                    window = x[b, c, h_start:h_end, w_start:w_end]
                    out[b, c, i, j] = torch.max(window)
    return out

def manual_avg_pool2d(x, kernel_size=2, stride=2):
    """Manual implementation of 2D Average Pooling for educational purposes."""
    B, C, H, W = x.shape
    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sH, sW = (stride, stride) if isinstance(stride, int) else stride
    
    out_H = (H - kH) // sH + 1
    out_W = (W - kW) // sW + 1
    
    out = torch.zeros(B, C, out_H, out_W)
    
    for b in range(B):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    h_start = i * sH
                    h_end = h_start + kH
                    w_start = j * sW
                    w_end = w_start + kW
                    
                    window = x[b, c, h_start:h_end, w_start:w_end]
                    out[b, c, i, j] = torch.mean(window)
    return out

def create_synthetic_feature_map():
    """Create a synthetic 6x6 feature map with distinct patterns."""
    # Base pattern: vertical edge on left, horizontal edge on right, noise in middle
    img = torch.zeros(1, 1, 6, 6)
    
    # Vertical edge (high values left, low right)
    img[0, 0, :, 0:3] = 1.0
    img[0, 0, :, 3:6] = 0.1
    
    # Horizontal edge (high values top, low bottom) - overlapping
    img[0, 0, 0:3, :] += 1.0
    img[0, 0, 3:6, :] += 0.1
    
    # Add a distinct peak at (1,1) and (4,4) to test max pooling sensitivity
    img[0, 0, 1, 1] = 5.0  # Sharp peak
    img[0, 0, 4, 4] = 0.5  # Mild peak
    
    # Add noise
    img += torch.randn_like(img) * 0.05
    
    return img

def visualize_pooling(input_tensor, max_out, avg_out, max_manual, avg_manual):
    """Plot input and pooling results."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Day 97: Pooling Layers Comparison (Max vs Average)', fontsize=16)
    
    data = [
        (input_tensor.squeeze(), 'Input (6x6)'),
        (max_out.squeeze(), 'MaxPool2d (3x3)'),
        (avg_out.squeeze(), 'AvgPool2d (3x3)'),
        (max_manual.squeeze(), 'Manual MaxPool (3x3)'),
        (avg_manual.squeeze(), 'Manual AvgPool (3x3)'),
        (torch.abs(max_out - max_manual).squeeze(), 'Diff (Built-in vs Manual Max)')
    ]
    
    for idx, (tensor, title) in enumerate(data):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        im = ax.imshow(tensor.detach().numpy(), cmap='viridis', interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Annotate values
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                ax.text(j, i, f'{tensor[i, j]:.2f}', ha='center', va='center', 
                        color='white' if tensor[i, j] > tensor.max()/2 else 'black', fontsize=8)

    plt.tight_layout()
    plt.savefig('day97_pooling_comparison.png', dpi=150)
    print("Visualization saved to 'day97_pooling_comparison.png'")
    plt.close()

def run_experiment():
    print("=" * 60)
    print("Day 97: Pooling Layers Mini-Experiment")
    print("=" * 60)
    
    # 1. Setup Data
    x = create_synthetic_feature_map()
    print(f"Input shape: {x.shape}")
    print(f"Input:\n{x.squeeze().numpy()}")
    
    # 2. Built-in PyTorch Layers
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    max_out_builtin = maxpool(x)
    avg_out_builtin = avgpool(x)
    
    # 3. Manual Implementations
    max_out_manual = manual_max_pool2d(x, kernel_size=2, stride=2)
    avg_out_manual = manual_avg_pool2d(x, kernel_size=2, stride=2)
    
    # 4. Verification
    print("\n--- Verification ---")
    max_diff = torch.max(torch.abs(max_out_builtin - max_out_manual))
    avg_diff = torch.max(torch.abs(avg_out_builtin - avg_out_manual))
    print(f"MaxPool Diff (Built-in vs Manual): {max_diff.item():.6f} {'PASS' if max_diff < 1e-5 else 'FAIL'}")
    print(f"AvgPool Diff (Built-in vs Manual): {avg_diff.item():.6f} {'PASS' if avg_diff < 1e-5 else 'FAIL'}")
    
    # 5. Analysis
    print("\n--- Analysis ---")
    print(f"Input Range:  [{x.min():.2f}, {x.max():.2f}]")
    print(f"MaxPool Out:  [{max_out_builtin.min():.2f}, {max_out_builtin.max():.2f}] (Preserves sharp peaks: {max_out_builtin.max():.2f})")
    print(f"AvgPool Out:  [{avg_out_builtin.min():.2f}, {avg_out_builtin.max():.2f}] (Smooths peaks: {avg_out_builtin.max():.2f})")
    
    # 6. Gradient Flow Check (Backprop)
    print("\n--- Gradient Flow Test ---")
    x_req = x.clone().requires_grad_(True)
    loss_max = maxpool(x_req).sum()
    loss_max.backward()
    grad_max = x_req.grad.clone()
    
    x_req = x.clone().requires_grad_(True)
    loss_avg = avgpool(x_req).sum()
    loss_avg.backward()
    grad_avg = x_req.grad.clone()
    
    print(f"MaxPool Gradient Sparsity (zero grads): {(grad_max == 0).float().mean().item()*100:.1f}%")
    print(f"AvgPool Gradient Sparsity (zero grads): {(grad_avg == 0).float().mean().item()*100:.1f}%")
    print("MaxPool routes gradient ONLY to max index. AvgPool distributes gradient EVENLY.")
    
    # 7. Visualization
    visualize_pooling(x, max_out_builtin, avg_out_builtin, max_out_manual, avg_out_manual)

if __name__ == "__main__":
    run_experiment()