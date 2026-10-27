import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

# ============================================================
# Day 98: CNN Receptive Field & Spatial Hierarchy Experiment
# ============================================================

def compute_receptive_field(layer_configs, input_size=224):
    """
    Compute theoretical receptive field size and stride at each layer.
    
    Args:
        layer_configs: List of dicts with keys: kernel, stride, padding, dilation
        input_size: Input image size (assumed square)
    
    Returns:
        List of (rf_size, stride, center_offset) for each layer
    """
    rf = 1          # receptive field size
    stride = 1      # cumulative stride
    offset = 0.5    # center offset (0-indexed)
    
    results = [(rf, stride, offset)]
    
    for cfg in layer_configs:
        k = cfg['kernel']
        s = cfg['stride']
        p = cfg['padding']
        d = cfg.get('dilation', 1)
        
        # Effective kernel size with dilation
        k_eff = k + (k - 1) * (d - 1)
        
        # Update offset (center of receptive field in input coordinates)
        offset = offset + ((k_eff - 1) / 2 - p) * stride
        
        # Update receptive field
        rf = rf + (k_eff - 1) * stride
        
        # Update cumulative stride
        stride = stride * s
        
        results.append((rf, stride, offset))
    
    return results


class HierarchyCNN(nn.Module):
    """Simple CNN to demonstrate spatial hierarchy."""
    
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: Low-level features (edges, colors)
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 1/2
            
            # Stage 2: Mid-level features (textures, patterns)
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 1/4
            
            # Stage 3: High-level features (object parts)
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 1/8
            
            # Stage 4: Semantic features (objects)
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(base_channels*8, 10)
        
        # Register hooks to capture feature maps
        self.feature_maps = {}
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        # Hook after each stage's last conv (before pooling)
        layer_names = ['stage1', 'stage2', 'stage3', 'stage4']
        conv_indices = [3, 9, 15, 19]  # Indices of last conv in each stage
        
        for name, idx in zip(layer_names, conv_indices):
            self.features[idx].register_forward_hook(hook_fn(name))
    
    def forward(self, x):
        self.feature_maps.clear()
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def visualize_receptive_fields(layer_configs, input_size=224):
    """Visualize how receptive field grows through the network."""
    results = compute_receptive_field(layer_configs, input_size)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Receptive field size vs depth
    depths = list(range(len(results)))
    rf_sizes = [r[0] for r in results]
    strides = [r[1] for r in results]
    
    ax1 = axes[0]
    ax1.plot(depths, rf_sizes, 'b-o', label='Receptive Field', linewidth=2, markersize=6)
    ax1.plot(depths, strides, 'r-s', label='Cumulative Stride', linewidth=2, markersize=6)
    ax1.axhline(y=input_size, color='k', linestyle='--', alpha=0.5, label=f'Input Size ({input_size})')
    ax1.set_xlabel('Layer Depth')
    ax1.set_ylabel('Pixels')
    ax1.set_title('Receptive Field Growth Through Network')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(depths)
    
    # Plot 2: Visual representation of receptive fields
    ax2 = axes[1]
    ax2.set_xlim(0, input_size)
    ax2.set_ylim(0, input_size)
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    ax2.set_title('Receptive Field Coverage (Centered at Image Center)')
    ax2.set_xlabel('Input Width')
    ax2.set_ylabel('Input Height')
    
    # Draw input image boundary
    ax2.add_patch(Rectangle((0, 0), input_size, input_size, 
                            fill=False, edgecolor='black', linewidth=2, label='Input Image'))
    
    # Draw receptive fields at key layers
    key_layers = [0, 2, 4, 6, 8]  # Input, after stage1, stage2, stage3, stage4
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(key_layers)))
    
    for i, layer_idx in enumerate(key_layers):
        if layer_idx < len(results):
            rf, stride, offset = results[layer_idx]
            # Center of receptive field at image center
            center = input_size / 2
            half_rf = rf / 2
            x = center - half_rf
            y = center - half_rf
            
            rect = Rectangle((x, y), rf, rf, 
                           fill=True, facecolor=colors[i], alpha=0.3,
                           edgecolor=colors[i], linewidth=2,
                           label=f'Layer {layer_idx}: RF={rf:.0f}px')
            ax2.add_patch(rect)
    
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    return fig


def visualize_feature_maps(model, input_tensor, max_channels=16):
    """Visualize feature maps at each hierarchy level."""
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    feature_maps = model.feature_maps
    n_stages = len(feature_maps)
    
    fig, axes = plt.subplots(n_stages, max_channels, figsize=(max_channels*1.2, n_stages*1.5))
    if n_stages == 1:
        axes = axes.reshape(1, -1)
    
    stage_names = ['Stage 1\n(Low-level: Edges, Colors)', 
                   'Stage 2\n(Mid-level: Textures, Patterns)',
                   'Stage 3\n(High-level: Object Parts)',
                   'Stage 4\n(Semantic: Objects)']
    
    for stage_idx, (name, fmap) in enumerate(feature_maps.items()):
        # fmap shape: [1, C, H, W]
        fmap = fmap[0]  # Remove batch dim
        C, H, W = fmap.shape
        
        # Select channels to visualize (first max_channels or evenly spaced)
        if C <= max_channels:
            ch_indices = list(range(C))
        else:
            ch_indices = np.linspace(0, C-1, max_channels, dtype=int)
        
        for ch_idx, c in enumerate(ch_indices):
            ax = axes[stage_idx, ch_idx]
            channel_map = fmap[c].cpu().numpy()
            
            # Normalize for visualization
            vmin, vmax = np.percentile(channel_map, [1, 99])
            channel_map = np.clip(channel_map, vmin, vmax)
            channel_map = (channel_map - vmin) / (vmax - vmin + 1e-8)
            
            im = ax.imshow(channel_map, cmap='viridis', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if ch_idx == 0:
                ax.set_ylabel(stage_names[stage_idx], rotation=0, labelpad=40, fontsize=9, va='center')
            if stage_idx == 0:
                ax.set_title(f'Ch {c}', fontsize=8)
    
    plt.suptitle('Feature Maps Across Spatial Hierarchy', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def visualize_receptive_field_on_image(input_size=224, rf_size=64, stride=8, layer_name="Layer"):
    """Show receptive field grid on input image."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Create a synthetic input pattern
    img = np.zeros((input_size, input_size, 3))
    # Add grid lines
    for i in range(0, input_size, 16):
        img[i, :, :] = 0.3
        img[:, i, :] = 0.3
    # Add center marker
    center = input_size // 2
    img[center-2:center+2, center-2:center+2] = [1, 0, 0]
    
    ax.imshow(img)
    ax.set_xlim(0, input_size)
    ax.set_ylim(input_size, 0)
    ax.set_aspect('equal')
    
    # Draw receptive field grid
    half_rf = rf_size // 2
    for y in range(half_rf, input_size - half_rf + 1, stride):
        for x in range(half_rf, input_size - half_rf + 1, stride):
            rect = Rectangle((x - half_rf, y - half_rf), rf_size, rf_size,
                           fill=False, edgecolor='cyan', linewidth=0.5, alpha=0.6)
            ax.add_patch(rect)
    
    # Highlight center receptive field
    center_rf = Rectangle((center - half_rf, center - half_rf), rf_size, rf_size,
                         fill=False, edgecolor='red', linewidth=3, label=f'Center RF ({rf_size}px)')
    ax.add_patch(center_rf)
    
    ax.set_title(f'{layer_name}: Receptive Field = {rf_size}×{rf_size}, Stride = {stride}')
    ax.legend()
    ax.axis('off')
    plt.tight_layout()
    return fig


def analyze_effective_receptive_field(model, input_size=224, num_samples=100):
    """
    Estimate effective receptive field (ERF) via gradient visualization.
    ERF is typically Gaussian-shaped and smaller than theoretical RF.
    """
    model.eval()
    
    # Create input requiring gradients
    x = torch.randn(1, 3, input_size, input_size, requires_grad=True)
    
    # Forward pass to get output
    output = model(x)
    # Use center pixel of final feature map (before global pool)
    # We need to hook the last conv layer output
    
    # Simplified: compute gradient w.r.t. input for a specific output channel
    target_channel = 0
    loss = output[0, target_channel]
    
    model.zero_grad()
    loss.backward()
    
    # Gradient magnitude shows ERF
    grad = x.grad.abs().sum(dim=1).squeeze().cpu().numpy()  # [H, W]
    
    # Normalize
    grad = grad / (grad.max() + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Heatmap
    im = axes[0].imshow(grad, cmap='hot', interpolation='bilinear')
    axes[0].set_title('Effective Receptive Field (Gradient Magnitude)')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Cross-section through center
    center = input_size // 2
    axes[1].plot(grad[center, :], 'b-', label='Horizontal')
    axes[1].plot(grad[:, center], 'r--', label='Vertical')
    axes[1].set_xlabel('Pixel Position')
    axes[1].set_ylabel('Normalized Gradient')
    axes[1].set_title('ERF Cross-Section (Gaussian-like)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, grad


def print_receptive_field_table(layer_configs, input_size=224):
    """Print detailed receptive field computation table."""
    results = compute_receptive_field(layer_configs, input_size)
    
    print("\n" + "="*80)
    print("RECEPTIVE FIELD COMPUTATION TABLE")
    print("="*80)
    print(f"{'Layer':<6} {'Kernel':<8} {'Stride':<8} {'Pad':<6} {'Dil':<6} {'RF Size':<10} {'Cum Stride':<12} {'Center Offset':<14}")
    print("-"*80)
    
    rf = 1
    stride = 1
    offset = 0.5
    
    print(f"{'Input':<6} {'-':<8} {'-':<8} {'-':<6} {'-':<6} {rf:<10.1f} {stride:<12} {offset:<14.1f}")
    
    for i, cfg in enumerate(layer_configs):
        k = cfg['kernel']
        s = cfg['stride']
        p = cfg['padding']
        d = cfg.get('dilation', 1)
        k_eff = k + (k - 1) * (d - 1)
        
        offset = offset + ((k_eff - 1) / 2 - p) * stride
        rf = rf + (k_eff - 1) * stride
        stride = stride * s
        
        print(f"{i+1:<6} {k:<8} {s:<8} {p:<6} {d:<6} {rf:<10.1f} {stride:<12} {offset:<14.1f}")
    
    print("-"*80)
    print(f"Final Receptive Field: {rf:.1f} × {rf:.1f} pixels")
    print(f"Final Stride: {stride} pixels")
    print(f"Output Feature Map Size: {input_size // stride} × {input_size // stride}")
    print("="*80 + "\n")


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("="*60)
    print("DAY 98: CNN Receptive Field & Spatial Hierarchy")
    print("="*60)
    
    # Configuration for a typical CNN (like VGG-style)
    layer_configs = [
        # Stage 1
        {'kernel': 3, 'stride': 1, 'padding': 1},  # Conv1
        {'kernel': 3, 'stride': 1, 'padding': 1},  # Conv2
        {'kernel': 2, 'stride': 2, 'padding': 0},  # Pool1
        # Stage 2
        {'kernel': 3, 'stride': 1, 'padding': 1},  # Conv3
        {'kernel': 3, 'stride': 1, 'padding': 1},  # Conv4
        {'kernel': 2, 'stride': 2, 'padding': 0},  # Pool2
        # Stage 3
        {'kernel': 3, 'stride': 1, 'padding': 1},  # Conv5
        {'kernel': 3, 'stride': 1, 'padding': 1},  # Conv6
        {'kernel': 2, 'stride': 2, 'padding': 0},  # Pool3
        # Stage 4
        {'kernel': 3, 'stride': 1, 'padding': 1},  # Conv7
        {'kernel': 3, 'stride': 1, 'padding': 1},  # Conv8
    ]
    
    INPUT_SIZE = 224
    
    # 1. Print theoretical receptive field table
    print_receptive_field_table(layer_configs, INPUT_SIZE)
    
    # 2. Visualize receptive field growth
    print("Generating receptive field growth visualization...")
    fig1 = visualize_receptive_fields(layer_configs, INPUT_SIZE)
    fig1.savefig('day98_rf_growth.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 3. Create model and visualize feature hierarchy
    print("Building model and extracting feature maps...")
    model = HierarchyCNN(in_channels=3, base_channels=16)  # Smaller for speed
    
    # Create synthetic input with patterns
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    # Add some structured patterns
    x[0, 0, ::8, :] = 2.0  # Vertical lines
    x[0, 1, :, ::8] = 2.0  # Horizontal lines
    x[0, 2, 50:100, 50:100] = 3.0  # Square blob
    
    fig2 = visualize_feature_maps(model, x, max_channels=8)
    fig2.savefig('day98_feature_hierarchy.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 4. Visualize receptive field grid on image
    print("Visualizing receptive field grid...")
    # Get final layer RF and stride
    results = compute_receptive_field(layer_configs, INPUT_SIZE)
    final_rf = results[-1][0]
    final_stride = results[-1][1]
    
    fig3 = visualize_receptive_field_on_image(INPUT_SIZE, int(final_rf), int(final_stride), "Final Conv Layer")
    fig3.savefig('day98_rf_grid.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 5. Effective Receptive Field analysis
    print("Computing Effective Receptive Field (ERF)...")
    fig4, erf_map = analyze_effective_receptive_field(model, INPUT_SIZE)
    fig4.savefig('day98_erf.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    # 6. ERF Statistics
    print("\n" + "="*60)
    print("EFFECTIVE RECEPTIVE FIELD ANALYSIS")
    print("="*60)
    print(f"Theoretical RF: {final_rf:.1f} × {final_rf:.1f}")
    
    # Find 95% energy radius
    center = INPUT_SIZE // 2
    erf_flat = erf_map.flatten()
    erf_sorted = np.sort(erf_flat)[::-1]
    cumsum = np.cumsum(erf_sorted)
    threshold_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0][0]
    threshold_val = erf_sorted[threshold_idx]
    
    # Approximate ERF diameter
    erf_binary = (erf_map >= threshold_val).astype(float)
    y_coords, x_coords = np.where(erf_binary > 0)
    if len(y_coords) > 0:
        erf_h = y_coords.max() - y_coords.min()
        erf_w = x_coords.max() - x_coords.min()
        print(f"Effective RF (95% energy): ~{erf_h:.0f} × {erf_w:.0f} pixels")
        print(f"ERF / Theoretical RF ratio: {erf_h/final_rf:.2f}")
    
    print("\nKey Insights:")
    print("  • Theoretical RF grows linearly with depth")
    print("  • Effective RF is Gaussian-shaped and smaller than theoretical")
    print("  • Early layers: Small RF → detect edges, colors (local)")
    print("  • Middle layers: Medium RF → detect textures, patterns")
    print("  • Deep layers: Large RF → detect object parts, semantics (global)")
    print("  • Stride reduces spatial resolution but increases RF coverage")
    print("="*60)
    
    print("\nExperiment complete. Check generated PNG files:")
    print("  - day98_rf_growth.png")
    print("  - day98_feature_hierarchy.png")
    print("  - day98_rf_grid.png")
    print("  - day98_erf.png")


if __name__ == "__main__":
    main()