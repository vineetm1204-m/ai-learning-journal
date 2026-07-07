import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# PART 1: RECEPTIVE FIELD CALCULATION (THEORETICAL)
# ============================================================
def compute_receptive_field(layers, input_size=224):
    """
    Compute receptive field size and stride at each layer.
    layers: list of dicts with keys: 'type', 'kernel', 'stride', 'padding', 'dilation'
    Returns list of (rf_size, effective_stride, center_offset) per layer.
    """
    rf = 1
    stride = 1
    offset = 0.5  # center of receptive field in input coordinates
    results = [(rf, stride, offset)]
    
    for layer in layers:
        k = layer.get('kernel', 1)
        s = layer.get('stride', 1)
        p = layer.get('padding', 0)
        d = layer.get('dilation', 1)
        
        if layer['type'] in ('conv', 'pool'):
            rf = rf + (k - 1) * d * stride
            offset = offset + ((k - 1) * d * stride) / 2
            stride = stride * s
        results.append((rf, stride, offset))
    return results

# Define a VGG-like architecture for demonstration
vgg_layers = [
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv1_1
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv1_2
    {'type': 'pool', 'kernel': 2, 'stride': 2, 'padding': 0, 'dilation': 1},  # pool1
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv2_1
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv2_2
    {'type': 'pool', 'kernel': 2, 'stride': 2, 'padding': 0, 'dilation': 1},  # pool2
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv3_1
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv3_2
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv3_3
    {'type': 'pool', 'kernel': 2, 'stride': 2, 'padding': 0, 'dilation': 1},  # pool3
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv4_1
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv4_2
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv4_3
    {'type': 'pool', 'kernel': 2, 'stride': 2, 'padding': 0, 'dilation': 1},  # pool4
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv5_1
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv5_2
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1, 'dilation': 1},  # conv5_3
    {'type': 'pool', 'kernel': 2, 'stride': 2, 'padding': 0, 'dilation': 1},  # pool5
]

rf_results = compute_receptive_field(vgg_layers)
layer_names = ['input'] + [f"{i//2+1}_{'p' if l['type']=='pool' else 'c'}{i%2+1}" for i, l in enumerate(vgg_layers)]

print("=" * 70)
print("THEORETICAL RECEPTIVE FIELD ANALYSIS (VGG-style)")
print("=" * 70)
print(f"{'Layer':>8} | {'RF Size':>8} | {'Eff. Stride':>12} | {'Center Offset':>14}")
print("-" * 70)
for name, (rf, stride, offset) in zip(layer_names, rf_results):
    print(f"{name:>8} | {rf:>8} | {stride:>12} | {offset:>14.1f}")

# ============================================================
# PART 2: EMPIRICAL RECEPTIVE FIELD VIA GRADIENT BACKPROP
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.layer_names = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
        ]
    
    def forward(self, x, return_features=False):
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if return_features and isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                feats.append((self.layer_names[len(feats)], x.clone()))
        return (x, feats) if return_features else x

def empirical_receptive_field(model, layer_idx, input_size=224, device=DEVICE):
    """
    Compute empirical receptive field by backpropagating gradient
    from a single activation at the center of the feature map.
    """
    model.eval()
    x = torch.zeros(1, 3, input_size, input_size, device=device, requires_grad=True)
    
    # Forward pass with feature extraction
    _, feats = model(x, return_features=True)
    target_name, target_feat = feats[layer_idx]
    
    # Select center pixel of first channel
    c, h, w = target_feat.shape[1], target_feat.shape[2], target_feat.shape[3]
    center_h, center_w = h // 2, w // 2
    
    # Create gradient target: 1 at center, 0 elsewhere
    grad_output = torch.zeros_like(target_feat)
    grad_output[0, 0, center_h, center_w] = 1.0
    
    # Backward
    target_feat.backward(grad_output, retain_graph=True)
    
    # Gradient w.r.t input shows receptive field
    grad_map = x.grad[0].abs().sum(0).cpu().numpy()  # (H, W)
    grad_map = grad_map / (grad_map.max() + 1e-8)
    
    # Threshold to find effective receptive field
    threshold = 0.01
    mask = grad_map > threshold
    if mask.any():
        rows, cols = np.where(mask)
        rf_h = rows.max() - rows.min() + 1
        rf_w = cols.max() - cols.min() + 1
        center_y = (rows.max() + rows.min()) / 2
        center_x = (cols.max() + cols.min()) / 2
    else:
        rf_h = rf_w = 0
        center_y = center_x = input_size / 2
    
    return grad_map, (rf_h, rf_w), (center_y, center_x)

# ============================================================
# PART 3: SPATIAL HIERARCHY VISUALIZATION
# ============================================================
def visualize_spatial_hierarchy(model, image_tensor, layer_indices, device=DEVICE):
    """
    Visualize feature maps at different depths to show spatial hierarchy.
    """
    model.eval()
    with torch.no_grad():
        _, feats = model(image_tensor.to(device), return_features=True)
    
    fig, axes = plt.subplots(len(layer_indices), 4, figsize=(12, 3 * len(layer_indices)))
    if len(layer_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(layer_indices):
        name, feat = feats[idx]
        feat = feat[0].cpu()  # (C, H, W)
        
        # Show first 4 channels
        for ch in range(min(4, feat.shape[0])):
            ax = axes[row, ch]
            im = feat[ch].numpy()
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)
            ax.imshow(im, cmap='viridis')
            ax.set_title(f"{name}\nCh {ch} ({feat.shape[1]}x{feat.shape[2]})", fontsize=9)
            ax.axis('off')
    
    plt.suptitle("Spatial Hierarchy: Feature Maps Across Depth", fontsize=14)
    plt.tight_layout()
    return fig

def visualize_receptive_fields(model, layer_indices, input_size=224, device=DEVICE):
    """
    Plot empirical receptive field heatmaps for selected layers.
    """
    fig, axes = plt.subplots(1, len(layer_indices), figsize=(4 * len(layer_indices), 4))
    if len(layer_indices) == 1:
        axes = [axes]
    
    for ax, idx in zip(axes, layer_indices):
        grad_map, rf_size, center = empirical_receptive_field(model, idx, input_size, device)
        im = ax.imshow(grad_map, cmap='hot', interpolation='nearest')
        ax.set_title(f"Layer {idx}\nRF: {rf_size[0]}x{rf_size[1]}\nCenter: ({center[0]:.0f}, {center[1]:.0f})", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle("Empirical Receptive Fields (Gradient Backprop)", fontsize=14)
    plt.tight_layout()
    return fig

# ============================================================
# PART 4: RECEPTIVE FIELD GROWTH COMPARISON
# ============================================================
def plot_rf_growth(theoretical_rfs, empirical_rfs, layer_names):
    """
    Compare theoretical vs empirical receptive field growth.
    """
    theo_sizes = [rf for rf, _, _ in theoretical_rfs]
    emp_sizes = [rf[0] for rf in empirical_rfs]  # use height
    
    # Only plot layers that have both
    valid_indices = [i for i, (t, e) in enumerate(zip(theo_sizes, emp_sizes)) if e > 0]
    theo_valid = [theo_sizes[i] for i in valid_indices]
    emp_valid = [emp_sizes[i] for i in valid_indices]
    names_valid = [layer_names[i] for i in valid_indices]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(valid_indices))
    width = 0.35
    
    ax.bar(x - width/2, theo_valid, width, label='Theoretical', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, emp_valid, width, label='Empirical', alpha=0.8, color='coral')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names_valid, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Receptive Field Size (pixels)')
    ax.set_title('Theoretical vs Empirical Receptive Field Growth')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

# ============================================================
# PART 5: MAIN EXPERIMENT RUNNER
# ============================================================
def run_experiment():
    print("\n" + "=" * 70)
    print("DAY 16: CNN RECEPTIVE FIELD & SPATIAL HIERARCHY EXPERIMENT")
    print("=" * 70)
    
    # Initialize model
    model = SimpleCNN().to(DEVICE)
    print(f"Model loaded on {DEVICE}")
    print(f"Total layers with features: {len(model.layer_names)}")
    
    # Create a test image (simple pattern)
    test_img = torch.zeros(1, 3, 224, 224)
    # Add some structure: gradients, edges, shapes
    for c in range(3):
        test_img[0, c] = torch.linspace(0, 1, 224).unsqueeze(0).repeat(224, 1)
        if c == 1:
            test_img[0, c] += torch.linspace(0, 1, 224).unsqueeze(1).repeat(1, 224)
        if c == 2:
            # Add a circle
            y, x = torch.meshgrid(torch.linspace(-1, 1, 224), torch.linspace(-1, 1, 224), indexing='ij')
            test_img[0, c] += (x**2 + y**2 < 0.5).float()
    test_img = torch.clamp(test_img, 0, 1)
    
    # Save input visualization
    plt.figure(figsize=(4, 4))
    plt.imshow(test_img[0].permute(1, 2, 0).numpy())
    plt.title("Input Test Image")
    plt.axis('off')
    plt.savefig("day16_input.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved input visualization: day16_input.png")
    
    # Select key layers for analysis (conv layers after each block)
    key_layer_indices = [0, 3, 6, 10, 14, 18, 22]  # conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_3, conv4_1
    key_layer_names = [model.layer_names[i] for i in key_layer_indices]
    
    print("\n[1/4] Computing empirical receptive fields...")
    empirical_rfs = []
    for idx in key_layer_indices:
        _, rf_size, center = empirical_receptive_field(model, idx, 224, DEVICE)
        empirical_rfs.append(rf_size)
        print(f"  {model.layer_names[idx]:>8}: RF={rf_size}, Center=({center[0]:.0f}, {center[1]:.0f})")
    
    print("\n[2/4] Visualizing spatial hierarchy (feature maps)...")
    fig1 = visualize_spatial_hierarchy(model, test_img, key_layer_indices, DEVICE)
    fig1.savefig("day16_spatial_hierarchy.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("Saved: day16_spatial_hierarchy.png")
    
    print("\n[3/4] Visualizing empirical receptive fields...")
    fig2 = visualize_receptive_fields(model, key_layer_indices, 224, DEVICE)
    fig2.savefig("day16_empirical_rf.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("Saved: day16_empirical_rf.png")
    
    print("\n[4/4] Comparing theoretical vs empirical RF growth...")
    fig3    fig3 = plot_rf_growth(rf_results, empirical_rfs, layer_names)
    fig3.savefig("day16_rf_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("Saved: day16_rf_comparison.png")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Input resolution: 224x224")
    print(f"Final theoretical RF: {rf_results[-1][0]}x{rf_results[-1][0]} (covers entire input)")
    print(f"Final effective stride: {rf_results[-1][1]} pixels")
    print("\nKey Insight: Receptive field grows linearly with depth.")
    print("Early layers: small RF -> edges, colors, textures")
    print("Middle layers: medium RF -> patterns, parts")
    print("Late layers: large RF -> objects, scenes")
    print("\nAll visualizations saved as day16_*.png")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()