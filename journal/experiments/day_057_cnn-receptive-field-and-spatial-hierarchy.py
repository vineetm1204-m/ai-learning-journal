import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# ============================================================
# 1. Model Definition: A simple CNN to track hierarchy
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: RF 3x3 -> 3x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # RF 4x4
        
        # Block 2: RF 4x4 -> 10x10
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # RF 20x20
        
        # Block 3: RF 20x20 -> 44x44
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # RF 88x88
        
        # Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
        
        # Store layer names for hooking
        self.layer_blocks = OrderedDict([
            ('conv1', self.conv1), ('pool1', self.pool1),
            ('conv2', self.conv2), ('pool2', self.pool2),
            ('conv3', self.conv3), ('pool3', self.pool3),
        ])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================================================
# 2. Receptive Field Calculator (Theoretical)
# ============================================================
def compute_receptive_field(model, input_size=(224, 224)):
    """
    Computes theoretical receptive field size for each layer.
    Assumes standard Conv(k=3, p=1, s=1) and Pool(k=2, s=2).
    """
    # (r, j, start) -> receptive field, stride (jump), start offset
    # Initialize for input image
    r, j, start = 1, 1, 0.5
    layer_stats = []
    
    # We manually trace the SimpleCNN architecture defined above
    layers = [
        ('conv1', 'conv', 3, 1, 1),
        ('pool1', 'pool', 2, 2, 0),
        ('conv2', 'conv', 3, 1, 1),
        ('pool2', 'pool', 2, 2, 0),
        ('conv3', 'conv', 3, 1, 1),
        ('pool3', 'pool', 2, 2, 0),
    ]
    
    for name, ltype, k, s, p in layers:
        if ltype == 'conv':
            r_new = r + (k - 1) * j
            start_new = start + ((k - 1) / 2 - p) * j
            j_new = j * s
        else: # pool
            r_new = r + (k - 1) * j
            start_new = start + ((k - 1) / 2 - p) * j
            j_new = j * s
            
        r, j, start = r_new, j_new, start_new
        layer_stats.append((name, int(r), int(j)))
        
    return layer_stats

# ============================================================
# 3. Hook Mechanism for Feature Map Extraction
# ============================================================
class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        # Detach and move to CPU for plotting
        self.features = output.detach().cpu()
    
    def close(self):
        self.hook.remove()

# ============================================================
# 4. Synthetic Input Generation (Structured Patterns)
# ============================================================
def generate_synthetic_input(size=224):
    """Creates an image with distinct frequency/orientation patterns."""
    img = np.zeros((3, size, size), dtype=np.float32)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Channel 0: Low freq vertical stripes (Global structure)
    img[0] = 0.5 * (np.sin(4 * np.pi * X) + 1)
    
    # Channel 1: High freq checkerboard (Texture/Detail)
    img[1] = 0.5 * (np.sign(np.sin(32 * np.pi * X) * np.sin(32 * np.pi * Y)) + 1)
    
    # Channel 2: Centered Gaussian blob (Object-like)
    img[2] = np.exp(-(X**2 + Y**2) / 0.1)
    
    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return torch.from_numpy(img).unsqueeze(0) # (1, 3, H, W)

# ============================================================
# 5. Visualization Utilities
# ============================================================
def plot_feature_maps(features_dict, max_maps=16):
    """Plots a grid of feature maps for each layer."""
    n_layers = len(features_dict)
    fig, axes = plt.subplots(n_layers, max_maps, figsize=(max_maps * 1.2, n_layers * 1.5), squeeze=False)
    fig.suptitle("Spatial Hierarchy: Feature Maps Across Depth", fontsize=16, y=1.0)
    
    for row_idx, (name, fmap) in enumerate(features_dict.items()):
        # fmap shape: (1, C, H, W)
        fmap = fmap[0] # (C, H, W)
        n_channels = min(fmap.shape[0], max_maps)
        
        # Normalize each channel independently for visibility
        for c in range(n_channels):
            ch = fmap[c].numpy()
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            axes[row_idx, c].imshow(ch, cmap='viridis')
            axes[row_idx, c].axis('off')
            if c == 0:
                axes[row_idx, c].set_ylabel(f"{name}\n{fmap.shape[1]}x{fmap.shape[2]}", rotation=0, labelpad=40, fontsize=10)
        
        # Hide unused axes
        for c in range(n_channels, max_maps):
            axes[row_idx, c].axis('off')
            
    plt.tight_layout()
    return fig

def plot_rf_growth(rf_stats):
    """Plots Receptive Field size vs Layer Depth."""
    names = [s[0] for s in rf_stats]
    rfs = [s[1] for s in rf_stats]
    strides = [s[2] for s in rf_stats]
    
    fig, ax1 = plt.subplots(figsize=(8, 4))
    color = 'tab:blue'
    ax1.set_xlabel('Layer Depth')
    ax1.set_ylabel('Receptive Field (pixels)', color=color)
    ax1.plot(names, rfs, 'o-', color=color, label='Receptive Field')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Stride (Jump)', color=color)
    ax2.plot(names, strides, 's--', color=color, label='Stride')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.suptitle("Theoretical Receptive Field Growth", fontsize=14)
    fig.tight_layout()
    return fig

def plot_input_image(img_tensor):
    """Plots the input synthetic image channels."""
    img = img_tensor[0].numpy() # (3, H, W)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    titles = ["Ch 0: Low Freq Stripes", "Ch 1: High Freq Checkerboard", "Ch 2: Gaussian Blob"]
    for i in range(3):
        axes[i].imshow(img[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    fig.suptitle("Synthetic Input Structure", fontsize=14)
    plt.tight_layout()
    return fig

# ============================================================
# 6. Main Experiment Execution
# ============================================================
if __name__ == "__main__":
    print("--- Day 57: CNN Receptive Field & Spatial Hierarchy ---")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.eval()
    
    # 1. Theoretical Analysis
    print("\n[1] Computing Theoretical Receptive Fields...")
    rf_stats = compute_receptive_field(model)
    print(f"{'Layer':<10} | {'RF Size':<10} | {'Stride (Jump)':<15}")
    print("-" * 40)
    for name, rf, stride in rf_stats:
        print(f"{name:<10} | {rf:<10} | {stride:<15}")
    
    # 2. Prepare Input
    print("\n[2] Generating Synthetic Input...")
    input_tensor = generate_synthetic_input(224).to(device)
    
    # 3. Register Hooks
    print("[3] Registering Forward Hooks...")
    hooks = {}
    for name, layer in model.layer_blocks.items():
        hooks[name] = FeatureHook(layer)
    
    # 4. Forward Pass
    print("[4] Running Forward Pass...")
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 5. Collect Features
    print("[5] Collecting Feature Maps...")
    captured_features = {}
    for name, hook in hooks.items():
        if hook.features is not None:
            captured_features[name] = hook.features
        hook.close()
    
    # 6. Visualize
    print("[6] Rendering Visualizations...")
    
    # Input
    fig_in = plot_input_image(input_tensor.cpu())
    
    # Feature Maps
    fig_fm = plot_feature_maps(captured_features, max_maps=8)
    
    # RF Growth
    fig_rf = plot_rf_growth(rf_stats)
    
    # Show all
    plt.show()
    
    print("\n--- Experiment Complete ---")