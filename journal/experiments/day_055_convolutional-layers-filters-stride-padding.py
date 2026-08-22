import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# 1. Synthetic Input: A 6x6 image with a vertical edge pattern
# ------------------------------------------------------------
def create_synthetic_input():
    """Creates a batch=1, channel=1, 6x6 image with a vertical line."""
    img = torch.zeros(1, 1, 6, 6)
    img[:, :, :, 2:4] = 1.0  # Vertical bar in the middle
    return img

# ------------------------------------------------------------
# 2. Manual Convolution (NumPy) - "Under the Hood"
# ------------------------------------------------------------
def manual_conv2d(input_np, kernel_np, stride=1, padding=0):
    """
    Naive implementation of 2D cross-correlation (standard DL 'convolution').
    Input: (H, W), Kernel: (kH, kW)
    """
    H, W = input_np.shape
    kH, kW = kernel_np.shape
    
    # Pad input
    if padding > 0:
        input_padded = np.pad(input_np, ((padding, padding), (padding, padding)), mode='constant')
    else:
        input_padded = input_np
    
    # Output dimensions
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    
    output = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + kH
            w_end = w_start + kW
            
            patch = input_padded[h_start:h_end, w_start:w_end]
            output[i, j] = np.sum(patch * kernel_np)
            
    return output

# ------------------------------------------------------------
# 3. Visualization Helpers
# ------------------------------------------------------------
def plot_feature_maps(images, titles, suptitle, cmap='viridis'):
    """Plots a list of 2D arrays as a grid."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1: axes = [axes]
    fig.suptitle(suptitle, fontsize=14)
    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, cmap=cmap, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def print_tensor_info(name, tensor):
    print(f"{name}: shape={tuple(tensor.shape)}, min={tensor.min():.2f}, max={tensor.max():.2f}")

# ------------------------------------------------------------
# 4. Experiment Runs
# ------------------------------------------------------------
def run_experiments():
    input_img = create_synthetic_input().to(DEVICE)
    input_np = input_img.squeeze().cpu().numpy()
    
    # Define a Vertical Edge Detection Filter (Sobel-ish)
    # High response for vertical transitions
    kernel_v = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(DEVICE)
    kernel_v_np = kernel_v.squeeze().cpu().numpy()

    print("="*60)
    print("DAY 55: CONVOLUTIONAL LAYERS - FILTERS, STRIDE, PADDING")
    print("="*60)
    print(f"Input Image Shape: {tuple(input_img.shape)} (Batch, Channels, H, W)")
    print(f"Kernel Shape:      {tuple(kernel_v.shape)} (Out_Ch, In_Ch, kH, kW)")
    print("-" * 60)

    # --------------------------------------------------------
    # Experiment A: Stride Comparison (Padding=0)
    # --------------------------------------------------------
    print("\n[EXPERIMENT A] Varying Stride (Padding=0, Kernel=3x3)")
    print("Formula: Out = floor((W - K) / S) + 1")
    
    strides = [1, 2]
    results_a = []
    titles_a = []
    
    for s in strides:
        # PyTorch
        conv = nn.Conv2d(1, 1, kernel_size=3, stride=s, padding=0, bias=False).to(DEVICE)
        conv.weight.data = kernel_v.clone()
        out_pt = conv(input_img)
        
        # Manual
        out_manual = manual_conv2d(input_np, kernel_v_np, stride=s, padding=0)
        
        print(f"  Stride={s}: PyTorch Out Shape={tuple(out_pt.shape)} | Manual Out Shape={out_manual.shape}")
        assert np.allclose(out_pt.squeeze().cpu().numpy(), out_manual, atol=1e-5), "Mismatch!"
        
        results_a.append(out_pt.squeeze().cpu().numpy())
        titles_a.append(f"Stride={s} (Out: {out_manual.shape[0]}x{out_manual.shape[1]})")

    plot_feature_maps([input_np] + results_a, ["Input"] + titles_a, "Experiment A: Stride Effect on Spatial Resolution")

    # --------------------------------------------------------
    # Experiment B: Padding Comparison (Stride=1)
    # --------------------------------------------------------
    print("\n[EXPERIMENT B] Varying Padding (Stride=1, Kernel=3x3)")
    print("Formula: Out = floor((W + 2P - K) / S) + 1")
    print("'Same' Padding (P=1) preserves spatial dims (6x6 -> 6x6).")
    print("'Valid' Padding (P=0) reduces spatial dims (6x6 -> 4x4).")
    
    paddings = [0, 1]
    results_b = []
    titles_b = []
    
    for p in paddings:
        conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=p, bias=False).to(DEVICE)
        conv.weight.data = kernel_v.clone()
        out_pt = conv(input_img)
        
        out_manual = manual_conv2d(input_np, kernel_v_np, stride=1, padding=p)
        
        print(f"  Padding={p}: PyTorch Out Shape={tuple(out_pt.shape)} | Manual Out Shape={out_manual.shape}")
        assert np.allclose(out_pt.squeeze().cpu().numpy(), out_manual, atol=1e-5), "Mismatch!"
        
        results_b.append(out_pt.squeeze().cpu().numpy())
        titles_b.append(f"Padding={p} (Out: {out_manual.shape[0]}x{out_manual.shape[1]})")

    plot_feature_maps([input_np] + results_b, ["Input"] + titles_b, "Experiment B: Padding Effect (Border Handling)")

    # --------------------------------------------------------
    # Experiment C: Multiple Filters (Channels Out)
    # --------------------------------------------------------
    print("\n[EXPERIMENT C] Multiple Filters (Out_Channels=3)")
    print("Stacking different kernels: Vertical Edge, Horizontal Edge, Blur.")
    
    # Kernel 1: Vertical Edge (defined above)
    # Kernel 2: Horizontal Edge
    kernel_h = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=torch.float32)
    # Kernel 3: Blur / Smoothing
    kernel_b = torch.tensor([[1, 2, 1],
                             [2, 4, 2],
                             [1, 2, 1]], dtype=torch.float32) / 16.0
    
    # Stack: (Out_C, In_C, kH, kW)
    multi_kernel = torch.stack([kernel_v.squeeze(), kernel_h, kernel_b], dim=0).unsqueeze(1).to(DEVICE)
    
    conv_multi = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False).to(DEVICE)
    conv_multi.weight.data = multi_kernel.clone()
    out_multi = conv_multi(input_img) # Shape: (1, 3, 6, 6)
    
    print(f"  Output Shape: {tuple(out_multi.shape)} (Batch, 3 Filters, H, W)")
    
    # Visualize each filter output
    filter_names = ["Vertical Edge", "Horizontal Edge", "Blur"]
    imgs_c = [out_multi[0, i].cpu().numpy() for i in range(3)]
    plot_feature_maps(imgs_c, filter_names, "Experiment C: 3 Different Filters (Padding=1, Stride=1)")

    # --------------------------------------------------------
    # Experiment D: Receptive Field Visualization
    # --------------------------------------------------------
    print("\n[EXPERIMENT D] Receptive Field Growth (Stacking Layers)")
    print("Two 3x3 conv layers (stride=1, pad=1) -> Effective Receptive Field 5x5.")
    
    # Layer 1
    l1 = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False).to(DEVICE)
    l1.weight.data = kernel_v.clone()
    # Layer 2
    l2 = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False).to(DEVICE)
    # Identity-ish kernel for L2 to see propagation clearly
    l2.weight.data = torch.tensor([[[[0,0,0],[0,1,0],[0,0,0]]]], dtype=torch.float32).to(DEVICE)
    
    out_l1 = l1(input_img)
    out_l2 = l2(out_l1)
    
    print(f"  Input:  {tuple(input_img.shape)}")
    print(f"  After L1 (3x3 kernel): {tuple(out_l1.shape)} -> Receptive Field: 3x3")
    print(f"  After L2 (3x3 kernel): {tuple(out_l2.shape)} -> Receptive Field: 5x5")
    
    plot_feature_maps(
        [input_np, out_l1.squeeze().cpu().numpy(), out_l2.squeeze().cpu().numpy()],
        ["Input", "L1 Output (RF 3x3)", "L2 Output (RF 5x5)"],
        "Experiment D: Receptive Field Expansion"
    )

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE.")
    print("="*60)

if __name__ == "__main__":
    run_experiments()