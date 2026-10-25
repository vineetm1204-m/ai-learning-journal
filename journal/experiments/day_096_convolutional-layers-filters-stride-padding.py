import numpy as np
import matplotlib.pyplot as plt

def conv2d_manual(input_img, kernel, stride=1, padding=0):
    """
    Manual 2D convolution implementation.
    Args:
        input_img: (H, W) numpy array
        kernel: (kH, kW) numpy array
        stride: int
        padding: int (symmetric zero-padding)
    Returns:
        output: (out_H, out_W) numpy array
    """
    h, w = input_img.shape
    kh, kw = kernel.shape
    
    # Apply padding
    if padding > 0:
        input_padded = np.pad(input_img, ((padding, padding), (padding, padding)), mode='constant')
    else:
        input_padded = input_img
    
    h_pad, w_pad = input_padded.shape
    
    # Calculate output dimensions
    out_h = (h_pad - kh) // stride + 1
    out_w = (w_pad - kw) // stride + 1
    
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Kernel too large for input size with given stride/padding.")
    
    output = np.zeros((out_h, out_w))
    
    # Slide kernel
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + kh
            w_end = w_start + kw
            
            patch = input_padded[h_start:h_end, w_start:w_end]
            output[i, j] = np.sum(patch * kernel)
            
    return output

def create_synthetic_image(size=28):
    """Creates a simple synthetic image with edges and shapes."""
    img = np.zeros((size, size), dtype=np.float32)
    # Vertical bar
    img[:, 10:14] = 1.0
    # Horizontal bar
    img[10:14, :] = 1.0
    # Diagonal line
    for i in range(size):
        if 0 <= i < size and 0 <= (size - 1 - i) < size:
            img[i, size - 1 - i] = 0.8
    # Box
    img[20:26, 20:26] = 0.5
    return img

def run_experiment():
    print("--- Day 96: Convolutional Layers Mini-Experiment ---")
    
    # 1. Setup Input
    img = create_synthetic_image(28)
    print(f"Input Image Shape: {img.shape}")
    
    # 2. Define Filters (Kernels)
    filters = {
        "Sobel_X (Vertical Edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        "Sobel_Y (Horizontal Edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        "Blur (Box)": np.ones((3, 3), dtype=np.float32) / 9.0,
        "Sharpen (Laplacian)": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    }
    
    # 3. Experiment A: Filter Effects (Stride=1, Padding=1 'same')
    print("\n[Experiment A] Filter Effects (Stride=1, Padding=1)")
    fig_a, axes_a = plt.subplots(1, len(filters) + 1, figsize=(15, 3))
    axes_a[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes_a[0].set_title("Original\n28x28")
    axes_a[0].axis('off')
    
    for idx, (name, kernel) in enumerate(filters.items()):
        out = conv2d_manual(img, kernel, stride=1, padding=1)
        # Normalize for visualization
        out_vis = (out - out.min()) / (out.max() - out.min() + 1e-8)
        axes_a[idx + 1].imshow(out_vis, cmap='gray')
        axes_a[idx + 1].set_title(f"{name}\n{out.shape[0]}x{out.shape[1]}")
        axes_a[idx + 1].axis('off')
    plt.tight_layout()
    plt.savefig("day96_filters.png")
    print("Saved 'day96_filters.png'")
    plt.close(fig_a)
    
    # 4. Experiment B: Stride Effects
    print("\n[Experiment B] Stride Effects (Sobel_X, Padding=1)")
    strides = [1, 2, 3]
    fig_b, axes_b = plt.subplots(1, len(strides) + 1, figsize=(12, 3))
    axes_b[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes_b[0].set_title("Original\n28x28")
    axes_b[0].axis('off')
    
    kernel = filters["Sobel_X (Vertical Edges)"]
    for idx, s in enumerate(strides):
        out = conv2d_manual(img, kernel, stride=s, padding=1)
        out_vis = (out - out.min()) / (out.max() - out.min() + 1e-8)
        axes_b[idx + 1].imshow(out_vis, cmap='gray')
        axes_b[idx + 1].set_title(f"Stride={s}\n{out.shape[0]}x{out.shape[1]}")
        axes_b[idx + 1].axis('off')
    plt.tight_layout()
    plt.savefig("day96_stride.png")
    print("Saved 'day96_stride.png'")
    plt.close(fig_b)
    
    # 5. Experiment C: Padding Effects
    print("\n[Experiment C] Padding Effects (Sobel_X, Stride=1)")
    paddings = [0, 1, 2] # Valid, Same (for 3x3), Extra
    fig_c, axes_c = plt.subplots(1, len(paddings) + 1, figsize=(12, 3))
    axes_c[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes_c[0].set_title("Original\n28x28")
    axes_c[0].axis('off')
    
    for idx, p in enumerate(paddings):
        out = conv2d_manual(img, kernel, stride=1, padding=p)
        out_vis = (out - out.min()) / (out.max() - out.min() + 1e-8)
        pad_name = "Valid (0)" if p == 0 else f"Padding={p}"
        axes_c[idx + 1].imshow(out_vis, cmap='gray')
        axes_c[idx + 1].set_title(f"{pad_name}\n{out.shape[0]}x{out.shape[1]}")
        axes_c[idx + 1].axis('off')
    plt.tight_layout()
    plt.savefig("day96_padding.png")
    print("Saved 'day96_padding.png'")
    plt.close(fig_c)
    
    # 6. Numerical Verification of Output Shapes
    print("\n[Verification] Output Shape Formula: Floor((W - K + 2P) / S) + 1")
    H, W = img.shape
    K = 3
    for S in [1, 2]:
        for P in [0, 1]:
            calc_h = (H - K + 2 * P) // S + 1
            calc_w = (W - K + 2 * P) // S + 1
            actual = conv2d_manual(img, kernel, stride=S, padding=P).shape
            match = "OK" if (calc_h, calc_w) == actual else "MISMATCH"
            print(f"  Stride={S}, Pad={P}: Formula=({calc_h},{calc_w}) Actual={actual} [{match}]")

    print("\nExperiment Complete.")

if __name__ == "__main__":
    run_experiment()