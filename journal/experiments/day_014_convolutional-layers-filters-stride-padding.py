import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================
# Day 14: Convolutional Layers - Filters, Stride, Padding
# Self-contained mini-experiment
# ============================================================

def manual_conv2d(input_matrix, kernel, stride=1, padding=0):
    """
    Manual implementation of 2D convolution to understand the mechanics.
    """
    # Add padding
    if padding > 0:
        input_padded = np.pad(input_matrix, pad_width=padding, mode='constant', constant_values=0)
    else:
        input_padded = input_matrix
    
    h_in, w_in = input_padded.shape
    k_h, k_w = kernel.shape
    
    # Calculate output dimensions
    h_out = (h_in - k_h) // stride + 1
    w_out = (w_in - k_w) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride
            receptive_field = input_padded[h_start:h_start+k_h, w_start:w_start+k_w]
            output[i, j] = np.sum(receptive_field * kernel)
    
    return output

def visualize_conv_process(input_img, kernel, stride, padding, title):
    """Visualize input, kernel, and output."""
    output = manual_conv2d(input_img, kernel, stride, padding)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title(f'Input ({input_img.shape[0]}x{input_img.shape[1]})')
    axes[0].axis('off')
    
    axes[1].imshow(kernel, cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title(f'Kernel ({kernel.shape[0]}x{kernel.shape[1]})')
    axes[1].axis('off')
    
    im = axes[2].imshow(output, cmap='viridis')
    axes[2].set_title(f'Output ({output.shape[0]}x{output.shape[1]})\nStride={stride}, Padding={padding}')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return output

def demonstrate_padding_modes():
    """Show different padding modes in PyTorch."""
    print("=" * 60)
    print("PADDING MODES IN PYTORCH")
    print("=" * 60)
    
    # Create a simple 5x5 input with border pattern
    x = torch.zeros(1, 1, 5, 5)
    x[0, 0, 0, :] = 1  # Top border
    x[0, 0, -1, :] = 1  # Bottom border
    x[0, 0, :, 0] = 1  # Left border
    x[0, 0, :, -1] = 1  # Right border
    x[0, 0, 2, 2] = 2  # Center point
    
    kernel = torch.ones(1, 1, 3, 3)
    conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)
    conv.weight.data = kernel
    
    modes = ['zeros', 'reflect', 'replicate', 'circular']
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(x[0, 0].numpy(), cmap='gray')
    axes[0, 0].set_title('Input (5x5)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(kernel[0, 0].numpy(), cmap='RdBu')
    axes[0, 1].set_title('Kernel (3x3, all ones)')
    axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')
    
    for idx, mode in enumerate(modes):
        row = 1 if idx >= 2 else 0
        col = idx % 2 + (1 if row == 0 else 0)
        if row == 0 and col == 2:
            continue
            
        # Need to use functional conv for padding_mode
        padded = F.pad(x, (1, 1, 1, 1), mode=mode)
        out = F.conv2d(padded, kernel)
        
        ax = axes[row, col] if row == 1 else axes[0, col]
        ax.imshow(out[0, 0].detach().numpy(), cmap='viridis')
        ax.set_title(f'padding_mode={mode}\nOutput: {out.shape[2]}x{out.shape[3]}')
        ax.axis('off')
    
    plt.suptitle('Effect of Different Padding Modes')
    plt.tight_layout()
    plt.show()

def demonstrate_stride_effects():
    """Show how stride affects output size and receptive field."""
    print("\n" + "=" * 60)
    print("STRIDE EFFECTS ON OUTPUT SIZE")
    print("=" * 60)
    
    input_sizes = [7, 8, 9, 10]
    kernel_size = 3
    strides = [1, 2, 3]
    
    print(f"{'Input':>6} | {'Kernel':>6} | {'Stride':>6} | {'Padding':>7} | {'Output':>6} | Formula")
    print("-" * 70)
    
    for h_in in input_sizes:
        for stride in strides:
            for padding in [0, 1]:
                h_out = (h_in + 2*padding - kernel_size) // stride + 1
                if h_out > 0:
                    formula = f"({h_in} + 2×{padding} - {kernel_size})/{stride} + 1 = {h_out}"
                    print(f"{h_in:>6} | {kernel_size:>6} | {stride:>6} | {padding:>7} | {h_out:>6} | {formula}")
    
    # Visual demonstration
    x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
    kernel = torch.ones(1, 1, 2, 2)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(x[0, 0].numpy(), cmap='Blues')
    axes[0].set_title('Input (4x4)')
    axes[0].axis('off')
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f'{int(x[0,0,i,j])}', ha='center', va='center', fontsize=12)
    
    for idx, stride in enumerate([1, 2, 3]):
        conv = nn.Conv2d(1, 1, kernel_size=2, stride=stride, bias=False)
        conv.weight.data = kernel
        out = conv(x)
        
        axes[idx+1].imshow(out[0, 0].detach().numpy(), cmap='Oranges')
        axes[idx+1].set_title(f'Stride={stride}, Output: {out.shape[2]}x{out.shape[3]}')
        axes[idx+1].axis('off')
        for i in range(out.shape[2]):
            for j in range(out.shape[3]):
                axes[idx+1].text(j, i, f'{out[0,0,i,j]:.0f}', ha='center', va='center', fontsize=12)
    
    plt.suptitle('Stride Effect on 4x4 Input with 2x2 Kernel')
    plt.tight_layout()
    plt.show()

def demonstrate_filter_learning():
    """Show how filters detect different patterns."""
    print("\n" + "=" * 60)
    print("FILTER PATTERNS: EDGE DETECTORS")
    print("=" * 60)
    
    # Classic edge detection kernels
    kernels = {
        'Vertical Edge (Sobel)': np.array([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]]),
        'Horizontal Edge (Sobel)': np.array([[-1, -2, -1],
                                              [0, 0, 0],
                                              [1, 2, 1]]),
        'Diagonal Edge': np.array([[0, 1, 2],
                                    [-1, 0, 1],
                                    [-2, -1, 0]]),
        'Laplacian (Sharpen)': np.array([[0, -1, 0],
                                          [-1, 4, -1],
                                          [0, -1, 0]]),
        'Gaussian Blur': np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]]) / 16,
        'Emboss': np.array([[-2, -1, 0],
                             [-1, 1, 1],
                             [0, 1, 2]])
    }
    
    # Create test image: square with gradient
    img = np.zeros((20, 20))
    img[5:15, 5:15] = 1  # White square
    img[8:12, 8:12] = 0.5  # Gray inner square
    # Add diagonal gradient
    for i in range(20):
        for j in range(20):
            img[i, j] += (i + j) / 80
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Test Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    
    for idx, (name, kernel) in enumerate(kernels.items()):
        row = 1 if idx >= 3 else 0
        col = idx % 3 + (1 if row == 0 else 0)
        if row == 0 and col > 1:
            col += 1
        
        out = manual_conv2d(img, kernel, stride=1, padding=1)
        
        ax = axes[row, col]
        im = ax.imshow(out, cmap='RdBu', vmin=-np.max(np.abs(out)), vmax=np.max(np.abs(out)))
        ax.set_title(name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Different Filters Detect Different Features')
    plt.tight_layout()
    plt.show()

def pytorch_conv_analysis():
    """Analyze PyTorch Conv2d parameters and behavior."""
    print("\n" + "=" * 60)
    print("PYTORCH CONV2D PARAMETER ANALYSIS")
    print("=" * 60)
    
    # Different configurations
    configs = [
        {'in_channels': 3, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'in_channels': 3, 'out_channels': 32, 'kernel_size': 5, 'stride': 2, 'padding': 2},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0},
        {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1, 'stride': 1, 'padding': 0},  # 1x1 conv
    ]
    
    print(f"{'Config':<40} | {'Params':>10} | {'Input':>12} | {'Output':>12}")
    print("-" * 85)
    
    for i, cfg in enumerate(configs):
        conv = nn.Conv2d(**cfg, bias=True)
        params = sum(p.numel() for p in conv.parameters())
        
        # Test with sample input
        h_in, w_in = 32, 32
        x = torch.randn(1, cfg['in_channels'], h_in, w_in)
        out = conv(x)
        
        print(f"Conv2d({cfg['in_channels']}->{cfg['out_channels']}, k={cfg['kernel_size']}, "
              f"s={cfg['stride']}, p={cfg['padding']}) | "
              f"{params:>10,} | "
              f"{cfg['in_channels']}x{h_in}x{w_in:>3} | "
              f"{cfg['out_channels']}x{out.shape[2]}x{out.shape[3]:>3}")
    
    # Demonstrate 1x1 convolution (channel mixing)
    print("\n--- 1x1 Convolution: Channel Mixing ---")
    x = torch.randn(1, 64, 8, 8)  # 64 channels
    conv1x1 = nn.Conv2d(64, 32, kernel_size=1)  # Reduce to 32 channels
    out = conv1x1(x)
    print(f"Input:  {x.shape}  ->  Output: {out.shape}")
    print("1x1 conv learns linear combinations across channels (no spatial mixing)")

def receptive_field_calculator():
    """Calculate receptive field for stacked conv layers."""
    print("\n" + "=" * 60)
    print("RECEPTIVE FIELD CALCULATOR")
    print("=" * 60)
    
    def calc_receptive_field(layers, input_size=224):
        """
        layers: list of (kernel_size, stride, padding)
        Returns receptive field size and output size
        """
        rf = 1  # receptive field
        jump = 1  # effective stride
        size = input_size
        
        print(f"{'Layer':<6} | {'Kernel':>6} | {'Stride':>6} | {'Pad':>4} | {'RF':>6} | {'Jump':>6} | {'Out Size':>8}")
        print("-" * 65)
        
        for i, (k, s, p) in enumerate(layers):
            rf = rf + (k - 1) * jump
            jump = jump * s
            size = (size + 2*p - k) // s + 1
            print(f"{i+1:<6} | {k:>6} | {s:>6} | {p:>4} | {rf:>6} | {jump:>6} | {size:>8}")
        
        return rf, size
    
    # VGG-like architecture
    print("VGG-style (3x3 convs, stride 1, pad 1, pool stride 2):")
    vgg_layers = [
        (3, 1, 1), (3, 1, 1),  # conv1_1, conv1_2
        (2, 2, 0),              # pool1
        (3, 1, 1), (3, 1, 1),  # conv2_1, conv2_2
        (2, 2, 0),              # pool2
        (3, 1, 1), (3, 1, 1), (3, 1, 1),  # conv3
        (2, 2, 0),              # pool3
    ]
    calc_receptive_field(vgg_layers)
    
    print("\nResNet-style (7x7 stride 2, then 3x3):")
    resnet_layers = [
        (7, 2, 3),   # conv1
        (3, 2, 1),   # maxpool
        (3, 1, 1), (3, 1, 1),  # layer1
        (3, 2, 1), (3, 1, 1),  # layer2 (first block stride 2)
        (3, 2, 1), (3, 1, 1),  # layer3
        (3, 2, 1), (3, 1, 1),  # layer4
    ]
    calc_receptive_field(resnet_layers)

def main():
    print("=" * 60)
    print("DAY 14: CONVOLUTIONAL LAYERS - FILTERS, STRIDE, PADDING")
    print("=" * 60)
    
    # 1. Manual convolution basics
    print("\n1. MANUAL CONVOLUTION DEMONSTRATION")
    print("-" * 40)
    
    # Simple 5x5 input, 3x3 kernel
    input_img = np.array([
        [1, 2, 3, 0, 1],
        [4, 5, 6, 1, 0],
        [7, 8, 9, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1]
    ], dtype=float)
    
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ], dtype=float)  # Vertical edge detector
    
    print("Input (5x5):")
    print(input_img.astype(int))
    print("\nKernel (3x3) - Vertical Edge Detector:")
    print(kernel.astype(int))
    
    # No padding, stride 1
    out_valid = manual_conv2d(input_img, kernel, stride=1, padding=0)
    print("\nOutput (Valid, stride=1):")
    print(np.round(out_valid, 2))
    print(f"Shape: {out_valid.shape}  (Formula: (5-3)/1+1 = 3)")
    
    # With padding, stride 1
    out_same = manual_conv2d(input_img, kernel, stride=1, padding=1)
    print("\nOutput (Same, stride=1, padding=1):")
    print(np.round(out_same, 2))
    print(f"Shape: {out_same.shape}  (Formula: (5+2-3)/1+1 = 5)")
    
    # Stride 2
    out_stride2 = manual_conv2d(input_img, kernel, stride=2, padding=0)
    print("\nOutput (Valid, stride=2):")
    print(np.round(out_stride2, 2))
    print(f"Shape: {out_stride2.shape}  (Formula: (5-3)/2+1 = 2)")
    
    # 2. Visual demonstrations
    print("\n2. VISUALIZING CONVOLUTION PROCESS")
    print("   (Close plot windows to continue...)")
    
    # Create a more interesting test image
    test_img = np.zeros((16, 16))
    test_img[4:12, 4:12] = 1
    test_img[6:10, 6:10] = 0.3
    
    edge_kernel = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])
    
    visualize_conv_process(test_img, edge_kernel, stride=1, padding=1, 
                          title="Stride=1, Padding=1 (Same)")
    visualize_conv_process(test_img, edge_kernel, stride=2, padding=0,
                          title="Stride=2, Padding=0 (Downsample)")
    
    # 3. Padding modes
    demonstrate_padding_modes()
    
    # 4. Stride effects
    demonstrate_stride_effects()
    
    # 5. Filter patterns
    demonstrate_filter_learning()
    
    # 6. PyTorch analysis
    pytorch_conv_analysis()
    
    # 7. Receptive field
    receptive_field_calculator()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
1. OUTPUT SIZE FORMULA: H_out = (H_in + 2*padding - kernel_size) / stride + 1

2. PADDING TYPES:
   - Valid (padding=0): Output shrinks
   - Same (padding=(k-1)/2): Output same size (for stride=1)
   - Modes: zeros, reflect, replicate, circular

3. STRIDE:
   - Stride=1: Dense sampling, same resolution (with padding)
   - Stride>1: Downsampling, reduces spatial dimensions
   - Stride=2 common for reducing resolution

4. FILTERS/KERNELS:
   - Learnable parameters (weights)
   - Detect specific patterns (edges, textures, shapes)
   - 1x1 conv: Channel mixing without spatial change
   - 3x3 conv: Standard for capturing local patterns

5. RECEPTIVE FIELD:
   - Grows with depth and kernel size
   - Stacking 3x3 convs gives larger RF than single 7x7
   - Effective stride (jump) increases with pooling/strided conv

6. PARAMETER COUNT:
   - Conv2d: (in_channels * out_channels * k_h * k_w) + out_channels (bias)
   - Independent of input image size!
""")

if __name__ == "__main__":
    main()