import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# DAY 62: Object Detection Architecture Overview (Conceptual)
# Self-contained: Runs on CPU, no external weights/data needed.
# ============================================================

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------------------------------------------------------
# 1. BACKBONE (Shared Feature Extractor)
# ------------------------------------------------------------
class TinyBackbone(nn.Module):
    """Simulates a CNN backbone (e.g., ResNet50, Darknet53) outputting feature map."""
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1), nn.ReLU(), # /2
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),       # /4
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),      # /8
            nn.Conv2d(256, out_channels, 3, stride=2, padding=1), nn.ReLU() # /16
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# 2. R-CNN FAMILY CONCEPTS (Two-Stage)
# ------------------------------------------------------------

class RCNN_Head(nn.Module):
    """Simulates the 'Crop -> Warp -> FC' head of original R-CNN (Slow)."""
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 1024), nn.ReLU(),
            nn.Linear(1024, num_classes + 1), # +1 background
            nn.Linear(1024, 4 * num_classes)  # BBox regression per class
        )
    def forward(self, roi_features):
        # roi_features: (N_rois, C, H, H) -> Flattened
        return self.fc(roi_features)

class RoIPool(nn.Module):
    """RoI Pooling (Fast R-CNN): Quantizes coords, Max pools to fixed size."""
    def __init__(self, output_size=(7, 7)):
        super().__init__()
        self.output_size = output_size
    def forward(self, feature_map, rois):
        # feature_map: (B, C, H, W), rois: (N, 5) -> [batch_idx, x1, y1, x2, y2] in image coords
        # This is a simplified differentiable approximation using AdaptiveMaxPool2d per ROI
        # Real implementation uses RoIAlign (bilinear interp) for gradient flow.
        B, C, H, W = feature_map.shape
        pooled = []
        spatial_scale = H / 224.0 # Assuming input img 224
        
        for roi in rois:
            b_idx, x1, y1, x2, y2 = roi.int()
            # Map to feature map coords
            x1, y1, x2, y2 = [int(c * spatial_scale) for c in (x1, y1, x2, y2)]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1: 
                patch = torch.zeros(C, *self.output_size)
            else:
                patch = feature_map[b_idx, :, y1:y2, x1:x2]
                patch = F.adaptive_max_pool2d(patch.unsqueeze(0), self.output_size).squeeze(0)
            pooled.append(patch)
        return torch.stack(pooled) if pooled else torch.empty(0, C, *self.output_size)

class RPNHead(nn.Module):
    """Region Proposal Network (Faster R-CNN): Slides 3x3 window, predicts objectness + bbox delta."""
    def __init__(self, in_channels, num_anchors=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, 1) # Obj/Background
        self.bbox_reg = nn.Conv2d(512, num_anchors * 4, 1)   # dx, dy, dw, dh
    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.cls_logits(x), self.bbox_reg(x)

class FastRCNN_Head(nn.Module):
    """Fast/Faster R-CNN Head: RoI Pool -> FC -> Cls + BBox."""
    def __init__(self, in_channels, roi_size, num_classes):
        super().__init__()
        self.roi_pool = RoIPool(roi_size)
        feat_dim = in_channels * roi_size[0] * roi_size[1]
        self.fc1 = nn.Linear(feat_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls = nn.Linear(1024, num_classes + 1)
        self.bbox = nn.Linear(1024, 4 * (num_classes + 1))
    def forward(self, feat_map, rois):
        x = self.roi_pool(feat_map, rois) # (N, C, 7, 7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.cls(x), self.bbox(x)

# ------------------------------------------------------------
# 3. YOLO CONCEPT (Single-Stage)
# ------------------------------------------------------------

class YOLOHead(nn.Module):
    """
    YOLOv1 Style: Divides image into SxS grid.
    Each cell predicts B boxes (x,y,w,h,conf) + C class probs.
    Output: (B, S, S, B*5 + C)
    """
    def __init__(self, in_channels, S=7, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        # Simplified head: 1x1 conv to reduce channels, then FC (like v1) or Conv (like v3+)
        # Here we use a Conv head for fully convolutional behavior (YOLOv3/v5/v8 style)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
            nn.Conv2d(512, B * 5 + C, 1) # 5 = tx, ty, tw, th, to (objectness)
        )
    def forward(self, x):
        # x: (B, C, H, W) -> Output: (B, S, S, B*5+C) 
        # Note: In modern YOLO, H,W == S (stride 32). We assume backbone outputs SxS.
        out = self.head(x) # (B, Ch, S, S)
        return out.permute(0, 2, 3, 1).contiguous() # (B, S, S, Ch)

# ------------------------------------------------------------
# 4. EXPERIMENT RUNNER
# ------------------------------------------------------------

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # Config
    B, C_in, H_img, W_img = 1, 3, 224, 224
    num_classes = 20 # VOC style
    S = 7 # Grid size for YOLO / RoI output
    
    dummy_img = torch.randn(B, C_in, H_img, W_img).to(device)
    # Dummy ROIs: [batch_idx, x1, y1, x2, y2] in image coords (0-224)
    dummy_rois = torch.tensor([
        [0, 10, 10, 100, 100],
        [0, 50, 50, 150, 200],
        [0, 120, 120, 220, 220]
    ], dtype=torch.float32).to(device)

    # --- Shared Backbone ---
    print_header("1. SHARED BACKBONE")
    backbone = TinyBackbone(out_channels=256).to(device)
    feat_map = backbone(dummy_img)
    print(f"Input:  {dummy_img.shape}")
    print(f"FeatMap: {feat_map.shape} (Stride ~{H_img // feat_map.shape[-1]})")
    print(f"Backbone Params: {count_params(backbone):,}")

    # --- R-CNN (Conceptual: Region Proposals -> Crop -> Classify) ---
    print_header("2. R-CNN (Original) - Conceptual Flow")
    print("Logic: Selective Search (CPU) -> Warp Regions -> CNN (Backbone) -> SVM/FC")
    print("Bottleneck: Runs Backbone N times per image (N=2000). Extremely Slow.")
    rcnn_head = RCNN_Head(256 * 7 * 7, num_classes).to(device)
    # Simulate warped regions (N, C, 7, 7)
    warped_regions = torch.randn(3, 256, 7, 7).to(device)
    cls_scores, bbox_preds = rcnn_head(warped_regions)
    print(f"Warped Regions: {warped_regions.shape} -> Cls: {cls_scores.shape}, BBox: {bbox_preds.shape}")
    print(f"Head Params: {count_params(rcnn_head):,}")

    # --- Fast R-CNN (RoI Pooling) ---
    print_header("3. FAST R-CNN (RoI Pooling)")
    print("Innovation: Run Backbone ONCE. RoI Pool shares computation.")
    fast_rcnn = FastRCNN_Head(256, (7, 7), num_classes).to(device)
    cls_f, bbox_f = fast_rcnn(feat_map, dummy_rois)
    print(f"FeatMap: {feat_map.shape}, ROIs: {dummy_rois.shape}")
    print(f"Output Cls: {cls_f.shape}, BBox: {bbox_f.shape}")
    print(f"Head Params: {count_params(fast_rcnn):,}")

    # --- Faster R-CNN (RPN) ---
    print_header("4. FASTER R-CNN (Region Proposal Network)")
    print("Innovation: Learn proposals via small conv net on feature map (RPN). End-to-end.")
    rpn = RPNHead(256, num_anchors=9).to(device)
    rpn_cls, rpn_reg = rpn(feat_map)
    print(f"FeatMap: {feat_map.shape}")
    print(f"RPN Cls Logits: {rpn_cls.shape} (H, W, 9*2)")
    print(f"RPN Reg Deltas: {rpn_reg.shape} (H, W, 9*4)")
    print(f"RPN Params: {count_params(rpn):,}")
    print("-> Top scoring anchors -> NMS -> ROIs -> Fast R-CNN Head (above)")

    # --- YOLO (Single Stage) ---
    print_header("5. YOLO (You Only Look Once) - Single Stage")
    print("Philosophy: Detection as Regression. Grid cells predict boxes directly. No RoI pooling.")
    yolo = YOLOHead(256, S=7, B=2, C=num_classes).to(device)
    yolo_out = yolo(feat_map)
    print(f"FeatMap: {feat_map.shape} -> YOLO Out: {yolo_out.shape}")
    print(f"Interpretation: (Batch, Grid_H, Grid_W, B*5 + C) = (1, 7, 7, 2*5 + 20) = (1, 7, 7, 30)")
    print(f"  Per Cell: 2 Boxes * (x, y, w, h, conf) + 20 Class Probs")
    print(f"YOLO Head Params: {count_params(yolo):,}")

    # --- Summary Comparison ---
    print_header("6. ARCHITECTURE COMPARISON SUMMARY")
    
    summary = {
        "Paradigm": ["Two-Stage (R-CNN)", "Two-Stage (Fast R-CNN)", "Two-Stage (Faster R-CNN)", "Single-Stage (YOLO)"],
        "Proposal": ["Selective Search (CPU)", "Selective Search (CPU)", "RPN (GPU, Learned)", "None (Dense Grid)"],
        "Feature Sharing": ["No (Crop per region)", "Yes (RoI Pool)", "Yes (RoI Pool/Align)", "Yes (Full Conv)"],
        "Speed (Relative)": ["~0.02 FPS", "~0.5 FPS", "~5-7 FPS", "~30-150+ FPS"],
        "Accuracy Trend": ["High (Region based)", "High", "High (SOTA 2015)", "Lower (Small objects), Improved in v3-v10"],
        "Key Component": ["Warping + SVM", "RoI Pooling", "RPN + RoI Align", "Grid Regression + Anchor Boxes"]
    }
    
    # Print Table
    col_widths = [22, 22, 22, 22, 18, 35]
    headers = list(summary.keys())
    rows = list(zip(*summary.values()))
    
    header_str = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * sum(col_widths))
    for row in rows:
        row_str = "".join(f"{str(c):<{w}}" for c, w in zip(row, col_widths))
        print(row_str)

    print_header("7. LOSS FUNCTION INTUITION")
    print("YOLO Loss (v1 simplified):")
    print("  L = lambda_coord * Sum_{obj} [(x-gx)^2 + (y-gy)^2 + (sqrt(w)-sqrt(gw))^2 + (sqrt(h)-sqrt(gh))^2]")
    print("    + Sum_{obj} (C - C_hat)^2  (Objectness Confidence)")
    print("    + lambda_noobj * Sum_{noobj} (C - 0)^2 (No-object Confidence)")
    print("    + Sum_{obj} Sum_{classes} (p(c) - p_hat(c))^2 (Classification)")
    print("\nFaster R-CNN Loss (Multi-task):")
    print("  L = L_cls (p, p*) + lambda * [u >= 1] L_loc (t, t*)  (Smooth L1)")
    print("  RPN Loss: Same structure (Anchor vs GT IoU matching)")

if __name__ == "__main__":
    run_experiment()