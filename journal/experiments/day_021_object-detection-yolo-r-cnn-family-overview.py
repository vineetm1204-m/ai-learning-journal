import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time

# ============================================================
# 1. CORE GEOMETRY UTILITIES (Shared by all detectors)
# ============================================================

def box_iou(boxes1, boxes2, format='xyxy'):
    """
    Calculate IoU between two sets of boxes.
    boxes: (N, 4) or (M, 4)
    format: 'xyxy' (x1, y1, x2, y2) or 'cxcywh' (center_x, center_y, w, h)
    Returns: (N, M) IoU matrix
    """
    if format == 'cxcywh':
        boxes1 = cxcywh_to_xyxy(boxes1)
        boxes2 = cxcywh_to_xyxy(boxes2)

    # boxes1: (N, 4) -> (N, 1, 4)
    # boxes2: (M, 4) -> (1, M, 4)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou

def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)

def nms(boxes, scores, iou_threshold=0.5):
    """Pure PyTorch NMS implementation."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    _, order = scores.sort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        # Compute IoU of the picked box with the rest
        ious = box_iou(boxes[i:i+1], boxes[order[1:]]).squeeze(0)
        # Keep only boxes with IoU < threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

# ============================================================
# 2. R-CNN FAMILY SIMULATION (Two-Stage Pipeline)
# ============================================================

class MockBackbone(nn.Module):
    """Simulates a CNN Backbone (e.g., ResNet50) outputting feature maps."""
    def __init__(self, in_channels=3, feat_dim=256, stride=16):
        super().__init__()
        self.stride = stride
        # Simple conv stack to reduce spatial dims and increase channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, feat_dim, 3, stride=2, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class ROIAlign(nn.Module):
    """Simplified ROI Align: RoI -> Fixed Size Feature Vector (7x7)."""
    def __init__(self, output_size=7, spatial_scale=1/16):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, proposals):
        """
        features: (B, C, H, W)
        proposals: (N, 5) -> (batch_idx, x1, y1, x2, y2) in IMAGE coordinates
        """
        # In a real impl, this uses bilinear interpolation.
        # Here we simulate by adaptive pooling on the feature map region.
        # This is a MOCK for demonstration of the *concept*, not a numerically accurate ROIAlign.
        B, C, H, W = features.shape
        device = features.device
        pooled_feats = []

        for prop in proposals:
            b_idx, x1, y1, x2, y2 = prop.int()
            # Map image coords to feature map coords
            fx1 = int(x1 * self.spatial_scale)
            fy1 = int(y1 * self.spatial_scale)
            fx2 = int(x2 * self.spatial_scale)
            fy2 = int(y2 * self.spatial_scale)
            
            # Clamp
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(W, fx2), min(H, fy2)
            
            if fx2 <= fx1 or fy2 <= fy1:
                pooled_feats.append(torch.zeros(C, self.output_size, self.output_size, device=device))
                continue

            roi_feat = features[b_idx, :, fy1:fy2, fx1:fx2] # (C, h, w)
            # Adaptive pool to fixed size
            pooled = F.adaptive_avg_pool2d(roi_feat.unsqueeze(0), self.output_size).squeeze(0)
            pooled_feats.append(pooled)

        return torch.stack(pooled_feats) if pooled_feats else torch.empty(0, C, self.output_size, self.output_size, device=device)

class RCNNHead(nn.Module):
    """Classification + BBox Regression Head (Fast R-CNN style)."""
    def __init__(self, in_channels=256, roi_size=7, num_classes=20):
        super().__init__()
        self.flatten_dim = in_channels * roi_size * roi_size
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU()
        )
        self.cls_head = nn.Linear(1024, num_classes + 1) # +1 for background
        self.reg_head = nn.Linear(1024, (num_classes + 1) * 4) # class-agnostic or specific

    def forward(self, roi_feats):
        x = roi_feats.flatten(1)
        x = self.fc(x)
        cls_logits = self.cls_head(x)
        bbox_deltas = self.reg_head(x)
        return cls_logits, bbox_deltas

def run_rcnn_simulation():
    print("\n" + "="*60)
    print("SIMULATION: R-CNN Family (Two-Stage: Region Proposal -> ROI Pool -> Head)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Setup
    backbone = MockBackbone().to(device)
    roi_align = ROIAlign(spatial_scale=1/16).to(device)
    head = RCNNHead().to(device)
    
    # Dummy Image Batch (B=1, 3, 224, 224)
    img = torch.randn(1, 3, 224, 224).to(device)
    
    # 2. Stage 1: Backbone Feature Extraction
    start = time.time()
    feat_map = backbone(img) # (1, 256, 7, 7) roughly
    t_backbone = time.time() - start
    print(f"[Backbone] Output shape: {feat_map.shape} | Time: {t_backbone*1000:.1f}ms")

    # 3. Stage 1.5: Region Proposal Network (RPN) / Selective Search MOCK
    # Generate ~100 random proposals (batch_idx, x1, y1, x2, y2)
    num_proposals = 100
    proposals = []
    for _ in range(num_proposals):
        x1 = random.randint(0, 100)
        y1 = random.randint(0, 100)
        x2 = random.randint(x1+10, 224)
        y2 = random.randint(y1+10, 224)
        proposals.append([0, x1, y1, x2, y2])
    proposals = torch.tensor(proposals, dtype=torch.float32, device=device)
    print(f"[RPN/SelectiveSearch] Generated {len(proposals)} proposals.")

    # 4. Stage 2: ROI Align (RoI Pooling)
    start = time.time()
    roi_feats = roi_align(feat_map, proposals) # (100, 256, 7, 7)
    t_roi = time.time() - start
    print(f"[ROI Align] Output shape: {roi_feats.shape} | Time: {t_roi*1000:.1f}ms")

    # 5. Stage 3: Detection Head (Classification + Regression)
    start = time.time()
    cls_logits, bbox_deltas = head(roi_feats)
    t_head = time.time() - start
    print(f"[RCNN Head] Cls: {cls_logits.shape}, Reg: {bbox_deltas.shape} | Time: {t_head*1000:.1f}ms")

    # 6. Post-processing (Mock NMS per class)
    scores = F.softmax(cls_logits, dim=-1)[:, 1:] # Ignore background (idx 0)
    max_scores, labels = scores.max(dim=1)
    
    # Decode boxes (simplified: assume deltas are direct offsets for demo)
    # Real impl: apply deltas to proposals
    decoded_boxes = proposals[:, 1:] + bbox_deltas[torch.arange(len(proposals)), labels*4:(labels*4+4)].sigmoid() * 50 # Mock decode
    
    keep = nms(decoded_boxes, max_scores, iou_threshold=0.5)
    print(f"[NMS] Kept {len(keep)} final detections.")
    print(f"Total Pipeline Time: {(t_backbone+t_roi+t_head)*1000:.1f}ms")
    print("Key Characteristic: High Accuracy, Slow (Sequential stages), Complex Training Pipeline.")

# ============================================================
# 3. YOLO FAMILY SIMULATION (One-Stage Pipeline)
# ============================================================

class YOLOHead(nn.Module):
    """YOLOv1/v5/v8 style Head: Predicts (x, y, w, h, obj_conf, class_probs) per grid cell/anchor."""
    def __init__(self, in_channels=256, num_anchors=3, num_classes=20, grid_size=7):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.grid_size = grid_size
        # Output: (B, num_anchors * (5 + num_classes), Grid, Grid)
        # 5 = tx, ty, tw, th, to (objectness)
        out_channels = num_anchors * (5 + num_classes)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, A*(5+C), H, W)
        return self.conv(x)

def decode_yolo_predictions(pred, anchors, stride, grid_size):
    """
    Decodes raw YOLO output to absolute xyxy boxes.
    pred: (B, A*(5+C), G, G)
    anchors: (A, 2) -> (w, h) in grid units
    """
    B, _, G, _ = pred.shape
    A = len(anchors)
    C = (pred.shape[1] // A) - 5
    
    pred = pred.view(B, A, 5 + C, G, G).permute(0, 1, 3, 4, 2).contiguous() # (B, A, G, G, 5+C)
    
    # Grid offsets
    grid_y, grid_x = torch.meshgrid(torch.arange(G), torch.arange(G), indexing='ij')
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().to(pred.device).view(1, 1, G, G, 2)
    
    # Anchor wh
    anchor_wh = anchors.float().to(pred.device).view(1, A, 1, 1, 2)
    
    # Decode
    # tx, ty -> sigmoid -> offset from grid top-left
    xy = pred[..., 0:2].sigmoid() + grid_xy
    # tw, th -> exp * anchor
    wh = pred[..., 2:4].exp() * anchor_wh
    # to -> sigmoid (objectness)
    obj = pred[..., 4:5].sigmoid()
    # cls -> sigmoid (multi-label) or softmax
    cls = pred[..., 5:].sigmoid()
    
    # Scale to image coordinates
    xy *= stride
    wh *= stride
    
    # Convert to xyxy
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    boxes = torch.cat([x1y1, x2y2], dim=-1) # (B, A, G, G, 4)
    
    return boxes.view(B, -1, 4), obj.view(B, -1, 1), cls.view(B, -1, C)

def run_yolo_simulation():
    print("\n" + "="*60)
    print("SIMULATION: YOLO Family (One-Stage: Dense Prediction -> Decode -> NMS)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Setup
    backbone = MockBackbone(feat_dim=256, stride=32).to(device) # YOLO usually stride 32
    # Anchors (width, height) relative to grid cell size (typical COCO priors scaled)
    anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32) # 3 anchors
    head = YOLOHead(in_channels=256, num_anchors=3, num_classes=20, grid_size=7).to(device)
    
    img = torch.randn(1, 3, 224, 224).to(device)
    stride = 32 # 224 / 7 = 32

    # 2. Single Forward Pass
    start = time.time()
    feat_map = backbone(img) # (1, 256, 7, 7)
    raw_pred = head(feat_map) # (1, 3*(5+20), 7, 7)
    t_forward = time.time() - start
    print(f"[Backbone+Head] Feat: {feat_map.shape}, Raw Pred: {raw_pred.shape} | Time: {t_forward*1000:.1f}ms")

    # 3. Decode (Post-processing logic, part of inference)
    start = time.time()
    boxes, obj_scores, cls_scores = decode_yolo_predictions(raw_pred, anchors, stride, grid_size=7)
    # boxes: (1, 3*7*7, 4) -> (1, 147, 4)
    # Filter by objectness
    mask = obj_scores.squeeze(-1) > 0.1 # Confidence threshold
    boxes = boxes[mask]
    obj_scores = obj_scores[mask]
    cls_scores = cls_scores[mask]
    t_decode = time.time() - start
    print(f"[Decode] Candidate boxes: {boxes.shape[0]} | Time: {t_decode*1000:.1f}ms")

    # 4. NMS (Class-agnostic or per-class)
    # For speed, class-agnostic NMS on max class score
    if boxes.shape[0] > 0:
        max_cls_scores, labels = cls_scores.max(dim=1)
        final_scores = obj_scores.squeeze(-1) * max_cls_scores
        keep = nms(boxes, final_scores, iou_threshold=0.45)
        print(f"[NMS] Kept {len(keep)} final detections.")
    else:
        print("[NMS] No candidates above threshold.")

    print(f"Total Pipeline Time: {(t_forward+t_decode)*1000:.1f}ms")
    print("Key Characteristic: Fast (Single Pass), End-to-End Trainable, Struggles with Small/Overlapping Objects.")

# ============================================================
# 4. COMPARATIVE ANALYSIS & VISUALIZATION (Text-based)
# ============================================================

def print_architecture_comparison():
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON: R-CNN FAMILY vs YOLO FAMILY")
    print("="*60)
    
    headers = ["Aspect", "R-CNN / Fast / Faster R-CNN", "YOLO (v1 -> v8/NAS)", "SSD / RetinaNet"]
    rows = [
        ["Paradigm", "Two-Stage (Region Proposal -> Classification)", "One-Stage (Dense Prediction)", "One-Stage (Multi-scale Dense)"],
        ["Proposals", "External (Selective Search) / Learned (RPN)", "None (Grid + Anchors / Anchor-Free)", "None (Default Boxes / Anchors)"],
        ["Feature Extraction", "ROI Pool/Align (Region-wise)", "Whole Image (Backbone Neck)", "Multi-scale Feature Maps (FPN)"],
        ["Speed/Accuracy", "High Acc / Slow (Faster R-CNN ~5-15 FPS)", "Fast / Good Acc (YOLOv8 ~30-100+ FPS)", "Balanced (SSD ~20-40 FPS)"],
        ["Small Objects", "Good (ROI Align preserves spatial info)", "Historically Weak (Stride 32), Better in v5/v8 (PAN/FPN)", "Good (Explicit Multi-scale heads)"],
        ["Training", "Complex (Multi-stage, Multi-loss)", "End-to-End (Single Loss: Cls+Reg+Obj)", "End-to-End (Focal Loss for imbalance)"],
        ["Key Innovation", "RPN (Faster), ROI Align (Mask R-CNN)", "Unified Detection, Anchor-Free (v8/v10)", "Focal Loss (RetinaNet), Default Boxes (SSD)"],
        ["Deployment", "Harder (Two graphs / Custom Ops)", "Easier (Single Graph, ONNX/TensorRT friendly)", "Easy"],
    ]
    
    col_widths = [20, 35, 35, 35]
    header_fmt = "".join([f"{{:<{w}}}" for w in col_widths])
    row_fmt = "".join([f"{{:<{w}}}" for w in col_widths])
    
    print(header_fmt.format(*headers))
    print("-" * sum(col_widths))
    for row in rows:
        print(row_fmt.format(*row))

def demonstrate_iou_nms():
    print("\n" + "="*60)
    print("CORE MATH DEMO: IoU Calculation & NMS Logic")
    print("="*60)
    
    # Ground Truth
    gt = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32) # (1, 4)
    # Predictions: Good, Partial, Bad, Duplicate
    preds = torch.tensor([
        [12, 12, 48, 48],  # High IoU
        [10, 10, 30, 30],  # Medium IoU (contained)
        [60, 60, 100, 100],# Zero IoU
        [11, 11, 49, 49],  # Duplicate of first
    ], dtype=torch.float32)
    
    scores = torch.tensor([0.9, 0.75, 0.8, 0.85])
    
    ious = box_iou(preds, gt).squeeze(1)
    print(f"GT Box: {gt.tolist()}")
    for i, (box, iou, sc) in enumerate(zip(preds, ious, scores)):
        print(f"  Pred {i}: {box.tolist()} | IoU: {iou:.3f} | Score: {sc:.2f}")
    
    print("\nApplying NMS (iou_thresh=0.5)...")
    keep = nms(preds, scores, 0.5)
    print(f"Indices Kept: {keep.tolist()}")
    print(f"Boxes Kept: {preds[keep].tolist()}")
    print("Logic: Sort by score -> Pick highest -> Suppress overlaps > thresh -> Repeat.")

# ============================================================
# 5. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("DAY 21: OBJECT DETECTION MINI-EXPERIMENT")
    print("Topic: YOLO vs R-CNN Family Overview")
    
    # Run Simulations
    run_rcnn_simulation()
    run_yolo_simulation()
    
    # Core Math
    demonstrate_iou_nms()
    
    # Comparison Table
    print_architecture_comparison()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE.")
    print("Next Steps: Implement Focal Loss, try Anchor-Free head (FCOS),")
    print("or train on VOC/COCO with a real backbone (ResNet/EfficientNet).")
    print("="*60)