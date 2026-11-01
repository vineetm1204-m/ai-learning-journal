import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# ============================================================
# Day 103: Object Detection Architecture Overview
# Self-contained comparison of R-CNN Family vs YOLO concepts
# ============================================================

# ------------------------------------------------------------
# 1. Shared Backbone (Simulated Feature Extractor)
# ------------------------------------------------------------
class TinyBackbone(nn.Module):
    """Simulates a CNN backbone (e.g., ResNet50 up to C4/C5)."""
    def __init__(self, in_channels=3, feat_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, feat_dim, 2, stride=2)
        
    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        ))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x # Shape: (B, 256, H/32, W/32)

# ------------------------------------------------------------
# 2. R-CNN Family Components (Two-Stage Logic)
# ------------------------------------------------------------

class RoIPool(nn.Module):
    """RoI Pooling (Fast R-CNN): Quantizes coords, MaxPools to fixed size."""
    def __init__(self, output_size=(7, 7), spatial_scale=1.0/16.0):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, proposals):
        """
        features: (B, C, H, W)
        proposals: List of (N_i, 5) tensors [batch_idx, x1, y1, x2, y2] in image coords
        """
        pooled = []
        for prop in proposals:
            # Scale to feature map coords
            prop_scaled = prop.clone()
            prop_scaled[:, 1:] *= self.spatial_scale
            
            batch_indices = prop_scaled[:, 0].long()
            rois = prop_scaled[:, 1:]
            
            # Simple RoI Align approximation via adaptive pooling per roi
            # (Real impl uses bilinear interp; here we loop for clarity)
            for i in range(len(rois)):
                b_idx = batch_indices[i]
                x1, y1, x2, y2 = rois[i].long().clamp(min=0)
                # Ensure valid region
                x2 = max(x2, x1 + 1)
                y2 = max(y2, y1 + 1)
                roi_feat = features[b_idx:b_idx+1, :, y1:y2, x1:x2]
                pooled.append(F.adaptive_max_pool2d(roi_feat, self.output_size))
        
        return torch.cat(pooled, dim=0) if pooled else torch.empty(0)

class RCNNHead(nn.Module):
    """Classification + BBox Regression Head (Shared for Fast/Faster R-CNN)."""
    def __init__(self, in_dim=256*7*7, num_classes=21):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.cls_score(x), self.bbox_pred(x)

class RegionProposalNetwork(nn.Module):
    """RPN (Faster R-CNN): Sliding window on feature map -> Anchors -> Objectness + Delta."""
    def __init__(self, in_channels=256, num_anchors=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, 1) # bg/fg
        self.bbox_reg = nn.Conv2d(512, num_anchors * 4, 1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x) # (B, 2*A, H, W)
        deltas = self.bbox_reg(x)   # (B, 4*A, H, W)
        return logits, deltas

# ------------------------------------------------------------
# 3. YOLO Family Components (Single-Stage Logic)
# ------------------------------------------------------------

class YOLOHead(nn.Module):
    """
    YOLOv1/v2/v3 style head.
    Predicts: (B, S, S, B*(5+C)) -> x, y, w, h, conf, class_probs
    Simplified: Direct regression on grid cells + Anchor priors.
    """
    def __init__(self, in_channels=256, grid_size=13, num_anchors=5, num_classes=20):
        super().__init__()
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Output channels: num_anchors * (4 coords + 1 obj + C classes)
        out_ch = num_anchors * (5 + num_classes)
        self.conv = nn.Conv2d(in_channels, out_ch, 1)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, A*(5+C))
        B, _, H, W = x.shape
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H, W, self.num_anchors, 5 + self.num_classes)
        
        # Split predictions
        # tx, ty (sigmoid) -> center offset within cell
        # tw, th (exp) -> scale relative to anchor
        # to (sigmoid) -> objectness
        # cls (sigmoid/softmax) -> class probs
        pred_xy = torch.sigmoid(x[..., 0:2])
        pred_wh = torch.exp(x[..., 2:4]) # Requires anchor multiplication later
        pred_obj = torch.sigmoid(x[..., 4:5])
        pred_cls = torch.sigmoid(x[..., 5:])
        
        return torch.cat([pred_xy, pred_wh, pred_obj, pred_cls], dim=-1)

# ------------------------------------------------------------
# 4. Experiment Runner: Shape & Latency Comparison
# ------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, input_tensor, iterations=50, warmup=10):
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
    return (end - start) / iterations * 1000 # ms

def run_experiment():
    print("=" * 60)
    print("DAY 103: OBJECT DETECTION ARCHITECTURE MINI-EXPERIMENT")
    print("Topic: R-CNN Family (Two-Stage) vs YOLO (Single-Stage)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Config
    B, C, H, W = 1, 3, 416, 416 # Standard YOLO input size
    dummy_img = torch.randn(B, C, H, W, device=device)
    num_classes = 20 # VOC
    
    # 1. Backbone
    backbone = TinyBackbone(feat_dim=256).to(device)
    print(f"\n[Backbone] Params: {count_params(backbone)/1e6:.2f}M")
    with torch.no_grad():
        feats = backbone(dummy_img)
    print(f"  Feature Map Shape: {feats.shape} (Stride 32)")
    feat_h, feat_w = feats.shape[2], feats.shape[3]
    
    # --------------------------------------------------------
    # PIPELINE A: FASTER R-CNN (Conceptual Forward Pass)
    # --------------------------------------------------------
    print("\n--- [Pipeline A: Faster R-CNN Style] ---")
    rpn = RegionProposalNetwork(in_channels=256, num_anchors=9).to(device)
    roi_pool = RoIPool(output_size=(7,7), spatial_scale=1/32.0)
    rcnn_head = RCNNHead(in_dim=256*7*7, num_classes=num_classes+1).to(device) # +1 bg
    
    print(f"  RPN Params: {count_params(rpn)/1e6:.2f}M")
    print(f"  Head Params: {count_params(rcnn_head)/1e6:.2f}M")
    
    # RPN Forward
    rpn_logits, rpn_deltas = rpn(feats)
    print(f"  RPN Logits Shape: {rpn_logits.shape} (Per anchor: BG/FG)")
    print(f"  RPN Deltas Shape: {rpn_deltas.shape} (Per anchor: dx,dy,dw,dh)")
    
    # Simulate Proposals (Post-NMS top 100)
    # In reality: decode anchors + deltas -> NMS. Here: dummy proposals.
    num_proposals = 100
    dummy_proposals = [torch.cat([
        torch.zeros(num_proposals, 1, device=device), # batch idx
        torch.rand(num_proposals, 4, device=device) * torch.tensor([W, H, W, H], device=device)
    ], dim=1)]
    
    # RoI Pooling
    roi_feats = roi_pool(feats, dummy_proposals)
    print(f"  RoI Pooled Feats Shape: {roi_feats.shape} (N_proposals, C, 7, 7)")
    
    # RCNN Head
    cls_scores, bbox_preds = rcnn_head(roi_feats)
    print(f"  Final Cls Shape: {cls_scores.shape} (N, Classes)")
    print(f"  Final BBox Shape: {bbox_preds.shape} (N, Classes*4)")
    
    # Latency (Backbone + RPN + Head)
    class FasterRCNNWrapper(nn.Module):
        def __init__(self, bb, rpn, pool, head):
            super().__init__()
            self.bb, self.rpn, self.pool, self.head = bb, rpn, pool, head
        def forward(self, x):
            f = self.bb(x)
            logits, deltas = self.rpn(f)
            # Dummy proposals for speed test
            props = [torch.cat([torch.zeros(100,1,device=x.device), 
                                torch.rand(100,4,device=x.device)*416], 1)]
            rf = self.pool(f, props)
            return self.head(rf)
            
    wrapper_rcnn = FasterRCNNWrapper(backbone, rpn, roi_pool, rcnn_head).to(device)
    latency_rcnn = measure_latency(wrapper_rcnn, dummy_img)
    print(f"  Estimated Latency (Full Forward): {latency_rcnn:.2f} ms")
    
    # --------------------------------------------------------
    # PIPELINE B: YOLO (Single Stage Forward Pass)
    # --------------------------------------------------------
    print("\n--- [Pipeline B: YOLO Style (v2/v3)] ---")
    yolo_head = YOLOHead(in_channels=256, grid_size=feat_h, num_anchors=5, num_classes=num_classes).to(device)
    print(f"  YOLO Head Params: {count_params(yolo_head)/1e6:.2f}M")
    
    yolo_out = yolo_head(feats)
    print(f"  Output Shape: {yolo_out.shape}") 
    print(f"  Layout: (Batch, Grid_H, Grid_W, Anchors, 5+Classes)")
    print(f"  Grid Cells: {feat_h}x{feat_w} = {feat_h*feat_w}")
    print(f"  Total Predictions/Img: {feat_h*feat_w*5} (vs ~100 RoIs in R-CNN)")
    
    class YOLOWrapper(nn.Module):
        def __init__(self, bb, head):
            super().__init__()
            self.bb, self.head = bb, head
        def forward(self, x):
            return self.head(self.bb(x))
            
    wrapper_yolo = YOLOWrapper(backbone, yolo_head).to(device)
    latency_yolo = measure_latency(wrapper_yolo, dummy_img)
    print(f"  Estimated Latency (Full Forward): {latency_yolo:.2f} ms")
    
    # --------------------------------------------------------
    # SUMMARY & ANALYSIS
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Faster R-CNN (2-Stage)':<25} | {'YOLO (1-Stage)':<20}")
    print("-" * 80)
    print(f"{'Paradigm':<30} | {'Region Proposal + Classification':<25} | {'Dense Grid Regression':<20}")
    print(f"{'Proposals / Image':<30} | {'~100-300 (Sparse)':<25} | {'Grid_H x Grid_W x Anchors (Dense)':<20}")
    print(f"{'Context Usage':<30} | {'RoI Pooling (Adaptive)':<25} | {'Fixed Receptive Field':<20}")
    print(f"{'Small Objects':<30} | {'Better (RoI Align + FPN)':<25} | {'Struggles (Stride 32)':<20}")
    print(f"{'Speed (Latency)':<30} | {f'{latency_rcnn:.1f} ms (Heavier Head)':<25} | {f'{latency_yolo:.1f} ms (Lighter Head)':<20}")
    print(f"{'Training Complexity':<30} | {'Multi-stage loss (RPN+RCNN)':<25} | {'Single Joint Loss':<20}")
    print(f"{'Deployment':<30} | {'Complex (NMS, Two Heads)' :<25} | {'Simple (Single Tensor Out)' :<20}")
    print("-" * 80)
    print("\nKey Takeaway: R-CNN favors accuracy (esp. small obj) via RoI ops;")
    print("              YOLO favors speed/simplicity via dense prediction.")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()