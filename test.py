"""
Full end-to-end PyTorch implementation for DeepFashion-MultiModal
- Dataloader that reads images, parsing masks, keypoints, and label files
- ResNet50 backbone + keypoint MLP + multi-task classification heads
- Training loop, evaluation, and inference
- At inference: simple 3-region color detection using computer vision
  (upper_body, lower_body, shoes) based on vertical bands.

Usage examples:
  python deepfashion_multimodal_training.py --mode train --data_root /path/to/DeepFashion-MultiModal
  python deepfashion_multimodal_training.py --mode infer --data_root /path/to/DeepFashion-MultiModal --image example.jpg

Notes:
- You must adapt file names/paths if the dataset layout differs.
- This is a single-file baseline. For production: split into modules.

Requirements:
  python 3.8+
  pip install torch torchvision tqdm pillow numpy

"""

import os
import json
import argparse
from typing import Optional, Tuple, Dict, List

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score



# -----------------------------
# Helper label maps (from dataset README)
# -----------------------------
SHAPE_LABELS = {
    'sleeve_length': [
        'sleeveless','short-sleeve','medium-sleeve','long-sleeve','not-long-sleeve','NA'
    ], # 6
    'lower_length': ['three-point','medium-short','three-quarter','long','NA'], # 5
    'socks': ['no','socks','leggings','NA'], #4
    'hat': ['no','yes','NA'], #3
    'glasses': ['no','eyeglasses','sunglasses','have-glasses','NA'], #5
    'neckwear': ['no','yes','NA'], #3
    'wristwear': ['no','yes','NA'], #3
    'ring': ['no','yes','NA'], #3
    'waist_accessory': ['no','belt','have-clothing','hidden','NA'], #5
    'neckline': ['V-shape','square','round','standing','lapel','suspenders','NA'], #7
    'outer_cardigan': ['yes','no','NA'], #3
    'upper_cover_navel': ['no','yes','NA'] #3
}

FABRIC_LABELS = ['denim','cotton','leather','furry','knitted','chiffon','other','NA']
COLOR_LABELS = ['floral','graphic','striped','pure-color','lattice','other','color-block','NA']

# For convenience, build sizes dict
SHAPE_SIZES = {k: len(v) for k,v in SHAPE_LABELS.items()}

# -----------------------------
# Dataset
# -----------------------------
class DeepFashionMM(Dataset):
    """
    Expects directory structure pointed by data_root with subfolders:
      images/ or image/ (JPG/PNG)
      parsing/ (PNG) optional (NOT used for color in this version)
      keypoints_loc.txt
      keypoints_vis.txt
      shape_label.txt
      fabric_label.txt
      color_label.txt
    File formats are as described in dataset README.
    """
    def __init__(self, data_root: str, split: str = 'all', transforms_img=None, use_parsing: bool = True, use_keypoints: bool = True):
        super().__init__()
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, 'image') if os.path.isdir(os.path.join(data_root, 'image')) else os.path.join(data_root, 'images')
        self.parsing_dir = os.path.join(data_root, 'parsing')
        self.transforms_img = transforms_img
        # parsing is still read into dataset, but NOT used for color in this version
        self.use_parsing = use_parsing and os.path.isdir(self.parsing_dir)
        self.use_keypoints = use_keypoints

        # load image list
        self.images = sorted([f for f in os.listdir(self.img_dir)
                              if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')])

        # load keypoints
        self.kp_loc = {}
        self.kp_vis = {}
        kp_loc_path = os.path.join(data_root, 'keypoints_loc.txt')
        kp_vis_path = os.path.join(data_root, 'keypoints_vis.txt')
        if os.path.isfile(kp_loc_path):
            with open(kp_loc_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 1+42:
                        continue
                    name = parts[0]
                    coords = list(map(float, parts[1:1+42]))
                    self.kp_loc[name] = coords
        if os.path.isfile(kp_vis_path):
            with open(kp_vis_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split()
                    name = parts[0]
                    vs = list(map(int, parts[1:1+21]))
                    self.kp_vis[name] = vs

        # load labels (shape, fabric, color)
        self.shape_labels = {}
        shape_path = os.path.join(data_root, 'labels', 'shape_label.txt') if os.path.isdir(os.path.join(data_root, 'labels')) else os.path.join(data_root, 'shape_label.txt')
        if os.path.isfile(shape_path):
            with open(shape_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    name = parts[0]
                    vals = list(map(int, parts[1:]))
                    # dataset says 12 shape attributes
                    self.shape_labels[name] = vals

        self.fabric_labels = {}
        fabric_path = os.path.join(data_root, 'labels', 'fabric_label.txt') if os.path.isdir(os.path.join(data_root, 'labels')) else os.path.join(data_root, 'fabric_label.txt')
        if os.path.isfile(fabric_path):
            with open(fabric_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split()
                    name = parts[0]
                    vals = list(map(int, parts[1:1+3]))
                    self.fabric_labels[name] = vals

        self.color_labels = {}
        color_path = os.path.join(data_root, 'labels', 'color_label.txt') if os.path.isdir(os.path.join(data_root, 'labels')) else os.path.join(data_root, 'color_label.txt')
        if os.path.isfile(color_path):
            with open(color_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split()
                    name = parts[0]
                    vals = list(map(int, parts[1:1+3]))
                    self.color_labels[name] = vals

        # final list: only images that have at least one label entry
        filtered = []
        for img in self.images:
            base = os.path.splitext(img)[0]
            # Check BOTH full name (img) and base name (base)
            has_shape = (img in self.shape_labels) or (base in self.shape_labels)
            has_fabric = (img in self.fabric_labels) or (base in self.fabric_labels)
            has_color = (img in self.color_labels) or (base in self.color_labels)
            if has_shape or has_fabric or has_color:
                filtered.append(img)
        self.images = filtered
        print(f"Dataset initialized. Found {len(self.images)} valid images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transforms_img:
            img_t = self.transforms_img(img)
        else:
            img_t = transforms.ToTensor()(img)

        # parsing mask (not used for color in this version, but kept for compatibility)
        _, h, w = img_t.shape
        parsing_mask = torch.zeros((h, w), dtype=torch.long)
        if self.use_parsing:
            ppath = os.path.join(self.parsing_dir, base + '.png')
            if os.path.isfile(ppath):
                pm = Image.open(ppath)
                pm = pm.resize((w, h), Image.NEAREST)
                parsing_mask = torch.from_numpy(np.array(pm)).long()

        # keypoints
        kp = None
        if self.kp_loc and (base in self.kp_loc):
            coords = np.array(self.kp_loc[base]).astype(np.float32)
            # normalize by image size
            w0, h0 = img.size
            coords[0::2] /= float(w0)
            coords[1::2] /= float(h0)
            kp = torch.from_numpy(coords).float()
        else:
            kp = torch.full((42,), -1.0, dtype=torch.float32)

        # Shape
        s_val = self.shape_labels.get(img_name, self.shape_labels.get(base, None))
        if s_val is not None:
            shape = torch.tensor(s_val, dtype=torch.long)
        else:
            shape = torch.full((12,), -1, dtype=torch.long)

        # Fabric
        f_val = self.fabric_labels.get(img_name, self.fabric_labels.get(base, None))
        if f_val is not None:
            fabric = torch.tensor(f_val, dtype=torch.long)
        else:
            fabric = torch.full((3,), -1, dtype=torch.long)

        # Color
        c_val = self.color_labels.get(img_name, self.color_labels.get(base, None))
        if c_val is not None:
            color = torch.tensor(c_val, dtype=torch.long)
        else:
            color = torch.full((3,), -1, dtype=torch.long)

        sample = {
            'image': img_t,
            'parsing': parsing_mask,
            'keypoints': kp,
            'shape': shape,
            'fabric': fabric,
            'color': color,
            'name': img_name
        }
        return sample

# -----------------------------
# Model
# -----------------------------
class MultiTaskFashionModel(nn.Module):
    def __init__(self, backbone_name: str = 'resnet50', use_keypoints: bool = True, use_segmentation: bool = False, use_parsing_attention: bool = False):
        super().__init__()
        assert backbone_name in ['resnet50','resnet18']
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feat_dim = 2048
        else:
            self.backbone = models.resnet18(pretrained=True)
            feat_dim = 512

        # Store original forward for backward compatibility
        self._original_forward = self.backbone.forward
        
        # For region-aware features, we need spatial features before global pooling
        self.use_parsing_attention = use_parsing_attention
        if use_parsing_attention:
            # Remove avgpool and fc to get spatial features
            if backbone_name == 'resnet50':
                # ResNet50 structure: conv -> bn -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
                # We'll use layer4 output which is [B, 2048, H/32, W/32]
                self.backbone.avgpool = nn.Identity()
                self.backbone.fc = nn.Identity()
                # We'll add our own pooling later
                self.spatial_feat_dim = 2048
            else:
                self.backbone.avgpool = nn.Identity()
                self.backbone.fc = nn.Identity()
                self.spatial_feat_dim = 512
            
            # Region-aware feature projection (if spatial dim != feat_dim)
            if self.spatial_feat_dim != feat_dim:
                self.region_proj = nn.Linear(self.spatial_feat_dim, feat_dim)
            else:
                self.region_proj = nn.Identity()
        else:
            # Original behavior: global average pooling
            self.backbone.fc = nn.Identity()
            self.backbone.avgpool = nn.AdaptiveAvgPool2d(1) if not hasattr(self.backbone, 'avgpool') or self.backbone.avgpool is None else self.backbone.avgpool

        self.use_keypoints = use_keypoints
        kp_dim = 64
        if use_keypoints:
            self.kp_mlp = nn.Sequential(
                nn.Linear(42, 128),
                nn.ReLU(),
                nn.Linear(128, kp_dim),
                nn.ReLU()
            )
        else:
            kp_dim = 0

        fusion_dim = feat_dim + kp_dim
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # create heads for each shape attribute
        self.shape_heads = nn.ModuleDict()
        for k,v in SHAPE_SIZES.items():
            self.shape_heads[k] = nn.Linear(fusion_dim, v)

        # fabric (3 heads)
        self.fabric_heads = nn.ModuleList([nn.Linear(fusion_dim, len(FABRIC_LABELS)) for _ in range(3)])
        self.color_heads  = nn.ModuleList([nn.Linear(fusion_dim, len(COLOR_LABELS)) for _ in range(3)])

        # seg_decoder is not used in this code path (kept for compatibility)
        self.use_segmentation = use_segmentation
        if use_segmentation:
            self.seg_decoder = nn.Sequential(
                nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                nn.Conv2d(256, 24, kernel_size=1) # 24 parsing classes (0..23)
            )
        else:
            self.seg_decoder = None

    def forward(self, x_image, x_kp=None, parsing_mask=None):
        # backbone expects 3xHxW
        if self.use_parsing_attention and parsing_mask is not None:
            # Extract spatial features manually
            x = self.backbone.conv1(x_image)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            spatial_feat = self.backbone.layer4(x)  # [B, C, H', W']
            
            # Use parsing mask to weight features
            # Resize parsing mask to match spatial feature size
            B, C, H, W = spatial_feat.shape
            if len(parsing_mask.shape) == 2:  # [H, W] -> [1, 1, H, W]
                parsing_mask = parsing_mask.unsqueeze(0).unsqueeze(0)
            elif len(parsing_mask.shape) == 3:  # [B, H, W] -> [B, 1, H, W]
                parsing_mask = parsing_mask.unsqueeze(1)
            
            parsing_resized = F.interpolate(
                parsing_mask.float(), 
                size=(H, W), 
                mode='nearest'
            )  # [B, 1, H', W']
            
            # Create region attention weights
            # Focus on relevant regions: top (1), outer (2), pants (5), shoes (11)
            relevant_regions = torch.zeros_like(parsing_resized)
            for region_id in [1, 2, 5, 11]:  # top, outer, pants, shoes
                relevant_regions += (parsing_resized == region_id).float()
            
            # Weight features by relevant regions (boost clothing regions by 50%)
            weighted_feat = spatial_feat * (1.0 + 0.5 * relevant_regions)  # Boost relevant regions
            feat = F.adaptive_avg_pool2d(weighted_feat, 1).flatten(1)  # [B, spatial_feat_dim]
            
            # Project to original feature dimension
            feat = self.region_proj(feat)  # [B, feat_dim]
        else:
            # Original behavior: global average pooling
            feat = self.backbone(x_image)  # [B, feat_dim]

        if self.use_keypoints and x_kp is not None:
            kp_feat = self.kp_mlp(x_kp)
            fused = torch.cat([feat, kp_feat], dim=1)
        else:
            fused = feat

        fused = self.fusion_norm(fused)

        outputs = {}
        for k,head in self.shape_heads.items():
            outputs[f'shape_{k}'] = head(fused)

        for i,head in enumerate(self.fabric_heads):
            outputs[f'fabric_{i}'] = head(fused)

        for i,head in enumerate(self.color_heads):
            outputs[f'color_{i}'] = head(fused)

        return outputs

# -----------------------------
# Utilities: loss, metrics, human readable
# -----------------------------

def multitask_loss(outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], device: torch.device, weights: Optional[Dict[str,float]] = None):
    # outputs: keys like 'shape_sleeve_length' -> [B, C]
    # labels: dict contains tensors
    total = 0.0
    loss = 0.0
    weights = weights or {}

    # shape labels
    for i, (k,size) in enumerate(SHAPE_SIZES.items()):
        out = outputs[f'shape_{k}']
        lab = labels['shape'][:, i].to(device)  # -1 is missing
        mask = lab >= 0
        if mask.sum() == 0:
            continue
        l = F.cross_entropy(out[mask], lab[mask])
        w = weights.get(f'shape_{k}', 1.0)
        loss = loss + w * l
        total += 1

    # fabric
    for i in range(3):
        out = outputs[f'fabric_{i}']
        lab = labels['fabric'][:, i].to(device)
        mask = lab >= 0
        if mask.sum() == 0: continue
        l = F.cross_entropy(out[mask], lab[mask])
        loss = loss + l
        total += 1

    # color
    for i in range(3):
        out = outputs[f'color_{i}']
        lab = labels['color'][:, i].to(device)
        mask = lab >= 0
        if mask.sum() == 0: continue
        l = F.cross_entropy(out[mask], lab[mask])
        loss = loss + l
        total += 1

    if total == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)
    return loss / float(total)


def decode_predictions(outputs: Dict[str, torch.Tensor]) -> Dict[str,str]:
    # outputs contain logits per head for a single sample (batch dim stripped)
    out = {}
    # shape
    for i,(k,labels) in enumerate(SHAPE_LABELS.items()):
        logits = outputs[f'shape_{k}']
        idx = int(logits.argmax().item())
        out[k] = labels[idx] if idx < len(labels) else str(idx)

    # fabric
    for i in range(3):
        logits = outputs[f'fabric_{i}']
        idx = int(logits.argmax().item())
        out[f'fabric_{i}'] = FABRIC_LABELS[idx]

    # color patterns (not pixel color, but category)
    for i in range(3):
        logits = outputs[f'color_{i}']
        idx = int(logits.argmax().item())
        out[f'color_{i}'] = COLOR_LABELS[idx]

    return out


# -----------------------------
# Training and evaluation
# -----------------------------

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc='train'):
        imgs = batch['image'].to(device)
        kps = batch['keypoints'].to(device)
        labels = {'shape': batch['shape'].to(device),
                  'fabric': batch['fabric'].to(device),
                  'color': batch['color'].to(device)}

        outputs = model(imgs, kps)
        loss = multitask_loss(outputs, labels, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Returns:
      avg_loss: float
      per_attr_acc: dict[attr_name -> accuracy]
      macro_acc: float
      per_attr_f1: dict[attr_name -> f1]
      macro_f1: float
      confusion: dict[attr_name -> 2D np.array]
    """
    model.eval()
    total_loss = 0.0

    # Only do detailed metrics for SHAPE attributes (for presentation)
    shape_attrs = list(SHAPE_LABELS.keys())

    # For accuracy
    correct = {k: 0 for k in shape_attrs}
    total = {k: 0 for k in shape_attrs}

    # For F1 + confusion
    all_targets = {k: [] for k in shape_attrs}
    all_preds = {k: [] for k in shape_attrs}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='eval'):
            imgs = batch['image'].to(device)
            kps = batch['keypoints'].to(device)
            labels = {
                'shape': batch['shape'].to(device),
                'fabric': batch['fabric'].to(device),
                'color': batch['color'].to(device)
            }
            outputs = model(imgs, kps)
            loss = multitask_loss(outputs, labels, device)
            total_loss += loss.item()

            # ----- Metrics for shape attributes -----
            for i, attr in enumerate(shape_attrs):
                logits = outputs[f'shape_{attr}']       # [B, C]
                preds = logits.argmax(dim=1)            # [B]
                labs = labels['shape'][:, i]            # [B]
                mask = labs >= 0                        # ignore -1

                if mask.sum() == 0:
                    continue

                p = preds[mask].cpu().numpy()
                t = labs[mask].cpu().numpy()

                correct[attr] += (p == t).sum()
                total[attr] += len(t)
                all_preds[attr].extend(p.tolist())
                all_targets[attr].extend(t.tolist())

    avg_loss = total_loss / len(dataloader)

    # Per-attribute accuracy
    per_attr_acc = {}
    for attr in shape_attrs:
        if total[attr] > 0:
            per_attr_acc[attr] = correct[attr] / total[attr]
        else:
            per_attr_acc[attr] = None  # no labels available

    # Macro accuracy (ignore None)
    valid_accs = [v for v in per_attr_acc.values() if v is not None]
    macro_acc = float(np.mean(valid_accs)) if valid_accs else None

    # Per-attribute F1 + macro F1
    per_attr_f1 = {}
    for attr in shape_attrs:
        if len(all_targets[attr]) > 0:
            per_attr_f1[attr] = f1_score(
                all_targets[attr],
                all_preds[attr],
                average='macro'
            )
        else:
            per_attr_f1[attr] = None

    valid_f1s = [v for v in per_attr_f1.values() if v is not None]
    macro_f1 = float(np.mean(valid_f1s)) if valid_f1s else None

    # Confusion matrices for each shape attribute
    confusion = {}
    for attr in shape_attrs:
        if len(all_targets[attr]) > 0:
            confusion[attr] = confusion_matrix(
                all_targets[attr],
                all_preds[attr]
            )
        else:
            confusion[attr] = None

    return avg_loss, per_attr_acc, macro_acc, per_attr_f1, macro_f1, confusion

# -----------------------------
# Inference util
# -----------------------------

def infer_single(model, image_path: str, device: torch.device, transforms_img, parsing_mask=None):
    """
    Run attribute prediction + region-based color detection
    on a single image.
    
    Args:
        model: The classification model
        image_path: Path to input image
        device: torch device
        transforms_img: Image transforms
        parsing_mask: Optional parsing mask tensor [H, W] or [1, H, W] for region-aware attention
    """
    model.eval()

    # Load original image for region color detection
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32)  # H x W x 3

    # Preprocess image for model
    x = transforms_img(img).unsqueeze(0).to(device)
    kp = torch.full((1,42), -1.0, dtype=torch.float32).to(device)
    
    # Prepare parsing mask if provided
    parsing_tensor = None
    if parsing_mask is not None:
        if isinstance(parsing_mask, np.ndarray):
            parsing_tensor = torch.from_numpy(parsing_mask).long().to(device)
        else:
            parsing_tensor = parsing_mask.to(device)
        # Ensure it matches the model's expected input size
        _, _, h_model, w_model = x.shape
        if len(parsing_tensor.shape) == 2:  # [H, W]
            parsing_tensor = parsing_tensor.unsqueeze(0)  # [1, H, W]
        # Resize to match model input
        parsing_tensor = F.interpolate(
            parsing_tensor.float().unsqueeze(1), 
            size=(h_model, w_model), 
            mode='nearest'
        ).squeeze(1).long()  # [1, H, W]

    with torch.no_grad():
        outs = model(x, kp, parsing_mask=parsing_tensor)

    outs_squeezed = {k: v[0].cpu() for k,v in outs.items()}
    decoded = decode_predictions(outs_squeezed)

    # region colors from raw image
    return decoded

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default="./dataset",
                        help='Path to dataset')
    parser.add_argument('--mode', type=str, choices=['train','infer'], default='infer')
    parser.add_argument('--image', type=str, default=None, help='Path to input image for inference')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--backbone', type=str, choices=['resnet50','resnet18'], default='resnet50')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transforms
    transforms_img = transforms.Compose([
        transforms.Resize((384,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    if args.mode == 'train':
        ds = DeepFashionMM(args.data_root, transforms_img=transforms_img)
        val_split = int(len(ds)*0.1)
        if val_split < 1:
            val_split = max(1, int(len(ds)*0.1))
        indices = list(range(len(ds)))
        np.random.shuffle(indices)
        val_idx = indices[:val_split]
        train_idx = indices[val_split:]
        train_ds = torch.utils.data.Subset(ds, train_idx)
        val_ds = torch.utils.data.Subset(ds, val_idx)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=4)

        model = MultiTaskFashionModel(backbone_name=args.backbone,
                                      use_keypoints=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        os.makedirs(args.save_dir, exist_ok=True)
        best_val = 1e9

        # --- history for plots ---
        train_losses = []
        val_losses = []
        macro_accs = []
        macro_f1s = []
        last_per_attr_acc = None
        last_confusion = None

        for epoch in range(1, args.epochs+1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            (val_loss,
             per_attr_acc,
             macro_acc,
             per_attr_f1,
             macro_f1,
             confusion) = evaluate(model, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            macro_accs.append(macro_acc if macro_acc is not None else 0.0)
            macro_f1s.append(macro_f1 if macro_f1 is not None else 0.0)

            last_per_attr_acc = per_attr_acc
            last_confusion = confusion

            print(f"Epoch {epoch}: "
                  f"train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} "
                  f"macro_acc={macro_acc:.4f} "
                  f"macro_f1={macro_f1:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict()},
                           os.path.join(args.save_dir, 'best.pth'))

        # ========== AFTER TRAINING: MAKE PLOTS ==========

        epochs = range(1, args.epochs+1)

        # 1) Loss curve
        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training / Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
        plt.close()

        # 2) Macro accuracy & macro F1 curve
        plt.figure()
        plt.plot(epochs, macro_accs, label="Macro Accuracy")
        plt.plot(epochs, macro_f1s, label="Macro F1")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Macro Accuracy / Macro F1")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "macro_metrics.png"))
        plt.close()

        # 3) Per-attribute accuracy (bar chart, last epoch)
        if last_per_attr_acc is not None:
            attrs = list(last_per_attr_acc.keys())
            accs = [last_per_attr_acc[a] if last_per_attr_acc[a] is not None else 0.0
                    for a in attrs]

            plt.figure(figsize=(10, 4))
            x = np.arange(len(attrs))
            plt.bar(x, accs)
            plt.xticks(x, attrs, rotation=45, ha="right")
            plt.ylabel("Accuracy")
            plt.ylim(0.0, 1.0)
            plt.title("Per-Attribute Accuracy (Shape, last epoch)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, "per_attribute_accuracy.png"))
            plt.close()

        # 4) Confusion matrix for one key attribute (e.g., sleeve_length)
        if last_confusion is not None and last_confusion["sleeve_length"] is not None:
            cm = last_confusion["sleeve_length"]
            labels = SHAPE_LABELS["sleeve_length"]

            plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix - sleeve_length")
            plt.colorbar()
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45, ha="right")
            plt.yticks(tick_marks, labels)

            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, "cm_sleeve_length.png"))
            plt.close()


    else: # infer
        assert args.image is not None, 'Provide --image for inference'
        model = MultiTaskFashionModel(backbone_name=args.backbone, use_keypoints=True)
        ckpt = os.path.join(args.save_dir, 'best.pth')
        if os.path.isfile(ckpt):
            d = torch.load(ckpt, map_location='cpu')
            model.load_state_dict(d['model_state'])
        else:
            print(f"[Warning] Checkpoint not found at {ckpt}. Using randomly initialized model.")
        model = model.to(device)

        decoded= infer_single(model, args.image, device, transforms_img)

        result = {
            "attributes": decoded,
        }
        print('Inference results:')
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # also dump a visualization overlay to inspect regions


if __name__ == '__main__':
    main()
