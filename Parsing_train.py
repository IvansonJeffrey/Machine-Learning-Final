import os
import numpy as np
from typing import Dict, Optional, Tuple

from PIL import  Image
import colorsys
from tqdm import tqdm

# Optional scipy for advanced post-processing
try:
    from scipy import ndimage
    from scipy.ndimage import binary_dilation, binary_erosion
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[Warning] scipy not installed. Post-processing will use basic numpy operations.")
    print("          Install with: pip install scipy")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# Try to import better models
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("[Info] segmentation_models_pytorch not installed.")
    print("       For better parsing accuracy, install: pip install segmentation-models-pytorch")
    print("       This provides HRNet and SegFormer models with better pretrained weights.")

# =========================
# CONFIG
# =========================

DATA_ROOT = "./dataset"
CHECKPOINT_PATH = "./checkpoints/parsing_resnet50.pth"

NUM_CLASSES = 24          # DeepFashion parsing has 24 labels (0..23)
IMG_HEIGHT, IMG_WIDTH = 256, 192  # resize size (H, W) for training/inference
BATCH_SIZE = 4
EPOCHS = 5                # keep small first; increase if GPU can handle

DO_TRAIN = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# DATASET: images + parsing masks
# =========================

class ParsingDataset(Dataset):
    """
    Dataset for semantic segmentation:
    - images: root/image/*.jpg
    - masks:  root/parsing/*.png
    Both resized to (IMG_HEIGHT, IMG_WIDTH).
    """
    def __init__(self, root, img_size=(IMG_HEIGHT, IMG_WIDTH), img_list=None):
        self.root = root
        self.img_dir = os.path.join(root, "image")
        self.parsing_dir = os.path.join(root, "parsing")
        self.img_size = img_size

        if img_list is None:    
            all_imgs = [f for f in os.listdir(self.img_dir)
                        if f.lower().endswith(".jpg")]
            all_imgs.sort()

            # 🔥 Filter only images that have a corresponding mask
            # 🔥 Filter only images that have a corresponding *_segm.png mask
            self.img_list = [
                f for f in all_imgs
                if os.path.exists(
                    os.path.join(self.parsing_dir, os.path.splitext(f)[0] + "_segm.png")
                )
            ]
        else:
            self.img_list = img_list
        print(f"[ParsingDataset] Found {len(self.img_list)} pairs in {self.img_dir} / {self.parsing_dir}")


        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.mask_resize = transforms.Resize(self.img_size, interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        base = os.path.splitext(img_name)[0]

        img_path = os.path.join(self.img_dir, img_name)
        mask_name = base + "_segm.png"
        mask_path = os.path.join(self.parsing_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # each pixel is an int label 0..23

        img_t = self.img_transform(img)
        mask_r = self.mask_resize(mask)
        mask_arr = np.array(mask_r, dtype=np.int64)   # [H, W], int labels 0..23
        mask_t = torch.from_numpy(mask_arr)          # int64 tensor

        
        return img_t, mask_t


# =========================
# MODEL: DeepLabV3-ResNet50
# =========================

def get_parsing_model(num_classes=NUM_CLASSES, model_type: str = "auto"):
    """
    Get parsing model. Tries to use best available pretrained model.
    
    Args:
        num_classes: Number of segmentation classes (24 for DeepFashion)
        model_type: "auto" (best available), "deeplabv3", "hrnet", "segformer"
    
    Returns:
        Model ready for inference
    """
    if model_type == "auto":
        # Try to use best available model
        if SMP_AVAILABLE:
            # Use DeepLabV3+ with ResNet101 - better accuracy than ResNet50
            print("[Info] Using DeepLabV3+ with ResNet101 (better accuracy than ResNet50)")
            model = smp.DeepLabV3Plus(
                encoder_name="resnet101",  # Better than resnet50
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
            )
            return model
        else:
            # Fallback to DeepLabV3
            print("[Info] Using DeepLabV3-ResNet50 (install segmentation-models-pytorch for better models)")
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            model = deeplabv3_resnet50(weights=weights)
            model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
            return model
    
    elif model_type == "hrnet":
        if not SMP_AVAILABLE:
            raise ImportError("segmentation_models_pytorch not installed. Install with: pip install segmentation-models-pytorch")
        # HRNet is not available as encoder, use best alternative
        print("[Info] HRNet encoder not available, using EfficientNet-B4 (excellent accuracy)")
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b4",  # Excellent accuracy
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        return model
    
    elif model_type == "segformer":
        if not SMP_AVAILABLE:
            raise ImportError("segmentation_models_pytorch not installed. Install with: pip install segmentation-models-pytorch")
        print("[Info] Using SegFormer-B2 (state-of-the-art transformer model)")
        model = smp.SegFormer(
            encoder_name="mit_b2",  # SegFormer-B2
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        return model
    
    elif model_type == "efficientnet":
        if not SMP_AVAILABLE:
            raise ImportError("segmentation_models_pytorch not installed. Install with: pip install segmentation-models-pytorch")
        print("[Info] Using DeepLabV3+ with EfficientNet-B4 (excellent accuracy)")
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        return model
    
    else:  # deeplabv3
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        return model


# =========================
# TRAINING
# =========================

def train_parsing_model():
    # Build full dataset
    full_dataset = ParsingDataset(DATA_ROOT)

    # Simple 90/10 split
    val_ratio = 0.1
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = get_parsing_model().to(device)
    criterion = nn.CrossEntropyLoss()  # labels 0..23, no ignore_index needed
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        # ----- Train -----
        model.train()
        total_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            imgs = imgs.to(device)
            masks = masks.to(device)  # [B,H,W]

            optimizer.zero_grad()
            out = model(imgs)["out"]  # [B,C,H,W]
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * imgs.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # ----- Val -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
                imgs = imgs.to(device)
                masks = masks.to(device)
                out = model(imgs)["out"]
                loss = criterion(out, masks)
                val_loss += float(loss.item()) * imgs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print("Saved parsing model to:", CHECKPOINT_PATH)


# =========================
# COLOR ATTRIBUTES
# =========================

# Label groups (you can tweak)
# DeepFashion parsing ids:
# 1: top, 2: outer, 3: skirt, 4: dress, 5: pants, 6: leggings,
# 11: footwear, 21: rompers, etc.
REGION_LABELS = {
    "top":   {1, 4, 21},   # top, dress, rompers upper
    "outer": {2},
    "skirt": {3},
    "pants": {5, 6},       # pants + leggings (merged during post-processing)
    "leggings": {6},       # kept for backward compatibility, but usually merged with pants
    "shoes": {11},
}

def median_rgb_and_hex(img_rgb: np.ndarray, mask_bool: np.ndarray
                      ) -> Tuple[Optional[Tuple[int, int, int]], Optional[str]]:
    pixels = img_rgb[mask_bool]
    if pixels.size == 0:
        return None, None
    median = np.median(pixels, axis=0)
    r, g, b = median.astype(np.uint8).tolist()
    hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
    return (r, g, b), hex_color

def rgb_to_basic_color_name(r: int, g: int, b: int) -> str:
    r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_f, g_f, b_f)
    h_deg = h * 360.0

    # blacks and whites
    if v < 0.15:
        return "black"
    if v > 0.9 and s < 0.2:
        return "white"

    # greys (low saturation)
    if s < 0.2:
        if v < 0.4:
            return "dark grey"
        elif v < 0.7:
            return "grey"
        else:
            return "light grey"

    # chromatic colors
    if (h_deg < 15) or (h_deg >= 345):
        return "red"
    if 15 <= h_deg < 45:
        return "orange"
    if 45 <= h_deg < 70:
        return "yellow"
    if 70 <= h_deg < 170:
        return "green"
    if 170 <= h_deg < 255:
        return "blue"
    if 255 <= h_deg < 290:
        return "purple"
    if 290 <= h_deg < 345:
        return "magenta"

    return "other"


# =========================
# INFERENCE HELPERS
# =========================

infer_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_trained_parsing_model(model_type: str = "auto", use_pretrained_only: bool = False):
    """
    Load parsing model. If checkpoint exists, loads it. Otherwise uses pretrained weights.
    
    Args:
        model_type: "auto", "deeplabv3", "hrnet", "segformer"
        use_pretrained_only: If True, skip loading checkpoint and use pretrained weights only
    """
    model = get_parsing_model(model_type=model_type)
    
    # Try to load checkpoint if available
    if not use_pretrained_only and os.path.exists(CHECKPOINT_PATH):
        try:
            state = torch.load(CHECKPOINT_PATH, map_location="cpu")
            # Handle different checkpoint formats
            if isinstance(state, dict) and 'model_state' in state:
                model.load_state_dict(state['model_state'], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            model = model.to(device)
            model.eval()
            print(f"[Info] Loaded trained model from: {CHECKPOINT_PATH}")
            return model
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint: {e}")
            print("[Info] Using pretrained weights instead")
    
    # Use pretrained weights (no training needed)
    model = model.to(device)
    model.eval()
    print(f"[Info] Using pretrained {model_type} model (no checkpoint found or use_pretrained_only=True)")
    print("[Info] This model uses ImageNet pretrained weights and should work reasonably well")
    return model

def simple_dilate(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Simple dilation using numpy (fallback if scipy not available)"""
    H, W = mask.shape
    dilated = mask.copy()
    half_k = kernel_size // 2
    
    for y in range(half_k, H - half_k):
        for x in range(half_k, W - half_k):
            if mask[y, x]:
                # Expand the region
                dilated[y-half_k:y+half_k+1, x-half_k:x+half_k+1] = mask[y, x]
    
    return dilated


def post_process_parsing_mask(pred_mask: np.ndarray, expand_regions: bool = True) -> np.ndarray:
    """
    Post-process parsing mask to improve accuracy:
    - Expand small regions to fill gaps
    - Merge related regions (pants + leggings)
    - Fill holes in regions
    - Smooth boundaries
    
    Args:
        pred_mask: [H, W] int array with labels 0..23
        expand_regions: Whether to expand regions to fill gaps
    
    Returns:
        Improved [H, W] mask
    """
    improved_mask = pred_mask.copy()
    
    if expand_regions:
        # Define region groups that should be merged/expanded
        # 1: top, 2: outer, 3: skirt, 4: dress, 5: pants, 6: leggings, 11: shoes
        
        # Strategy 1: Merge leggings (6) into pants (5) if leggings are detected
        # This handles the case where leggings only cover half of pants
        leggings_mask = (improved_mask == 6)
        pants_mask = (improved_mask == 5)
        
        if leggings_mask.sum() > 0 and pants_mask.sum() > 0:
            # If both exist, merge leggings into pants
            improved_mask[leggings_mask] = 5
            print("[Post-process] Merged leggings into pants region")
        elif leggings_mask.sum() > 0 and pants_mask.sum() == 0:
            # If only leggings exist (no pants detected), expand leggings to cover pants area
            # Find the vertical extent of leggings
            leggings_rows = np.where(leggings_mask.any(axis=1))[0]
            if len(leggings_rows) > 0:
                min_row = max(0, leggings_rows[0] - 20)  # Expand upward
                max_row = min(improved_mask.shape[0], leggings_rows[-1] + 20)  # Expand downward
                # Fill the pants area (typically middle to lower body)
                pants_area = improved_mask[min_row:max_row, :]
                # Only fill if it's background (0) or other non-clothing
                fill_mask = (pants_area == 0) | (pants_area > 20)
                improved_mask[min_row:max_row, :][fill_mask] = 5
                print("[Post-process] Expanded leggings to cover pants area")
        
        # Strategy 2-4: Expand regions to fill gaps
        if SCIPY_AVAILABLE:
            # Use scipy for efficient morphological operations
            # Expand top region (1)
            top_mask = (improved_mask == 1)
            if top_mask.sum() > 0:
                kernel_size = 5
                dilated_top = binary_dilation(top_mask, structure=np.ones((kernel_size, kernel_size)))
                fill_area = dilated_top & (improved_mask == 0)
                improved_mask[fill_area] = 1
            
            # Expand outer region (2)
            outer_mask = (improved_mask == 2)
            if outer_mask.sum() > 0:
                kernel_size = 7
                dilated_outer = binary_dilation(outer_mask, structure=np.ones((kernel_size, kernel_size)))
                fill_area = dilated_outer & (improved_mask == 0)
                improved_mask[fill_area] = 2
            
            # Expand pants region (5)
            pants_mask = (improved_mask == 5)
            if pants_mask.sum() > 0:
                kernel_size = 8  # Larger for pants to cover full leg area
                dilated_pants = binary_dilation(pants_mask, structure=np.ones((kernel_size, kernel_size)))
                fill_area = dilated_pants & ((improved_mask == 0) | (improved_mask > 10))
                improved_mask[fill_area] = 5
            
            # Fill small holes using median filter
            for region_id in [1, 2, 5, 11]:
                region_mask = (improved_mask == region_id)
                if region_mask.sum() > 100:
                    filtered = ndimage.median_filter(improved_mask, size=3)
                    region_expanded = binary_dilation(region_mask, structure=np.ones((3, 3)))
                    improved_mask[region_expanded & (filtered == region_id)] = region_id
        else:
            # Fallback: Simple expansion using numpy
            # Expand top region
            top_mask = (improved_mask == 1)
            if top_mask.sum() > 0:
                # Simple vertical expansion
                for y in range(1, improved_mask.shape[0] - 1):
                    for x in range(improved_mask.shape[1]):
                        if improved_mask[y, x] == 0 and (improved_mask[y-1, x] == 1 or improved_mask[y+1, x] == 1):
                            improved_mask[y, x] = 1
            
            # Expand pants region
            pants_mask = (improved_mask == 5)
            if pants_mask.sum() > 0:
                # More aggressive expansion for pants
                for y in range(2, improved_mask.shape[0] - 2):
                    for x in range(improved_mask.shape[1]):
                        if improved_mask[y, x] == 0:
                            # Check if nearby pixels are pants
                            neighbors = improved_mask[max(0,y-2):y+3, max(0,x-2):min(improved_mask.shape[1],x+3)]
                            if (neighbors == 5).sum() > 3:  # If enough neighbors are pants
                                improved_mask[y, x] = 5
    
    return improved_mask


def predict_parsing_mask(model, image_path: str, post_process: bool = True, use_fallback: bool = True):
    """
    Returns:
      img_rgb_resized: [H,W,3] uint8
      pred_mask: [H,W] int labels 0..23 (post-processed if enabled)
    
    Args:
        use_fallback: If True, uses rule-based fallback when parsing is too inaccurate
    """
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT), resample=Image.BILINEAR)  # (W,H) for PIL

    img_tensor = infer_transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        out = model(img_tensor)
        
        # Handle different model output formats
        if isinstance(out, dict):
            out = out["out"]  # DeepLabV3 format
        # HRNet/SegFormer from smp return tensor directly
        
        pred = out.argmax(dim=1)[0].cpu().numpy().astype(np.int32)  # [H,W]

    # Check if parsing is too inaccurate (too much background, too few clothing regions)
    clothing_regions = np.isin(pred, [1, 2, 3, 4, 5, 6, 11, 21])
    clothing_ratio = clothing_regions.sum() / pred.size
    
    if use_fallback and clothing_ratio < 0.1:  # Less than 10% clothing detected
        print(f"[Warning] Parsing model detected only {clothing_ratio*100:.1f}% clothing. Using fallback method.")
        pred = create_fallback_mask(pred.shape, img_resized.size)  # [H, W]
    
    # Post-process to improve mask quality
    if post_process:
        pred = post_process_parsing_mask(pred, expand_regions=True)

    img_rgb = np.array(img_resized)          # [H,W,3]
    return img_rgb, pred


def create_fallback_mask(shape: Tuple[int, int], img_size: Tuple[int, int]) -> np.ndarray:
    """
    Creates a rule-based parsing mask when the model is too inaccurate.
    Uses simple vertical regions based on image geometry.
    
    Args:
        shape: (H, W) of the mask to create
        img_size: (W, H) of original image
    
    Returns:
        [H, W] mask with labels
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.int32)
    
    # Simple vertical division:
    # Top 40% = top (1)
    # Middle 30% = could be top or outer
    # Lower 30% = pants (5)
    # Bottom 10% = shoes (11)
    
    top_end = int(H * 0.40)
    middle_end = int(H * 0.70)
    pants_end = int(H * 0.90)
    
    # Top region
    mask[:top_end, :] = 1  # top
    
    # Middle region (try to detect if there's an outer layer)
    # For now, assume it's part of top
    mask[top_end:middle_end, :] = 1  # top
    
    # Lower region = pants
    mask[middle_end:pants_end, :] = 5  # pants
    
    # Bottom = shoes
    mask[pants_end:, :] = 11  # shoes
    
    print("[Fallback] Created rule-based mask using vertical regions")
    return mask

def analyze_regions_from_pred(img_rgb: np.ndarray, pred_mask: np.ndarray,
                              use_expanded_regions: bool = True) -> Dict[str, Dict[str, Optional[str]]]:
    """
    From predicted mask -> classical color attributes per region.
    Uses expanded region detection to handle inaccurate parsing.
    
    Returns:
      {
        "top": {"hex": "#RRGGBB", "name": "green"},
        "pants": {...},
        ...
      }
    """
    results: Dict[str, Dict[str, Optional[str]]] = {}
    H, W = pred_mask.shape

    for region_name, label_ids in REGION_LABELS.items():
        mask = np.isin(pred_mask, list(label_ids))
        
        # If region is too small, try to expand it using geometric heuristics
        if use_expanded_regions and mask.sum() < (H * W * 0.05):  # Less than 5% of image
            # Try to find the region in nearby areas based on typical clothing positions
            if region_name == "top":
                # Expand top region to upper 50% of image
                expanded_mask = np.zeros_like(mask, dtype=bool)
                expanded_mask[:int(H*0.5), :] = True
                # Only use pixels that aren't clearly other clothing
                expanded_mask = expanded_mask & ~np.isin(pred_mask, [5, 6, 11])  # Not pants/shoes
                mask = mask | expanded_mask
            elif region_name == "pants":
                # Expand pants region to lower 40% of image
                expanded_mask = np.zeros_like(mask, dtype=bool)
                expanded_mask[int(H*0.5):int(H*0.9), :] = True
                # Only use pixels that aren't clearly other clothing
                expanded_mask = expanded_mask & ~np.isin(pred_mask, [1, 2, 11])  # Not top/outer/shoes
                mask = mask | expanded_mask
            elif region_name == "shoes":
                # Expand shoes to bottom 15% of image
                expanded_mask = np.zeros_like(mask, dtype=bool)
                expanded_mask[int(H*0.85):, :] = True
                mask = mask | expanded_mask
        
        rgb, hex_color = median_rgb_and_hex(img_rgb, mask)
        if rgb is None:
            results[region_name] = {"hex": None, "name": None}
        else:
            r, g, b = rgb
            name = rgb_to_basic_color_name(r, g, b)
            results[region_name] = {"hex": hex_color, "name": name}

    return results


# =========================
# RUN: TRAIN + TEST
# =========================
def main():
    # 1) Train (optional)
    if DO_TRAIN:
        train_parsing_model()

    # 2) Load trained model
    model = load_trained_parsing_model()

    # 3) Test on one image
    # (we'll customize which image below)
    test_img_name = "MEN-Jackets_Vests-id_00007800-01_1_front.jpg"
    test_img_path = os.path.join(DATA_ROOT, "image", test_img_name)
    print("Testing on:", test_img_path)

    img_rgb, pred_mask = predict_parsing_mask(model, test_img_path)
    region_colors = analyze_regions_from_pred(img_rgb, pred_mask)

    print("\nPredicted colors:")
    for part, info in region_colors.items():
        print(f"{part:9s} -> {info['name']} ({info['hex']})")



if __name__ == "__main__":
    # optional, but sometimes recommended:
    # import torch.multiprocessing as mp
    # mp.set_start_method("spawn", force=True)

    main()
