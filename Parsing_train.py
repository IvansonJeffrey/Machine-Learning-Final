import os
import numpy as np
from typing import Dict, Optional, Tuple

from PIL import  Image
import colorsys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# =========================
# CONFIG
# =========================

DATA_ROOT = r"C:\Users\Jeffrey M. Ivanson\Documents\MLDEMO\dataset"
CHECKPOINT_PATH = r"C:\Users\Jeffrey M. Ivanson\Documents\MLDEMO\checkpoints\parsing_resnet50.pth"

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

def get_parsing_model(num_classes=NUM_CLASSES):
    """
    DeepLabV3-ResNet50:
    - pretrained backbone
    - final classifier changed to num_classes
    """
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    # Replace the classifier head last conv: 256->num_classes
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
    "pants": {5},
    "leggings": {6},
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

def load_trained_parsing_model():
    model = get_parsing_model()
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print("Loaded model from:", CHECKPOINT_PATH)
    return model

def predict_parsing_mask(model, image_path: str):
    """
    Returns:
      img_rgb_resized: [H,W,3] uint8
      pred_mask: [H,W] int labels 0..23
    """
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT), resample=Image.BILINEAR)  # (W,H) for PIL

    img_tensor = infer_transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        out = model(img_tensor)["out"]        # [1,C,H,W]
        pred = out.argmax(dim=1)[0].cpu().numpy().astype(np.int32)  # [H,W]

    img_rgb = np.array(img_resized)          # [H,W,3]
    return img_rgb, pred

def analyze_regions_from_pred(img_rgb: np.ndarray, pred_mask: np.ndarray
                             ) -> Dict[str, Dict[str, Optional[str]]]:
    """
    From predicted mask -> classical color attributes per region.
    Returns:
      {
        "top": {"hex": "#RRGGBB", "name": "green"},
        "pants": {...},
        ...
      }
    """
    results: Dict[str, Dict[str, Optional[str]]] = {}

    for region_name, label_ids in REGION_LABELS.items():
        mask = np.isin(pred_mask, list(label_ids))
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
