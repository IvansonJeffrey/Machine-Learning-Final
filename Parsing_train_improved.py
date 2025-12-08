"""
Improved Parsing Model Training Script
Supports multiple architectures: DeepLabV3+, HRNet, SegFormer
Better training configuration with augmentation, larger input size, more epochs
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple
import argparse

from PIL import Image
import colorsys
from tqdm import tqdm

# Optional scipy for advanced post-processing
try:
    from scipy import ndimage
    from scipy.ndimage import binary_dilation, binary_erosion
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50, 
    deeplabv3_resnet101,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights
)

# Try to import HRNet (if available)
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("[Info] segmentation_models_pytorch not installed. HRNet unavailable.")
    print("       Install with: pip install segmentation-models-pytorch")

# =========================
# CONFIG
# =========================

DATA_ROOT = "./dataset"
CHECKPOINT_PATH = "./checkpoints/parsing_resnet50.pth"

NUM_CLASSES = 24          # DeepFashion parsing has 24 labels (0..23)

# Improved training config
IMG_HEIGHT, IMG_WIDTH = 512, 384  # Larger input size (was 256x192)
BATCH_SIZE = 4  # Adjust based on GPU memory
EPOCHS = 30     # More epochs (was 5)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Data augmentation
USE_AUGMENTATION = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# DATASET: images + parsing masks
# =========================

class ParsingDataset(Dataset):
    """
    Dataset for semantic segmentation with augmentation.
    - images: root/images/*.jpg
    - masks:  root/parsing/*_segm.png
    """
    def __init__(self, root, img_size=(IMG_HEIGHT, IMG_WIDTH), img_list=None, augment=False):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.parsing_dir = os.path.join(root, "parsing")
        self.img_size = img_size
        self.augment = augment

        if img_list is None:    
            all_imgs = [f for f in os.listdir(self.img_dir)
                        if f.lower().endswith(".jpg")]
            all_imgs.sort()

            # Filter only images that have a corresponding mask
            self.img_list = [
                f for f in all_imgs
                if os.path.exists(
                    os.path.join(self.parsing_dir, os.path.splitext(f)[0] + "_segm.png")
                )
            ]
        else:
            self.img_list = img_list
        
        print(f"[ParsingDataset] Found {len(self.img_list)} pairs in {self.img_dir} / {self.parsing_dir}")
        
        if len(self.img_list) == 0:
            print("[WARNING] No parsing masks found! Cannot train without ground truth masks.")
            print(f"         Expected masks in: {self.parsing_dir}")
            print(f"         Format: {{image_name}}_segm.png")

        # Base transforms (always applied)
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        # Augmentation transforms (only for training)
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.aug_transform = None

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

        # Apply augmentation if enabled
        if self.aug_transform:
            # Apply same augmentation to both image and mask
            seed = np.random.randint(2147483647)
            np.random.seed(seed)
            torch.manual_seed(seed)
            img = self.aug_transform(img)
            # For mask, only apply horizontal flip (same seed ensures consistency)
            if np.random.random() < 0.5:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img_t = self.img_transform(img)
        mask_r = self.mask_resize(mask)
        mask_arr = np.array(mask_r, dtype=np.int64)   # [H, W], int labels 0..23
        mask_t = torch.from_numpy(mask_arr)          # int64 tensor

        return img_t, mask_t


# =========================
# MODELS: Multiple architectures
# =========================

def get_deeplabv3_resnet50(num_classes=NUM_CLASSES):
    """DeepLabV3 with ResNet50 backbone"""
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

def get_deeplabv3_resnet101(num_classes=NUM_CLASSES):
    """DeepLabV3 with ResNet101 backbone (more accurate, slower)"""
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

def get_hrnet(num_classes=NUM_CLASSES):
    """HRNet - High Resolution Network (best accuracy for segmentation)"""
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch not installed. Install with: pip install segmentation-models-pytorch")
    
    model = smp.DeepLabV3Plus(
        encoder_name="hrnet_w48",  # HRNet Wide-48
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model

def get_segformer(num_classes=NUM_CLASSES):
    """SegFormer - Transformer-based (state-of-the-art)"""
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch not installed")
    
    # SegFormer-B2 variant
    model = smp.SegFormer(
        encoder_name="mit_b2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model

def get_model(model_name: str = "deeplabv3_resnet50", num_classes: int = NUM_CLASSES):
    """
    Get model by name.
    
    Options:
    - deeplabv3_resnet50: Fast, good accuracy (default)
    - deeplabv3_resnet101: Better accuracy, slower
    - hrnet: Best accuracy, moderate speed
    - segformer: State-of-the-art, slower
    """
    model_name = model_name.lower()
    
    if model_name == "deeplabv3_resnet50":
        return get_deeplabv3_resnet50(num_classes)
    elif model_name == "deeplabv3_resnet101":
        return get_deeplabv3_resnet101(num_classes)
    elif model_name == "hrnet":
        return get_hrnet(num_classes)
    elif model_name == "segformer":
        return get_segformer(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: deeplabv3_resnet50, deeplabv3_resnet101, hrnet, segformer")


# =========================
# TRAINING
# =========================

def train_parsing_model(model_name: str = "deeplabv3_resnet50", 
                       epochs: int = EPOCHS,
                       batch_size: int = BATCH_SIZE,
                       lr: float = LEARNING_RATE,
                       use_augmentation: bool = USE_AUGMENTATION):
    """
    Train parsing model with improved configuration.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {epochs} epochs")
    print(f"Input size: {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Augmentation: {use_augmentation}")
    print(f"{'='*60}\n")
    
    # Build dataset
    full_dataset = ParsingDataset(DATA_ROOT, augment=use_augmentation)
    
    if len(full_dataset) == 0:
        print("\n[ERROR] No training data found!")
        print("        You need parsing masks to train.")
        print(f"        Expected: {os.path.join(DATA_ROOT, 'parsing', '*_segm.png')}")
        return None

    # Train/val split
    val_ratio = 0.1
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # Get model
    try:
        model = get_model(model_name, NUM_CLASSES).to(device)
        print(f"Model: {model_name}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")
    except Exception as e:
        print(f"[ERROR] Failed to create model: {e}")
        return None

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore invalid labels
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_val_loss = float('inf')
    best_miou = 0.0
    
    for epoch in range(1, epochs + 1):
        # ----- Train -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            imgs = imgs.to(device)
            masks = masks.to(device)  # [B,H,W]

            optimizer.zero_grad()
            out = model(imgs)["out"] if isinstance(out, dict) else model(imgs)  # [B,C,H,W]
            
            # Handle different model outputs
            if isinstance(out, dict):
                out = out["out"]
            
            loss = criterion(out, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            
            # Calculate accuracy
            pred = out.argmax(dim=1)
            valid_mask = masks != 255
            train_correct += (pred[valid_mask] == masks[valid_mask]).sum().item()
            train_total += valid_mask.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        class_correct = torch.zeros(NUM_CLASSES).to(device)
        class_total = torch.zeros(NUM_CLASSES).to(device)
        
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                out = model(imgs)
                if isinstance(out, dict):
                    out = out["out"]
                
                loss = criterion(out, masks)
                val_loss += loss.item()
                
                pred = out.argmax(dim=1)
                valid_mask = masks != 255
                val_correct += (pred[valid_mask] == masks[valid_mask]).sum().item()
                val_total += valid_mask.sum().item()
                
                # Per-class accuracy
                for c in range(NUM_CLASSES):
                    class_mask = (masks == c) & valid_mask
                    if class_mask.sum() > 0:
                        class_correct[c] += (pred[class_mask] == c).sum().item()
                        class_total[c] += class_mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Calculate mIoU
        class_iou = class_correct / (class_total + 1e-6)
        miou = class_iou.mean().item()

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, mIoU: {miou*100:.2f}%")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_name = f"parsing_{model_name}_best.pth"
            checkpoint_path = os.path.join(os.path.dirname(CHECKPOINT_PATH), checkpoint_name)
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'miou': miou,
                'model_name': model_name,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss={avg_val_loss:.4f}, mIoU={miou*100:.2f}%)")

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best mIoU: {best_miou*100:.2f}%")
    print(f"{'='*60}\n")
    
    return model


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved parsing model")
    parser.add_argument(
        "--model", 
        type=str, 
        default="deeplabv3_resnet50",
        choices=["deeplabv3_resnet50", "deeplabv3_resnet101", "hrnet", "segformer"],
        help="Model architecture"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Dataset root directory")
    
    args = parser.parse_args()
    
    DATA_ROOT = args.data_root
    
    # Train model
    model = train_parsing_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_augmentation=not args.no_aug
    )

