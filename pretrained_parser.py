"""
Pre-trained Human Parsing Model for Clothing Segmentation
Segments: head, shirt, pants, shoes
Uses pre-trained models that work out-of-the-box without training.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Tuple, Optional
import colorsys

# Matplotlib for visualization (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# OpenCV for image processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[Warning] OpenCV not available. Install with: pip install opencv-python")

# Try to use segmentation-models-pytorch (best option)
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

# Try to use detectron2 (alternative)
try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False


def is_skin_color(r: int, g: int, b: int) -> bool:
    """
    Detect if a color is likely skin/flesh tone.
    Skin colors typically have:
    - Hue in orange/peach range (8-45 degrees is most common for skin)
    - Moderate saturation (0.2-0.6) - skin is not very saturated
    - Medium to high brightness (0.35-0.95)
    - R > G > B (typical skin tone gradient)
    """
    r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_f, g_f, b_f)
    h_deg = h * 360.0
    
    # Skin color detection - more specific to avoid false positives on clothing
    # Hue: typically in the orange/peach range (8-45 degrees is most common)
    # Also allow near-red (350-360) for some skin tones
    hue_ok = (h_deg >= 8 and h_deg <= 45) or (h_deg >= 350)
    
    # Saturation: moderate but not too high (0.15-0.65) - skin is not very vibrant
    # Lower bound allows darker skin tones, upper bound avoids pure red clothing
    sat_ok = s >= 0.15 and s <= 0.65
    
    # Brightness: medium to high (0.25-0.95) - allow darker skin tones
    val_ok = v >= 0.25 and v <= 0.95
    
    # Critical: R should be higher than B, and G should be closer to R than B
    # This distinguishes skin (peach/orange) from pure red clothing
    # For skin: typically R > G > B, and G is closer to R
    rgb_ratio_ok = (r > b) and (g > b * 0.7) and (abs(r - g) < abs(g - b) * 2.0)
    
    # Additional check: avoid very saturated reds (pure red clothing)
    # Skin has lower saturation than bright red clothing
    not_pure_red = not ((h_deg < 15 or h_deg > 345) and s > 0.7 and r > g * 1.5)
    
    return hue_ok and sat_ok and val_ok and rgb_ratio_ok and not_pure_red


def rgb_to_color_name(r: int, g: int, b: int) -> str:
    """Convert RGB to color name that matches classifier labels"""
    r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_f, g_f, b_f)
    h_deg = h * 360.0

    # Blacks and whites
    if v < 0.15:
        return "black"
    if v > 0.9 and s < 0.2:
        return "white"

    # Greys
    if s < 0.2:
        if v < 0.4:
            return "dark grey"
        elif v < 0.7:
            return "grey"
        else:
            return "light grey"

    # Chromatic colors
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


def extract_color_from_region(img_rgb: np.ndarray, mask: np.ndarray, region_name: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    Extract dominant color from a region mask.
    
    Args:
        img_rgb: RGB image array [H, W, 3]
        mask: Boolean mask for the region [H, W]
        region_name: Optional region name (e.g., "pants", "shirt", "top") to apply region-specific filtering
                     For pants/shirt/top regions, skin-colored pixels are filtered out (helps with shorts/short sleeves)
    """
    if mask.sum() == 0:
        return {"hex": None, "name": None, "rgb": None}
    
    # Get pixels in the region
    pixels = img_rgb[mask]
    
    # Filter out extreme values (background, shadows)
    brightness = pixels.mean(axis=1)
    valid_mask = (brightness > 20) & (brightness < 235)
    valid_pixels = pixels[valid_mask]
    
    # For pants region, filter out skin-colored pixels (common with shorts)
    # For shirt/top region, filter out skin-colored pixels (common with short sleeves)
    if region_name in ["pants", "shirt", "top"] and len(valid_pixels) > 0:
        # Check each pixel for skin color
        skin_mask = np.array([
            not is_skin_color(int(p[0]), int(p[1]), int(p[2])) 
            for p in valid_pixels
        ])
        non_skin_pixels = valid_pixels[skin_mask]
        
        # Only use filtered pixels if we have enough (at least 10% of original)
        if len(non_skin_pixels) >= len(valid_pixels) * 0.1:
            valid_pixels = non_skin_pixels
        # If too many pixels were filtered out, likely not shorts/short sleeves, use original
    
    if len(valid_pixels) == 0:
        return {"hex": None, "name": None, "rgb": None}
    
    # Use median (more robust than mean)
    median_color = np.median(valid_pixels, axis=0).astype(np.uint8)
    r, g, b = median_color.tolist()
    hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
    color_name = rgb_to_color_name(r, g, b)
    
    return {
        "hex": hex_color,
        "name": color_name,
        "rgb": [int(r), int(g), int(b)]
    }


# =========================
# METHOD 1: Using segmentation-models-pytorch (HRNet/DeepLabV3)
# =========================

def load_smp_parser(model_name: str = "deeplabv3plus_resnet50") -> Optional[torch.nn.Module]:
    """Load pre-trained segmentation model from smp"""
    if not SMP_AVAILABLE:
        return None
    
    try:
        # Use a model pre-trained on ImageNet (we'll adapt it)
        # For human parsing, we'll use a general segmentation model
        model = smp.create_model(
            model_name,
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=4,  # head, shirt, pants, shoes
            activation=None,
        )
        model.eval()
        return model
    except Exception as e:
        print(f"[Warning] Could not load SMP model: {e}")
        return None


def parse_with_smp(model: torch.nn.Module, image_path: str, device: torch.device) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Parse image using SMP model"""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    
    if not CV2_AVAILABLE:
        return img_np, parse_with_geometric(img_np)
    
    # Resize for model (typical input size)
    img_resized = cv2.resize(img_np, (512, 512))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
    
    # Resize prediction back to original size
    pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create masks for each region
    masks = {
        "head": (pred == 0),
        "shirt": (pred == 1),
        "pants": (pred == 2),
        "shoes": (pred == 3),
    }
    
    return img_np, masks


# =========================
# METHOD 2: Simple Geometric Parser (Fallback - Always Works)
# =========================

def parse_with_geometric(img_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Simple geometric parser that divides image into regions.
    This is a fallback that always works, though less accurate.
    """
    h, w = img_rgb.shape[:2]
    masks = {}
    
    # Head: top 20% of image
    head_mask = np.zeros((h, w), dtype=bool)
    head_mask[:int(h * 0.20), :] = True
    masks["head"] = head_mask
    
    # Shirt: 20% to 50% of image
    shirt_mask = np.zeros((h, w), dtype=bool)
    shirt_mask[int(h * 0.20):int(h * 0.50), :] = True
    masks["shirt"] = shirt_mask
    
    # Pants: 50% to 85% of image
    pants_mask = np.zeros((h, w), dtype=bool)
    pants_mask[int(h * 0.50):int(h * 0.85), :] = True
    masks["pants"] = pants_mask
    
    # Shoes: bottom 15% of image
    shoes_mask = np.zeros((h, w), dtype=bool)
    shoes_mask[int(h * 0.85):, :] = True
    masks["shoes"] = shoes_mask
    
    return masks


# =========================
# METHOD 3: Using MediaPipe (if available)
# =========================

def parse_with_mediapipe(img_rgb: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """Parse using MediaPipe Selfie Segmentation"""
    try:
        import mediapipe as mp
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 0: general, 1: landscape
        )
        
        # MediaPipe expects RGB
        results = selfie_segmentation.process(img_rgb)
        person_mask = results.segmentation_mask > 0.5
        
        if person_mask.sum() == 0:
            return None
        
        # Divide person mask into regions
        h, w = img_rgb.shape[:2]
        masks = {}
        
        # Head: top 25% of person
        head_y = int(h * 0.25)
        masks["head"] = person_mask & (np.arange(h)[:, None] < head_y)
        
        # Shirt: 25% to 50% of person
        shirt_y0, shirt_y1 = int(h * 0.25), int(h * 0.50)
        masks["shirt"] = person_mask & (np.arange(h)[:, None] >= shirt_y0) & (np.arange(h)[:, None] < shirt_y1)
        
        # Pants: 50% to 85% of person
        pants_y0, pants_y1 = int(h * 0.50), int(h * 0.85)
        masks["pants"] = person_mask & (np.arange(h)[:, None] >= pants_y0) & (np.arange(h)[:, None] < pants_y1)
        
        # Shoes: bottom 15% of person
        shoes_y = int(h * 0.85)
        masks["shoes"] = person_mask & (np.arange(h)[:, None] >= shoes_y)
        
        return masks
    except ImportError:
        return None
    except Exception as e:
        print(f"[Warning] MediaPipe parsing failed: {e}")
        return None


# =========================
# MAIN PARSING FUNCTION
# =========================

def remove_legs_from_pants_mask(img_rgb: np.ndarray, pants_mask: np.ndarray, is_shorts: bool = False) -> np.ndarray:
    """
    Remove leg/skin areas from pants mask when shorts are detected.
    
    Args:
        img_rgb: RGB image array [H, W, 3]
        pants_mask: Boolean mask for pants region [H, W]
        is_shorts: If True, actively removes legs. If False, uses skin detection.
    
    Returns:
        Adjusted pants mask with leg areas removed
    """
    if pants_mask.sum() == 0:
        return pants_mask
    
    h, w = img_rgb.shape[:2]
    adjusted_mask = pants_mask.copy()
    
    # Get pixels in the pants region
    pants_pixels = img_rgb[pants_mask]
    
    if len(pants_pixels) == 0:
        return adjusted_mask
    
    # Focus on the lower 60% of the pants region (where legs typically appear in shorts)
    # Find the vertical bounds of the pants region
    pants_rows = np.where(pants_mask.any(axis=1))[0]
    if len(pants_rows) == 0:
        return adjusted_mask
    
    # If shorts are detected, remove legs more aggressively
    if is_shorts:
        print("[Parsing] Shorts detected by classifier. Removing legs from pants mask.")
        
        # Remove skin-colored pixels from the entire pants mask
        pants_pixels_rgb = img_rgb[pants_mask]  # [N, 3]
        
        # Vectorized skin color detection
        skin_mask_vectorized = np.array([
            is_skin_color(int(p[0]), int(p[1]), int(p[2])) for p in pants_pixels_rgb
        ])
        
        # Find which pixels in pants_mask are skin-colored
        pants_indices = np.where(pants_mask)
        skin_pants_indices = (pants_indices[0][skin_mask_vectorized], 
                             pants_indices[1][skin_mask_vectorized])
        
        # Remove skin pixels from adjusted mask
        adjusted_mask[skin_pants_indices] = False
        
        # Additionally, remove the lower portion of pants region (where legs typically are)
        pants_top = pants_rows[0]
        pants_bottom = pants_rows[-1]
        pants_height = pants_bottom - pants_top
        # Remove bottom 50% of pants region (typical leg area in shorts)
        leg_cutoff = pants_top + int(pants_height * 0.5)
        # Create a mask for the lower portion that was originally part of pants
        lower_leg_mask = pants_mask.copy()
        lower_leg_mask[:leg_cutoff, :] = False  # Only keep bottom portion
        # Remove this lower leg area from adjusted mask
        adjusted_mask[lower_leg_mask] = False
        
    else:
        # Original skin detection logic for auto-detection
        pants_top = pants_rows[0]
        pants_bottom = pants_rows[-1]
        pants_height = pants_bottom - pants_top
        lower_pants_start = pants_top + int(pants_height * 0.4)  # Lower 60% of pants region
        
        # Create mask for lower pants area
        lower_pants_mask = pants_mask.copy()
        lower_pants_mask[:lower_pants_start, :] = False
        
        # Check pixels in lower pants area for skin color
        if lower_pants_mask.sum() > 0:
            lower_pants_pixels = img_rgb[lower_pants_mask]
            
            # Count how many pixels in lower region are skin-colored
            skin_pixel_count = 0
            total_pixel_count = len(lower_pants_pixels)
            
            for pixel in lower_pants_pixels:
                if is_skin_color(int(pixel[0]), int(pixel[1]), int(pixel[2])):
                    skin_pixel_count += 1
            
            # If more than 30% of lower pants region is skin-colored, likely shorts with exposed legs
            skin_ratio = skin_pixel_count / total_pixel_count if total_pixel_count > 0 else 0
            
            if skin_ratio > 0.3:
                print(f"[Parsing] Auto-detected shorts (skin ratio in lower pants: {skin_ratio:.1%}). Removing legs from pants mask.")
                
                # Remove skin-colored pixels from the entire pants mask (vectorized for speed)
                pants_pixels_rgb = img_rgb[pants_mask]  # [N, 3]
                
                # Vectorized skin color detection
                skin_mask_vectorized = np.array([
                    is_skin_color(int(p[0]), int(p[1]), int(p[2])) for p in pants_pixels_rgb
                ])
                
                # Find which pixels in pants_mask are skin-colored
                pants_indices = np.where(pants_mask)
                skin_pants_indices = (pants_indices[0][skin_mask_vectorized], 
                                     pants_indices[1][skin_mask_vectorized])
                
                # Remove skin pixels from adjusted mask
                adjusted_mask[skin_pants_indices] = False
    
    return adjusted_mask


def parse_clothing_regions(
    image_path: str, 
    device: torch.device = None,
    visualize: bool = False,
    vis_save_path: str = "parsed_regions_visualization.png"
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Parse image into clothing regions: head, shirt, pants, shoes
    Tries multiple methods in order of preference.
    Automatically removes legs from pants mask when shorts are detected.
    
    Returns:
        img_rgb: Original image as numpy array
        masks: Dict with keys "head", "shirt", "pants", "shoes", each containing a boolean mask
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_rgb = np.array(img)
    
    # Try MediaPipe first (fast, good for person segmentation)
    masks = parse_with_mediapipe(img_rgb)
    
    if masks is None:
        # Fallback to geometric parser (always works)
        print("[Info] Using geometric parser (fallback method)")
        masks = parse_with_geometric(img_rgb)
    else:
        print("[Info] Using MediaPipe parser")
    
    # Note: Leg removal will be done after classification if shorts are detected
    # (See adjust_masks_based_on_classifier in output.py)
    
    # Visualize if requested
    if visualize:
        try:
            visualize_parsed_regions(img_rgb, masks, save_path=vis_save_path, show=False)
        except Exception as e:
            print(f"[Warning] Could not create matplotlib visualization: {e}")
            print("[Info] Trying simple PIL visualization...")
            try:
                visualize_parsed_regions_simple(img_rgb, masks, save_path=vis_save_path.replace('.png', '_simple.png'))
            except Exception as e2:
                print(f"[Warning] Could not create simple visualization: {e2}")
    
    return img_rgb, masks


def extract_colors_from_regions(img_rgb: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Extract colors from parsed regions.
    
    Returns:
        {
            "head": {"hex": "#...", "name": "black", "rgb": [r, g, b]},
            "shirt": {...},
            "pants": {...},
            "shoes": {...}
        }
    """
    colors = {}
    for region_name, mask in masks.items():
        colors[region_name] = extract_color_from_region(img_rgb, mask, region_name=region_name)
    
    return colors


def calculate_color_extraction_confidence(
    masks: Dict[str, np.ndarray],
    matched_colors: Dict[str, Dict],
    classifier_colors: Dict[str, str]
) -> Dict[str, float]:
    """
    Calculate confidence for color extraction per region.
    Based on:
    - Parser region quality (size, coverage)
    - Color extraction success
    - Classifier prediction agreement
    """
    confidences = {}
    h, w = masks.get("shirt", np.zeros((1, 1), dtype=bool)).shape
    total_pixels = h * w if h > 0 and w > 0 else 1
    
    color_to_region = {
        "color_0": "shirt",
        "color_1": "pants",
        "color_2": "shoes"
    }
    
    for color_key, region_name in color_to_region.items():
        if region_name not in masks:
            confidences[region_name] = 0.0
            continue
        
        mask = masks[region_name]
        region_size = mask.sum()
        coverage = region_size / total_pixels if total_pixels > 0 else 0.0
        
        # Parser confidence: based on region coverage
        # Good coverage: >5% of image = high confidence
        parser_conf = min(coverage * 20, 1.0)  # 5% coverage = 1.0
        
        # Color extraction confidence
        region_color = matched_colors.get(region_name, {})
        color_extracted = 1.0 if region_color.get("hex") else 0.0
        
        # Classifier confidence: if classifier says "NA", lower confidence
        classifier_pattern = classifier_colors.get(color_key, "NA")
        classifier_conf = 1.0 if classifier_pattern != "NA" else 0.3
        
        # Combined: weighted average
        # Parser quality (40%) + Color extraction (30%) + Classifier agreement (30%)
        combined = (parser_conf * 0.4 + color_extracted * 0.3 + classifier_conf * 0.3)
        confidences[region_name] = combined
    
    return confidences


def get_overall_color_confidence(confidences: Dict[str, float]) -> float:
    """Get overall color extraction confidence"""
    if not confidences:
        return 0.0
    # Average of shirt, pants, shoes (exclude head)
    clothing_regions = [confidences.get(r, 0.0) for r in ["shirt", "pants", "shoes"]]
    return sum(clothing_regions) / len(clothing_regions) if clothing_regions else 0.0


def extract_colors_guided_by_classifier(
    img_rgb: np.ndarray,
    masks: Dict[str, np.ndarray],
    classifier_colors: Dict[str, str]  # {"color_0": "pure-color", "color_1": "striped", "color_2": "NA"}
) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    """
    Extract colors using classifier predictions as primary guide.
    Uses classifier's color_0, color_1, color_2 to determine which regions to extract from.
    
    Strategy:
    - color_0 → shirt (top region)
    - color_1 → pants (lower region)
    - color_2 → shoes (footwear region)
    - Only extract RGB from regions that classifier identified (skip "NA")
    - Use pattern info (pure-color vs striped) to adjust extraction
    
    Args:
        img_rgb: Original image as numpy array [H, W, 3]
        masks: Dict with keys "head", "shirt", "pants", "shoes", each containing boolean mask
        classifier_colors: {"color_0": "pure-color", "color_1": "striped", "color_2": "NA"}
    
    Returns:
        {
            "shirt": {
                "hex": "#...",
                "name": "blue",
                "rgb": [r, g, b],
                "pattern": "pure-color",  # from classifier
                "classifier_source": "color_0"
            },
            ...
        }
    """
    # Direct mapping: classifier color predictions → regions
    color_to_region = {
        "color_0": "shirt",  # Primary top color
        "color_1": "pants",  # Primary bottom color
        "color_2": "shoes",  # Footwear color
    }
    
    result = {}
    
    # Initialize all regions
    for region_name in ["head", "shirt", "pants", "shoes"]:
        result[region_name] = {
            "hex": None,
            "name": None,
            "rgb": None,
            "pattern": None,
            "classifier_source": None
        }
    
    # Extract colors only from regions that classifier identified as "pure-color"
    for color_key, region_name in color_to_region.items():
        pattern = classifier_colors.get(color_key, "NA")
        
        # Skip if classifier says "NA" (no color detected)
        if pattern == "NA":
            continue
        
        # Only extract RGB colors when pattern is "pure-color"
        # For other patterns (striped, graphic, floral, etc.), don't extract a single color
        if pattern != "pure-color":
            # Store pattern info but don't extract RGB color
            result[region_name] = {
                "hex": None,
                "name": None,
                "rgb": None,
                "pattern": pattern,  # Store pattern info
                "classifier_source": color_key
            }
            continue
        
        # Skip if region mask doesn't exist or is empty
        if region_name not in masks or masks[region_name].sum() == 0:
            continue
        
        # Extract RGB color from this region (pass region_name to filter skin for pants)
        # Only do this for "pure-color" patterns
        region_color = extract_color_from_region(img_rgb, masks[region_name], region_name=region_name)
        
        if region_color["hex"] is not None:
            result[region_name] = {
                "hex": region_color["hex"],
                "name": region_color["name"],
                "rgb": region_color["rgb"],
                "pattern": pattern,  # Classifier's pattern prediction
                "classifier_source": color_key
            }
    
    # For head region, extract color if available (not guided by classifier)
    if "head" in masks and masks["head"].sum() > 0:
        head_color = extract_color_from_region(img_rgb, masks["head"], region_name="head")
        if head_color["hex"] is not None:
            result["head"] = {
                "hex": head_color["hex"],
                "name": head_color["name"],
                "rgb": head_color["rgb"],
                "pattern": None,
                "classifier_source": None
            }
    
    # Calculate confidence for each region
    confidences = calculate_color_extraction_confidence(masks, result, classifier_colors)
    
    return result, confidences


# =========================
# VISUALIZATION
# =========================

def visualize_parsed_regions(
    img_rgb: np.ndarray,
    masks: Dict[str, np.ndarray],
    save_path: str = "parsed_regions_visualization.png",
    show: bool = False,
    matched_colors: Optional[Dict[str, Dict]] = None,
    color_confidences: Optional[Dict[str, float]] = None
) -> None:
    """
    Visualize parsed clothing regions with color overlays using matplotlib.
    
    Args:
        img_rgb: Original image as numpy array [H, W, 3]
        masks: Dict with keys "head", "shirt", "pants", "shoes", each containing boolean mask
        save_path: Path to save visualization
        show: Whether to display the visualization
        matched_colors: Optional dict with color info per region (from extract_colors_guided_by_classifier)
        color_confidences: Optional dict with confidence scores per region
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[Warning] Matplotlib not available. Using simple PIL visualization instead.")
        visualize_parsed_regions_simple(img_rgb, masks, save_path)
        return
    
    # Region colors for visualization (semi-transparent)
    region_colors = {
        "head": [255, 0, 0, 128],      # Red (semi-transparent)
        "shirt": [0, 255, 0, 128],     # Green (semi-transparent)
        "pants": [0, 0, 255, 128],     # Blue (semi-transparent)
        "shoes": [255, 255, 0, 128],   # Yellow (semi-transparent)
    }
    
    # Create overlay image
    overlay = img_rgb.copy().astype(np.float32)
    
    # Apply colored overlays for each region
    for region_name, color in region_colors.items():
        if region_name in masks:
            mask = masks[region_name]
            if mask.sum() > 0:  # Only if region exists
                # Create colored overlay
                color_overlay = np.zeros_like(img_rgb, dtype=np.float32)
                color_overlay[mask] = color[:3]  # RGB only
                
                # Blend with original image
                alpha = color[3] / 255.0
                overlay[mask] = overlay[mask] * (1 - alpha) + color_overlay[mask] * alpha
    
    # Convert back to uint8
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Create visualization with matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Original image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Right: Parsed regions with overlay
    axes[1].imshow(overlay)
    axes[1].set_title("Parsed Regions (Overlay)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add legend with hex codes, color squares, and confidence
    legend_elements = []
    for region_name, color in region_colors.items():
        if region_name in masks and masks[region_name].sum() > 0:
            # Count pixels in region
            pixel_count = masks[region_name].sum()
            percentage = (pixel_count / masks[region_name].size) * 100
            
            # Build label with hex code and confidence
            label_parts = [f"{region_name.capitalize()}", f"({pixel_count} px, {percentage:.1f}%)"]
            
            # Get actual extracted color for the patch if available, otherwise use overlay color
            patch_color = [c/255.0 for c in color[:3]]
            hex_code = None
            
            if matched_colors and region_name in matched_colors:
                region_color_data = matched_colors[region_name]
                hex_code = region_color_data.get('hex')
                if hex_code:
                    # Parse hex code to RGB for patch color
                    hex_clean = hex_code.lstrip('#')
                    if len(hex_clean) == 6:
                        try:
                            r = int(hex_clean[0:2], 16)
                            g = int(hex_clean[2:4], 16)
                            b = int(hex_clean[4:6], 16)
                            patch_color = [r/255.0, g/255.0, b/255.0]
                            label_parts.append(f"Hex: {hex_code}")
                        except ValueError:
                            pass  # Use overlay color if hex parsing fails
            
            # Add confidence if available
            if color_confidences and region_name in color_confidences:
                conf = color_confidences[region_name]
                conf_percent = conf * 100
                label_parts.append(f"Conf: {conf_percent:.1f}%")
            
            label_text = "\n".join(label_parts)
            
            legend_elements.append(
                mpatches.Patch(
                    facecolor=patch_color,
                    label=label_text,
                    edgecolor='black',
                    linewidth=1.5
                )
            )
    
    if legend_elements:
        # Position legend with better spacing
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9, 
                      framealpha=0.95, handlelength=2.5, handletextpad=0.8,
                      borderpad=0.8, labelspacing=1.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Visualization] Saved parsed regions to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_colors_with_classifier(
    img_rgb: np.ndarray,
    masks: Dict[str, np.ndarray],
    matched_colors: Dict[str, Dict],
    save_path: str = "color_extraction_visualization.png"
) -> None:
    """
    Visualize extracted colors with classifier mappings.
    Shows which regions have colors and their classifier source (color_0, color_1, color_2).
    """
    print(f"[Color Visualization] Starting visualization...")
    print(f"[Color Visualization] Save path: {save_path}")
    
    try:
        if not MATPLOTLIB_AVAILABLE:
            print("[Color Visualization] ERROR: Matplotlib not available. Skipping color visualization.")
            return
        
        print("[Color Visualization] Matplotlib is available")
        
        # Validate inputs
        if img_rgb is None:
            print("[Color Visualization] ERROR: img_rgb is None")
            return
        if len(img_rgb.shape) != 3 or img_rgb.shape[2] != 3:
            print(f"[Color Visualization] ERROR: Invalid img_rgb shape: {img_rgb.shape}, expected (H, W, 3)")
            return
        
        h, w = img_rgb.shape[:2]
        print(f"[Color Visualization] Image dimensions: {h}x{w}")
        print(f"[Color Visualization] Image dtype: {img_rgb.dtype}, min: {img_rgb.min()}, max: {img_rgb.max()}")
        
        # If image is very large, resize it for visualization to avoid memory/rendering issues
        MAX_DISPLAY_SIZE = 1024
        if h > MAX_DISPLAY_SIZE or w > MAX_DISPLAY_SIZE:
            scale = min(MAX_DISPLAY_SIZE / h, MAX_DISPLAY_SIZE / w)
            new_h, new_w = int(h * scale), int(w * scale)
            print(f"[Color Visualization] Resizing image from {h}x{w} to {new_h}x{new_w} for display")
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(img_rgb)
            img_pil_resized = img_pil.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            img_rgb_display = np.array(img_pil_resized)
            # Also resize masks to match
            masks_display = {}
            for region_name, mask in masks.items():
                mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255)
                mask_pil_resized = mask_pil.resize((new_w, new_h), PILImage.Resampling.NEAREST)
                masks_display[region_name] = np.array(mask_pil_resized) > 128
            img_rgb = img_rgb_display
            masks = masks_display
            h, w = img_rgb.shape[:2]
            print(f"[Color Visualization] After resize: {h}x{w}")
        
        if masks is None:
            print("[Color Visualization] ERROR: masks is None")
            return
        print(f"[Color Visualization] Masks keys: {list(masks.keys())}")
        for region_name, mask in masks.items():
            if mask is not None:
                print(f"[Color Visualization]   {region_name}: shape={mask.shape}, dtype={mask.dtype}, sum={mask.sum()}")
            else:
                print(f"[Color Visualization]   {region_name}: None")
        
        if matched_colors is None:
            print("[Color Visualization] ERROR: matched_colors is None")
            return
        print(f"[Color Visualization] Matched colors keys: {list(matched_colors.keys())}")
        for region_name, color_data in matched_colors.items():
            print(f"[Color Visualization]   {region_name}: {color_data}")
        
        # Create figure with subplots - use reasonable size to avoid huge files
        print("[Color Visualization] Creating matplotlib figure...")
        fig = plt.figure(figsize=(16, 8))
        print(f"[Color Visualization] Figure created: {fig}")
        
        # 1. Original image
        print("[Color Visualization] Creating subplot 1: Original image...")
        ax1 = plt.subplot(2, 3, 1)
        # Ensure image is in correct format for imshow (uint8, RGB)
        img_display = img_rgb.copy()
        if img_display.dtype != np.uint8:
            img_display = np.clip(img_display, 0, 255).astype(np.uint8)
        print(f"[Color Visualization]   Displaying image: shape={img_display.shape}, dtype={img_display.dtype}")
        im1 = ax1.imshow(img_display)
        ax1.set_title("Original Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        print("[Color Visualization] Subplot 1 created and image displayed")
        
        # 2. Parsed regions overlay
        print("[Color Visualization] Creating overlay image...")
        overlay = img_rgb.copy().astype(np.float32)
        region_colors_viz = {
            "head": [255, 0, 0, 128],      # Red
            "shirt": [0, 255, 0, 128],     # Green
            "pants": [0, 0, 255, 128],     # Blue
            "shoes": [255, 255, 0, 128],   # Yellow
        }
        
        for region_name, color in region_colors_viz.items():
            if region_name in masks and masks[region_name].sum() > 0:
                mask = masks[region_name]
                print(f"[Color Visualization]   Applying overlay for {region_name}: {mask.sum()} pixels")
                color_overlay = np.zeros_like(img_rgb, dtype=np.float32)
                color_overlay[mask] = color[:3]
                alpha = color[3] / 255.0
                overlay[mask] = overlay[mask] * (1 - alpha) + color_overlay[mask] * alpha
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        print(f"[Color Visualization] Overlay created: shape={overlay.shape}, dtype={overlay.dtype}")
        ax2 = plt.subplot(2, 3, 2)
        print(f"[Color Visualization]   Displaying overlay: shape={overlay.shape}, dtype={overlay.dtype}")
        im2 = ax2.imshow(overlay)
        ax2.set_title("Parsed Regions", fontsize=14, fontweight='bold')
        ax2.axis('off')
        print("[Color Visualization] Subplot 2 created and overlay displayed")
        
        # 3-6. Individual region colors with classifier info
        print("[Color Visualization] Creating color swatches for regions...")
        regions_to_show = ["shirt", "pants", "shoes", "head"]
        positions = [(2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]
        
        for idx, region_name in enumerate(regions_to_show):
            print(f"[Color Visualization] Processing region {idx+1}/4: {region_name}")
            ax = plt.subplot(*positions[idx])
            
            region_data = matched_colors.get(region_name, {})
            hex_color = region_data.get("hex")
            color_name = region_data.get("name", "N/A")
            pattern = region_data.get("pattern", "N/A")
            classifier_source = region_data.get("classifier_source", "N/A")
            
            print(f"[Color Visualization]   {region_name}: hex={hex_color}, name={color_name}, pattern={pattern}, source={classifier_source}")
            
            if hex_color and hex_color != "None":
                try:
                    print(f"[Color Visualization]   Creating color swatch for {region_name}...")
                    # Create color swatch
                    color_swatch = np.ones((150, 150, 3), dtype=np.uint8)
                    # Parse hex color
                    hex_clean = hex_color.lstrip('#')
                    print(f"[Color Visualization]   Parsing hex: '{hex_color}' -> cleaned: '{hex_clean}'")
                    if len(hex_clean) >= 6:
                        r = int(hex_clean[0:2], 16)
                        g = int(hex_clean[2:4], 16)
                        b = int(hex_clean[4:6], 16)
                        print(f"[Color Visualization]   RGB values: ({r}, {g}, {b})")
                        color_swatch[:, :] = [r, g, b]
                        
                        print(f"[Color Visualization]   Displaying color swatch: shape={color_swatch.shape}, dtype={color_swatch.dtype}")
                        im_swatch = ax.imshow(color_swatch)
                        ax.set_title(f"{region_name.capitalize()}\n{color_name}", fontsize=11, fontweight='bold')
                        
                        # Add classifier info as text (positioned better)
                        info_text = f"Hex: {hex_color}\n"
                        if classifier_source and classifier_source != "N/A":
                            info_text += f"Source: {classifier_source}\n"
                        if pattern and pattern != "N/A":
                            info_text += f"Pattern: {pattern}"
                        
                        ax.text(75, 180, info_text, ha='center', va='top', fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5))
                        print(f"[Color Visualization]   Color swatch created for {region_name}")
                    else:
                        raise ValueError(f"Invalid hex color length: {len(hex_clean)} (expected >= 6)")
                except Exception as e:
                    # Fallback if color parsing fails
                    print(f"[Color Visualization]   ERROR parsing color for {region_name}: {e}")
                    ax.text(75, 75, f"{region_name.capitalize()}\nColor parse error", 
                           ha='center', va='center', fontsize=11,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    ax.set_facecolor('lightgray')
            else:
                # No color detected
                print(f"[Color Visualization]   No color detected for {region_name}")
                ax.text(75, 75, f"{region_name.capitalize()}\nNo color detected", 
                       ha='center', va='center', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax.set_facecolor('lightgray')
            
            ax.axis('off')
        
        # Use subplots_adjust instead of tight_layout for better control
        print("[Color Visualization] Adjusting subplot layout...")
        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.15, hspace=0.2)
        
        # Force figure to render before saving
        print("[Color Visualization] Forcing figure to draw...")
        fig.canvas.draw()
        
        print(f"[Color Visualization] Saving figure to: {save_path}")
        print(f"[Color Visualization] Figure size: {fig.get_size_inches()}")
        print(f"[Color Visualization] DPI: 100, bbox_inches='tight', pad_inches=0.1")
        print(f"[Color Visualization] Facecolor: white")
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1, facecolor='white', format='png')
        print(f"[Color Visualization] Savefig completed")
        
        # Verify file was created
        import os
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"[Color Visualization] SUCCESS: File saved! Size: {file_size} bytes ({file_size/1024:.2f} KB)")
        else:
            print(f"[Color Visualization] ERROR: File was not created at {save_path}")
        
        plt.close()
        print("[Color Visualization] Figure closed, visualization complete")
    except Exception as e:
        print(f"[Color Visualization] ERROR: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        # Try to close figure even if error occurred
        try:
            plt.close('all')
            print("[Color Visualization] Closed all figures after error")
        except:
            pass


def visualize_parsed_regions_simple(
    img_rgb: np.ndarray,
    masks: Dict[str, np.ndarray],
    save_path: str = "parsed_regions_simple.png"
) -> None:
    """
    Simple visualization using PIL (no matplotlib required).
    Creates a side-by-side comparison.
    """
    h, w = img_rgb.shape[:2]
    
    # Create a new image with 2x width (original + overlay)
    result_img = Image.new('RGB', (w * 2, h))
    
    # Left side: Original image
    img_pil = Image.fromarray(img_rgb)
    result_img.paste(img_pil, (0, 0))
    
    # Right side: Overlay with regions
    overlay = img_rgb.copy().astype(np.float32)
    
    region_colors = {
        "head": [255, 0, 0],      # Red
        "shirt": [0, 255, 0],     # Green
        "pants": [0, 0, 255],     # Blue
        "shoes": [255, 255, 0],   # Yellow
    }
    
    # Apply colored overlays
    for region_name, color in region_colors.items():
        if region_name in masks:
            mask = masks[region_name]
            if mask.sum() > 0:
                alpha = 0.4  # 40% opacity
                overlay[mask] = overlay[mask] * (1 - alpha) + np.array(color) * alpha
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay)
    result_img.paste(overlay_pil, (w, 0))
    
    # Add text labels
    draw = ImageDraw.Draw(result_img)
    try:
        # Try to use a default font
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Add labels
    y_offset = 10
    for region_name, color in region_colors.items():
        if region_name in masks and masks[region_name].sum() > 0:
            pixel_count = masks[region_name].sum()
            percentage = (pixel_count / masks[region_name].size) * 100
            label = f"{region_name.capitalize()}: {pixel_count}px ({percentage:.1f}%)"
            draw.text((w + 10, y_offset), label, fill=tuple(color), font=font)
            y_offset += 25
    
    result_img.save(save_path)
    print(f"[Visualization] Saved simple visualization to: {save_path}")

