"""
Improved Color Extraction Methods (No Parsing Required)
Multiple robust methods for extracting clothing colors without relying on parsing masks.
"""

import numpy as np
from PIL import Image
from typing import Dict, Optional, List, Tuple
import colorsys

# Optional: for K-Means clustering
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional: for saliency detection
try:
    from skimage import segmentation, filters
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def rgb_to_basic_color_name(r: int, g: int, b: int) -> str:
    """Convert RGB to basic color name"""
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
# METHOD 1: Improved Vertical Regions (Better than current)
# =========================

def extract_colors_vertical_improved(img_np: np.ndarray) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Improved vertical region color extraction with better region definitions.
    More accurate than simple vertical bands.
    """
    h, w, _ = img_np.shape
    
    # Use wider central region (50% instead of 30%)
    x0 = int(w * 0.25)
    x1 = int(w * 0.75)
    
    # More refined vertical regions
    regions = {
        "top": {
            "y_range": (int(h * 0.15), int(h * 0.50)),  # Upper body
            "exclude_bottom": int(h * 0.40)  # Don't include lower parts
        },
        "outer": {
            "y_range": (int(h * 0.10), int(h * 0.55)),  # Outer garments (slightly higher)
            "exclude_bottom": int(h * 0.45)
        },
        "pants": {
            "y_range": (int(h * 0.45), int(h * 0.85)),  # Lower body
            "exclude_top": int(h * 0.50)  # Don't include upper parts
        },
        "shoes": {
            "y_range": (int(h * 0.85), int(h * 0.98)),  # Feet
            "exclude_top": int(h * 0.88)
        }
    }
    
    results = {}
    
    for region_name, config in regions.items():
        y0, y1 = config["y_range"]
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        
        if y1 <= y0:
            results[region_name] = {"hex": None, "name": None}
            continue
        
        # Extract region
        region = img_np[y0:y1, x0:x1].copy()
        
        # Remove background (very dark or very light pixels at edges)
        # This helps focus on clothing
        if region.size > 0:
            # Use median instead of mean (more robust to outliers)
            # Reshape to [N, 3] for easier processing
            pixels = region.reshape(-1, 3)
            
            # Filter out extreme values (likely background)
            v_values = pixels.max(axis=1) - pixels.min(axis=1)  # Rough brightness
            valid_mask = (v_values > 10) & (v_values < 240)  # Not pure black/white
            valid_pixels = pixels[valid_mask]
            
            if len(valid_pixels) > 0:
                # Use median for robustness
                median_color = np.median(valid_pixels, axis=0).astype(np.uint8)
                r, g, b = median_color.tolist()
                hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
                color_name = rgb_to_basic_color_name(r, g, b)
                results[region_name] = {"hex": hex_color, "name": color_name}
            else:
                results[region_name] = {"hex": None, "name": None}
        else:
            results[region_name] = {"hex": None, "name": None}
    
    return results


# =========================
# METHOD 2: Keypoint-Based Regions
# =========================

def extract_colors_keypoint_based(img_np: np.ndarray, keypoints: Optional[Dict] = None) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Extract colors based on keypoint positions.
    Uses human pose keypoints to identify clothing regions more accurately.
    """
    h, w, _ = img_np.shape
    results = {}
    
    if keypoints is None:
        # Fallback to vertical regions if no keypoints
        return extract_colors_vertical_improved(img_np)
    
    # Keypoint indices (COCO format, 21 keypoints)
    # 0: nose, 1: left_eye, 2: right_eye, 5: left_shoulder, 6: right_shoulder
    # 11: left_hip, 12: right_hip, 15: left_ankle, 16: right_ankle
    
    # Extract keypoint coordinates
    kp_coords = {}
    if 'left_shoulder' in keypoints:
        kp_coords['shoulder_y'] = int(keypoints['left_shoulder'][1] * h)
    if 'right_hip' in keypoints:
        kp_coords['hip_y'] = int(keypoints['right_hip'][1] * h)
    if 'left_ankle' in keypoints:
        kp_coords['ankle_y'] = int(keypoints['left_ankle'][1] * h)
    
    # Define regions based on keypoints
    x0, x1 = int(w * 0.25), int(w * 0.75)
    
    # Top region: from head to shoulders
    if 'shoulder_y' in kp_coords:
        top_y0 = max(0, int(h * 0.10))
        top_y1 = min(h, kp_coords['shoulder_y'] + int(h * 0.10))
        top_region = img_np[top_y0:top_y1, x0:x1]
        if top_region.size > 0:
            pixels = top_region.reshape(-1, 3)
            valid_pixels = pixels[(pixels.max(axis=1) - pixels.min(axis=1)) > 10]
            if len(valid_pixels) > 0:
                median = np.median(valid_pixels, axis=0).astype(np.uint8)
                r, g, b = median.tolist()
                results["top"] = {
                    "hex": "#{:02X}{:02X}{:02X}".format(r, g, b),
                    "name": rgb_to_basic_color_name(r, g, b)
                }
    
    # Pants region: from hips to ankles
    if 'hip_y' in kp_coords and 'ankle_y' in kp_coords:
        pants_y0 = max(0, kp_coords['hip_y'] - int(h * 0.05))
        pants_y1 = min(h, kp_coords['ankle_y'])
        pants_region = img_np[pants_y0:pants_y1, x0:x1]
        if pants_region.size > 0:
            pixels = pants_region.reshape(-1, 3)
            valid_pixels = pixels[(pixels.max(axis=1) - pixels.min(axis=1)) > 10]
            if len(valid_pixels) > 0:
                median = np.median(valid_pixels, axis=0).astype(np.uint8)
                r, g, b = median.tolist()
                results["pants"] = {
                    "hex": "#{:02X}{:02X}{:02X}".format(r, g, b),
                    "name": rgb_to_basic_color_name(r, g, b)
                }
    
    # Shoes region: below ankles
    if 'ankle_y' in kp_coords:
        shoes_y0 = max(0, kp_coords['ankle_y'])
        shoes_y1 = min(h, int(h * 0.98))
        shoes_region = img_np[shoes_y0:shoes_y1, x0:x1]
        if shoes_region.size > 0:
            pixels = shoes_region.reshape(-1, 3)
            valid_pixels = pixels[(pixels.max(axis=1) - pixels.min(axis=1)) > 10]
            if len(valid_pixels) > 0:
                median = np.median(valid_pixels, axis=0).astype(np.uint8)
                r, g, b = median.tolist()
                results["shoes"] = {
                    "hex": "#{:02X}{:02X}{:02X}".format(r, g, b),
                    "name": rgb_to_basic_color_name(r, g, b)
                }
    
    # Fill missing regions with vertical method
    vertical_colors = extract_colors_vertical_improved(img_np)
    for region in ["top", "pants", "shoes", "outer"]:
        if region not in results:
            results[region] = vertical_colors.get(region, {"hex": None, "name": None})
    
    return results


# =========================
# METHOD 3: Color Clustering (K-Means)
# =========================

def extract_colors_kmeans(img_np: np.ndarray, n_clusters: int = 5) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Extract dominant colors using K-Means clustering.
    Finds the most prominent colors in the image and assigns them to regions.
    """
    if not SKLEARN_AVAILABLE:
        # Fallback to vertical method if sklearn not available
        return extract_colors_vertical_improved(img_np)
    
    h, w, _ = img_np.shape
    
    # Reshape image to [N, 3]
    pixels = img_np.reshape(-1, 3).astype(np.float32)
    
    # Filter out extreme values (background, shadows)
    brightness = pixels.mean(axis=1)
    valid_mask = (brightness > 20) & (brightness < 235)
    valid_pixels = pixels[valid_mask]
    
    if len(valid_pixels) < n_clusters:
        # Fallback to vertical method
        return extract_colors_vertical_improved(img_np)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(valid_pixels)
    
    # Get cluster centers (dominant colors)
    colors = kmeans.cluster_centers_.astype(np.uint8)
    
    # Get cluster sizes (how many pixels belong to each)
    labels = kmeans.labels_
    cluster_sizes = np.bincount(labels)
    
    # Sort by size (most common first)
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    
    # Map colors to regions based on vertical position
    # Find where each cluster is most common
    labels_full = np.full(h * w, -1, dtype=int)  # Use -1 for invalid pixels
    labels_full[valid_mask] = labels
    
    # Track which pixels are valid in the full image
    valid_mask_full = np.zeros((h, w), dtype=bool)
    valid_mask_full.reshape(-1)[valid_mask] = True
    
    # Define vertical regions
    top_region_mask = np.zeros((h, w), dtype=bool)
    top_region_mask[:int(h*0.5), :] = True
    
    pants_region_mask = np.zeros((h, w), dtype=bool)
    pants_region_mask[int(h*0.5):int(h*0.9), :] = True
    
    shoes_region_mask = np.zeros((h, w), dtype=bool)
    shoes_region_mask[int(h*0.85):, :] = True
    
    results = {}
    
    # Find dominant color in each region
    for region_name, region_mask in [("top", top_region_mask), ("pants", pants_region_mask), ("shoes", shoes_region_mask)]:
        # Only consider valid pixels in this region
        valid_region_mask = region_mask & valid_mask_full
        region_labels = labels_full[valid_region_mask.reshape(-1)]
        
        # Filter out invalid labels (-1)
        valid_region_labels = region_labels[region_labels >= 0]
        
        if len(valid_region_labels) > 0:
            # Find most common cluster in this region
            region_clusters, counts = np.unique(valid_region_labels, return_counts=True)
            if len(region_clusters) > 0:
                dominant_cluster = region_clusters[np.argmax(counts)]
                r, g, b = colors[dominant_cluster].tolist()
                results[region_name] = {
                    "hex": "#{:02X}{:02X}{:02X}".format(r, g, b),
                    "name": rgb_to_basic_color_name(r, g, b)
                }
            else:
                results[region_name] = {"hex": None, "name": None}
        else:
            results[region_name] = {"hex": None, "name": None}
    
    # Outer region: use second most common color from top region
    if "top" in results and results["top"]["hex"]:
        # Find second most common cluster in top region (only valid pixels)
        valid_top_mask = top_region_mask & valid_mask_full
        top_labels = labels_full[valid_top_mask.reshape(-1)]
        valid_top_labels = top_labels[top_labels >= 0]
        if len(valid_top_labels) > 0:
            top_clusters, top_counts = np.unique(valid_top_labels, return_counts=True)
            if len(top_clusters) > 1:
                # Get second most common
                sorted_top = np.argsort(top_counts)[::-1]
                second_cluster = top_clusters[sorted_top[1]]
                r, g, b = colors[second_cluster].tolist()
                results["outer"] = {
                    "hex": "#{:02X}{:02X}{:02X}".format(r, g, b),
                    "name": rgb_to_basic_color_name(r, g, b)
                }
    
    return results


# =========================
# METHOD 4: Saliency-Based (Automatic Clothing Detection)
# =========================

def extract_colors_saliency_based(img_np: np.ndarray) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Extract colors from salient (prominent) regions in the image.
    Automatically finds clothing regions without parsing.
    """
    if not SKIMAGE_AVAILABLE:
        # Fallback to vertical method
        return extract_colors_vertical_improved(img_np)
    
    h, w, _ = img_np.shape
    
    # Convert to grayscale for saliency
    gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Simple saliency: edges and texture
    edges = filters.sobel(gray)
    saliency = edges / (edges.max() + 1e-6)
    
    # Threshold to get prominent regions
    threshold = np.percentile(saliency, 70)  # Top 30% most salient
    salient_mask = saliency > threshold
    
    # Combine with vertical regions
    vertical_colors = extract_colors_vertical_improved(img_np)
    
    # Refine colors using saliency
    results = {}
    for region_name in ["top", "pants", "shoes", "outer"]:
        if region_name in vertical_colors and vertical_colors[region_name]["hex"]:
            results[region_name] = vertical_colors[region_name]
        else:
            results[region_name] = {"hex": None, "name": None}
    
    return results


# =========================
# METHOD 5: Combined Approach (Best Results)
# =========================

def extract_colors_combined(img_np: np.ndarray, 
                           keypoints: Optional[Dict] = None,
                           use_kmeans: bool = True,
                           use_saliency: bool = False) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Combines multiple methods for robust color extraction.
    Uses voting/consensus from different methods.
    """
    methods_results = []
    
    # Method 1: Improved vertical regions
    methods_results.append(("vertical", extract_colors_vertical_improved(img_np)))
    
    # Method 2: Keypoint-based (if available)
    if keypoints:
        methods_results.append(("keypoint", extract_colors_keypoint_based(img_np, keypoints)))
    
    # Method 3: K-Means clustering
    if use_kmeans:
        try:
            methods_results.append(("kmeans", extract_colors_kmeans(img_np)))
        except:
            pass
    
    # Method 4: Saliency-based (if available)
    if use_saliency and SKIMAGE_AVAILABLE:
        methods_results.append(("saliency", extract_colors_saliency_based(img_np)))
    
    # Combine results: use most common color name across methods
    final_results = {}
    
    for region_name in ["top", "pants", "shoes", "outer"]:
        color_votes = {}
        hex_values = []
        
        for method_name, method_result in methods_results:
            if region_name in method_result and method_result[region_name].get("hex"):
                hex_val = method_result[region_name]["hex"]
                color_name = method_result[region_name].get("name", "unknown")
                hex_values.append(hex_val)
                color_votes[color_name] = color_votes.get(color_name, 0) + 1
        
        if hex_values:
            # Use most common color name
            if color_votes:
                most_common_color = max(color_votes, key=color_votes.get)
            else:
                most_common_color = "unknown"
            
            # Use median hex value (most stable)
            # Convert hex to RGB, get median, convert back
            rgb_values = []
            for hex_val in hex_values:
                hex_val = hex_val.lstrip('#')
                r = int(hex_val[0:2], 16)
                g = int(hex_val[2:4], 16)
                b = int(hex_val[4:6], 16)
                rgb_values.append([r, g, b])
            
            median_rgb = np.median(rgb_values, axis=0).astype(np.uint8)
            r, g, b = median_rgb.tolist()
            final_hex = "#{:02X}{:02X}{:02X}".format(r, g, b)
            
            # Use most common color name or compute from median
            if most_common_color != "unknown":
                final_name = most_common_color
            else:
                final_name = rgb_to_basic_color_name(r, g, b)
            
            final_results[region_name] = {
                "hex": final_hex,
                "name": final_name
            }
        else:
            final_results[region_name] = {"hex": None, "name": None}
    
    return final_results

