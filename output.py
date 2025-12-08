import os
import argparse
import json
import numpy as np
from PIL import Image
from typing import Optional, Dict

# (Optional but useful for your OpenMP issue)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torchvision import transforms

# For loading .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # python-dotenv not installed, will use system environment variables

# For ChatGPT API
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[Warning] OpenAI package not installed. Install with: pip install openai")

# From your classification file (test.py)
from test import MultiTaskFashionModel, infer_single

# From your parsing/color file (Parsing_train.py)
from Parsing_train import (
    load_trained_parsing_model,
    predict_parsing_mask,
    analyze_regions_from_pred,
)

# Pre-trained parser for clothing regions
from pretrained_parser import (
    parse_clothing_regions,
    extract_colors_guided_by_classifier,
    visualize_colors_with_classifier,
    get_overall_color_confidence,
)


def generate_clothing_categories(pred_mask, parsing_region_colors, decoded_attrs):
    """
    Maps parsing regions + colors + attributes to category names using ALL classifier information.
    
    Categories:
    - T-shirt: Short/long-sleeve casual shirt
    - Polo: Shirt with lapel neckline
    - Formal_Shirt: Long-sleeve shirt with standing/collared neckline
    - Tank_Top: Sleeveless top
    - Outer: Outer layer garment (hoodie, jacket, sweater, etc.)
    - Long_jeans: Denim pants with long length
    - Short_jeans: Denim pants with short length
    - long_sweat_pants: Knitted/cotton pants with long length (including leggings)
    - short_sweat_pants: Knitted/cotton pants with short length
    - Shorts: Short-length pants (non-denim, non-leggings)
    - Shoes: Footwear
    
    Uses ALL classifier attributes:
    - sleeve_length: Determines T-shirt vs Formal_Shirt vs Tank_Top
    - neckline: Determines Polo (lapel) vs Formal_Shirt (standing)
    - outer_cardigan: Determines Outer vs inner layer
    - lower_length: Determines Shorts vs long pants
    - socks: Detects leggings (socks='leggings')
    - fabric_0, fabric_1: Determines material (denim, knitted, cotton) for jeans/sweat pants
    
    Only uses colors when pattern is 'pure-color' (no RGB for striped/graphic patterns).
    
    Returns list like: ["grey T-shirt", "blue Long_jeans", "black Shoes"]
    """
    categories = []
    
    # Helper function to get color with fallback
    def get_color(region_key):
        region_data = parsing_region_colors.get(region_key, {})
        # Only use color if pattern is pure-color
        if region_data.get('pattern') == 'pure-color':
            return region_data.get('name')
        return None
    
    # Get fabric information
    # fabric_0: top/shirt fabric, fabric_1: pants/lower fabric, fabric_2: shoes/third fabric
    top_fabric = decoded_attrs.get('fabric_0') if decoded_attrs.get('fabric_0') != 'NA' else None
    pants_fabric = decoded_attrs.get('fabric_1') if decoded_attrs.get('fabric_1') != 'NA' else None
    # primary_fabric is used for top garments (from fabric_0)
    primary_fabric = top_fabric
    
    sleeve_length = decoded_attrs.get('sleeve_length', 'NA')
    neckline = decoded_attrs.get('neckline', 'NA')
    outer_cardigan = decoded_attrs.get('outer_cardigan', 'NA')
    lower_length = decoded_attrs.get('lower_length', 'NA')
    socks = decoded_attrs.get('socks', 'NA')
    
    # === TOP/OUTER GARMENTS ===
    # Check for outer garments first (jackets, hoodies)
    outer_color = get_color('outer')
    top_color = get_color('top')
    shirt_color = get_color('shirt') if 'shirt' in parsing_region_colors else top_color
    
    # Determine if outer layer exists (check parsing mask for label 2 = outer)
    has_outer_in_mask = pred_mask is not None and np.isin(pred_mask, [2]).sum() > 0
    
    # Check if top/shirt region exists (even without color)
    # Check if top/shirt has pattern or exists in parsing results
    top_region_exists = (
        'top' in parsing_region_colors or 
        'shirt' in parsing_region_colors or
        (pred_mask is not None and np.isin(pred_mask, [1, 4, 21]).sum() > 0)  # Labels 1,4,21 = top regions
    )
    top_has_pattern = (
        parsing_region_colors.get('top', {}).get('pattern') is not None or
        parsing_region_colors.get('shirt', {}).get('pattern') is not None
    )
    # Top detected if region exists OR classifier detected sleeve_length
    top_detected = top_region_exists or (sleeve_length != 'NA' and sleeve_length is not None)
    
    # Outer layer logic: use outer_cardigan attribute to determine if outer layer exists
    # If outer_cardigan is 'yes', treat the top/outer as outer layer
    if outer_cardigan == 'yes' and (outer_color or top_color or top_detected):
        outer_layer_color = outer_color if outer_color else (top_color if top_color else None)
        # All outer layers categorized as "Outer"
        if outer_layer_color:
            categories.append(f"{outer_layer_color} Outer")
        else:
            categories.append("Outer")
    
    # If outer_cardigan is 'no', check if we have a separate outer layer in the mask
    elif has_outer_in_mask and outer_color:
        # Separate outer layer detected - categorize as "Outer"
        categories.append(f"{outer_color} Outer")
    
    # Check for tops (shirts, t-shirts, polo, tank top, sweater)
    # Add inner layer if:
    # 1. outer_cardigan is 'no' (inner layer exists)
    # 2. OR we have separate top color different from outer
    # 3. OR no outer layer was detected
    # 4. OR top region detected (even without color)
    if top_color or top_detected:
        # Determine if we should add inner layer
        # If outer_cardigan is 'yes', we already added it as outer, don't add again
        # Unless colors are different (separate layers) OR top detected separately
        add_inner_layer = True
        if outer_cardigan == 'yes':
            # Only add if we have a separate color for inner layer OR top detected separately
            add_inner_layer = (
                (outer_color is not None and outer_color != top_color) or 
                (outer_color is None and has_outer_in_mask == False) or
                (not outer_color and top_detected and top_color is None)
            )
        
        if add_inner_layer:
            top_type = None
            
            # Tank Top: sleeveless
            if sleeve_length == 'sleeveless':
                top_type = "Tank_Top"
            
            # Polo: lapel neckline
            elif neckline == 'lapel':
                top_type = "Polo"
            
            # T-shirt: short-sleeve, casual neckline
            elif sleeve_length == 'short-sleeve':
                if neckline in ['round', 'square', 'V-shape', 'NA']:
                    top_type = "T-shirt"
                else:
                    top_type = "Polo"  # Fallback if lapel but short-sleeve
            
            # Long-sleeve shirts: determine between Formal_Shirt, Outer, T-shirt
            elif sleeve_length in ['long-sleeve', 'medium-sleeve']:
                # Outer: knitted fabric and not outer layer (sweater-like)
                if primary_fabric == 'knitted' and outer_cardigan != 'yes':
                    top_type = "Outer"
                # Formal Shirt: standing neckline indicates formal/collared shirt
                elif neckline == 'standing':
                    top_type = "Formal_Shirt"
                # Long-sleeve casual: could be T-shirt or casual shirt
                elif neckline in ['round', 'square', 'V-shape', 'NA']:
                    top_type = "T-shirt"  # Long-sleeve T-shirt
                else:
                    top_type = "Formal_Shirt"  # Other necklines suggest more formal
            
            # Default fallback
            if top_type is None:
                if sleeve_length == 'short-sleeve':
                    top_type = "T-shirt"
                elif sleeve_length in ['long-sleeve', 'medium-sleeve']:
                    if primary_fabric == 'knitted':
                        top_type = "Outer"
                    else:
                        top_type = "Formal_Shirt"
                else:
                    top_type = "T-shirt"
            
            if top_type:
                # Add with color if available, otherwise without
                if top_color:
                    categories.append(f"{top_color} {top_type}")
                else:
                    categories.append(top_type)
    
    # === BOTTOM GARMENTS ===
    pants_color = get_color('pants')
    
    if pants_color:
        bottom_type = None
        
        # Priority 1: Check if leggings (from socks attribute)
        if socks == 'leggings':
            # Leggings are treated as sweat pants in this context
            if lower_length in ['three-point', 'medium-short']:
                bottom_type = "short_sweat_pants"
            else:
                bottom_type = "long_sweat_pants"
        
        # Priority 2: Shorts (short length, not leggings)
        elif lower_length in ['three-point', 'medium-short']:
            bottom_type = "Shorts"
        
        # Priority 3: Jeans (denim fabric)
        # Use fabric_1 for pants (lower garment fabric)
        elif pants_fabric == 'denim':
            if lower_length == 'long':
                bottom_type = "Long_jeans"
            else:
                # Short jeans (denim but not long length)
                bottom_type = "Short_jeans"
        
        # Priority 4: Sweat pants (knitted/cotton fabric)
        # Use fabric_1 for pants (lower garment fabric)
        elif pants_fabric in ['knitted', 'cotton']:
            if lower_length == 'long':
                bottom_type = "long_sweat_pants"
            else:
                bottom_type = "short_sweat_pants"
        
        # Default classification based on length
        else:
            if lower_length == 'long':
                # Default to long sweat pants if no fabric info
                bottom_type = "long_sweat_pants"
            elif lower_length in ['three-point', 'medium-short']:
                bottom_type = "Shorts"
            else:
                # Default fallback
                bottom_type = "long_sweat_pants"
        
        if bottom_type:
            categories.append(f"{pants_color} {bottom_type}")
    
    # === SHOES ===
    # Check if shoes were detected (either by parsing mask or region colors)
    shoes_detected = False
    if pred_mask is not None:
        shoes_detected = np.isin(pred_mask, [11]).sum() > 0  # Label 11 = shoes
    # Also check if shoes region exists in parsing results
    if not shoes_detected and 'shoes' in parsing_region_colors:
        shoes_region_data = parsing_region_colors.get('shoes', {})
        # Shoes detected if region data exists (even if no color extracted)
        shoes_detected = shoes_region_data is not None
    
    shoes_color = get_color('shoes')
    if shoes_detected:
        if shoes_color:
            categories.append(f"{shoes_color} Shoes")
        else:
            # Shoes detected but no color available - add without color
            categories.append("Shoes")
    
    return categories if categories else ["clothing"]


def format_style_data_for_chatgpt(
    categories, 
    parsing_region_colors, 
    decoded_attrs,
    color_confidences: Optional[Dict[str, float]] = None,
    confidence_threshold: float = 0.5
):
    """
    Formats the detailed clothing attributes into a structured description for ChatGPT.
    Always includes classifier predictions (attributes, fabrics, patterns).
    Only includes color RGB values if confidence is above threshold.
    
    Args:
        categories: List of category strings
        parsing_region_colors: Dict with region color info
        decoded_attrs: Classifier predictions (always included)
        color_confidences: Dict with confidence scores per region
        confidence_threshold: Minimum confidence to include color RGB (default 0.5)
    """
    # Build detailed attribute description
    details = []
    
    # === SHAPE ATTRIBUTES ===
    details.append("=== SHAPE & FIT ATTRIBUTES ===")
    if decoded_attrs.get('sleeve_length') and decoded_attrs.get('sleeve_length') != 'NA':
        details.append(f"Sleeve Length: {decoded_attrs['sleeve_length']}")
    if decoded_attrs.get('lower_length') and decoded_attrs.get('lower_length') != 'NA':
        details.append(f"Lower Garment Length: {decoded_attrs['lower_length']}")
    if decoded_attrs.get('neckline') and decoded_attrs.get('neckline') != 'NA':
        details.append(f"Neckline: {decoded_attrs['neckline']}")
    if decoded_attrs.get('upper_cover_navel') and decoded_attrs.get('upper_cover_navel') != 'NA':
        details.append(f"Upper Covers Navel: {decoded_attrs['upper_cover_navel']}")
    if decoded_attrs.get('outer_cardigan') and decoded_attrs.get('outer_cardigan') != 'NA':
        details.append(f"Outer Cardigan: {decoded_attrs['outer_cardigan']}")
    
    # === FABRIC & MATERIAL ===
    details.append("\n=== FABRIC & MATERIAL ===")
    fabrics = []
    for i in range(3):
        fabric_key = f'fabric_{i}'
        if decoded_attrs.get(fabric_key) and decoded_attrs.get(fabric_key) != 'NA':
            fabrics.append(decoded_attrs[fabric_key])
    if fabrics:
        details.append(f"Fabrics: {', '.join(fabrics)}")
    
    # === COLOR PATTERNS ===
    details.append("\n=== COLOR PATTERNS ===")
    color_patterns = []
    for i in range(3):
        color_key = f'color_{i}'
        if decoded_attrs.get(color_key) and decoded_attrs.get(color_key) != 'NA':
            color_patterns.append(decoded_attrs[color_key])
    if color_patterns:
        details.append(f"Color Patterns: {', '.join(color_patterns)}")
    
    # === REGION-SPECIFIC COLORS ===
    # Only include RGB colors if confidence is high enough
    if color_confidences:
        overall_conf = get_overall_color_confidence(color_confidences)
        
        if overall_conf >= confidence_threshold:
            details.append("\n=== REGION COLORS (from segmentation) ===")
            region_order = ['shirt', 'pants', 'shoes', 'outer', 'top', 'leggings', 'skirt']
            for region in region_order:
                region_data = parsing_region_colors.get(region, {})
                region_conf = color_confidences.get(region, 0.0)
                
                # Only include if confidence is above threshold AND pattern is pure-color
                pattern = region_data.get('pattern')
                if region_data.get('name') and region_conf >= confidence_threshold and pattern == 'pure-color':
                    hex_color = region_data.get('hex', 'N/A')
                    match_info = ""
                    if region_data.get('classifier_source'):
                        match_info = f" [source: {region_data['classifier_source']}]"
                    details.append(f"{region.capitalize()}: {region_data['name']} ({hex_color}){match_info}")
                elif pattern and pattern != 'pure-color':
                    # Include pattern info for non-pure-color patterns
                    details.append(f"{region.capitalize()}: pattern={pattern} (no single RGB color available)")
        else:
            details.append("\n=== REGION COLORS ===")
            details.append(f"[Note: Color extraction confidence ({overall_conf:.1%}) below threshold ({confidence_threshold:.1%}). "
                          f"Only classifier color patterns are available. RGB color values may be unreliable.]")
    else:
        # Fallback: include colors anyway if no confidence info available
        details.append("\n=== REGION COLORS (from segmentation) ===")
        region_order = ['shirt', 'pants', 'shoes', 'outer', 'top', 'leggings', 'skirt']
        for region in region_order:
            region_data = parsing_region_colors.get(region, {})
            pattern = region_data.get('pattern')
            if region_data.get('name') and pattern == 'pure-color':
                hex_color = region_data.get('hex', 'N/A')
                match_info = ""
                if region_data.get('classifier_source'):
                    match_info = f" [source: {region_data['classifier_source']}]"
                details.append(f"{region.capitalize()}: {region_data['name']} ({hex_color}){match_info}")
            elif pattern and pattern != 'pure-color':
                # Include pattern info for non-pure-color patterns
                details.append(f"{region.capitalize()}: pattern={pattern} (no single RGB color available)")
    
    # === ACCESSORIES ===
    details.append("\n=== ACCESSORIES ===")
    accessories_list = []
    if decoded_attrs.get('hat') == 'yes':
        accessories_list.append("hat")
    if decoded_attrs.get('glasses') and decoded_attrs.get('glasses') not in ['no', 'NA']:
        accessories_list.append(f"glasses ({decoded_attrs['glasses']})")
    if decoded_attrs.get('wristwear') == 'yes':
        accessories_list.append("wristwear")
    if decoded_attrs.get('neckwear') == 'yes':
        accessories_list.append("neckwear")
    if decoded_attrs.get('ring') == 'yes':
        accessories_list.append("ring")
    if decoded_attrs.get('socks') and decoded_attrs.get('socks') not in ['no', 'NA']:
        accessories_list.append(f"socks ({decoded_attrs['socks']})")
    if decoded_attrs.get('waist_accessory') and decoded_attrs.get('waist_accessory') not in ['no', 'NA']:
        accessories_list.append(f"waist accessory ({decoded_attrs['waist_accessory']})")
    
    if accessories_list:
        details.append(f"Accessories: {', '.join(accessories_list)}")
    else:
        details.append("Accessories: none")
    
    # Combine all details
    detailed_description = "\n".join(details)
    
    return detailed_description


def evaluate_style_with_chatgpt(detailed_attributes, api_key=None):
    """
    Uses ChatGPT API to evaluate if the outfit style is good or not based on detailed attributes.
    Returns: dict with 'rating', 'score', 'feedback', 'suggestions'
    """
    if not OPENAI_AVAILABLE:
        return {
            "rating": "unknown",
            "score": 0,
            "feedback": "OpenAI package not installed",
            "suggestions": []
        }
    
    # Get API key from environment or parameter
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "rating": "unknown",
            "score": 0,
            "feedback": "OpenAI API key not found. Set OPENAI_API_KEY environment variable.",
            "suggestions": []
        }
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are a professional fashion stylist. Analyze the following detailed clothing attributes detected from an outfit and provide a comprehensive style evaluation.

DETAILED OUTFIT ATTRIBUTES:
{detailed_attributes}

Based on these abstract attributes (sleeve length, fabrics, colors, accessories, etc.), evaluate the overall style coherence, color harmony, and fashion sense of this outfit.

Please provide:
1. A rating: "excellent", "good", "fair", or "poor"
2. A score from 1-10 (where 10 is perfect)
3. Brief feedback analyzing:
   - Color coordination and harmony
   - Fabric combinations and appropriateness
   - Accessory choices and balance
   - Overall style coherence
   - What works well and what doesn't
4. Specific, actionable suggestions for improvement (if any)

Consider:
- How well the colors work together
- Whether the fabrics complement each other
- If the accessories enhance or clash with the outfit
- The appropriateness of the style combination
- Modern fashion principles and trends

Respond in JSON format with these exact keys:
{{
    "rating": "excellent|good|fair|poor",
    "score": <number 1-10>,
    "feedback": "<detailed explanation>",
    "suggestions": ["<suggestion1>", "<suggestion2>", ...]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4" for better quality, "gpt-3.5-turbo" for cheaper
            messages=[
                {"role": "system", "content": "You are a professional fashion stylist. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON from response
        # Sometimes GPT wraps JSON in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        return result
        
    except json.JSONDecodeError as e:
        return {
            "rating": "error",
            "score": 0,
            "feedback": f"Failed to parse ChatGPT response: {str(e)}",
            "suggestions": []
        }
    except Exception as e:
        return {
            "rating": "error",
            "score": 0,
            "feedback": f"ChatGPT API error: {str(e)}",
            "suggestions": []
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image for inference",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory containing best.pth for the attribute model",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet50", "resnet18"],
        default="resnet50",
        help="Backbone architecture for the attribute model (must match training)",
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var). If not provided, style evaluation will be skipped.",
    )
    parser.add_argument(
        "--evaluate_style",
        action="store_true",
        help="Enable ChatGPT-based style evaluation",
    )
    parser.add_argument(
        "--no_visualize_parsing",
        action="store_true",
        help="Disable automatic visualization of parsed regions (visualization is on by default)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === 1) Transforms for the attribute model (same as in test.py) ===
    transforms_img = transforms.Compose(
        [
            transforms.Resize((384, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # === 2) Load attribute model (shape/fabric/color_0..2) ===
    attr_model = MultiTaskFashionModel(
        backbone_name=args.backbone,
        use_keypoints=True,
        use_parsing_attention=False,  # Not using parser integration
    )
    ckpt_path = os.path.join(args.save_dir, "best.pth")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            attr_model.load_state_dict(ckpt["model_state"], strict=False)
            print(f"[Info] Loaded attribute model from {ckpt_path}")
            print("[Info] Note: Using region-aware attention with parsing masks")
        except Exception as e:
            print(f"[Warning] Error loading checkpoint: {e}")
            print("[Info] Using randomly initialized model")
    else:
        print(
            f"[Warning] Attribute checkpoint not found at {ckpt_path}. "
            "Using randomly initialized model."
        )
    attr_model = attr_model.to(device)

    # === 3) Parse clothing regions using pre-trained parser ===
    print("[Info] Parsing clothing regions (shirt, pants, shoes)...")
    vis_path = "parsed_regions_visualization.png"
    img_rgb, region_masks = parse_clothing_regions(
        args.image, 
        device, 
        visualize=not args.no_visualize_parsing,  # Visualize by default unless disabled
        vis_save_path=vis_path
    )
    
    # Print region statistics
    print("\n=== PARSING STATISTICS ===")
    for region_name, mask in region_masks.items():
        pixel_count = mask.sum()
        percentage = (pixel_count / mask.size) * 100
        print(f"{region_name.capitalize()}: {pixel_count} pixels ({percentage:.1f}% of image)")
    
    if not args.no_visualize_parsing:
        print(f"\n[Visualization] Check '{vis_path}' to verify if regions are correctly parsed!")
    
    # === 4) Run attribute classification ===
    # Classifier runs independently without parser integration
    print("[Info] Running attribute classification...")
    decoded_attrs = infer_single(
        attr_model, args.image, device, transforms_img, parsing_mask=None
    )
    vertical_region_colors = {}  # Legacy feature, not used anymore
    
    # === 4.5) Adjust masks based on classifier results (e.g., remove legs for shorts) ===
    from pretrained_parser import remove_legs_from_pants_mask
    is_shorts = decoded_attrs.get('lower_length') in ['three-point', 'medium-short']
    if is_shorts and "pants" in region_masks and region_masks["pants"].sum() > 0:
        print(f"[Info] Classifier detected shorts (lower_length: {decoded_attrs.get('lower_length')}). Adjusting pants mask...")
        original_pants_pixels = region_masks["pants"].sum()
        region_masks["pants"] = remove_legs_from_pants_mask(img_rgb, region_masks["pants"], is_shorts=True)
        removed_pixels = original_pants_pixels - region_masks["pants"].sum()
        if removed_pixels > 0:
            print(f"[Info] Removed {removed_pixels} leg/skin pixels from pants mask ({removed_pixels/original_pants_pixels:.1%} reduction)")
    
    # === 5) Extract colors using classifier predictions as guide ===
    # Get classifier color predictions (color_0, color_1, color_2)
    classifier_colors = {
        "color_0": decoded_attrs.get("color_0", "NA"),
        "color_1": decoded_attrs.get("color_1", "NA"),
        "color_2": decoded_attrs.get("color_2", "NA"),
    }
    
    print("[Info] Extracting colors guided by classifier predictions...")
    print(f"  Classifier: color_0={classifier_colors['color_0']}, color_1={classifier_colors['color_1']}, color_2={classifier_colors['color_2']}")
    
    # Extract colors using classifier as primary guide
    # This only extracts from regions that classifier identified (skips "NA")
    matched_colors, color_confidences = extract_colors_guided_by_classifier(img_rgb, region_masks, classifier_colors)
    
    # Calculate overall color confidence
    overall_color_confidence = get_overall_color_confidence(color_confidences)
    print(f"[Info] Color extraction confidence: {overall_color_confidence:.1%}")
    for region, conf in color_confidences.items():
        if region in ["shirt", "pants", "shoes"]:
            print(f"  {region.capitalize()}: {conf:.1%}")
    
    # Visualize color extraction with classifier mappings
    if not args.no_visualize_parsing:
        try:
            color_viz_path = "color_extraction_visualization.png"
            visualize_colors_with_classifier(img_rgb, region_masks, matched_colors, color_viz_path)
            print(f"[Visualization] Check '{color_viz_path}' to see extracted colors with classifier mappings!")
        except Exception as e:
            print(f"[Warning] Could not create color visualization: {e}")
    
    # Convert to format expected by category generation
    # Map: shirt -> top, pants -> pants, shoes -> shoes
    # Preserve pattern information from classifier
    parsing_region_colors = {
        "top": {
            "hex": matched_colors.get("shirt", {}).get("hex"),
            "name": matched_colors.get("shirt", {}).get("name"),
            "pattern": matched_colors.get("shirt", {}).get("pattern"),
        },
        "pants": {
            "hex": matched_colors.get("pants", {}).get("hex"),
            "name": matched_colors.get("pants", {}).get("name"),
            "pattern": matched_colors.get("pants", {}).get("pattern"),
        },
        "shoes": {
            "hex": matched_colors.get("shoes", {}).get("hex"),
            "name": matched_colors.get("shoes", {}).get("name"),
            "pattern": matched_colors.get("shoes", {}).get("pattern"),
        },
        "outer": {
            "hex": matched_colors.get("shirt", {}).get("hex"),  # Use shirt as outer
            "name": matched_colors.get("shirt", {}).get("name"),
            "pattern": matched_colors.get("shirt", {}).get("pattern"),
        },
        "leggings": {
            "hex": matched_colors.get("pants", {}).get("hex"),  # Use pants
            "name": matched_colors.get("pants", {}).get("name"),
            "pattern": matched_colors.get("pants", {}).get("pattern"),
        },
        "skirt": {"hex": None, "name": None, "pattern": None},  # Not detected by parser
    }
    
    # === 7) Generate clothing categories ===
    # Create a simple mask for category generation from parsed regions
    h_img, w_img = img_rgb.shape[:2]
    pred_mask = np.zeros((h_img, w_img), dtype=np.int32)
    if region_masks["shirt"].sum() > 0:
        pred_mask[region_masks["shirt"]] = 1  # Top
    if region_masks["pants"].sum() > 0:
        pred_mask[region_masks["pants"]] = 5  # Pants
    if region_masks["shoes"].sum() > 0:
        pred_mask[region_masks["shoes"]] = 11  # Shoes
    
    categories = generate_clothing_categories(
        pred_mask, 
        parsing_region_colors, 
        decoded_attrs
    )

    # === 9) Combine everything into one JSON result ===
    result = {
        "categories": categories,  # ["grey jacket", "purple pants", "orange shoes"]
        "attributes": decoded_attrs,                # sleeve_length, color_0, color_1, color_2, etc. (always included)
        "region_colors": matched_colors,            # shirt, pants, shoes with classifier matches
        "color_confidence": {
            "overall": overall_color_confidence,
            "per_region": color_confidences
        },
        "region_colors_vertical": vertical_region_colors,  # upper/lower/shoes (simple bands, legacy)
    }
    
    # === 10) Style evaluation with ChatGPT (optional) ===
    if args.evaluate_style:
        print("\n[Style Evaluation] Analyzing outfit style...")
        print(f"[Info] Color confidence: {overall_color_confidence:.1%} (threshold: 50%)")
        
        # Always send classifier predictions, conditionally include colors based on confidence
        detailed_attributes = format_style_data_for_chatgpt(
            categories, 
            matched_colors, 
            decoded_attrs,
            color_confidences=color_confidences,
            confidence_threshold=0.5  # Only include colors if confidence > 50%
        )
        
        # Always call API (classifier predictions are always included)
        style_evaluation = evaluate_style_with_chatgpt(
            detailed_attributes, 
            api_key=args.openai_key
        )
        
        # Add confidence info to evaluation
        style_evaluation["color_confidence"] = overall_color_confidence
        style_evaluation["color_confidence_note"] = (
            "High - color information included" if overall_color_confidence >= 0.5 
            else "Low - only classifier color patterns included, RGB values may be unreliable"
        )
        
        result["style_evaluation"] = style_evaluation
        
        print("\n=== STYLE EVALUATION ===")
        print(f"Rating: {style_evaluation.get('rating', 'N/A')}")
        print(f"Score: {style_evaluation.get('score', 'N/A')}/10")
        print(f"Feedback: {style_evaluation.get('feedback', 'N/A')}")
        if style_evaluation.get('suggestions'):
            print("Suggestions:")
            for suggestion in style_evaluation['suggestions']:
                print(f"  - {suggestion}")

    print("\nInference results:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    output_path = "output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[Saved] Results written to {output_path}")


if __name__ == "__main__":
    main()
