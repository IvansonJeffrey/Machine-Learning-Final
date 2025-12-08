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
    Maps parsing regions + colors + attributes to category names.
    Returns list like: ["grey jacket", "purple pants", "orange shoes"]
    """
    categories = []
    
    # Check for outer garments (jackets, coats, hoodies)
    if parsing_region_colors.get('outer', {}).get('name'):
        color = parsing_region_colors['outer']['name']
        outer_mask = np.isin(pred_mask, [2])  # label 2 = outer in DeepFashion
        
        if outer_mask.sum() > 0:
            # Use attributes to refine category
            if decoded_attrs.get('sleeve_length') == 'long-sleeve':
                if decoded_attrs.get('fabric_0') in ['cotton', 'knitted']:
                    categories.append(f"{color} hoodie")
                else:
                    categories.append(f"{color} jacket")
            else:
                categories.append(f"{color} jacket")
    
    # Check for tops (shirts, t-shirts, polo shirts, tank tops)
    if parsing_region_colors.get('top', {}).get('name'):
        color = parsing_region_colors['top']['name']
        top_mask = np.isin(pred_mask, [1, 4, 21])  # top, dress, rompers
        
        if top_mask.sum() > 0:
            top_type = "shirt"  # default
            
            # Refine based on attributes
            if decoded_attrs.get('neckline') == 'lapel':
                top_type = "polo shirt"
            elif decoded_attrs.get('sleeve_length') == 'sleeveless':
                top_type = "tank top"
            elif decoded_attrs.get('sleeve_length') == 'short-sleeve':
                top_type = "t-shirt"
            elif decoded_attrs.get('sleeve_length') == 'long-sleeve':
                top_type = "long-sleeve shirt"
            
            categories.append(f"{color} {top_type}")
    
    # Check for pants/jeans/shorts
    if parsing_region_colors.get('pants', {}).get('name'):
        color = parsing_region_colors['pants']['name']
        pants_mask = np.isin(pred_mask, [5])  # label 5 = pants
        
        if pants_mask.sum() > 0:
            pants_type = "pants"  # default
            
            # Check length
            if decoded_attrs.get('lower_length') in ['three-point', 'medium-short']:
                pants_type = "shorts"
            elif decoded_attrs.get('fabric_0') == 'denim':
                pants_type = "jeans"
            elif decoded_attrs.get('fabric_0') == 'cotton':
                pants_type = "pants"
            
            categories.append(f"{color} {pants_type}")
    
    # Check for leggings
    if parsing_region_colors.get('leggings', {}).get('name'):
        color = parsing_region_colors['leggings']['name']
        if color:
            categories.append(f"{color} leggings")
    
    # Check for shoes
    if parsing_region_colors.get('shoes', {}).get('name'):
        color = parsing_region_colors['shoes']['name']
        if color:
            categories.append(f"{color} shoes")
    
    # Check for skirts
    if parsing_region_colors.get('skirt', {}).get('name'):
        color = parsing_region_colors['skirt']['name']
        if color:
            categories.append(f"{color} skirt")
    
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
            region_order = ['head', 'shirt', 'pants', 'shoes', 'outer', 'top', 'leggings', 'skirt']
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
        region_order = ['head', 'shirt', 'pants', 'shoes', 'outer', 'top', 'leggings', 'skirt']
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
    print("[Info] Parsing clothing regions (head, shirt, pants, shoes)...")
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
        "region_colors": matched_colors,            # head, shirt, pants, shoes with classifier matches
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
