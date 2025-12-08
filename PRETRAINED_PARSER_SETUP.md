# Pre-trained Parser Setup

The new color extraction system uses a **pre-trained parser** that segments clothing into: **head, shirt, pants, shoes**.

## How It Works

1. **Parsing**: Uses MediaPipe (if available) or geometric fallback to segment clothing regions
2. **Color Extraction**: Extracts RGB colors from each region (head, shirt, pants, shoes)
3. **Classifier Matching**: Matches extracted colors with classifier's color predictions (`color_0`, `color_1`, `color_2`)

## Installation

### Required
```bash
pip install opencv-python
```

### Optional (for better parsing accuracy)
```bash
pip install mediapipe
```

MediaPipe provides better person segmentation, which improves region detection.

## Usage

Simply run:
```bash
python output.py --image path/to/image.jpg
```

The parser will automatically:
- Try MediaPipe first (if installed)
- Fall back to geometric parser (always works)

## Output Format

The output includes `region_colors` with classifier matches:

```json
{
  "region_colors": {
    "head": {
      "hex": "#4A4A4A",
      "name": "grey",
      "rgb": [74, 74, 74],
      "classifier_match": null
    },
    "shirt": {
      "hex": "#800080",
      "name": "purple",
      "rgb": [128, 0, 128],
      "classifier_match": "color_0"  // Matches classifier's color_0 prediction
    },
    "pants": {
      "hex": "#FFA500",
      "name": "orange",
      "rgb": [255, 165, 0],
      "classifier_match": "color_1"  // Matches classifier's color_1 prediction
    },
    "shoes": {
      "hex": "#000000",
      "name": "black",
      "rgb": [0, 0, 0],
      "classifier_match": "color_2"  // Matches classifier's color_2 prediction
    }
  },
  "attributes": {
    "color_0": "pure-color",
    "color_1": "striped",
    "color_2": "NA",
    ...
  }
}
```

## Classifier Color Matching

The system matches extracted RGB colors with the classifier's color pattern predictions:
- `color_0`, `color_1`, `color_2` are matched to `shirt`, `pants`, `shoes` respectively
- This ensures the extracted colors correlate with the classifier's predictions

## Benefits

✅ **No training required** - Uses pre-trained models  
✅ **Always works** - Geometric fallback ensures it never fails  
✅ **Correlates with classifier** - Colors are matched to classifier predictions  
✅ **Simple regions** - Only segments: head, shirt, pants, shoes (no complex parsing)

