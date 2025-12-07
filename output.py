import os
import argparse
import json

# (Optional but useful for your OpenMP issue)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchvision import transforms

# From your classification file (test.py)
from test import MultiTaskFashionModel, infer_single, visualize_color_regions

# From your parsing/color file (Parsing_train.py)
from Parsing_train import (
    load_trained_parsing_model,
    predict_parsing_mask,
    analyze_regions_from_pred,
)


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
    )
    ckpt_path = os.path.join(args.save_dir, "best.pth")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        attr_model.load_state_dict(ckpt["model_state"])
        print(f"[Info] Loaded attribute model from {ckpt_path}")
    else:
        print(
            f"[Warning] Attribute checkpoint not found at {ckpt_path}. "
            "Using randomly initialized model."
        )
    attr_model = attr_model.to(device)

    # === 3) Run attribute inference + simple 3-vertical-region colors ===
    decoded_attrs, vertical_region_colors = infer_single(
        attr_model, args.image, device, transforms_img
    )

    # === 4) Load parsing model for segmentation-based color detection ===
    parsing_model = load_trained_parsing_model()  # from Parsing_train.py

    # === 5) Predict parsing mask and analyze colors by clothing region ===
    img_rgb, pred_mask = predict_parsing_mask(parsing_model, args.image)
    parsing_region_colors = analyze_regions_from_pred(img_rgb, pred_mask)

    # === 6) Combine everything into one JSON result ===
    result = {
        "attributes": decoded_attrs,                # sleeve_length, etc.
        "region_colors_vertical": vertical_region_colors,  # upper/lower/shoes (simple bands)
        "region_colors_parsing": parsing_region_colors,    # top/pants/shoes from segmentation
    }

    print("\nInference results:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    output_path = "output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[Saved] Results written to {output_path}")

    # === 7) Optional visualization of vertical regions (from test.py) ===
    visualize_color_regions(args.image, "vis_color_regions.png")
    print("[Viz] Saved vertical region overlay as vis_color_regions.png")


if __name__ == "__main__":
    main()
