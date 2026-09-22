"""
debug_prediction.py
-------------------
Command-line debugging tool matching Step 11 specifications.
Usage:
    python debug_prediction.py path/to/image.jpg
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cat_dog_classifier"))

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from cat_dog_classifier.config import (
    MODEL_PATH, IDX_TO_CLASS, CLASS_TO_IDX, DECISION_THRESHOLD, UNCERTAINTY_THRESHOLD
)
from cat_dog_classifier.preprocessing import preprocess_for_inference, get_tensor_statistics
from cat_dog_classifier.model import build_cat_dog_cnn


def run_debug(image_path, model_path=MODEL_PATH):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file does not exist: {image_path}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        sys.exit(1)

    # 1. Load image and get original stats
    raw_img = Image.open(image_path)
    orig_mode = raw_img.mode
    orig_size = raw_img.size  # (W, H)

    # 2. Preprocess with single source of truth
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = preprocess_for_inference(image_path).to(device)
    stats = get_tensor_statistics(input_tensor)

    # 3. Load model and run inference
    model = build_cat_dog_cnn().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        raw_sigmoid = float(model(input_tensor)[0][0].item())

    # 4. Interpret class mapping
    dog_prob = raw_sigmoid
    cat_prob = 1.0 - raw_sigmoid

    if dog_prob >= DECISION_THRESHOLD:
        pred_class = "Dog"
        confidence = dog_prob * 100.0
    else:
        pred_class = "Cat"
        confidence = cat_prob * 100.0

    is_uncertain = (confidence / 100.0) < UNCERTAINTY_THRESHOLD
    uncertain_tag = " (⚠ LOW CONFIDENCE / UNCERTAIN)" if is_uncertain else ""

    print("\n================================")
    print("CAT/DOG MODEL DEBUG")
    print("================================")
    print(f"Image: {os.path.basename(image_path)} (Full: {image_path})")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Original Shape: ({orig_size[1]}, {orig_size[0]}, {3 if orig_mode=='RGB' else orig_mode}) [Mode: {orig_mode}]")
    print(f"Input shape: {stats['shape']} [dtype: {stats['dtype']}, Range: {stats['min_val']:.4f} to {stats['max_val']:.4f}]")
    print(f"Raw output: {raw_sigmoid:.5f}")
    print("\nClass mapping:")
    print(f"0 = {IDX_TO_CLASS[0].capitalize()}")
    print(f"1 = {IDX_TO_CLASS[1].capitalize()}")
    print(f"\nProbabilities:")
    print(f"Cat (0): {cat_prob*100:.2f}%")
    print(f"Dog (1): {dog_prob*100:.2f}%")
    print(f"\nPrediction: {pred_class}{uncertain_tag}")
    print(f"Confidence: {confidence:.2f}%")
    print("================================\n")

    return {
        "image": image_path,
        "raw_output": raw_sigmoid,
        "prediction": pred_class,
        "confidence": confidence,
        "is_uncertain": is_uncertain
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug single image prediction.")
    parser.add_argument("image", nargs="?", default=None, help="Path to image file.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .keras model.")

    args = parser.parse_args()
    img_path = args.image

    if not img_path:
        for p in ["test_images/sample_dog_1.jpg", "test_images/sample_cat_1.jpg"]:
            if os.path.exists(p):
                img_path = p
                break

    if not img_path:
        print("[ERROR] Please provide an image path: python debug_prediction.py path/to/image.jpg")
        sys.exit(1)

    run_debug(img_path, model_path=args.model)
