"""
cat_dog_classifier / predict.py
--------------------------------
Single image inference module using unified preprocessing and confidence calibration.

Usage:
------
1. CLI:
   python predict.py test_images/sample_cat_1.jpg

2. In Python:
   from predict import predict_image
   result = predict_image("path/to/img.jpg")
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Ensure stdout supports UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from config import (
        MODEL_PATH, IDX_TO_CLASS, CLASS_EMOJIS,
        DECISION_THRESHOLD, UNCERTAINTY_THRESHOLD,
        OUTPUTS_DIR
    )
    from preprocessing import preprocess_for_inference
    from model import build_cat_dog_cnn
except ImportError:
    from cat_dog_classifier.config import (
        MODEL_PATH, IDX_TO_CLASS, CLASS_EMOJIS,
        DECISION_THRESHOLD, UNCERTAINTY_THRESHOLD,
        OUTPUTS_DIR
    )
    from cat_dog_classifier.preprocessing import preprocess_for_inference
    from cat_dog_classifier.model import build_cat_dog_cnn


def predict_image(
    image_path,
    model_path=MODEL_PATH,
    show_plot=True,
    save_plot_path=os.path.join(OUTPUTS_DIR, "last_prediction.png")
):
    """
    Classifies a single input image with confidence calibration and uncertainty check.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image file does not exist: {image_path}")

    if not os.path.exists(model_path):
        parent_model = os.path.join("cat_dog_classifier", model_path)
        if os.path.exists(parent_model):
            model_path = parent_model
        else:
            raise FileNotFoundError(f"[ERROR] Model file not found at {model_path}. Run train.py first.")

    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cat_dog_cnn().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Preprocess with Unified Pipeline
    input_tensor = preprocess_for_inference(image_path).to(device)

    # 3. Model Inference
    with torch.no_grad():
        raw_sigmoid = float(model(input_tensor)[0][0].item())

    # 4. Probabilities
    dog_prob = raw_sigmoid
    cat_prob = 1.0 - raw_sigmoid

    if dog_prob >= DECISION_THRESHOLD:
        pred_class = "dog"
        confidence = dog_prob * 100.0
    else:
        pred_class = "cat"
        confidence = cat_prob * 100.0

    is_uncertain = bool((confidence / 100.0) < UNCERTAINTY_THRESHOLD)
    emoji = CLASS_EMOJIS.get(pred_class, "🐾") if not is_uncertain else CLASS_EMOJIS["uncertain"]
    verdict_label = pred_class.upper() if not is_uncertain else f"{pred_class.upper()} (UNCERTAIN)"

    # 5. Console Output
    print("\n" + "="*50)
    print(" BINARY IMAGE CLASSIFICATION RESULT")
    print("="*50)
    print(f"  Image Path       : {image_path}")
    print(f"  Prediction       : {verdict_label} {emoji}")
    print(f"  Confidence       : {confidence:.2f}%")
    print(f"  Cat Probability  : {cat_prob*100:.2f}%")
    print(f"  Dog Probability  : {dog_prob*100:.2f}%")
    print(f"  Raw Sigmoid      : {raw_sigmoid:.5f}")
    print(f"  Uncertainty Flag : {is_uncertain} (Threshold: {UNCERTAINTY_THRESHOLD*100:.0f}%)")
    print("="*50 + "\n")

    # 6. Plotting
    raw_img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(raw_img)
    ax.axis("off")

    color_box = '#fee2e2' if is_uncertain else ('#dcfce7' if pred_class == 'dog' else '#fef3c7')
    color_edge = '#ef4444' if is_uncertain else ('#22c55e' if pred_class == 'dog' else '#f59e0b')

    title_text = f"Prediction: {verdict_label}\nConfidence: {confidence:.1f}%\n(Cat: {cat_prob*100:.1f}% | Dog: {dog_prob*100:.1f}%)"
    ax.set_title(
        title_text,
        fontsize=12,
        fontweight='bold',
        pad=15,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=color_box, edgecolor=color_edge, linewidth=2)
    )

    if save_plot_path:
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
        print(f"[PLOT] Prediction visual saved to: {save_plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        "prediction": pred_class,
        "label": verdict_label,
        "emoji": emoji,
        "confidence": round(confidence, 2),
        "cat_probability": round(cat_prob, 5),
        "dog_probability": round(dog_prob, 5),
        "raw_sigmoid": round(raw_sigmoid, 5),
        "is_uncertain": is_uncertain
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Cat vs Dog using trained CNN.")
    parser.add_argument("image", nargs="?", default=None, help="Path to image file.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to saved model.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot popup.")

    args = parser.parse_args()
    img_path = args.image

    if not img_path:
        for p in ["test_images/sample_dog_1.jpg", "test_images/sample_cat_1.jpg"]:
            if os.path.exists(p):
                img_path = p
                break

    if not img_path or not os.path.exists(img_path):
        print("[ERROR] Please provide an image path: python predict.py path/to/image.jpg")
        sys.exit(1)

    predict_image(img_path, model_path=args.model, show_plot=not args.no_show)
