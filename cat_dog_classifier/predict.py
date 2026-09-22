"""
cat_dog_classifier / predict.py
--------------------------------
Inference module for single image prediction.

Usage:
------
1. As a Python module:
    from predict import predict_image
    predict_image("path/to/image.jpg")

2. From Command Line (CLI):
    python predict.py test_images/sample_cat_1.jpg
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# Ensure stdout supports UTF-8 on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def predict_image(image_path, model_path="models/cat_dog_cnn.keras", show_plot=True, save_plot_path="outputs/last_prediction.png"):
    """
    Classifies a single input image into Cat (0) or Dog (1) using the trained custom CNN.

    Parameters:
    -----------
    image_path : str
        Path to the input image file (.jpg, .jpeg, .png).
    model_path : str
        Path to the saved Keras model file.
    show_plot : bool
        Whether to render the Matplotlib figure plot.
    save_plot_path : str
        File path to save the output visualization with prediction overlay.

    Returns:
    --------
    dict
        Dictionary containing label ('Cat'/'Dog'), confidence percentage, and raw sigmoid score.
    """

    # 1. Validate paths
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image not found at path: {image_path}")

    if not os.path.exists(model_path):
        parent_model_path = os.path.join("cat_dog_classifier", model_path)
        if os.path.exists(parent_model_path):
            model_path = parent_model_path
        else:
            raise FileNotFoundError(
                f"[ERROR] Saved model file not found at '{model_path}'. "
                f"Please run 'python train.py' first to train and save the model."
            )

    # 2. Load Trained Model
    model = tf.keras.models.load_model(model_path)

    # 3. Preprocess Input Image
    raw_img = Image.open(image_path).convert("RGB")
    resized_img = raw_img.resize((128, 128))
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    # 4. Pass through CNN for Inference
    raw_prediction = model.predict(input_tensor, verbose=0)[0][0]

    # 5. Interpret Sigmoid Output
    if raw_prediction >= 0.5:
        label = "DOG"
        class_name = "Dog"
        confidence = float(raw_prediction) * 100.0
    else:
        label = "CAT"
        class_name = "Cat"
        confidence = (1.0 - float(raw_prediction)) * 100.0

    # 6. Display Console Summary
    print("\n" + "="*45)
    print(" BINARY IMAGE CLASSIFICATION RESULT")
    print("="*45)
    print(f"  Image Path  : {image_path}")
    print(f"  Prediction  : {label}")
    print(f"  Confidence  : {confidence:.2f}%")
    print(f"  Raw Sigmoid : {raw_prediction:.4f}")
    print("="*45 + "\n")

    # 7. Visualization Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(raw_img)
    ax.axis("off")

    title_text = f"Prediction: {label}\nConfidence: {confidence:.1f}%"
    box_color = '#d4edda' if class_name == "Dog" else '#fff3cd'
    edge_color = '#28a745' if class_name == "Dog" else '#ffc107'

    ax.set_title(
        title_text,
        fontsize=14,
        fontweight='bold',
        pad=15,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, edgecolor=edge_color, linewidth=2)
    )

    if save_plot_path:
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
        print(f"[PLOT] Prediction visualization saved to: {save_plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        "class": class_name,
        "label": label,
        "confidence": round(confidence, 2),
        "raw_score": float(raw_prediction)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Cat or Dog from an input image.")
    parser.add_argument("image", nargs="?", default=None, help="Path to input image file.")
    parser.add_argument("--image", dest="image_opt", default=None, help="Path to input image file.")
    parser.add_argument("--model", default="models/cat_dog_cnn.keras", help="Path to saved .keras model.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot popup.")

    args = parser.parse_args()
    img_path = args.image or args.image_opt

    if not img_path:
        default_tests = [
            "test_images/sample_cat_1.jpg",
            "test_images/sample_dog_1.jpg",
            "../test_images/sample_cat_1.jpg",
            "cat_dog_classifier/test_images/sample_cat_1.jpg"
        ]
        for p in default_tests:
            if os.path.exists(p):
                img_path = p
                break

    if not img_path or not os.path.exists(img_path):
        print("[WARNING] No image specified or found! Generating sample dataset & test image...")
        from download_dataset import download_sample_images
        download_sample_images(num_per_class=10)
        img_path = "test_images/sample_cat_1.jpg"

    predict_image(
        image_path=img_path,
        model_path=args.model,
        show_plot=not args.no_show
    )
