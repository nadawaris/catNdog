"""
Top-level entrypoint for predict.py
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cat_dog_classifier"))
from cat_dog_classifier.predict import predict_image

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
            "cat_dog_classifier/test_images/sample_cat_1.jpg"
        ]
        for p in default_tests:
            if os.path.exists(p):
                img_path = p
                break

    if not img_path or not os.path.exists(img_path):
        print("⚠️ No image specified or found! Initializing sample dataset & test image...")
        from cat_dog_classifier.download_dataset import download_sample_images
        download_sample_images(num_per_class=10)
        img_path = "test_images/sample_cat_1.jpg"

    predict_image(
        image_path=img_path,
        model_path=args.model,
        show_plot=not args.no_show
    )
