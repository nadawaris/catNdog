"""
Top-level entrypoint for predict.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cat_dog_classifier"))
from cat_dog_classifier.predict import predict_image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict Cat vs Dog using trained CNN.")
    parser.add_argument("image", nargs="?", default=None, help="Path to image file.")
    parser.add_argument("--model", default="models/cat_dog_cnn.pth", help="Path to saved model.")
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
