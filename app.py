"""
app.py
------
Cat vs Dog Classifier - Web Application Backend (Flask)
Serves the localhost web UI and REST API for real-time image classification,
metrics visualization, and sample testing using unified preprocessing.
"""

import os
import io
import sys
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cat_dog_classifier"))

# Ensure stdout supports UTF-8 on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from cat_dog_classifier.config import (
    MODEL_PATH, OUTPUTS_DIR, TEST_IMAGES_DIR, BASE_DIR,
    DECISION_THRESHOLD, UNCERTAINTY_THRESHOLD, CLASS_EMOJIS
)
from cat_dog_classifier.preprocessing import (
    preprocess_for_inference,
    load_and_preprocess_single_image
)
from cat_dog_classifier.model import build_cat_dog_cnn

app = Flask(__name__, template_folder="templates", static_folder="static")

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_if_needed():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = build_cat_dog_cnn().to(device)
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                model.eval()
                print(f"[INFO] Successfully loaded PyTorch model from {MODEL_PATH}")
            except Exception as e:
                print(f"[ERROR] Could not load model: {e}")
                model = None
        else:
            print(f"[WARNING] Model not found at {MODEL_PATH}")

load_model_if_needed()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    load_model_if_needed()
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure models/cat_dog_cnn.pth exists."}), 500

    try:
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            image = Image.open(file.stream)
        elif request.is_json and "image_base64" in request.json:
            base64_data = request.json["image_base64"].split(",")[-1]
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
        elif request.is_json and "sample_name" in request.json:
            sample_name = os.path.basename(request.json["sample_name"])
            sample_path = os.path.join(TEST_IMAGES_DIR, sample_name)
            if not os.path.exists(sample_path):
                return jsonify({"error": f"Sample {sample_name} not found"}), 404
            image = Image.open(sample_path)
        else:
            return jsonify({"error": "No image provided"}), 400

        # Preprocess strictly with unified preprocessing pipeline
        tensor = preprocess_for_inference(image).to(device)
        with torch.no_grad():
            raw_sigmoid = float(model(tensor)[0][0].item())

        dog_prob = raw_sigmoid
        cat_prob = 1.0 - raw_sigmoid

        if dog_prob >= DECISION_THRESHOLD:
            pred_class = "Dog"
            confidence = dog_prob * 100.0
        else:
            pred_class = "Cat"
            confidence = cat_prob * 100.0

        is_uncertain = bool((confidence / 100.0) < UNCERTAINTY_THRESHOLD)
        emoji = "🐶" if pred_class == "Dog" else "🐱"
        if is_uncertain:
            emoji = "❓"

        # Generate thumbnail preview
        buffered = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "prediction": pred_class.lower(),
            "label": pred_class if not is_uncertain else f"{pred_class} (Uncertain)",
            "emoji": emoji,
            "confidence": round(confidence, 2),
            "raw_probability": round(raw_sigmoid, 5),
            "cat_percentage": round(cat_prob * 100.0, 2),
            "dog_percentage": round(dog_prob * 100.0, 2),
            "is_uncertain": is_uncertain,
            "uncertainty_threshold": UNCERTAINTY_THRESHOLD * 100.0,
            "preview_base64": f"data:image/jpeg;base64,{img_str}",
            "debug": {
                "model_name": "Custom_Cat_vs_Dog_CNN",
                "model_file": "models/cat_dog_cnn.pth",
                "input_size": "128 × 128 × 3 (RGB)",
                "class_mapping": "Class 0 = Cat | Class 1 = Dog",
                "raw_sigmoid_output": round(raw_sigmoid, 5),
                "predicted_class": pred_class,
                "confidence_score": f"{confidence:.2f}%",
                "preprocessing": "PIL RGB → Bilinear Resize (128,128) → float32 / 255.0",
                "decision_threshold": "≥ 0.50 (Dog) | < 0.50 (Cat)"
            }
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


@app.route("/api/samples", methods=["GET"])
def get_samples():
    if not os.path.exists(TEST_IMAGES_DIR):
        return jsonify({"samples": []})
    
    samples = []
    for f in sorted(os.listdir(TEST_IMAGES_DIR)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            samples.append({
                "name": f,
                "url": f"/test_images/{f}",
                "expected": "Cat" if "cat" in f.lower() else "Dog"
            })
    return jsonify({"samples": samples})


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/test_images/<path:filename>")
def serve_test_images(filename):
    return send_from_directory(TEST_IMAGES_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n=======================================================")
    print(" Cats vs Dogs Web Classifier Server Running!")
    print(f" Open in your browser: http://127.0.0.1:{port}")
    print("=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
