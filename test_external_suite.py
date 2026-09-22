"""
test_external_suite.py
-----------------------
Evaluates the trained model strictly on unseen external images in test_external/.
Produces:
- External prediction summary table
- CSV export: models/external_predictions.csv
- Confusion matrix and accuracy on external images
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cat_dog_classifier"))

from cat_dog_classifier.config import (
    MODEL_PATH, TEST_EXTERNAL_DIR, TEST_EXTERNAL_CSV,
    CLASS_TO_IDX, IDX_TO_CLASS, UNCERTAINTY_THRESHOLD
)
from cat_dog_classifier.preprocessing import preprocess_for_inference
from cat_dog_classifier.model import build_cat_dog_cnn

# Ensure stdout supports UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def evaluate_external_suite(model_path=MODEL_PATH, external_dir=TEST_EXTERNAL_DIR, csv_path=TEST_EXTERNAL_CSV):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Trained model not found at {model_path}. Run train.py first.")

    print(f"\n[EXTERNAL EVALUATION] Loading trained PyTorch model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cat_dog_cnn().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cats_dir = os.path.join(external_dir, "cats")
    dogs_dir = os.path.join(external_dir, "dogs")

    cat_files = sorted(glob.glob(os.path.join(cats_dir, "*.*")))
    dog_files = sorted(glob.glob(os.path.join(dogs_dir, "*.*")))

    records = []
    correct_count = 0
    total_count = 0

    print(f"[EVALUATING] Testing {len(cat_files)} external Cats and {len(dog_files)} external Dogs...\n")
    print(f"{'Filename':<25} | {'Actual':<6} | {'Predicted':<9} | {'Confidence':<10} | {'Status':<7}")
    print("-" * 65)

    def test_file_list(files, actual_label):
        nonlocal correct_count, total_count
        for fpath in files:
            fname = os.path.basename(fpath)
            try:
                tensor = preprocess_for_inference(fpath).to(device)
                with torch.no_grad():
                    prob = float(model(tensor)[0][0].item())
                
                pred_label = "dog" if prob >= 0.50 else "cat"
                conf = prob if pred_label == "dog" else (1.0 - prob)
                
                # Check uncertainty
                is_uncertain = bool(conf < UNCERTAINTY_THRESHOLD)
                display_pred = pred_label if not is_uncertain else f"{pred_label}?"

                is_correct = (pred_label == actual_label)
                if is_correct:
                    correct_count += 1
                total_count += 1

                status = "CORRECT" if is_correct else "WRONG"

                print(f"{fname:<25} | {actual_label:<6} | {display_pred:<9} | {conf*100:6.2f}%    | {status}")

                records.append({
                    "filename": fname,
                    "actual_label": actual_label,
                    "predicted_label": pred_label,
                    "confidence": round(conf * 100.0, 2),
                    "dog_probability": round(prob, 5),
                    "cat_probability": round((1.0 - prob), 5),
                    "is_uncertain": is_uncertain,
                    "status": status,
                    "correct": is_correct
                })
            except Exception as e:
                print(f"Error evaluating {fname}: {e}")

    test_file_list(cat_files, "cat")
    test_file_list(dog_files, "dog")

    ext_acc = (correct_count / total_count) if total_count > 0 else 0.0

    print("-" * 65)
    print(f"EXTERNAL GENERALIZATION ACCURACY: {ext_acc*100:.2f}% ({correct_count}/{total_count} correct)")
    print("=" * 65)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] External sanity evaluation table saved to: {csv_path}\n")

    return ext_acc, df


if __name__ == "__main__":
    evaluate_external_suite()
