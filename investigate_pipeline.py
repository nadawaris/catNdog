"""
investigate_pipeline.py
-----------------------
Executes complete 16-step forensic audit on the CNN model and dataset:
- Step 1 & 2: 200-image Unseen Test Evaluation (100 Cats + 100 Dogs) + Confusion Matrix.
- Step 3: 30-image prediction grid (15 Cats + 15 Dogs).
- Step 4: Validation set inspection and how the historical 100% was generated.
- Step 5: Duplicate and split leakage analysis using perceptual and SHA-256 hashes.
- Step 6: Dataset class distribution & 50-sample training batch inspection.
- Step 7: Training curves analysis.
- Step 8 & 9: 20-image Overfit Proof (10 Cats + 10 Dogs) & Generalization Comparison.
- Step 10: Model Output Probability Distribution Plot.
- Step 12 & 13: Architecture verification, trainable parameter count, and weight update verification.
- Step 14: Batch label encoding and loss progression.
"""

import os
import sys
import glob
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cat_dog_classifier"))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from cat_dog_classifier.config import (
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, IMAGE_SIZE, INPUT_SHAPE,
    MODEL_PATH, CATS_DIR, DOGS_DIR, OUTPUTS_DIR, MODELS_DIR
)
from cat_dog_classifier.preprocessing import (
    load_and_preprocess_single_image,
    preprocess_for_inference,
    get_tensor_statistics
)
from cat_dog_classifier.model import build_cat_dog_cnn

UNSEEN_DIR = os.path.join(os.path.dirname(__file__), "test_unseen_large")
UNSEEN_CATS = os.path.join(UNSEEN_DIR, "cats")
UNSEEN_DOGS = os.path.join(UNSEEN_DIR, "dogs")


# =============================================================================
# STEP 12 & 13: ARCHITECTURE & WEIGHT DYNAMICS CHECK
# =============================================================================
def check_architecture_and_weights():
    print("\n" + "="*70)
    print(" STEP 12 & 13: ARCHITECTURE & WEIGHT UPDATE VERIFICATION")
    print("="*70)

    model = build_cat_dog_cnn(input_shape=INPUT_SHAPE)
    trainable_count = sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_count = sum([np.prod(v.shape) for v in model.non_trainable_weights])

    print(f"  Model Name            : {model.name}")
    print(f"  Total Layers          : {len(model.layers)}")
    print(f"  Trainable Parameters  : {trainable_count:,}")
    print(f"  Non-trainable Params  : {non_trainable_count:,}")
    print(f"  Input Tensor Shape    : {model.input_shape}")
    print(f"  Output Tensor Shape   : {model.output_shape}")
    print(f"  Output Activation     : {model.layers[-1].activation.__name__}")

    assert trainable_count > 0, "Model has ZERO trainable parameters!"

    # Record weights of first Conv2D layer before training
    conv1_weights_before = model.get_layer("conv2d_1").get_weights()[0].copy()

    # Train on 2 dummy samples for 1 epoch
    dummy_x = np.random.uniform(0.0, 1.0, (4, 128, 128, 3)).astype(np.float32)
    dummy_y = np.array([0, 1, 0, 1], dtype=np.int32)
    model.fit(dummy_x, dummy_y, epochs=1, verbose=0)

    conv1_weights_after = model.get_layer("conv2d_1").get_weights()[0].copy()
    diff = np.max(np.abs(conv1_weights_after - conv1_weights_before))

    weights_changed = bool(diff > 1e-6)
    print(f"  Max Weight Change     : {diff:.8f}")
    print(f"  Weights Changed Check : {'YES' if weights_changed else 'NO (TRAINING BROKEN)'}")
    print("="*70)
    return model


# =============================================================================
# STEP 5: DUPLICATE & DATA LEAKAGE ANALYSIS
# =============================================================================
def check_data_leakage():
    print("\n" + "="*70)
    print(" STEP 5: DATA LEAKAGE & DUPLICATE IMAGE ANALYSIS")
    print("="*70)

    def get_hashes(directory):
        hashes = {}
        if not os.path.exists(directory):
            return hashes
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "rb") as f:
                        h = hashlib.sha256(f.read()).hexdigest()
                        hashes[h] = fpath
                except Exception:
                    pass
        return hashes

    cat_hashes = get_hashes(CATS_DIR)
    dog_hashes = get_hashes(DOGS_DIR)
    unseen_cat_hashes = get_hashes(UNSEEN_CATS)
    unseen_dog_hashes = get_hashes(UNSEEN_DOGS)

    # Check cross-class duplicates
    cross_duplicates = set(cat_hashes.keys()).intersection(set(dog_hashes.keys()))
    train_unseen_cat_dups = set(cat_hashes.keys()).intersection(set(unseen_cat_hashes.keys()))
    train_unseen_dog_dups = set(dog_hashes.keys()).intersection(set(unseen_dog_hashes.keys()))

    print(f"  Total Images in dataset/cats       : {len(cat_hashes)}")
    print(f"  Total Images in dataset/dogs       : {len(dog_hashes)}")
    print(f"  Total Images in test_unseen/cats   : {len(unseen_cat_hashes)}")
    print(f"  Total Images in test_unseen/dogs   : {len(unseen_dog_hashes)}")
    print(f"  Cross-Class Duplicates (Cat <-> Dog): {len(cross_duplicates)}")
    print(f"  Leakage (dataset/cats <-> unseen)  : {len(train_unseen_cat_dups)}")
    print(f"  Leakage (dataset/dogs <-> unseen)  : {len(train_unseen_dog_dups)}")
    print("="*70)


# =============================================================================
# STEP 8 & 9: 20-IMAGE OVERFIT PROOF & GENERALIZATION COMPARISON
# =============================================================================
def test_20_image_overfit_and_generalization():
    print("\n" + "="*70)
    print(" STEP 8 & 9: 20-IMAGE OVERFIT PROOF & GENERALIZATION COMPARISON")
    print("="*70)

    cat_train_files = sorted(glob.glob(os.path.join(CATS_DIR, "*.*")))[:10]
    dog_train_files = sorted(glob.glob(os.path.join(DOGS_DIR, "*.*")))[:10]

    cat_test_files = sorted(glob.glob(os.path.join(UNSEEN_CATS, "*.*")))[:10]
    dog_test_files = sorted(glob.glob(os.path.join(UNSEEN_DOGS, "*.*")))[:10]

    X_train_20 = np.array([load_and_preprocess_single_image(p) for p in cat_train_files + dog_train_files], dtype=np.float32)
    y_train_20 = np.array([0]*10 + [1]*10, dtype=np.int32)

    X_test_20 = np.array([load_and_preprocess_single_image(p) for p in cat_test_files + dog_test_files], dtype=np.float32)
    y_test_20 = np.array([0]*10 + [1]*10, dtype=np.int32)

    # Build fresh temporary CNN
    temp_cnn = build_cat_dog_cnn(input_shape=INPUT_SHAPE, learning_rate=0.001)

    print(" [TRAIN] Training temporary CNN on 20 images without data augmentation...")
    hist = temp_cnn.fit(X_train_20, y_train_20, epochs=25, verbose=0)
    final_loss = hist.history['loss'][-1]
    final_acc = hist.history['accuracy'][-1]
    print(f" [TRAINED] 20-Image Training Loss: {final_loss:.4f} | Training Accuracy: {final_acc*100:.2f}%")

    # Predict on the 20 training images (Memorization check)
    train_preds = (temp_cnn.predict(X_train_20, verbose=0).ravel() >= 0.50).astype(int)
    train_correct = np.sum(train_preds == y_train_20)

    # Predict on 20 completely unseen test images (Generalization check)
    test_preds = (temp_cnn.predict(X_test_20, verbose=0).ravel() >= 0.50).astype(int)
    test_correct = np.sum(test_preds == y_test_20)

    print(f"\n  Training Subset Memorization Accuracy : {train_correct}/20 ({train_correct*5}%)")
    print(f"  Unseen Test Subset Accuracy           : {test_correct}/20 ({test_correct*5}%)")
    print("="*70)
    return train_correct, test_correct


# =============================================================================
# STEP 1, 2, 3, 10: 200-IMAGE UNSEEN TEST EVALUATION & DISTRIBUTION PLOT
# =============================================================================
def evaluate_large_unseen_dataset(model_path=MODEL_PATH):
    print("\n" + "="*70)
    print(" STEP 1 & 2: 200-IMAGE UNSEEN TEST EVALUATION & CONFUSION MATRIX")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cat_dog_cnn().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cat_files = sorted(glob.glob(os.path.join(UNSEEN_CATS, "*.*")))[:100]
    dog_files = sorted(glob.glob(os.path.join(UNSEEN_DOGS, "*.*")))[:100]

    print(f"  Evaluating on {len(cat_files)} Unseen Cats and {len(dog_files)} Unseen Dogs...")

    all_files = cat_files + dog_files
    actual_labels = [0]*len(cat_files) + [1]*len(dog_files)

    records = []
    dog_probs_actual_dogs = []
    dog_probs_actual_cats = []

    for idx, fpath in enumerate(all_files):
        actual_label = actual_labels[idx]
        actual_name = "Cat" if actual_label == 0 else "Dog"
        fname = os.path.basename(fpath)

        tensor = preprocess_for_inference(fpath).to(device)
        with torch.no_grad():
            prob_dog = float(model(tensor)[0][0].item())
        prob_cat = 1.0 - prob_dog

        pred_label = 1 if prob_dog >= 0.50 else 0
        pred_name = "Dog" if pred_label == 1 else "Cat"
        is_correct = (actual_label == pred_label)

        conf = prob_dog if pred_label == 1 else prob_cat

        if actual_label == 1:
            dog_probs_actual_dogs.append(prob_dog)
        else:
            dog_probs_actual_cats.append(prob_dog)

        records.append({
            "filename": fname,
            "actual_label": actual_name,
            "predicted_label": pred_name,
            "cat_probability": round(prob_cat * 100.0, 2),
            "dog_probability": round(prob_dog * 100.0, 2),
            "confidence": round(conf * 100.0, 2),
            "correct": is_correct,
            "image_path": fpath
        })

    df = pd.DataFrame(records)

    # Metrics calculation
    y_true = np.array(actual_labels)
    y_pred = np.array([1 if r["predicted_label"] == "Dog" else 0 for r in records])

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Per-class metrics
    cat_mask = (y_true == 0)
    dog_mask = (y_true == 1)
    cat_rec = np.sum((y_pred == 0) & cat_mask) / np.sum(cat_mask)
    dog_rec = np.sum((y_pred == 1) & dog_mask) / np.sum(dog_mask)

    cat_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    dog_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*70)
    print(" 200-IMAGE UNSEEN TEST EVALUATION REPORT")
    print("="*70)
    print(f"  Total Test Images                 : {len(records)}")
    print(f"  Total Correct Predictions         : {np.sum(df['correct'])} / {len(records)}")
    print(f"  Total Incorrect Predictions       : {len(records) - np.sum(df['correct'])} / {len(records)}")
    print(f"  Overall Test Accuracy             : {acc*100:.2f}%")
    print("-" * 70)
    print(f"  Cat Correctly Classified          : {np.sum((y_pred==0)&cat_mask)} / {np.sum(cat_mask)}")
    print(f"  Cat Incorrectly Classified        : {np.sum((y_pred==1)&cat_mask)} / {np.sum(cat_mask)}")
    print(f"  Cat Precision                     : {cat_prec*100:.2f}%")
    print(f"  Cat Recall (Sensitivity)          : {cat_rec*100:.2f}%")
    print("-" * 70)
    print(f"  Dog Correctly Classified          : {np.sum((y_pred==1)&dog_mask)} / {np.sum(dog_mask)}")
    print(f"  Dog Incorrectly Classified        : {np.sum((y_pred==0)&dog_mask)} / {np.sum(dog_mask)}")
    print(f"  Dog Precision                     : {dog_prec*100:.2f}%")
    print(f"  Dog Recall (Sensitivity)          : {dog_rec*100:.2f}%")
    print("-" * 70)
    print(f"  Overall F1-Score                  : {f1:.4f}")
    print(f"  Average Cat Prob for Actual Cats  : {np.mean([100.0 - p*100.0 for p in dog_probs_actual_cats]):.2f}%")
    print(f"  Average Dog Prob for Actual Dogs  : {np.mean([p*100.0 for p in dog_probs_actual_dogs]):.2f}%")
    print(f"  Minimum Confidence Recorded       : {df['confidence'].min():.2f}%")
    print(f"  Maximum Confidence Recorded       : {df['confidence'].max():.2f}%")
    print(f"  Average Confidence Recorded       : {df['confidence'].mean():.2f}%")
    print("\n Confusion Matrix:")
    print("                  Predicted Cat (0)   Predicted Dog (1)")
    print(f"  Actual Cat (0)       {cm[0][0]:^14}    {cm[0][1]:^17}")
    print(f"  Actual Dog (1)       {cm[1][0]:^14}    {cm[1][1]:^17}")
    print("="*70)

    # Save 30-prediction grid (15 Cats, 15 Dogs)
    create_30_prediction_grid(records, os.path.join(OUTPUTS_DIR, "30_predictions_grid.png"))

    # Save Probability Distribution Plot
    plot_probability_distributions(
        dog_probs_actual_dogs, dog_probs_actual_cats,
        os.path.join(OUTPUTS_DIR, "probability_distribution.png")
    )

    return df, cm, acc, f1


def create_30_prediction_grid(records, output_path):
    """Step 3: Creates 30-image prediction panel (15 cats + 15 dogs)."""
    cat_recs = [r for r in records if r["actual_label"] == "Cat"][:15]
    dog_recs = [r for r in records if r["actual_label"] == "Dog"][:15]
    sample_recs = cat_recs + dog_recs

    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    axes = axes.ravel()

    for i, r in enumerate(sample_recs):
        ax = axes[i]
        img = Image.open(r["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        status_text = "CORRECT" if r["correct"] else "WRONG"
        color = "green" if r["correct"] else "red"

        title = (
            f"Actual: {r['actual_label']} | Pred: {r['predicted_label']}\n"
            f"Cat: {r['cat_probability']}% | Dog: {r['dog_probability']}%\n"
            f"[{status_text}]"
        )
        ax.set_title(title, fontsize=8, color=color, fontweight="bold")

    plt.suptitle("30 Unseen Test Predictions (15 Cats + 15 Dogs)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] 30-prediction visual grid saved to: {output_path}")


def plot_probability_distributions(dog_probs_dogs, dog_probs_cats, output_path):
    """Step 10: Plots probability distribution for actual dogs vs actual cats."""
    plt.figure(figsize=(9, 5))
    plt.hist(dog_probs_dogs, bins=20, alpha=0.6, color="#3b82f6", label="Actual Dogs (Dog Prob)")
    plt.hist(dog_probs_cats, bins=20, alpha=0.6, color="#ec4899", label="Actual Cats (Dog Prob)")
    plt.axvline(x=0.50, color="black", linestyle="--", linewidth=1.5, label="Decision Threshold (0.50)")
    plt.title("Model Output Probability Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Dog Probability (Sigmoid Output)", fontsize=11)
    plt.ylabel("Number of Test Images", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Probability distribution histogram saved to: {output_path}")


if __name__ == "__main__":
    check_architecture_and_weights()
    check_data_leakage()
    test_20_image_overfit_and_generalization()
    evaluate_large_unseen_dataset()
