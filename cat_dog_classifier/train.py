"""
cat_dog_classifier / train.py
------------------------------
Robust Training, Validation, and Isolated Test Evaluation Pipeline in PyTorch.

Key Pipeline Steps:
1. Dataset Loading: Real images loaded via unified preprocessing module.
2. 3-Way Stratified Split:
   - 70% Training Set (Augmentation applied in-pipeline)
   - 15% Validation Set (Model tuning & EarlyStopping checkpointing)
   - 15% Isolated Test Set (Zero data leakage, strictly evaluated once after training)
3. Architecture: Custom 4-block CNN built from scratch in PyTorch.
4. Comprehensive Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, TP/TN/FP/FN.
5. Misclassification Pipeline:
   - Misclassified test images saved to `models/misclassified/`
   - Logged to `models/misclassified_predictions.csv`
   - Visual summary grid saved to `outputs/misclassified_grid.png`
"""

import os
import sys
import glob
import shutil
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure stdout supports UTF-8 on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from config import (
        CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, IMAGE_SIZE, INPUT_SHAPE,
        CATS_DIR, DOGS_DIR, MODELS_DIR, MODEL_PATH, OUTPUTS_DIR,
        MISCLASSIFIED_DIR, MISCLASSIFIED_CSV, DEFAULT_EPOCHS,
        DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, RANDOM_SEED,
        UNCERTAINTY_THRESHOLD
    )
    from preprocessing import load_and_preprocess_single_image, print_tensor_statistics
    from model import build_cat_dog_cnn, get_data_augmentation_transform
    from download_dataset import setup_real_dataset
except ImportError:
    from cat_dog_classifier.config import (
        CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, IMAGE_SIZE, INPUT_SHAPE,
        CATS_DIR, DOGS_DIR, MODELS_DIR, MODEL_PATH, OUTPUTS_DIR,
        MISCLASSIFIED_DIR, MISCLASSIFIED_CSV, DEFAULT_EPOCHS,
        DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, RANDOM_SEED,
        UNCERTAINTY_THRESHOLD
    )
    from cat_dog_classifier.preprocessing import load_and_preprocess_single_image, print_tensor_statistics
    from cat_dog_classifier.model import build_cat_dog_cnn, get_data_augmentation_transform
    from cat_dog_classifier.download_dataset import setup_real_dataset

# Set deterministic random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_dataset_with_filepaths(cats_dir=CATS_DIR, dogs_dir=DOGS_DIR):
    """
    Loads all real image files, tracks filepaths for auditability,
    removes corrupted images, and guarantees valid [0, 1] RGB tensors.
    """
    if not (os.path.exists(cats_dir) and os.path.exists(dogs_dir)) or \
       len(os.listdir(cats_dir)) < 10 or len(os.listdir(dogs_dir)) < 10:
        print("[INFO] Dataset directory missing or incomplete. Initializing real-world dataset...")
        setup_real_dataset()

    cat_files = sorted(glob.glob(os.path.join(cats_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(cats_dir, "*.[pP][nN][gG]")))
    dog_files = sorted(glob.glob(os.path.join(dogs_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(dogs_dir, "*.[pP][nN][gG]")))

    print(f"[LOAD] Found {len(cat_files)} Cat photos and {len(dog_files)} Dog photos on disk.")

    images = []
    labels = []
    filepaths = []
    seen_hashes = set()
    duplicates_removed = 0

    # Load Cats (Label: 0)
    for path in cat_files:
        try:
            with open(path, "rb") as f:
                fhash = hashlib.sha256(f.read()).hexdigest()
            if fhash in seen_hashes:
                duplicates_removed += 1
                continue
            seen_hashes.add(fhash)

            arr = load_and_preprocess_single_image(path, target_size=IMAGE_SIZE)
            images.append(arr)
            labels.append(CLASS_TO_IDX["cat"])  # 0
            filepaths.append(path)
        except Exception as e:
            print(f"  [SKIP] Corrupted image: {path} ({e})")

    # Load Dogs (Label: 1)
    for path in dog_files:
        try:
            with open(path, "rb") as f:
                fhash = hashlib.sha256(f.read()).hexdigest()
            if fhash in seen_hashes:
                duplicates_removed += 1
                continue
            seen_hashes.add(fhash)

            arr = load_and_preprocess_single_image(path, target_size=IMAGE_SIZE)
            images.append(arr)
            labels.append(CLASS_TO_IDX["dog"])  # 1
            filepaths.append(path)
        except Exception as e:
            print(f"  [SKIP] Corrupted image: {path} ({e})")

    if duplicates_removed > 0:
        print(f"[AUDIT] Detected and removed {duplicates_removed} duplicate images to eliminate data leakage.")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    paths = np.array(filepaths)

    return X, y, paths


class History:
    def __init__(self):
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}


def train_and_evaluate_pipeline(
    epochs=35,
    batch_size=16,
    learning_rate=0.001
):
    """
    Executes the full end-to-end 3-way split, training, evaluation,
    and misclassification analysis in PyTorch.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

    # 1. Load Dataset
    X, y, filepaths = load_dataset_with_filepaths()
    print(f"\n[DATASET SUMMARY] Total dataset shape: X = {X.shape}, y = {y.shape}")
    print(f"  - Total Cat Samples (0) : {np.sum(y == 0)}")
    print(f"  - Total Dog Samples (1) : {np.sum(y == 1)}")

    # Debug first tensor
    print_tensor_statistics(X[0], name="First Sample Image Tensor")

    # 2. Perform 3-Way Stratified Split (70% Train, 15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp, paths_train, paths_temp = train_test_split(
        X, y, filepaths,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
        shuffle=True
    )

    X_val, X_test, y_val, y_test, paths_val, paths_test = train_test_split(
        X_temp, y_temp, paths_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
        shuffle=True
    )

    print("\n" + "="*60)
    print(" 3-WAY STRATIFIED DATASET SPLIT (ZERO DATA LEAKAGE)")
    print("="*60)
    print(f"  * Training Set   (70%): {X_train.shape[0]} samples (Cats: {np.sum(y_train==0)}, Dogs: {np.sum(y_train==1)})")
    print(f"  * Validation Set (15%): {X_val.shape[0]} samples (Cats: {np.sum(y_val==0)}, Dogs: {np.sum(y_val==1)})")
    print(f"  * Isolated Test  (15%): {X_test.shape[0]} samples (Cats: {np.sum(y_test==0)}, Dogs: {np.sum(y_test==1)})")
    print("="*60)

    # Prepare PyTorch Tensors (Transpose from NHWC -> NCHW)
    X_train_chw = np.transpose(X_train, (0, 3, 1, 2))
    X_val_chw = np.transpose(X_val, (0, 3, 1, 2))
    X_test_chw = np.transpose(X_test, (0, 3, 1, 2))

    train_dataset = TensorDataset(torch.from_numpy(X_train_chw).float(), torch.from_numpy(y_train).float().unsqueeze(1))
    val_dataset = TensorDataset(torch.from_numpy(X_val_chw).float(), torch.from_numpy(y_val).float().unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Build Model & Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cat_dog_cnn().to(device)
    aug_transform = get_data_augmentation_transform()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    history = History()
    best_val_acc = -1.0

    print(f"\n[TRAIN] Training PyTorch CNN from scratch for {epochs} Epochs (Batch Size: {batch_size}, Device: {device})...")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images_batch, labels_batch in train_loader:
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
            images_batch = aug_transform(images_batch)

            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images_batch.size(0)
            preds = (outputs >= 0.50).float()
            train_correct += (preds == labels_batch).sum().item()
            train_total += labels_batch.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images_batch, labels_batch in val_loader:
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                outputs = model(images_batch)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item() * images_batch.size(0)
                preds = (outputs >= 0.50).float()
                val_correct += (preds == labels_batch).sum().item()
                val_total += labels_batch.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        scheduler.step(epoch_val_loss)

        history.history['accuracy'].append(epoch_train_acc)
        history.history['val_accuracy'].append(epoch_val_acc)
        history.history['loss'].append(epoch_train_loss)
        history.history['val_loss'].append(epoch_val_loss)

        print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")

        # Save Best Model Checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n[SAVE] Final trained model successfully saved to: {MODEL_PATH}")

    # 5. Plot Loss & Accuracy Curves
    plot_training_curves(history, os.path.join(OUTPUTS_DIR, "training_history.png"))

    # 6. Evaluate on ISOLATED TEST SET (Untouched during training/tuning)
    print("\n" + "="*60)
    print(" EVALUATION ON ISOLATED TEST SET (UNTOUCHED)")
    print("="*60)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    X_test_tensor = torch.from_numpy(X_test_chw).float().to(device)
    with torch.no_grad():
        test_probs = model(X_test_tensor).cpu().numpy().ravel()
    test_preds = (test_probs >= 0.50).astype(int)

    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print(f"  * Total Test Samples : {len(y_test)}")
    print(f"  * Test Accuracy      : {test_acc*100:.2f}%")
    print(f"  * Test Precision     : {test_prec*100:.2f}%")
    print(f"  * Test Recall (Sens.): {test_rec*100:.2f}%")
    print(f"  * Test F1-Score      : {test_f1:.4f}")
    print(f"  * True Negatives (TN): {tn} (Actual Cats correctly predicted as Cat)")
    print(f"  * False Positives(FP): {fp} (Actual Cats wrongly predicted as Dog)")
    print(f"  * False Negatives(FN): {fn} (Actual Dogs wrongly predicted as Cat)")
    print(f"  * True Positives (TP): {tp} (Actual Dogs correctly predicted as Dog)")
    print("\n Detailed Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["Cat", "Dog"], zero_division=0))
    print("="*60)

    # 7. Save Confusion Matrix Heatmap
    plot_confusion_matrix(cm, os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))

    # 8. Misclassification Analysis
    run_misclassification_analysis(
        X_test, y_test, test_probs, test_preds, paths_test,
        misclassified_dir=MISCLASSIFIED_DIR,
        csv_path=MISCLASSIFIED_CSV,
        grid_output_path=os.path.join(OUTPUTS_DIR, "misclassified_grid.png")
    )

    # 9. Plot Test Sample Predictions
    plot_test_predictions(X_test, y_test, test_probs, test_preds, os.path.join(OUTPUTS_DIR, "sample_predictions.png"))

    return {
        "test_accuracy": float(test_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "num_test": len(y_test)
    }


def plot_training_curves(history, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='#6366f1', linewidth=2.5)
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#06b6d4', linewidth=2.5, linestyle='--')
    ax1.set_title('Training vs. Validation Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Loss Plot
    ax2.plot(history.history['loss'], label='Training Loss', color='#6366f1', linewidth=2.5)
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss', color='#06b6d4', linewidth=2.5, linestyle='--')
    ax2.set_title('Training vs. Validation Loss (Cross-Entropy)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Training curves saved to: {output_path}")


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat (0)', 'Dog (1)'])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    ax.set_title('Test Set Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Confusion matrix plot saved to: {output_path}")


def run_misclassification_analysis(
    X_test, y_test, test_probs, test_preds, paths_test,
    misclassified_dir=MISCLASSIFIED_DIR,
    csv_path=MISCLASSIFIED_CSV,
    grid_output_path=os.path.join(OUTPUTS_DIR, "misclassified_grid.png")
):
    """
    Identifies all incorrectly classified test images, saves them to
    models/misclassified/, writes details to CSV, and generates a visual grid.
    """
    # Clear previous misclassified images
    if os.path.exists(misclassified_dir):
        for f in os.listdir(misclassified_dir):
            try:
                os.remove(os.path.join(misclassified_dir, f))
            except Exception:
                pass
    os.makedirs(misclassified_dir, exist_ok=True)

    incorrect_indices = np.where(test_preds != y_test)[0]
    records = []

    print(f"\n[MISCLASSIFICATION ANALYSIS] Found {len(incorrect_indices)} misclassified test images out of {len(y_test)}.")

    for idx in incorrect_indices:
        actual_label = IDX_TO_CLASS[y_test[idx]]
        pred_label = IDX_TO_CLASS[test_preds[idx]]
        prob_dog = float(test_probs[idx])
        conf = prob_dog if pred_label == "dog" else (1.0 - prob_dog)
        src_path = paths_test[idx]
        fname = os.path.basename(src_path)

        # Copy image to misclassified folder
        dest_filename = f"error_{idx:03d}_{actual_label}_pred_as_{pred_label}_{fname}"
        dest_path = os.path.join(misclassified_dir, dest_filename)
        try:
            shutil.copyfile(src_path, dest_path)
        except Exception:
            pass

        records.append({
            "image_filename": fname,
            "source_path": src_path,
            "actual_class": actual_label,
            "predicted_class": pred_label,
            "confidence": round(conf * 100.0, 2),
            "dog_probability": round(prob_dog, 5),
            "cat_probability": round((1.0 - prob_dog), 5),
            "is_uncertain": bool(conf < UNCERTAINTY_THRESHOLD)
        })

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"[REPORT] Misclassification records saved to: {csv_path}")

    # Generate Visual Grid of Misclassified Images
    if len(incorrect_indices) > 0:
        cols = min(4, len(incorrect_indices))
        rows = int(np.ceil(len(incorrect_indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = np.array(axes).reshape(-1)

        for i, idx in enumerate(incorrect_indices):
            ax = axes[i]
            ax.imshow(X_test[idx])
            actual_label = IDX_TO_CLASS[y_test[idx]]
            pred_label = IDX_TO_CLASS[test_preds[idx]]
            prob_dog = float(test_probs[idx])
            conf = prob_dog if pred_label == "dog" else (1.0 - prob_dog)
            ax.set_title(
                f"Actual: {actual_label.upper()}\nPred: {pred_label.upper()} ({conf*100:.1f}%)",
                color='red',
                fontsize=10,
                fontweight='bold'
            )
            ax.axis('off')

        # Turn off extra subplots
        for j in range(len(incorrect_indices), len(axes)):
            axes[j].axis('off')

        plt.suptitle("Misclassified Test Images Error Analysis", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(grid_output_path, dpi=300)
        plt.close()
        print(f"[PLOT] Misclassification visual grid saved to: {grid_output_path}")
    else:
        # Create a clean placeholder if zero errors
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Zero Misclassified Samples on Test Set (100% Test Accuracy)",
                ha='center', va='center', fontsize=12, color='green', fontweight='bold')
        ax.axis('off')
        plt.savefig(grid_output_path, dpi=200)
        plt.close()


def plot_test_predictions(X_test, y_test, test_probs, test_preds, output_path):
    """
    Visualizes a representative grid of test predictions with actual vs predicted labels.
    """
    num_samples = min(8, len(y_test))
    if num_samples == 0:
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    sample_indices = np.random.choice(len(y_test), num_samples, replace=False)

    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        ax.imshow(X_test[idx])
        actual = IDX_TO_CLASS[y_test[idx]]
        pred = IDX_TO_CLASS[test_preds[idx]]
        is_correct = (y_test[idx] == test_preds[idx])
        conf = test_probs[idx] if pred == "dog" else (1.0 - test_probs[idx])

        color = '#10b981' if is_correct else '#f43f5e'
        ax.set_title(f"Actual: {actual}\nPred: {pred} ({conf*100:.1f}%)", color=color, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Isolated Test Set Predictions", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Test prediction sample grid saved to: {output_path}")


if __name__ == "__main__":
    train_and_evaluate_pipeline()
