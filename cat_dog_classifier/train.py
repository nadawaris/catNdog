"""
cat_dog_classifier / train.py
------------------------------
Main training, evaluation, and visualization pipeline.

This script performs the following end-to-end steps:
1. Loads cat and dog image files from dataset/cats and dataset/dogs.
2. Performs an 80% Train / 20% Validation split with stratification to prevent data leakage.
3. Preprocesses images: 128x128 resize, RGB conversion, pixel normalization [0, 1].
4. Builds the custom CNN architecture defined in model.py.
5. Trains the CNN for 15-20 epochs with batch size 32 and data augmentation.
6. Saves accuracy/loss history plots and confusion matrix to outputs/.
7. Evaluates Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
8. Displays sample predictions and misclassified examples with confidence scores.
9. Saves trained model to models/cat_dog_cnn.keras.
"""

import os
import sys
import glob
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
    ConfusionMatrixDisplay
)
import tensorflow as tf

# Ensure stdout supports UTF-8 on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import custom model architecture & data augmentation
from model import build_cat_dog_cnn, get_data_augmentation_layer
from download_dataset import download_sample_images

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_image_dataset(dataset_dir="dataset", image_size=(128, 128)):
    """
    Loads cat and dog images, resizes to (128, 128), converts to RGB,
    normalizes pixel values to [0, 1], and assigns numerical labels.

    Label Assignment:
    - 0 : Cat
    - 1 : Dog

    Returns:
    --------
    X : np.ndarray of shape (N, 128, 128, 3), float32
        Preprocessed image array normalized between 0.0 and 1.0.
    y : np.ndarray of shape (N,), int32
        Binary target labels (0 for cat, 1 for dog).
    """
    cats_dir = os.path.join(dataset_dir, "cats")
    dogs_dir = os.path.join(dataset_dir, "dogs")

    # If dataset is empty or missing, download/generate sample data automatically
    if not (os.path.exists(cats_dir) and os.path.exists(dogs_dir)):
        print("[INFO] Dataset directory missing. Initializing sample dataset...")
        download_sample_images(dataset_dir, num_per_class=50)

    cat_paths = glob.glob(os.path.join(cats_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(cats_dir, "*.[pP][nN][gG]"))
    dog_paths = glob.glob(os.path.join(dogs_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(dogs_dir, "*.[pP][nN][gG]"))

    if len(cat_paths) == 0 or len(dog_paths) == 0:
        print("[WARNING] No images found in dataset folders! Fetching samples...")
        download_sample_images(dataset_dir, num_per_class=50)
        cat_paths = glob.glob(os.path.join(cats_dir, "*.jpg"))
        dog_paths = glob.glob(os.path.join(dogs_dir, "*.jpg"))

    images = []
    labels = []

    print(f"[LOAD] Loading {len(cat_paths)} cat images and {len(dog_paths)} dog images...")

    for path in cat_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(image_size)
            images.append(np.array(img, dtype=np.float32) / 255.0)  # Normalize 0-255 -> 0-1
            labels.append(0)  # Cat = 0
        except Exception as e:
            print(f"Skipping corrupted image: {path} ({e})")

    for path in dog_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(image_size)
            images.append(np.array(img, dtype=np.float32) / 255.0)  # Normalize 0-255 -> 0-1
            labels.append(1)  # Dog = 1
        except Exception as e:
            print(f"Skipping corrupted image: {path} ({e})")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    return X, y


def train_and_evaluate(
    dataset_dir="dataset",
    output_dir="outputs",
    model_dir="models",
    epochs=15,
    batch_size=32,
    learning_rate=0.001
):
    """
    Executes training, evaluation, plot generation, and model saving.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load Data
    X, y = load_image_dataset(dataset_dir=dataset_dir, image_size=(128, 128))
    print(f"[OK] Loaded total dataset: X shape = {X.shape}, y shape = {y.shape}")

    # 2. Train / Validation Split (80% Train, 20% Validation)
    # Stratified split ensures equal proportion of cats and dogs in both sets.
    # Splitting BEFORE augmentation prevents data leakage between train & validation sets!
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
        shuffle=True
    )

    print(f"[SPLIT] Dataset Split Complete:")
    print(f"   - Training Set   : {X_train.shape[0]} samples (Cats: {np.sum(y_train==0)}, Dogs: {np.sum(y_train==1)})")
    print(f"   - Validation Set : {X_val.shape[0]} samples (Cats: {np.sum(y_val==0)}, Dogs: {np.sum(y_val==1)})")

    # 3. Build Model with Data Augmentation Wrapper
    base_model = build_cat_dog_cnn(input_shape=(128, 128, 3), learning_rate=learning_rate)
    data_aug = get_data_augmentation_layer()

    # Wrap in sequential model for training with real-time augmentation
    training_model = tf.keras.Sequential([
        data_aug,
        base_model
    ], name="Augmented_Training_Pipeline")

    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n[MODEL] Custom CNN Architecture Summary:")
    base_model.summary()

    # 4. Train Model
    print(f"\n[TRAIN] Training CNN for {epochs} Epochs with Batch Size {batch_size}...")
    history = training_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Save standalone base model (without augmentation wrapper for clean inference)
    model_save_path = os.path.join(model_dir, "cat_dog_cnn.keras")
    base_model.save(model_save_path)
    print(f"\n[SAVE] Model successfully saved to: {model_save_path}")

    # 5. Plot Accuracy & Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='#1f77b4', linewidth=2.5)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linewidth=2.5, linestyle='--')
    ax1.set_title('Model Accuracy vs Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Loss Plot
    ax2.plot(history.history['loss'], label='Training Loss', color='#1f77b4', linewidth=2.5)
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2.5, linestyle='--')
    ax2.set_title('Model Loss vs Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    history_plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(history_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Training curves saved to: {history_plot_path}")

    # 6. Evaluate Model on Validation Set
    val_probs = base_model.predict(X_val, verbose=0).ravel()
    val_preds = (val_probs >= 0.5).astype(int)

    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds, zero_division=0)
    rec = recall_score(y_val, val_preds, zero_division=0)
    f1 = f1_score(y_val, val_preds, zero_division=0)
    cm = confusion_matrix(y_val, val_preds)

    # Print Detailed Viva Explanation of Metrics
    print("\n" + "="*60)
    print(" MODEL EVALUATION METRICS ON VALIDATION SET")
    print("="*60)
    print(f"  * Accuracy        : {acc*100:.2f}%  (Overall proportion of correct predictions)")
    print(f"  * Precision       : {prec*100:.2f}%  (Out of all predicted dogs, how many are actually dogs)")
    print(f"  * Recall (Sens.)  : {rec*100:.2f}%  (Out of all actual dogs, how many were correctly detected)")
    print(f"  * F1-Score        : {f1*100:.2f}%  (Harmonic mean of Precision and Recall)")
    print("="*60)

    print("\n CONFUSION MATRIX:")
    print("                  Predicted Cat (0)   Predicted Dog (1)")
    print(f"  Actual Cat (0)       {cm[0][0]:^14}    {cm[0][1]:^17}")
    print(f"  Actual Dog (1)       {cm[1][0]:^14}    {cm[1][1]:^17}")
    print("-" * 60)
    print("  Legend:")
    print(f"   - True Negatives (TN - Cat correctly identified as Cat)  : {cm[0][0]}")
    print(f"   - False Positives (FP - Cat wrongly predicted as Dog)    : {cm[0][1]}")
    print(f"   - False Negatives (FN - Dog wrongly predicted as Cat)    : {cm[1][0]}")
    print(f"   - True Positives (TP - Dog correctly identified as Dog)  : {cm[1][1]}")
    print("="*60 + "\n")

    # Plot Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Confusion matrix plot saved to: {cm_plot_path}")

    # 7. Display Sample Predictions & Mistakes
    plot_sample_predictions(X_val, y_val, val_probs, val_preds, output_dir=output_dir)

    return history, acc


def plot_sample_predictions(X_val, y_val, val_probs, val_preds, output_dir="outputs"):
    """
    Visualizes sample validation predictions, formatted as:
    Actual: Cat | Predicted: Cat | Confidence: 94%
    And displays any misclassified samples for error analysis.
    """
    class_names = {0: "Cat", 1: "Dog"}

    correct_indices = np.where(val_preds == y_val)[0]
    incorrect_indices = np.where(val_preds != y_val)[0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Validation Sample Predictions", fontsize=16, fontweight='bold')

    # Show 4 correct predictions
    sample_correct = np.random.choice(correct_indices, min(4, len(correct_indices)), replace=False)
    for i, idx in enumerate(sample_correct):
        ax = axes[0, i]
        ax.imshow(X_val[idx])
        actual_label = class_names[y_val[idx]]
        pred_label = class_names[val_preds[idx]]
        conf = val_probs[idx] if val_preds[idx] == 1 else (1.0 - val_probs[idx])
        ax.set_title(f"Actual: {actual_label}\nPred: {pred_label}\nConf: {conf*100:.1f}%", color='green', fontsize=10)
        ax.axis('off')

    # Show 4 misclassified or remaining predictions
    if len(incorrect_indices) > 0:
        sample_inc = np.random.choice(incorrect_indices, min(4, len(incorrect_indices)), replace=False)
        for i, idx in enumerate(sample_inc):
            ax = axes[1, i]
            ax.imshow(X_val[idx])
            actual_label = class_names[y_val[idx]]
            pred_label = class_names[val_preds[idx]]
            conf = val_probs[idx] if val_preds[idx] == 1 else (1.0 - val_probs[idx])
            ax.set_title(f"Actual: {actual_label}\nPred: {pred_label}\nConf: {conf*100:.1f}%", color='red', fontsize=10)
            ax.axis('off')
    else:
        # Fill remaining slots if 100% accurate
        sample_rem = np.random.choice(correct_indices, min(4, len(correct_indices)), replace=False)
        for i, idx in enumerate(sample_rem):
            ax = axes[1, i]
            ax.imshow(X_val[idx])
            actual_label = class_names[y_val[idx]]
            pred_label = class_names[val_preds[idx]]
            conf = val_probs[idx] if val_preds[idx] == 1 else (1.0 - val_probs[idx])
            ax.set_title(f"Actual: {actual_label}\nPred: {pred_label}\nConf: {conf*100:.1f}%", color='green', fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    sample_plot_path = os.path.join(output_dir, "sample_predictions.png")
    plt.savefig(sample_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Sample prediction visualization saved to: {sample_plot_path}")


if __name__ == "__main__":
    train_and_evaluate(epochs=15, batch_size=32, learning_rate=0.001)
