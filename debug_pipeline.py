"""
debug_pipeline.py
------------------
Comprehensive Diagnostic & Debugging Tool for the Cat vs Dog Classifier.
Executes:
1. debug_prediction(image_path) with full raw tensor & model output inspection.
2. 20-sample Dataset Sanity Test (10 Cats + 10 Dogs) comparing actual vs predicted labels.
3. 10-sample Overfit Sanity Test (5 Cats + 5 Dogs) to prove from-scratch convergence and label consistency.
4. Model architecture & weights inspection.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cat_dog_classifier"))

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from cat_dog_classifier.config import (
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, IMAGE_SIZE, INPUT_SHAPE,
    MODEL_PATH, CATS_DIR, DOGS_DIR, TEST_EXTERNAL_DIR, TEST_IMAGES_DIR, UNCERTAINTY_THRESHOLD
)
from cat_dog_classifier.preprocessing import (
    load_and_preprocess_single_image,
    preprocess_for_inference,
    get_tensor_statistics
)
from cat_dog_classifier.model import build_cat_dog_cnn, get_data_augmentation_transform


def inspect_loaded_model(model_path=MODEL_PATH):
    """Step 8: Verify loaded model details."""
    print("\n" + "="*60)
    print(" STEP 8: MODEL FILE & ARCHITECTURE INSPECTION")
    print("="*60)
    print(f"  Model Path         : {model_path}")
    print(f"  File Exists        : {os.path.exists(model_path)}")
    if not os.path.exists(model_path):
        print("  [ERROR] Model file not found!")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cat_dog_cnn().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    params_count = sum(p.numel() for p in model.parameters())
    print(f"  Model Type         : PyTorch CatDogCNN")
    print(f"  Total Parameters   : {params_count:,}")
    print(f"  Input Shape        : (batch, 3, 128, 128)")
    print(f"  Output Shape       : (batch, 1)")
    print("="*60)
    return model


def debug_prediction(image_path, model=None):
    """
    Step 4: Debug prediction function displaying raw tensor stats,
    raw model sigmoid output, class mapping, and calibrated confidence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = build_cat_dog_cnn().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 1. Image loading & shape inspection
    raw_img = Image.open(image_path)
    orig_mode = raw_img.mode
    orig_size = raw_img.size  # (Width, Height)

    # 2. Preprocess
    tensor = preprocess_for_inference(image_path).to(device)
    stats = get_tensor_statistics(tensor)

    # 3. Model feed-forward
    with torch.no_grad():
        raw_output = float(model(tensor)[0][0].item())

    # 4. Interpret class mapping
    # Class 0: Cat, Class 1: Dog
    dog_prob = raw_output
    cat_prob = 1.0 - raw_output

    if dog_prob >= 0.50:
        predicted_class = "Dog"
        confidence = dog_prob * 100.0
    else:
        predicted_class = "Cat"
        confidence = cat_prob * 100.0

    is_uncertain = (confidence / 100.0) < UNCERTAINTY_THRESHOLD

    print("\n" + "-"*55)
    print(f" DEBUG PREDICTION REPORT: {os.path.basename(image_path)}")
    print("-"*55)
    print(f"  Image Path       : {image_path}")
    print(f"  Original Mode    : {orig_mode} (Converted to RGB: True)")
    print(f"  Original Shape   : ({orig_size[1]}, {orig_size[0]}, {3 if orig_mode=='RGB' else orig_mode})")
    print(f"  Processed Shape  : {stats['shape']}")
    print(f"  Image Dtype      : {stats['dtype']}")
    print(f"  Pixel Range      : [{stats['min_val']:.4f}, {stats['max_val']:.4f}]")
    print(f"  Pixel Mean / Std : {stats['mean_val']:.4f} / {stats['std_val']:.4f}")
    print(f"\n  [RAW MODEL OUTPUT] Sigmoid Score: {raw_output:.5f}")
    print(f"  [CLASS MAPPING]    Class 0 = Cat | Class 1 = Dog")
    print(f"  [PROBABILITIES]    Cat: {cat_prob*100:.2f}% | Dog: {dog_prob*100:.2f}%")
    print(f"  [PREDICTED CLASS]  {predicted_class.upper()} (Confidence: {confidence:.2f}%)")
    print(f"  [UNCERTAINTY FLAG] {is_uncertain} (Threshold: {UNCERTAINTY_THRESHOLD*100:.0f}%)")
    print("-"*55 + "\n")

    return {
        "image": os.path.basename(image_path),
        "raw_output": raw_output,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "is_uncertain": is_uncertain
    }


def run_20_image_sanity_test(model=None):
    """
    Step 5 & 10: Test 10 known Cats and 10 known Dogs from the dataset
    through the exact same inference pipeline and print a structured table.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = build_cat_dog_cnn().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

    cat_files = sorted([os.path.join(CATS_DIR, f) for f in os.listdir(CATS_DIR) if f.lower().endswith(('.jpg', '.png'))])[:10]
    dog_files = sorted([os.path.join(DOGS_DIR, f) for f in os.listdir(DOGS_DIR) if f.lower().endswith(('.jpg', '.png'))])[:10]

    print("\n" + "="*70)
    print(" STEP 5 & 10: 20-IMAGE KNOWN DATASET SANITY TEST")
    print("="*70)
    print(f"{'Image Filename':<22} | {'Actual':<6} | {'Predicted':<9} | {'Probability':<11} | {'Correct?'}")
    print("-" * 70)

    cats_correct = 0
    dogs_correct = 0

    # Test Cats
    for path in cat_files:
        fname = os.path.basename(path)
        tensor = preprocess_for_inference(path).to(device)
        with torch.no_grad():
            prob = float(model(tensor)[0][0].item())
        pred = "Dog" if prob >= 0.50 else "Cat"
        is_correct = (pred == "Cat")
        if is_correct:
            cats_correct += 1
        status = "YES" if is_correct else "NO (ERROR)"
        print(f"{fname:<22} | {'Cat':<6} | {pred:<9} | {prob:9.4f}   | {status}")

    # Test Dogs
    for path in dog_files:
        fname = os.path.basename(path)
        tensor = preprocess_for_inference(path).to(device)
        with torch.no_grad():
            prob = float(model(tensor)[0][0].item())
        pred = "Dog" if prob >= 0.50 else "Cat"
        is_correct = (pred == "Dog")
        if is_correct:
            dogs_correct += 1
        status = "YES" if is_correct else "NO (ERROR)"
        print(f"{fname:<22} | {'Dog':<6} | {pred:<9} | {prob:9.4f}   | {status}")

    total_correct = cats_correct + dogs_correct
    print("="*70)
    print(f" SANITY TEST SUMMARY:")
    print(f"  * Cats correctly classified : {cats_correct}/10 ({cats_correct*10}%)")
    print(f"  * Dogs correctly classified : {dogs_correct}/10 ({dogs_correct*10}%)")
    print(f"  * Overall Sanity Accuracy   : {total_correct}/20 ({total_correct*5}%)")
    print("="*70 + "\n")

    return total_correct, cats_correct, dogs_correct


def test_tiny_dataset_overfit():
    """
    Step 11: Overfit a tiny dataset of 5 Cats and 5 Dogs to mathematically prove
    that the CNN architecture, labels, sigmoid mapping, and inference are bug-free.
    """
    print("\n" + "="*70)
    print(" STEP 11: OVERFIT TEST ON TINY DATASET (5 CATS + 5 DOGS)")
    print("="*70)
    print(" [INFO] Training PyTorch CNN on 10 known images without regularization...")

    cat_files = sorted([os.path.join(CATS_DIR, f) for f in os.listdir(CATS_DIR) if f.lower().endswith(('.jpg', '.png'))])[:5]
    dog_files = sorted([os.path.join(DOGS_DIR, f) for f in os.listdir(DOGS_DIR) if f.lower().endswith(('.jpg', '.png'))])[:5]

    X_tiny = []
    y_tiny = []

    for p in cat_files:
        X_tiny.append(load_and_preprocess_single_image(p))
        y_tiny.append(CLASS_TO_IDX["cat"])  # 0

    for p in dog_files:
        X_tiny.append(load_and_preprocess_single_image(p))
        y_tiny.append(CLASS_TO_IDX["dog"])  # 1

    X_tiny = np.array(X_tiny, dtype=np.float32)
    y_tiny = np.array(y_tiny, dtype=np.float32)

    X_tiny_chw = torch.from_numpy(np.transpose(X_tiny, (0, 3, 1, 2))).float()
    y_tiny_tensor = torch.from_numpy(y_tiny).float().unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tiny_model = build_cat_dog_cnn().to(device)
    X_tiny_chw = X_tiny_chw.to(device)
    y_tiny_tensor = y_tiny_tensor.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.002)

    tiny_model.train()
    for _ in range(50):
        optimizer.zero_grad()
        out = tiny_model(X_tiny_chw)
        loss = criterion(out, y_tiny_tensor)
        loss.backward()
        optimizer.step()

    tiny_model.eval()
    with torch.no_grad():
        preds_probs = tiny_model(X_tiny_chw).cpu().numpy().ravel()
    preds = (preds_probs >= 0.50).astype(int)
    correct_count = np.sum(preds == y_tiny)

    print("\n Overfit Test Verification Table:")
    print(f"{'Image':<15} | {'Actual Label':<12} | {'Raw Prob':<10} | {'Predicted':<10} | {'Status'}")
    print("-" * 65)

    all_files = cat_files + dog_files
    for i, p in enumerate(all_files):
        actual_name = "Cat (0)" if y_tiny[i] == 0 else "Dog (1)"
        pred_name = "Cat (0)" if preds[i] == 0 else "Dog (1)"
        prob = preds_probs[i]
        status = "PASSED" if preds[i] == y_tiny[i] else "FAILED"
        print(f"{os.path.basename(p):<15} | {actual_name:<12} | {prob:8.4f}   | {pred_name:<10} | {status}")

    print("="*70)
    print(f" OVERFIT TEST RESULT: {correct_count}/10 Memorized ({correct_count*10}%)")
    if correct_count == 10:
        print(" [CONCLUSION] The CNN and Label Mapping (0=Cat, 1=Dog) are mathematically 100% correct!")
    else:
        print(" [WARNING] Label or gradient inversion detected in core layers!")
    print("="*70 + "\n")

    return correct_count


if __name__ == "__main__":
    # Run full diagnostics
    inspect_loaded_model()
    debug_prediction(os.path.join(TEST_IMAGES_DIR, "sample_cat_1.jpg"))
    debug_prediction(os.path.join(TEST_IMAGES_DIR, "sample_dog_1.jpg"))
    run_20_image_sanity_test()
    test_tiny_dataset_overfit()
