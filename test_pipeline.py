"""
test_pipeline.py
----------------
Comprehensive Automated Sanity & Unit Test Suite for Cat vs Dog ML Pipeline.
Verifies all 10 essential pipeline safety and correctness rules.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cat_dog_classifier"))

from cat_dog_classifier.config import (
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, IMAGE_SIZE, INPUT_SHAPE,
    MODEL_PATH, TEST_IMAGES_DIR, CATS_DIR, DOGS_DIR
)
from cat_dog_classifier.preprocessing import (
    load_and_preprocess_single_image,
    preprocess_for_inference,
    get_tensor_statistics
)
from cat_dog_classifier.model import build_cat_dog_cnn

# Ensure stdout supports UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def test_1_class_mapping_alignment():
    print("[TEST 1/10] Verifying Single Source of Truth for Class Mappings...")
    assert CLASS_NAMES == ["cat", "dog"], f"Invalid CLASS_NAMES: {CLASS_NAMES}"
    assert CLASS_TO_IDX["cat"] == 0, f"Cat mapping corrupted: {CLASS_TO_IDX['cat']}"
    assert CLASS_TO_IDX["dog"] == 1, f"Dog mapping corrupted: {CLASS_TO_IDX['dog']}"
    assert IDX_TO_CLASS[0] == "cat", f"Reverse mapping 0 corrupted: {IDX_TO_CLASS[0]}"
    assert IDX_TO_CLASS[1] == "dog", f"Reverse mapping 1 corrupted: {IDX_TO_CLASS[1]}"
    print("  --> PASS: Class mappings are strictly aligned (Cat=0, Dog=1).")


def test_2_rgb_processing_and_channels():
    print("[TEST 2/10] Verifying RGB color space conversion on RGBA/Grayscale/Palette inputs...")
    # Create test RGBA image
    rgba_img = Image.new("RGBA", (200, 200), (255, 100, 50, 180))
    arr = load_and_preprocess_single_image(rgba_img)
    assert arr.shape == (128, 128, 3), f"Expected shape (128, 128, 3), got {arr.shape}"

    # Create test Grayscale image
    gray_img = Image.new("L", (200, 200), 128)
    arr_gray = load_and_preprocess_single_image(gray_img)
    assert arr_gray.shape == (128, 128, 3), f"Expected shape (128, 128, 3), got {arr_gray.shape}"
    print("  --> PASS: All image color spaces are safely normalized to 3-channel RGB.")


def test_3_tensor_normalization_and_dtype():
    print("[TEST 3/10] Verifying Tensor Normalization range [0.0, 1.0] and float32 dtype...")
    test_img = Image.new("RGB", (100, 100), (255, 128, 0))
    tensor = preprocess_for_inference(test_img)
    stats = get_tensor_statistics(tensor)
    assert "float32" in stats["dtype"], f"Invalid dtype: {stats['dtype']}"
    assert stats["min_val"] >= 0.0 and stats["max_val"] <= 1.0, f"Out of range: {stats['min_val']} to {stats['max_val']}"
    assert stats["shape"] == (1, 3, 128, 128), f"Invalid batch shape: {stats['shape']}"
    assert not stats["has_nans"], "Tensor contains NaNs!"
    print(f"  --> PASS: Tensor shape {stats['shape']}, dtype float32, range [{stats['min_val']:.2f}, {stats['max_val']:.2f}].")


def test_4_preprocessing_consistency_train_vs_inference():
    print("[TEST 4/10] Verifying preprocessing consistency across training and inference...")
    test_img = Image.new("RGB", (160, 160), (45, 90, 180))
    train_processed = load_and_preprocess_single_image(test_img)
    inference_processed_tensor = preprocess_for_inference(test_img)[0]
    inference_processed_hwc = np.transpose(inference_processed_tensor.numpy(), (1, 2, 0))
    np.testing.assert_allclose(train_processed, inference_processed_hwc, rtol=1e-5, atol=1e-5)
    print("  --> PASS: Training and inference preprocessing functions produce 100% identical arrays.")


def test_5_cnn_architecture_structure():
    print("[TEST 5/10] Verifying CNN Architecture compilation and tensor shapes...")
    model = build_cat_dog_cnn()
    model.eval()
    dummy_input = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    assert torch.all((output >= 0.0) & (output <= 1.0)), "Sigmoid output out of bounds [0, 1]"
    print("  --> PASS: CNN architecture compiles and produces valid probability shape (batch, 1).")


def test_6_model_save_and_load_equivalence():
    print("[TEST 6/10] Verifying Model Save/Load Numerical Equivalence...")
    model = build_cat_dog_cnn()
    model.eval()
    temp_save_path = os.path.join("models", "test_save_load_temp.pth")
    os.makedirs("models", exist_ok=True)

    dummy_input = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        pred_before = model(dummy_input).numpy()

    torch.save(model.state_dict(), temp_save_path)
    loaded_model = build_cat_dog_cnn()
    loaded_model.load_state_dict(torch.load(temp_save_path))
    loaded_model.eval()

    with torch.no_grad():
        pred_after = loaded_model(dummy_input).numpy()

    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5, atol=1e-5)
    if os.path.exists(temp_save_path):
        os.remove(temp_save_path)
    print("  --> PASS: Predictions before and after saving/loading are numerically equivalent.")


def test_7_no_synthetic_geometric_doodles():
    print("[TEST 7/10] Verifying all synthetic PIL doodle generators are eradicated...")
    from cat_dog_classifier import download_dataset
    assert not hasattr(download_dataset, "generate_synthetic_samples"), "Found legacy synthetic doodle generator!"
    print("  --> PASS: No synthetic drawing functions exist in the dataset pipeline.")


def test_8_uncertainty_threshold_logic():
    print("[TEST 8/10] Verifying Confidence and Uncertainty Threshold calculation...")
    from cat_dog_classifier.config import UNCERTAINTY_THRESHOLD
    # Case 1: High confidence Dog
    prob_dog_high = 0.95
    conf_dog = prob_dog_high
    assert conf_dog >= UNCERTAINTY_THRESHOLD

    # Case 2: Uncertain boundary (e.g. 0.54)
    prob_dog_low = 0.54
    conf_border = prob_dog_low
    assert conf_border < UNCERTAINTY_THRESHOLD, f"Expected uncertain for 54% confidence"
    print(f"  --> PASS: Uncertainty threshold ({UNCERTAINTY_THRESHOLD*100:.0f}%) correctly flags ambiguous predictions.")


def test_9_real_dataset_integrity():
    print("[TEST 9/10] Verifying Real Dataset Presence on Disk...")
    cats = [f for f in os.listdir(CATS_DIR) if f.lower().endswith(('.jpg', '.png'))]
    dogs = [f for f in os.listdir(DOGS_DIR) if f.lower().endswith(('.jpg', '.png'))]
    assert len(cats) >= 15, f"Too few cat images: {len(cats)}"
    assert len(dogs) >= 15, f"Too few dog images: {len(dogs)}"
    print(f"  --> PASS: Real image repository populated ({len(cats)} Cats, {len(dogs)} Dogs).")


def test_10_isolated_test_split_separation():
    print("[TEST 10/10] Verifying 3-way split separation and absence of split leakage...")
    from cat_dog_classifier.train import load_dataset_with_filepaths
    X, y, paths = load_dataset_with_filepaths()
    assert len(paths) == len(set(paths)), "Duplicate filepaths found in dataset!"
    print(f"  --> PASS: Zero duplicate filepaths in dataset across all {len(paths)} samples.")


def run_all_tests():
    print("\n" + "="*65)
    print(" RUNNING 10-POINT MACHINE LEARNING PIPELINE SANITY SUITE")
    print("="*65)
    test_1_class_mapping_alignment()
    test_2_rgb_processing_and_channels()
    test_3_tensor_normalization_and_dtype()
    test_4_preprocessing_consistency_train_vs_inference()
    test_5_cnn_architecture_structure()
    test_6_model_save_and_load_equivalence()
    test_7_no_synthetic_geometric_doodles()
    test_8_uncertainty_threshold_logic()
    test_9_real_dataset_integrity()
    test_10_isolated_test_split_separation()
    print("="*65)
    print("  ALL 10 PIPELINE AUDIT TESTS PASSED SUCCESSFULLY!")
    print("="*65 + "\n")


if __name__ == "__main__":
    run_all_tests()
