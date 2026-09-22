"""
cat_dog_classifier / config.py
------------------------------
Single Source of Truth for:
- Class label mapping (Cat: 0, Dog: 1)
- Image dimensions & tensor constraints
- Uncertainty decision thresholds
- Standard project paths and hyperparameters
"""

import os

# =============================================================================
# 1. CANONICAL CLASS MAPPING (SINGLE SOURCE OF TRUTH)
# =============================================================================
CLASS_NAMES = ["cat", "dog"]
CLASS_TO_IDX = {"cat": 0, "dog": 1}
IDX_TO_CLASS = {0: "cat", 1: "dog"}
CLASS_EMOJIS = {"cat": "🐱", "dog": "🐶", "uncertain": "❓"}

# =============================================================================
# 2. IMAGE PREPROCESSING SPECIFICATIONS
# =============================================================================
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
NUM_CHANNELS = 3
INPUT_SHAPE = (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
COLOR_MODE = "RGB"
NORMALIZATION_RANGE = (0.0, 1.0)  # Pixel range [0.0, 1.0]

# =============================================================================
# 3. CONFIDENCE & DECISION THRESHOLDS
# =============================================================================
# Binary classification threshold
DECISION_THRESHOLD = 0.50

# Predictions with confidence below UNCERTAINTY_THRESHOLD are flagged as Uncertain
UNCERTAINTY_THRESHOLD = 0.60

# =============================================================================
# 4. TRAINING HYPERPARAMETERS
# =============================================================================
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)  # Train (70%), Val (15%), Test (15%)
RANDOM_SEED = 42

# =============================================================================
# 5. DIRECTORY PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CATS_DIR = os.path.join(DATASET_DIR, "cats")
DOGS_DIR = os.path.join(DATASET_DIR, "dogs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "cat_dog_cnn.pth")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MISCLASSIFIED_DIR = os.path.join(MODELS_DIR, "misclassified")
MISCLASSIFIED_CSV = os.path.join(MODELS_DIR, "misclassified_predictions.csv")
TEST_EXTERNAL_DIR = os.path.join(BASE_DIR, "test_external")
TEST_EXTERNAL_CSV = os.path.join(MODELS_DIR, "external_predictions.csv")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")
