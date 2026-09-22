"""
Top-level entrypoint for model.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cat_dog_classifier"))
from cat_dog_classifier.model import build_cat_dog_cnn, get_data_augmentation_layer

if __name__ == "__main__":
    cnn = build_cat_dog_cnn()
    cnn.summary()
