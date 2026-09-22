"""
Top-level entrypoint for download_dataset.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cat_dog_classifier"))
from cat_dog_classifier.download_dataset import setup_real_dataset

if __name__ == "__main__":
    setup_real_dataset()
