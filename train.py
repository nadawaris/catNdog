"""
Top-level entrypoint for train.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cat_dog_classifier"))
from cat_dog_classifier.train import train_and_evaluate_pipeline

if __name__ == "__main__":
    train_and_evaluate_pipeline(epochs=25, batch_size=32, learning_rate=0.0005)
