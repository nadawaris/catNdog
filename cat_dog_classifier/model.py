"""
cat_dog_classifier / model.py
------------------------------
Custom Convolutional Neural Network (CNN) built 100% from scratch in PyTorch.
Optimized for robust convergence and discriminative feature separation on real-world photos.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms

try:
    from config import INPUT_SHAPE, DEFAULT_LEARNING_RATE
except ImportError:
    from cat_dog_classifier.config import INPUT_SHAPE, DEFAULT_LEARNING_RATE


class CatDogCNN(nn.Module):
    """
    Constructs a 3-stage feature extractor followed by a dense classification head.
    Input shape: (N, 3, 128, 128)
    """
    def __init__(self):
        super(CatDogCNN, self).__init__()
        
        # Block 1: Low-level edges, colors, and textures (3x128x128 -> 32x64x64)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.20)
        
        # Block 2: Mid-level fur patterns, contours, and corners (32x64x64 -> 64x32x32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.20)
        
        # Block 3: High-level ear, snout, and eye structures (64x32x32 -> 128x16x16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.30)
        
        # Classification Head (128 * 16 * 16 = 32768)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.relu_fc = nn.ReLU()
        self.drop_fc = nn.Dropout(0.40)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(self.pool1(self.relu1(self.conv1(x))))
        x = self.drop2(self.pool2(self.relu2(self.conv2(x))))
        x = self.drop3(self.pool3(self.relu3(self.conv3(x))))
        x = self.flatten(x)
        x = self.drop_fc(self.relu_fc(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x


def build_cat_dog_cnn():
    return CatDogCNN()


def get_data_augmentation_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.9, 1.0)),
    ])


if __name__ == "__main__":
    model = build_cat_dog_cnn()
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    print(f"[INFO] PyTorch Model Output Shape: {output.shape}")

