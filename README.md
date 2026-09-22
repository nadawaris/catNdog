# 🐱 vs 🐶 Cats vs Dogs Binary Image Classification (Built From Scratch)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0%2B-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

A complete, beginner-friendly **Machine Learning & Deep Learning project** that classifies input images into **Cats (0)** or **Dogs (1)** using a custom **Convolutional Neural Network (CNN) built entirely from scratch in Keras/TensorFlow**.

> ⚠️ **Strict Project Requirement**: **NO Transfer Learning or Pretrained Models** (such as VGG16, ResNet, MobileNet, EfficientNet) were used. Every layer, tensor shape transformation, activation function, and hyperparameter is constructed line-by-line to ensure complete transparency for **B.Tech lab exams and Viva Voce presentations**.

---

## 📌 Table of Contents
- [✨ Key Features](#-key-features)
- [📂 Project Hierarchy](#-project-hierarchy)
- [🧠 CNN Architecture & Layer Breakdown](#-cnn-architecture--layer-breakdown)
- [📐 Mathematical Foundations](#-mathematical-foundations)
- [⚙️ Installation & Setup](#%EF%B8%8F-installation--setup)
- [🚀 Quickstart & Execution Guide](#-quickstart--execution-guide)
  - [1. Dataset Preparation](#1-dataset-preparation)
  - [2. Model Training & Evaluation](#2-model-training--evaluation)
  - [3. Single-Image Inference](#3-single-image-inference)
- [📊 Evaluation Metrics & Output Visualizations](#-evaluation-metrics--output-visualizations)
- [🎓 B.Tech Viva Voce Companion (`VIVA.md`)](#-btech-viva-voce-companion-vivamd)
- [🤝 Contributing & License](#-contributing--license)

---

## ✨ Key Features

- 🏗️ **Custom CNN Built From Scratch**: Modular, 3-block convolutional neural network without relying on external pretrained weights.
- 🛡️ **Leakage-Free Train/Val Split**: 80% Training / 20% Validation split with stratification performed *before* data augmentation to guarantee zero data leakage.
- 🔄 **Real-Time Data Augmentation**: In-pipeline geometric transformations (Random Horizontal Flip, $\pm 10\%$ Rotation, $\pm 10\%$ Zoom) to combat overfitting.
- 📊 **End-to-End Metrics & Visualizations**: Automatically computes Accuracy, Precision, Recall, F1-Score, and generates Confusion Matrix heatmaps + Loss/Accuracy curves.
- 🤖 **Automated Dataset Generator**: Includes `download_dataset.py` to fetch/generate sample cat and dog images so the project runs out-of-the-box in under 1 minute.
- 🖼️ **CLI Prediction Tool**: Command-line tool and helper function `predict_image("path/to/image.jpg")` with percentage confidence output and graphical overlays.
- 🎓 **Viva Study Guide**: Accompanied by a dedicated 25-question [VIVA.md](VIVA.md) guide tailored for engineering students.

---

## 📂 Project Hierarchy

```text
catNdog/
│
├── dataset/                        # Image dataset directory
│   ├── cats/                       # Cat images (Label: 0)
│   └── dogs/                       # Dog images (Label: 1)
│
├── models/                         # Saved Keras model artifacts
│   └── cat_dog_cnn.keras           # Trained custom CNN model weights
│
├── outputs/                        # Generated plots & visual reports
│   ├── training_history.png        # Loss & Accuracy curves over 15 epochs
│   ├── confusion_matrix.png        # Confusion Matrix heatmap
│   ├── sample_predictions.png      # Validation predictions with confidence %
│   └── last_prediction.png         # Single-image prediction visual overlay
│
├── test_images/                    # Sample test images for inference
│   ├── sample_cat_1.jpg
│   └── sample_dog_1.jpg
│
├── cat_dog_classifier/             # Main Python package module
│   ├── model.py                    # Custom CNN architecture definition
│   ├── train.py                    # Preprocessing, data augmentation & training pipeline
│   ├── predict.py                  # Single-image prediction CLI & function
│   ├── download_dataset.py         # Automated dataset download & sample generator
│   ├── requirements.txt            # Package dependencies
│   ├── README.md                   # Module documentation
│   └── VIVA.md                     # 25 Viva Voce Q&A study guide
│
├── model.py                        # Top-level proxy script
├── train.py                        # Top-level training runner
├── predict.py                      # Top-level prediction CLI runner
├── download_dataset.py             # Top-level dataset runner
├── requirements.txt                # Root dependency list
├── .gitignore                      # Git exclusion rules
├── README.md                       # Comprehensive documentation (this file)
└── VIVA.md                         # 25 Viva Voce questions & detailed answers
```

---

## 🧠 CNN Architecture & Layer Breakdown

The neural network processes RGB input images of resolution $128 \times 128 \times 3$ through three feature-extraction convolutional blocks followed by a fully connected classification head.

```text
Input Image (128 × 128 × 3)
        │
        ▼
[ Conv2D (32 Filters, 3×3) ] ──► [ ReLU Activation ] ──► [ MaxPooling2D (2×2) ]
        │  Output Shape: (128, 128, 32)                          Output Shape: (64, 64, 32)
        ▼
[ Conv2D (64 Filters, 3×3) ] ──► [ ReLU Activation ] ──► [ MaxPooling2D (2×2) ]
        │  Output Shape: (64, 64, 64)                            Output Shape: (32, 32, 64)
        ▼
[ Conv2D (128 Filters, 3×3) ] ──► [ ReLU Activation ] ──► [ MaxPooling2D (2×2) ]
        │  Output Shape: (32, 32, 128)                           Output Shape: (16, 16, 128)
        ▼
[ Flatten Layer ] ───────────────────────────────────────► Vector Size: 32,768
        │
        ▼
[ Dense Layer (128 Units) ] ──► [ Dropout (Rate = 0.5) ]
        │
        ▼
[ Dense Output Layer (1 Unit, Sigmoid Activation) ]
        │
        ▼
Output Probability Score  ──►  [ < 0.5 : Cat 🐱 | ≥ 0.5 : Dog 🐶 ]
```

### Tensor Shape & Parameter Transformation Summary

| Layer Name | Type | Output Tensor Shape | Param # | Purpose / Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `data_augmentation` | Sequential | `(None, 128, 128, 3)` | 0 | Applies random flips, rotation & zoom during training |
| `conv2d_1` | Conv2D | `(None, 128, 128, 32)` | 896 | Extracts low-level features (edges, borders, lines) |
| `relu_1` | Activation | `(None, 128, 128, 32)` | 0 | Introduces non-linearity: $f(x) = \max(0, x)$ |
| `maxpooling_1` | MaxPooling2D | `(None, 64, 64, 32)` | 0 | Downsamples height & width by 50% |
| `conv2d_2` | Conv2D | `(None, 64, 64, 64)` | 18,496 | Extracts mid-level features (textures, fur patterns) |
| `relu_2` | Activation | `(None, 64, 64, 64)` | 0 | Non-linear activation |
| `maxpooling_2` | MaxPooling2D | `(None, 32, 32, 64)` | 0 | Downsamples spatial resolution |
| `conv2d_3` | Conv2D | `(None, 32, 32, 128)` | 73,856 | Extracts high-level features (ears, snout, eyes) |
| `relu_3` | Activation | `(None, 32, 32, 128)` | 0 | Non-linear activation |
| `maxpooling_3` | MaxPooling2D | `(None, 16, 16, 128)` | 0 | Final feature map downsampling |
| `flatten` | Flatten | `(None, 32768)` | 0 | Reshapes 3D tensor to 1D vector ($16 \times 16 \times 128$) |
| `dense_1` | Dense | `(None, 128)` | 4,194,432 | Fully connected decision-making hidden layer |
| `dropout_1` | Dropout | `(None, 128)` | 0 | Randomly drops 50% of units to prevent overfitting |
| `output_layer` | Dense | `(None, 1)` | 129 | Sigmoid neuron producing class probability |

---

## 📐 Mathematical Foundations

### 1. Convolution Operation
For an input image $I$ and a filter kernel $K$ of size $k \times k$:
$$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i - m, j - n) K(m, n)$$

### 2. Activation Functions
- **ReLU (Rectified Linear Unit)** (Hidden Layers):
  $$f(x) = \max(0, x)$$
- **Sigmoid Activation** (Output Layer):
  $$\sigma(z) = \frac{1}{1 + e^{-z}} \quad \in [0.0, 1.0]$$

### 3. Binary Cross-Entropy Loss Function
$$L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
where $y_i \in \{0, 1\}$ is the ground truth label and $\hat{y}_i \in [0, 1]$ is the predicted probability.

### 4. Evaluation Metrics
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- `pip` package manager

### Step-by-Step Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nadawaris/catNdog.git
   cd catNdog
   ```

2. **Create a Virtual Environment**:
   ```bash
   # On Windows
   py -3.13 -m venv .venv
   .\.venv\Scripts\activate

   # On Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quickstart & Execution Guide

### 1. Dataset Preparation
If you do not have a local dataset ready in `dataset/cats` and `dataset/dogs`, run the automated dataset utility:
```bash
python download_dataset.py
```
> This populates `dataset/cats/` with cat images and `dataset/dogs/` with dog images, and generates test images in `test_images/`.

### 2. Model Training & Evaluation
To run preprocessing, train the CNN from scratch, evaluate performance, generate output plots, and save the model:
```bash
python train.py
```

**Training Console Log Example**:
```text
[LOAD] Loading 50 cat images and 50 dog images...
[OK] Loaded total dataset: X shape = (100, 128, 128, 3), y shape = (100,)
[SPLIT] Dataset Split Complete:
   - Training Set   : 80 samples (Cats: 40, Dogs: 40)
   - Validation Set : 20 samples (Cats: 10, Dogs: 10)

[TRAIN] Training CNN for 15 Epochs with Batch Size 32...
Epoch 15/15 — loss: 0.0003 — accuracy: 1.0000 — val_loss: 0.0000 — val_accuracy: 1.0000

[SAVE] Model successfully saved to: models\cat_dog_cnn.keras
[PLOT] Training curves saved to: outputs\training_history.png

============================================================
 MODEL EVALUATION METRICS ON VALIDATION SET
============================================================
  * Accuracy        : 100.00%
  * Precision       : 100.00%
  * Recall (Sens.)  : 100.00%
  * F1-Score        : 100.00%
============================================================
```

### 3. Single-Image Inference
To classify any test image and view prediction confidence:
```bash
python predict.py test_images/sample_cat_1.jpg
```
or
```bash
python predict.py test_images/sample_dog_1.jpg
```

**Output Result**:
```text
=============================================
 BINARY IMAGE CLASSIFICATION RESULT
=============================================
  Image Path  : test_images/sample_dog_1.jpg
  Prediction  : DOG 🐶
  Confidence  : 100.00%
  Raw Sigmoid : 1.0000
=============================================
[PLOT] Prediction visualization saved to: outputs/last_prediction.png
```

---

## 📊 Evaluation Metrics & Output Visualizations

| Generated Report File | Location | Description |
| :--- | :--- | :--- |
| **Training History** | [`outputs/training_history.png`](outputs/training_history.png) | Dual plot of Training vs. Validation Accuracy and Loss curves over 15 epochs |
| **Confusion Matrix** | [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png) | Seaborn/Matplotlib heatmap showing True Positives, True Negatives, False Positives, False Negatives |
| **Validation Predictions** | [`outputs/sample_predictions.png`](outputs/sample_predictions.png) | 8-sample grid displaying actual label, predicted label, and confidence score |
| **Single Image Prediction** | [`outputs/last_prediction.png`](outputs/last_prediction.png) | Visual plot overlay generated during `predict.py` execution |

---

## 🎓 B.Tech Viva Voce Companion (`VIVA.md`)

A comprehensive 25-question exam preparation guide is available in [VIVA.md](VIVA.md). Key topics covered include:

1. **Binary Classification Definitions**: Why Cat vs. Dog is binary classification.
2. **CNN Principles**: Why CNNs outperform ANNs on images (Spatial structure preservation, Parameter sharing, Translation invariance).
3. **Layer Mechanics**: Convolution filters, stride, zero padding, and feature map dimensions.
4. **Activation Functions**: Why ReLU in hidden layers and Sigmoid in the output layer.
5. **Loss & Optimizer**: Binary Cross-Entropy (Log Loss) and Adam optimizer inner mechanics.
6. **Hyperparameters**: Epochs, batch size, learning rate (`0.001`).
7. **Overfitting & Generalization**: Dropout regularization and real-time data augmentation.
8. **Data Leakage**: Why splitting before augmentation prevents data leakage.
9. **Evaluation Metrics**: Precision, Recall, F1-Score, and Confusion Matrix interpretation.

---

## 🤝 Contributing & License

Contributions, issues, and feature requests are welcome!

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<p align="center">
Made with ❤️ for B.Tech Computer Science & AI/ML Students
</p>
