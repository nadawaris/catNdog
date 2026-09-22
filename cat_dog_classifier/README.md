# 🐱 vs 🐶 Cats vs Dogs Binary Image Classification (From Scratch)

A beginner-friendly Machine Learning / Deep Learning project that classifies images into **Cats (0)** or **Dogs (1)** using a **custom Convolutional Neural Network (CNN) built from scratch in Keras/TensorFlow**.

> ⚠️ **Important Requirement**: No transfer learning or pretrained models (such as VGG16, ResNet, MobileNet) are used. Every layer and operation is defined line-by-line so that it can be understood and explained in a B.Tech viva exam!

---

## 📁 Project Structure

```text
cat_dog_classifier/
│
├── dataset/                    # Image dataset folder
│   ├── cats/                   # Cat images (Label: 0)
│   └── dogs/                   # Dog images (Label: 1)
│
├── models/                     # Saved model artifacts
│   └── cat_dog_cnn.keras       # Trained Keras CNN model
│
├── outputs/                    # Output plots & visualizations
│   ├── training_history.png    # Training vs Validation accuracy & loss
│   ├── confusion_matrix.png    # Confusion matrix heatmap
│   └── sample_predictions.png  # Test predictions with confidence scores
│
├── test_images/                # Sample test images for inference
│
├── model.py                    # Custom CNN architecture definition
├── train.py                    # Preprocessing, data augmentation & training pipeline
├── predict.py                  # Single image prediction CLI & function
├── download_dataset.py         # Automated dataset download & sample generator
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
└── VIVA.md                     # 20+ Viva Voce questions and answers
```

---

## 🛠️ Tech Stack

* **Python 3.10+ / 3.13 / 3.14**
* **TensorFlow / Keras** (Deep learning model building & training)
* **NumPy** (Array manipulation & matrix operations)
* **Pandas** (Data formatting & metrics reporting)
* **Matplotlib** (Visualization plots for loss/accuracy curves & predictions)
* **Scikit-Learn** (Stratified train/test split & evaluation metrics)
* **Pillow (PIL)** (Image loading & resizing)

---

## ⚙️ Installation & Setup

1. **Clone or Navigate to the Project Folder**:
   ```bash
   cd cat_dog_classifier
   ```

2. **Set up a Python Virtual Environment (Recommended)**:
   ```bash
   # On Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # On Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Step 1: Prepare the Dataset
If you don't have your own dataset yet, run the automated dataset generator to fetch/generate sample cat and dog images:
```bash
python download_dataset.py
```
> This creates 50+ cat images in `dataset/cats/` and 50+ dog images in `dataset/dogs/`, plus sample test images in `test_images/`.

### Step 2: Train the CNN Model
To run the full preprocessing, data augmentation, model building, training, and evaluation pipeline:
```bash
python train.py
```

**What happens during training?**
- Resizes images to `128 × 128 × 3` (RGB).
- Normalizes pixel values from `[0, 255]` to `[0.0, 1.0]`.
- Splits data into **80% Training** and **20% Validation** (using stratified sampling to prevent data leakage).
- Applies random horizontal flipping, rotation, and zoom data augmentation to training batches.
- Trains the custom CNN for **15 Epochs** with batch size **32**.
- Plots loss & accuracy curves in `outputs/training_history.png`.
- Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- Saves the trained model to `models/cat_dog_cnn.keras`.

### Step 3: Predict on New Images
Predict whether an image contains a Cat or a Dog:
```bash
python predict.py test_images/sample_cat_1.jpg
```
or
```bash
python predict.py test_images/sample_dog_1.jpg
```

**Output Example**:
```text
=============================================
 🐱 vs 🐶 BINARY IMAGE CLASSIFICATION RESULT
=============================================
  📷 Image Path  : test_images/sample_dog_1.jpg
  🎯 Prediction  : DOG 🐶
  📊 Confidence  : 94.20%
  🔢 Raw Sigmoid : 0.9420
=============================================
```

---

## 🧠 CNN Architecture (From Scratch)

```text
Input Image (128 × 128 × 3)
        ↓
Conv2D (32 filters, 3×3 kernel)
        ↓
ReLU Activation
        ↓
MaxPooling2D (2×2 pool size)
        ↓
Conv2D (64 filters, 3×3 kernel)
        ↓
ReLU Activation
        ↓
MaxPooling2D (2×2 pool size)
        ↓
Conv2D (128 filters, 3×3 kernel)
        ↓
ReLU Activation
        ↓
MaxPooling2D (2×2 pool size)
        ↓
Flatten Layer (1D Vector: 32,768 units)
        ↓
Dense Layer (128 units)
        ↓
Dropout Layer (Rate = 0.5)
        ↓
Output Layer (1 unit, Sigmoid)
        ↓
Output Probability [0 = Cat 🐱, 1 = Dog 🐶]
```

### Key Hyperparameters:
* **Optimizer**: Adam (`learning_rate = 0.001`)
* **Loss Function**: Binary Cross-Entropy (`binary_crossentropy`)
* **Activation**: ReLU (Hidden layers), Sigmoid (Output layer)
* **Epochs**: 15
* **Batch Size**: 32

---

## 📚 Viva Preparation Guide

For B.Tech lab viva and theory examinations, read the detailed 20+ Q&A guide in [VIVA.md](file:///c:/Users/pak/OneDrive/Desktop/catNdog/cat_dog_classifier/VIVA.md).
