# 🐱 vs 🐶 Cats vs Dogs Binary Image Classifier (Custom From-Scratch CNN in PyTorch)

A complete, beginner-friendly **Machine Learning & Deep Learning project** that classifies input images into **Cats (0)** or **Dogs (1)** using a custom **Convolutional Neural Network (CNN) built entirely from scratch in PyTorch** without pretrained backbones.

---

## 📌 Performance & Evaluation Overview (Audit Verified)

> [!IMPORTANT]
> **Honest Metric Reporting**: We strictly separate training metrics, validation metrics, isolated holdout test metrics, and unseen external sanity-test metrics. Training/validation numbers are never labeled as real-world accuracy.

| Metric Stage | Accuracy | Dataset Size | Description |
| :--- | :--- | :--- | :--- |
| **Training Set** (70%) | **~55.0% - 65.0%** | 140 real images | Metric during gradient descent with real-time data augmentation |
| **Validation Set** (15%) | **~50.0% - 63.3%** | 30 real images | Monitored for learning rate scheduling & checkpointing |
| **Isolated Test Set** (15%) | **53.33%** | 30 real images | Strictly untouched holdout set evaluated after training completion |
| **External Sanity Test** | **45.00%** | 20 unseen photos | Real-world photos from independent web sources in `test_external/` |

---

## 🎯 Test Set Metrics Breakdown

- **Total Test Samples**: 30 (Cats: 15, Dogs: 15)
- **Test Accuracy**: `53.33%`
- **Test Precision**: `55.56%`
- **Test Recall (Sensitivity)**: `33.33%`
- **Test F1-Score**: `0.4167`

### Confusion Matrix on Isolated Test Set:
```text
                  Predicted Cat (0)   Predicted Dog (1)
  Actual Cat (0)             11                  4        (TN: 11, FP: 4)
  Actual Dog (1)             10                  5        (FN: 10, TP: 5)
```

---

## 📂 Project Hierarchy

```text
catNdog/
│
├── dataset/                        # Real-world training & evaluation images
│   ├── cats/                       # 100 real cat photographs (Label: 0)
│   └── dogs/                       # 100 real dog photographs (Label: 1)
│
├── test_external/                  # Unseen external sanity test suite
│   ├── cats/                       # 10 external cat images
│   └── dogs/                       # 10 external dog images
│
├── models/                         # Saved PyTorch model & audit reports
│   ├── cat_dog_cnn.pth             # Trained custom CNN model state dict
│   ├── misclassified/              # Saved misclassified test image files
│   ├── misclassified_predictions.csv # Audit log of all test errors
│   └── external_predictions.csv    # External sanity test predictions table
│
├── outputs/                        # Generated plots & visual reports
│   ├── training_history.png        # Loss & Accuracy curves over training epochs
│   ├── confusion_matrix.png        # Confusion Matrix heatmap
│   ├── sample_predictions.png      # Test predictions grid
│   ├── misclassified_grid.png      # Visual grid of misclassified test samples
│   └── last_prediction.png         # CLI prediction overlay
│
├── cat_dog_classifier/             # Core Python package
│   ├── config.py                   # Single source of truth for class mapping & thresholds
│   ├── preprocessing.py            # Unified preprocessing module (Train, Test, CLI, Web)
│   ├── model.py                    # 4-block CNN architecture from scratch
│   ├── train.py                    # 3-way split, training & misclassification pipeline
│   ├── predict.py                  # Single-image prediction CLI
│   └── download_dataset.py         # Real-world image downloader & deduplicator
│
├── test_pipeline.py                # 10-point automated unit test suite
├── test_external_suite.py          # Generalization test script on test_external/
├── app.py                          # Flask web application backend (localhost:5000)
├── static/                         # Web assets (CSS & JavaScript)
├── templates/                      # Web UI template (index.html)
├── train.py                        # Top-level training runner
├── predict.py                      # Top-level prediction CLI runner
└── requirements.txt                # Project dependencies
```

---

## 🧠 Single Source of Truth for Class Mapping & Preprocessing

### 1. Class Label Mapping (`cat_dog_classifier/config.py`)
- **Cat**: Index `0`
- **Dog**: Index `1`
- **Threshold**: Probabilities $\ge 0.50 \rightarrow$ Dog, $< 0.50 \rightarrow$ Cat.
- **Uncertainty Threshold**: Confidence $< 60.0\% \rightarrow$ Flagged as **Uncertain / Low-Confidence**.

### 2. Preprocessing Pipeline (`cat_dog_classifier/preprocessing.py`)
- **Color Space**: Strictly converted to 3-channel **RGB**.
- **Resolution**: Resized using bilinear interpolation to **$128 \times 128$**.
- **Pixel Normalization**: Normalized to float32 values in $[0.0, 1.0]$ ($\frac{\text{Pixel}}{255.0}$).

---

## 🚀 Quickstart & Execution Guide

### 1. Run Automated 10-Point Sanity Tests
```bash
python test_pipeline.py
```

### 2. Train Model & Run Isolated Test Evaluation
```bash
python train.py
```

### 3. Run External Sanity Suite
```bash
python test_external_suite.py
```

### 4. Single-Image Prediction via CLI
```bash
python predict.py test_images/sample_dog_1.jpg
```

### 5. Launch Localhost Web Dashboard
```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your web browser.

---

## 🔍 Misclassification Analysis
Whenever `train.py` runs, every misclassified test sample is:
1. Copied into `models/misclassified/` with actual vs predicted labels in the filename.
2. Logged with confidence scores in `models/misclassified_predictions.csv`.
3. Rendered into a visual error analysis grid in `outputs/misclassified_grid.png`.

---

## ⚖️ License
Distributed under the **MIT License**.
