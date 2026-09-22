# 🐱 vs 🐶 Cats vs Dogs Binary Image Classifier (Custom From-Scratch CNN)

A complete, beginner-friendly **Machine Learning & Deep Learning project** that classifies input images into **Cats (0)** or **Dogs (1)** using a custom **Convolutional Neural Network (CNN) built entirely from scratch in Keras/TensorFlow** without pretrained backbones.

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
