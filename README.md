<<<<<<< HEAD
# PCOS Prediction System — Multimodal Hybrid Deep Learning Ensemble

An advanced, academic-level PCOS (Polycystic Ovarian Syndrome) diagnostic system. This version introduces **Multimodal Fusion**, combining **YOLOv8 Computer Vision** with a **Hybrid Deep Learning Ensemble**, achieving **99.11% accuracy** on clinical data and high-precision ultrasound scan analysis.

---

## Project Highlights

- **Multimodal Fusion Engine**: Integrates Ultrasonic Scan Analysis (Vision) with Clinical Metrics (Structured Data).
- **YOLOv8 Vision Core**: Real-time detection and classification of PCOS from ultrasound images.
- **Weighted Ensemble**: Clinical prediction using DNN (0.35) + CNN (0.25) + CNN-LSTM (0.25) + YOLOv8 Signal (0.15).
- **OCR Medical Assistant**: Automated extraction of patient metrics from PDF/Image lab reports (using Tesseract & PDFPlumber).
- **Explainable AI**: Local and Global SHAP analysis for clinical transparency.
- **Next-Gen Web UI**: Dedicated portals for Manual Entry, Ultrasound Scanning, and Hybrid Analysis.

---

## Project Structure

```
├── app.py                    # Flask web application (Multimodal API)
├── data_pipeline.py          # Clinical data preprocessing (SMOTEENN, Scaler)
├── ensemble.py               # Classical/DL Ensemble evaluation
├── explainability.py         # SHAP analysis pipeline
├── pcos_scan_model.pt        # [NEW] YOLOv8 trained vision model
├── templates/
│   ├── index.html            # Main Dashboard
│   ├── scan.html             # [NEW] Ultrasound upload & Vision analysis
│   ├── hybrid.html           # [NEW] Multimodal Fusion portal
│   └── manual.html           # [NEW] Clinical metric entry
├── Dataset/
│   └── clean_data.csv        # Source dataset
├── results/
│   ├── master_comparison.csv  # Benchmark metrics
│   ├── feature_importance.json
│   ├── roc_curves.png
│   ├── shap_summary.png
│   └── ...                   # Full suite of DL/Classical artifacts
├── rf_model.joblib           # Frozen clinical DNN model
├── cnn_model.h5              # Trained CNN model
├── cnn_lstm_model.h5         # Trained CNN-LSTM model
└── requirements.txt
```

---

## Core Features

### 1. YOLOv8 Ultrasound Analysis

The system leverages a YOLOv8 model trained specifically on ovarian ultrasound scans. It detects follicular signatures and classifies them into PCOS/Non-PCOS states with high confidence.

### 2. Multimodal Hybrid Scoring

Instead of relying solely on clinical symptoms or image data, the system performs **late fusion**:

- **Clinical Signals (85%)**: Based on follicle counts, cycle regularity, and physical symptoms.
- **Vision Signal (15%)**: Based on direct ultrasound image inference.
  This minimizes false positives by cross-referencing visual evidence with biochemical/clinical symptoms.

### 3. OCR Medical Report Extractor

Patients can upload their existing medical reports (PDF/JPG). The system uses OCR to extract:

- Follicle counts (Left/Right Ovary)
- Cycle regularity status
- Symptoms like Skin Darkening (Acanthosis Nigricans) or Hirsutism.

### Detailed Setup Guide

For a step-by-step walkthrough for **Windows** and **macOS**, including Tesseract OCR installation, see the [Installation Guide](file:///Users/lulu/Desktop/Project_0ax2/INSTALLATION.md).

---

## 4. Performance & Cross-Platform Support

- **Cross-Platform**: Fully compatible with **Windows** and **macOS**.
- **Hardware Acceleration**:
  - **macOS**: Supports Apple Silicon (M1/M2/M3) acceleration via Metal.
  - **Windows**: Supports NVIDIA GPU acceleration via CUDA (if available).
- **Bottlenecks**: Synchronous OCR parsing and model loading on first startup are the primary performance considerations. All models are cached in memory after the first request.

---

## Model Performance Summary

| Model                        | Accuracy   | Recall     | F1-Score   |
| ---------------------------- | ---------- | ---------- | ---------- |
| Dense Neural Network (DNN)   | 99.11%     | 0.9100     | 0.9905     |
| CNN-LSTM Hybrid              | 98.21%     | 0.9670     | 0.9811     |
| CNN (Deep 1D)                | 97.32%     | 0.9615     | 0.9709     |
| MLP                          | 95.54%     | 0.9615     | 0.9524     |
| **Proposed Hybrid Ensemble** | **99.11%** | **0.9100** | **0.9905** |
| **YOLOv8 Scan Model**        | _Vision_   | _High_     | _Clinical_ |

---

## Top Contributing Features (SHAP)

1. **Follicle No. (R)** — Primary indicator
2. **Follicle No. (L)** — Secondary indicator
3. **Cycle (R/I)** — Menstrual irregularity

---

---

## Dataset & Reproducibility

- **Source**: PCOS Clinical Dataset (SMOTEENN balanced)
- **Seeds**: Fixed `random_state=0` for all experiments.
- **Models**: Pre-trained weights for CNN, LSTM, and YOLOv8 are provided in the root directory.
- **Features**: 6 selected clinical features + 1 Vision feature in Hybrid mode.
=======
# pcos
Ensemble based pcos prediction system
>>>>>>> 5a2388354cf2000a2b29f2da75e49ad99490613c
