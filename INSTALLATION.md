# PCOS Prediction System — Installation Guide

This guide provides step-by-step instructions for setting up the PCOS Prediction System on **Windows** and **macOS**.

---

## 1. Prerequisites

### Python
- Recommended Version: **Python 3.9 - 3.11**
- [Download Python](https://www.python.org/downloads/)

### OCR Engine (EasyOCR)
The system uses **EasyOCR** for report parsing. This is a pure-Python library and **does not require separate binary installation**. It will automatically download the necessary language models (English) on the first run.

---

## 2. Environment Setup

### Clone or Extract Project
Navigate to the project root directory in your terminal/command prompt.

### Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Apple Silicon (M1/M2/M3) Users**:
> For better performance with TensorFlow, you may optionally install:
> ```bash
> pip install tensorflow-macos tensorflow-metal
> ```

---

## 3. Running the Application

### Start the Flask Server
```bash
python app.py
```
- Port: `5001`
- URL: [http://localhost:5001](http://localhost:5001)

### Features to Explore
1. **Manual Entry**: Enter clinical metrics manually.
2. **Ultrasound Scan**: Upload `.jpg` or `.png` scans for YOLOv8 analysis.
3. **Hybrid Analysis**: Combine scan results with clinical data for maximum accuracy.
4. **Report Uploader**: Upload a PDF/Image lab report to auto-fill clinical fields.

---

### 5. (Special Step) Reassemble the Scan Model
Due to GitHub's file size limits, the YOLOv8 model (`pcos_scan_model.pt`) is stored in split chunks. You must reassemble them before the scan feature will work:

#### **Windows (Command Prompt)**
```cmd
copy /b pcos_scan_model.zip.aa + pcos_scan_model.zip.ab + pcos_scan_model.zip.ac + pcos_scan_model.zip.ad pcos_scan_model.zip
```

#### **macOS / Linux**
```bash
cat pcos_scan_model.zip.a* > pcos_scan_model.zip
```

**Finally**: Unzip `pcos_scan_model.zip` in the project root to obtain `pcos_scan_model.pt`.
