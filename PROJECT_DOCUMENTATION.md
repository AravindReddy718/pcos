# PCOS Prediction System - Project Documentation

## 1. Project Overview
This project implements a multimodal PCOS prediction system that combines:
- Clinical feature based prediction using machine learning and deep learning models
- Ultrasound scan based prediction using a YOLOv8 model
- Optional hybrid fusion of clinical and scan signals
- OCR based extraction of clinical hints from uploaded reports

The system is delivered as a Flask web application with training and evaluation pipelines for reproducible research.

## 2. Repository Structure
- app.py: Flask application, routes, API endpoints, model loading
- data_pipeline.py: data loading, cleaning, train test split, SMOTEENN, scaling
- train_classical.py: Logistic Regression, KNN, SVM, Random Forest, XGBoost
- train_models.py: baseline CNN and CNN-LSTM training
- train_dl_variants.py: additional DL variants (MLP, deep 1D CNN, pure LSTM)
- ensemble.py: weighted ensemble evaluation and RF artifact export
- ablation_study.py: with and without SMOTEENN comparisons
- explainability.py: SHAP analysis and feature importance exports
- generate_report.py: unified metrics and plots generation
- optimize_threshold.py: threshold sweep for ensemble probability
- reproduce_baseline.py: baseline reproducibility check
- templates/: web UI pages
- Dataset/: clinical source data
- pcos2.v2i.yolov8/: ultrasound dataset config and splits
- results/: generated metrics, plots, comparison files

## 3. Problem Definition
Given patient level input, predict whether PCOS is likely present.

Supported input modes:
- Manual mode: 6 clinical features
- Scan mode: ultrasound image
- Hybrid mode: combined image and clinical signals

Primary objective:
- High predictive performance with interpretable outputs suitable for clinical decision support workflows.

## 4. Data and Features
### 4.1 Clinical Data
Main source: Dataset/clean_data.csv

Pipeline selects 6 features:
- Follicle No. (R)
- Follicle No. (L)
- Skin darkening (Y/N)
- hair growth(Y/N)
- Weight gain(Y/N)
- Cycle(R/I)

Target:
- PCOS (Y/N)

### 4.2 Preprocessing Strategy
Implemented in data_pipeline.py:
1. Remove unnamed columns and coerce to numeric
2. Median imputation for missing values
3. Train test split with stratification (random_state=0)
4. Apply SMOTEENN to training set only
5. Fit StandardScaler on training set and transform test set
6. Reshape data for deep models to (samples, features, 1)

This design reduces data leakage risk and keeps evaluation realistic.

### 4.3 Scan Data
Ultrasound model artifact: pcos_scan_model.pt
Related dataset directory: pcos2.v2i.yolov8/

## 5. Model Architecture
### 5.1 Clinical Models
Classical models:
- Logistic Regression
- KNN
- SVM (RBF)
- Random Forest
- XGBoost

Deep models:
- CNN (Conv1D + pooling + dense)
- CNN-LSTM (Conv1D + LSTM)
- Additional variants: MLP, deep CNN, pure LSTM

### 5.2 Ensemble
Current ensemble in ensemble.py and app.py (clinical endpoint):
- 0.40 Random Forest probability
- 0.30 CNN probability
- 0.30 CNN-LSTM probability

Decision rule:
- Predict PCOS if combined probability is greater than or equal to 0.5

### 5.3 Hybrid Fusion in Scan Endpoint
In app.py hybrid flow:
- 0.35 Random Forest
- 0.25 CNN
- 0.25 CNN-LSTM
- 0.15 YOLO image signal

## 6. Explainability
explainability.py computes SHAP values for Random Forest and generates:
- SHAP summary plot
- SHAP bar plot
- Positive and negative waterfall plots
- results/feature_importance.json used by the web app

Top reported features are follicle counts and cycle regularity, consistent with clinical interpretation.

## 7. API Documentation
Flask routes in app.py:
- GET /
- GET /manual
- GET /scan
- GET /hybrid

Inference endpoints:
- POST /predict
- POST /predict_scan
- POST /extract_report

### 7.1 POST /predict
Input JSON fields:
- follicle_r
- follicle_l
- skin_darkening
- hair_growth
- weight_gain
- cycle_ri

Output:
- prediction string
- confidence (ensemble probability)
- top_features
- details object with rf_prob, cnn_prob, lstm_prob

### 7.2 POST /predict_scan
Input:
- multipart field scan_image
- optional form field clinical (JSON)

Output:
- image_prediction
- image_confidence
- detections
- optional hybrid section with combined confidence and extracted follicle values

### 7.3 POST /extract_report
Input:
- multipart field report_file (pdf, png, jpg, jpeg)

Output:
- extracted_data
- raw_text_snippet

Note: OCR path is optional and gracefully degrades when OCR dependencies are unavailable.

## 8. Training and Evaluation Workflow
Recommended execution order:
1. python data_pipeline.py
2. python train_classical.py
3. python train_models.py
4. python train_dl_variants.py
5. python ensemble.py
6. python ablation_study.py
7. python explainability.py
8. python generate_report.py
9. python reproduce_baseline.py
10. python optimize_threshold.py

Main outputs:
- rf_model.joblib
- cnn_model.h5
- cnn_lstm_model.h5
- scaler.pkl
- results/classical_metrics.json
- results/dl_metrics.json
- results/ablation_metrics.json
- results/master_comparison.csv
- final_results.json

## 9. Environment and Dependencies
Primary dependencies in requirements.txt:
- Flask, flask-cors
- numpy, pandas, joblib
- tensorflow, scikit-learn, ultralytics
- opencv-python, Pillow
- easyocr, pdfplumber

Install using:
- pip install -r requirements.txt

## 10. Frontend Overview
Templates:
- templates/index.html
- templates/manual.html
- templates/scan.html
- templates/hybrid.html

UI provides entry points for all major workflows and displays model outputs and confidence values.

## 11. Reproducibility and Governance Notes
- Most scripts enforce deterministic seeds, though some use seed 0 while others use 42
- Pipeline now applies split before resampling in data_pipeline.py
- app.py loads frozen artifacts at startup and runs inference only

Recommended improvement:
- Standardize all random seeds to one policy and document rationale.

## 12. Documentation Gaps to Close
High priority additions for final academic or production documentation:
1. Full API schema with validation rules and error catalog
2. Data dictionary for all dataset columns, units, and value ranges
3. YOLO training provenance, dataset composition, and validation metrics
4. Deployment guide (Docker and cloud)
5. Troubleshooting section for OCR, model loading, dependency conflicts
6. Test strategy and automated test report

## 13. Known Inconsistencies to Resolve
1. README ensemble description mentions DNN weight, while code currently uses Random Forest in ensemble.py and app.py clinical flow.
2. test_api.sh calls port 5000, but app.py runs on port 5001.
3. reproduce_baseline.py uses MinMaxScaler and resampling-before-split, while active training pipeline uses StandardScaler and split-before-resample.

## 14. Suggested Final Submission Package
For final project documentation submission, include:
- This file as system level documentation
- README.md as quick start
- PROJECT_RESULTS.md as benchmark summary
- A new API_SPEC.md with request and response schemas
- A new DATA_DICTIONARY.md with field definitions and allowed ranges
- A new DEPLOYMENT.md for container and cloud execution

## 15. Conclusion
The codebase is well organized around a clear training to inference lifecycle and supports clinical, scan, and hybrid prediction. The strongest current assets are model performance and modular pipeline scripts. The key work remaining is documentation hardening around APIs, data definitions, deployment, and consistency across narrative versus implementation.
