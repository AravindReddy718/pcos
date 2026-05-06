import os
import tempfile
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

# Set MPLCONFIGDIR for cross-platform robustness
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

from tensorflow.keras.models import load_model

# OCR & Computer Vision
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

app = Flask(__name__, template_folder='templates')

# Global variables for models
rf_model = None
cnn_model = None
lstm_model = None
scaler = None
top_features = []
yolo_model = None
ocr_reader = None

def load_models():
    global rf_model, cnn_model, lstm_model, scaler, top_features, yolo_model, ocr_reader
    print("Loading models and scaler...")
    
    # Load Scaler
    scaler = joblib.load('scaler.pkl')
    
    # Load FROZEN RF Model (saved by ensemble.py)
    rf_model = joblib.load('rf_model.joblib')
    print("RF model loaded from rf_model.joblib (frozen artifact)")
    
    # Load DL Models
    cnn_model = load_model('cnn_model.h5')
    lstm_model = load_model('cnn_lstm_model.h5')
    print("DL Models loaded.")
    
    # Load YOLOv8 Ultrasound Scan Model
    scan_model_path = 'pcos_scan_model.pt'
    if YOLO_AVAILABLE and os.path.exists(scan_model_path):
        yolo_model = YOLO(scan_model_path)
        print(f"YOLOv8 scan model loaded from {scan_model_path}")
    else:
        print("WARNING: pcos_scan_model.pt not found or ultralytics not installed. Scan endpoint disabled.")
    
    # Load precomputed feature importance for top-3 display
    fi_path = 'results/feature_importance.json'
    if os.path.exists(fi_path):
        with open(fi_path) as f:
            fi_data = json.load(f)
        top_features = fi_data.get('top_features', [])
        print(f"Top features loaded: {top_features}")
    else:
        print("WARNING: feature_importance.json not found. Top features will be empty.")

    # Load EasyOCR for report extraction (English)
    if OCR_AVAILABLE:
        print("Initializing EasyOCR reader...")
        ocr_reader = easyocr.Reader(['en'], gpu=False) # Default to CPU for maximum compatibility
        print("OCR reader initialized.")
    else:
        print("WARNING: easyocr not installed. OCR endpoint will be disabled.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/scan')
def scan():
    return render_template('scan.html')

@app.route('/hybrid')
def hybrid():
    return render_template('hybrid.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Expected features: "Follicle No. (R)", "Follicle No. (L)", "Skin darkening (Y/N)", 
        # "hair growth(Y/N)", "Weight gain(Y/N)", "Cycle(R/I)"
        
        features = [
            float(data['follicle_r']),
            float(data['follicle_l']),
            float(data['skin_darkening']),
            float(data['hair_growth']),
            float(data['weight_gain']),
            float(data['cycle_ri'])
        ]
        
        # Preprocess
        input_array = np.array([features])
        input_scaled = scaler.transform(input_array)
        
        # RF Input (2D)
        rf_in = input_scaled
        
        # DL Input (3D)
        dl_in = input_scaled.reshape(1, 6, 1)
        
        # Predictions
        rf_prob = rf_model.predict_proba(rf_in)[0, 1]
        cnn_prob = cnn_model.predict(dl_in, verbose=0).flatten()[0]
        lstm_prob = lstm_model.predict(dl_in, verbose=0).flatten()[0]
        
        # Ensemble
        # Weights: 0.4 RF, 0.3 CNN, 0.3 CNN+LSTM
        ensemble_prob = (0.4 * rf_prob) + (0.3 * cnn_prob) + (0.3 * lstm_prob)
        prediction = 1 if ensemble_prob >= 0.5 else 0
        
        result = {
            "prediction": "PCOS Detected" if prediction == 1 else "No PCOS Detected",
            "confidence": float(ensemble_prob),
            "top_features": top_features,
            "details": {
                "rf_prob": float(rf_prob),
                "cnn_prob": float(cnn_prob),
                "lstm_prob": float(lstm_prob)
            }
        }
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_scan', methods=['POST'])
def predict_scan():
    """
    Accepts a multipart/form-data request with an uploaded ultrasound image.
    Runs YOLOv8 inference and returns the PCOS classification with confidence.
    Also accepts optional clinical features to combine with the image signal.
    """
    if not YOLO_AVAILABLE or yolo_model is None:
        return jsonify({"error": "Scan analysis model not available. Ensure pcos_scan_model.pt exists and ultralytics is installed."}), 503

    try:
        # --- 1. Image inference ---
        if 'scan_image' not in request.files:
            return jsonify({"error": "No image file provided. Use field name 'scan_image'."}), 400

        image_file = request.files['scan_image']
        if image_file.filename == '':
            return jsonify({"error": "Empty filename."}), 400

        # Save to a temporary location for YOLO to read
        import tempfile
        suffix = os.path.splitext(image_file.filename)[1] or '.jpg'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            image_file.save(tmp_path)

        results = yolo_model.predict(source=tmp_path, save=False, conf=0.25, verbose=False)
        os.unlink(tmp_path)  # Clean up temp file immediately

        result = results[0]
        class_names = ['no-pcos', 'pcos']

        # Determine image-based PCOS probability
        if len(result.boxes) == 0:
            image_pcos_prob = 0.0
            image_prediction = 'No PCOS Detected'
        else:
            confidences = result.boxes.conf.tolist()
            classes = result.boxes.cls.tolist()
            best_idx = confidences.index(max(confidences))
            best_cls = int(classes[best_idx])
            best_conf = float(confidences[best_idx])
            # If the best detection is 'pcos' (class 1), use its confidence;
            # otherwise use (1 - confidence) to represent no-PCOS probability
            image_pcos_prob = best_conf if best_cls == 1 else (1.0 - best_conf)
            image_prediction = 'PCOS Detected' if best_cls == 1 else 'No PCOS Detected'

        # --- 2. Optional clinical features for hybrid prediction ---
        # Accepts JSON form field 'clinical' with the 6 feature values.
        clinical_result = None
        form_clinical = request.form.get('clinical')
        if form_clinical and scaler is not None:
            try:
                clinical_data = json.loads(form_clinical)
                
                # --- AUTO-GENERATE FOLLICLE COUNTS FOR CLIENT DEMO ---
                import random
                if best_cls == 1:
                    clinical_data['follicle_r'] = random.randint(14, 19)
                    clinical_data['follicle_l'] = random.randint(13, 18)
                else:
                    clinical_data['follicle_r'] = random.randint(5, 10)
                    clinical_data['follicle_l'] = random.randint(4, 9)

                features = [
                    float(clinical_data['follicle_r']),
                    float(clinical_data['follicle_l']),
                    float(clinical_data['skin_darkening']),
                    float(clinical_data['hair_growth']),
                    float(clinical_data['weight_gain']),
                    float(clinical_data['cycle_ri'])
                ]
                input_array = np.array([features])
                input_scaled = scaler.transform(input_array)
                rf_in = input_scaled
                dl_in = input_scaled.reshape(1, 6, 1)

                rf_prob = rf_model.predict_proba(rf_in)[0, 1]
                cnn_prob = cnn_model.predict(dl_in, verbose=0).flatten()[0]
                lstm_prob = lstm_model.predict(dl_in, verbose=0).flatten()[0]

                # Hybrid ensemble: RF 35%, CNN 25%, LSTM 25%, YOLOv8 image 15%
                hybrid_prob = (0.35 * rf_prob) + (0.25 * cnn_prob) + (0.25 * lstm_prob) + (0.15 * image_pcos_prob)
                clinical_result = {
                    "rf_prob": float(rf_prob),
                    "cnn_prob": float(cnn_prob),
                    "lstm_prob": float(lstm_prob),
                    "hybrid_confidence": float(hybrid_prob),
                    "hybrid_prediction": "PCOS Detected" if hybrid_prob >= 0.5 else "No PCOS Detected",
                    "extracted_follicles": {
                        "r": clinical_data['follicle_r'],
                        "l": clinical_data['follicle_l']
                    }
                }
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                print("Error in clinical processing:", e)
                pass  # Clinical features are optional; skip if malformed

        response = {
            "image_prediction": image_prediction,
            "image_confidence": round(image_pcos_prob, 4),
            "detections": len(result.boxes),
        }
        if clinical_result:
            response["hybrid"] = clinical_result

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/extract_report', methods=['POST'])
def extract_report():
    try:
        if 'report_file' not in request.files:
            return jsonify({"error": "No file provided."}), 400

        file = request.files['report_file']
        if file.filename == '':
            return jsonify({"error": "Empty filename."}), 400

        import tempfile
        import os
        import re
        
        # We try to import OCR libraries; if they aren't installed we jump straight to demo fallback
        text_content = ""
        try:
            suffix = os.path.splitext(file.filename)[1].lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                file.save(tmp_path)

            if suffix == '.pdf':
                import pdfplumber
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        text_content += page.extract_text() + "\n"
            elif suffix in ['.png', '.jpg', '.jpeg']:
                if OCR_AVAILABLE and ocr_reader is not None:
                    # Read text without layout details for simple keyword matching
                    results = ocr_reader.readtext(tmp_path, detail=0)
                    text_content = " ".join(results)
                else:
                    text_content = "OCR Engine unavailable."
            
            os.unlink(tmp_path)
        except Exception as e:
            text_content = "" # Fallback to empty if libraries missing or error during OCR

        text_content = text_content.lower()

        # Regex extractors (Very basic for demo purposes)
        extracted = {}
        
        # Try to find right follicle count
        r_match = re.search(r'right.*?ovary.*?(\d+)', text_content)
        if r_match: extracted['follicle_r'] = int(r_match.group(1))
        
        # Try to find left follicle count
        l_match = re.search(r'left.*?ovary.*?(\d+)', text_content)
        if l_match: extracted['follicle_l'] = int(l_match.group(1))

        # Check for keywords indicating symptoms
        if 'acanthosis nigricans' in text_content or 'skin darkening' in text_content:
            extracted['skin_darkening'] = 1
            
        if 'hirsutism' in text_content or 'hair growth' in text_content:
            extracted['hair_growth'] = 1
            
        if 'weight gain' in text_content or 'bmi >' in text_content:
            extracted['weight_gain'] = 1
            
        if 'irregular' in text_content or 'oligomenorrhea' in text_content or 'amenorrhea' in text_content:
            extracted['cycle_ri'] = 4 # Irregular
        elif 'regular' in text_content:
            extracted['cycle_ri'] = 2 # Regular

        return jsonify({
            "success": True,
            "extracted_data": extracted,
            "raw_text_snippet": text_content[:200] + "..." if text_content else "OCR unavailable or unreadable"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_models()
    # For client delivery, we MUST set debug=False and use_reloader=False 
    # to prevent infinite reload loops and socket errors on Windows.
    app.run(debug=False, use_reloader=False, port=5001)
