"""
Flask backend API for Lung Cancer Detection
Serves the pre-trained EfficientNet model for inference
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from pathlib import Path
import os
import sys
from PIL import Image
import io

# Add project root to path
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, str(PROJECT_ROOT))

from utils.transforms import val_transform

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
DEVICE = "cpu"
MODEL_PATH = PROJECT_ROOT / "models" / "dl" / "efficientnet_b0_hybrid_best.pth"
IMG_SIZE = 224

# Class labels (MATCH TRAINING ORDER)
CLASS_LABELS = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
]

# Model loading
model = None


def load_model():
    global model
    try:
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            len(CLASS_LABELS)
        )

        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

        if isinstance(ckpt, dict) and "model_state" in ckpt:
             model.load_state_dict(ckpt["model_state"])
        else:
             model.load_state_dict(ckpt)


        model.to(DEVICE)
        model.eval()
        print(f"✓ Model loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def preprocess_image(image_data):
    """
    EXACTLY replicate preprocess.py + dataset.py behavior
    """
    try:
        # PIL -> numpy
        img = np.array(image_data)

        # Convert to grayscale EXACTLY like training
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Resize EXACTLY like preprocess.py
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # Intensity normalization + quantization (CRITICAL)
        img = img.astype(np.float32) / 255.0
        img = (img * 255).astype(np.uint8)

        # Convert back to RGB (dataset.py)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Apply SAME val_transform
        aug = val_transform(image=img)
        img_tensor = aug["image"].unsqueeze(0).to(DEVICE)

        return img_tensor

    except Exception as e:
        print("Preprocess error:", e)
        raise



@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": model is not None,
        "classes": CLASS_LABELS
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_tensor = preprocess_image(img)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probabilities))

        class_confidences = {
            CLASS_LABELS[i]: float(probabilities[i])
            for i in range(len(CLASS_LABELS))
        }

        return jsonify({
            "predicted_class": CLASS_LABELS[pred_idx],
            "confidence": float(probabilities[pred_idx]),
            "class_probabilities": class_confidences,
            "success": True
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/sample-images', methods=['GET'])
def get_sample_images():
    try:
        import pandas as pd
        csv_path = PROJECT_ROOT / "data" / "meta" / "test.csv"

        df = pd.read_csv(csv_path)
        samples = df.drop_duplicates(subset=['image']).head(20).to_dict('records')

        return jsonify({
            "samples": samples,
            "success": True
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/sample-image/<path:image_path>', methods=['GET'])
def get_sample_image(image_path):
    try:
        from flask import send_file
        full_path = PROJECT_ROOT / image_path
        return send_file(str(full_path), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "model_name": "EfficientNet-B0",
        "accuracy": "96.7%",
        "classes": CLASS_LABELS,
        "device": DEVICE,
        "success": True
    })


if __name__ == '__main__':
    print("🫁 Lung Cancer Detection API")
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
