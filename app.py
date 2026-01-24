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
MODEL_PATH = PROJECT_ROOT / "models" / "dl" / "efficientnet_b0_best.pth"
IMG_SIZE = 224

# Class labels
CLASS_LABELS = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma"
]

# Model loading
model = None


def load_model():
    """Load the pre-trained EfficientNet model"""
    global model
    
    try:
        model = models.efficientnet_b0(pretrained=False)
        num_classes = len(CLASS_LABELS)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        # Load weights
        if MODEL_PATH.exists():
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            
        
        model.to(DEVICE)
        model.eval()
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def preprocess_image(image_data):
    """
    Preprocess image for model inference
    Args:
        image_data: PIL Image or numpy array
    Returns:
        torch tensor ready for model
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image_data, Image.Image):
            img = np.array(image_data)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = image_data
        
        # Ensure RGB format
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Apply transforms
        aug = val_transform(image=img)
        img_tensor = aug['image']
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": model is not None,
        "classes": CLASS_LABELS
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict lung cancer from uploaded image
    Expects: multipart/form-data with 'image' field
    Returns: JSON with predictions and confidence scores
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Read and preprocess image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_tensor = preprocess_image(img)
        
        # Model inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Prepare response
        pred_idx = predicted.item()
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        class_confidences = {
            CLASS_LABELS[i]: float(all_probs[i]) 
            for i in range(len(CLASS_LABELS))
        }
        
        response = {
            "predicted_class": CLASS_LABELS[pred_idx],
            "confidence": round(confidence_score, 4),
            "class_probabilities": {k: round(v, 4) for k, v in class_confidences.items()},
            "success": True
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/sample-images', methods=['GET'])
def get_sample_images():
    """Get list of available sample images from test dataset"""
    try:
        # Read test.csv to get sample images
        import pandas as pd
        
        csv_path = PROJECT_ROOT / "data" / "meta" / "test.csv"
        if not csv_path.exists():
            return jsonify({"error": "Test CSV not found"}), 404
        
        df = pd.read_csv(csv_path)
        
        # Get unique images and their labels
        samples = df.drop_duplicates(subset=['image']).head(20).to_dict('records')
        
        return jsonify({
            "samples": samples,
            "total_available": len(df),
            "success": True
        })
    
    except Exception as e:
        print(f"Error fetching samples: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/sample-image/<path:image_path>', methods=['GET'])
def get_sample_image(image_path):
    """Serve sample image file"""
    try:
        from flask import send_file
        
        # Prevent directory traversal
        if ".." in image_path:
            return jsonify({"error": "Invalid path"}), 403
        
        # Construct full path
        full_path = PROJECT_ROOT / image_path
        
        if not full_path.exists():
            return jsonify({"error": "Image not found"}), 404
        
        return send_file(str(full_path), mimetype='image/jpeg')
    
    except Exception as e:
        print(f"Error serving image: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_name": "EfficientNet-B0",
        "accuracy": "96.7%",
        "training_data": "Chest CT Scan Images",
        "dataset_source": "https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images",
        "classes": CLASS_LABELS,
        "optimization": "Blue Whale Optimization + Penguin Optimization",
        "input_size": IMG_SIZE,
        "device": DEVICE,
        "success": True
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🫁 Lung Cancer Detection API")
    print("=" * 60)
    
    # Load model
    if load_model():
        print("✓ Ready to serve predictions")
    else:
        print("⚠ Warning: Model failed to load - predictions may fail")
    
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASS_LABELS}")
    print("=" * 60)
    
    # Start server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False
    )
