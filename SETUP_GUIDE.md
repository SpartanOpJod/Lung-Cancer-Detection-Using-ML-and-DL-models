# 🫁 Lung Cancer Detection AI - Web Application Setup Guide

## Overview

This is a complete web application UI for your EfficientNet-B0 lung cancer detection model. The application consists of:

- **Flask Backend API** - Serves the pre-trained model for inference
- **React Frontend** - Beautiful, modern UI for image upload and analysis
- **Docker Support** - Easy containerized deployment

## Project Structure

```
project/
├── app.py                          # Flask API backend
├── frontend/                       # React application
│   ├── src/
│   │   ├── App.js                 # Main app component
│   │   ├── App.css                # Main styles
│   │   ├── index.js               # Entry point
│   │   └── components/
│   │       ├── ImageUploader.js   # Upload component
│   │       ├── PredictionResult.js # Results display
│   │       ├── SampleImages.js    # Sample images grid
│   │       └── ModelInfo.js       # Model info page
│   ├── package.json               # React dependencies
│   └── public/index.html          # HTML template
├── requirements-api.txt            # Python dependencies
├── docker-compose.yml             # Docker orchestration
├── Dockerfile                     # Backend container
└── .env.example                   # Environment variables
```

## Installation & Setup

### Option 1: Local Development (Recommended for Development)

#### Backend Setup

1. **Install Python dependencies:**
```bash
pip install -r requirements-api.txt
```

2. **Run Flask API:**
```bash
python app.py
```
The API will start on `http://localhost:5000`

#### Frontend Setup

1. **Navigate to frontend directory:**
```bash
cd frontend
```

2. **Install Node dependencies:**
```bash
npm install
```

3. **Start React development server:**
```bash
npm start
```
The UI will open at `http://localhost:3000`

4. **Build for production:**
```bash
npm run build
```

### Option 2: Docker Deployment (Recommended for Production)

1. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

2. **Access the application:**
   - Frontend: `http://localhost:3000`
   - API: `http://localhost:5000`

3. **Stop containers:**
```bash
docker-compose down
```

## API Endpoints

### 1. Health Check
```
GET /health
```
Check if the API and model are loaded and ready.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "classes": ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]
}
```

### 2. Predict (Main Endpoint)
```
POST /predict
Content-Type: multipart/form-data
Body: image (file)
```

Analyzes an uploaded image and returns predictions.

**Response:**
```json
{
  "predicted_class": "Normal",
  "confidence": 0.9567,
  "class_probabilities": {
    "Adenocarcinoma": 0.0123,
    "Large Cell Carcinoma": 0.0145,
    "Normal": 0.9567,
    "Squamous Cell Carcinoma": 0.0165
  },
  "success": true
}
```

### 3. Get Sample Images
```
GET /sample-images
```

Get list of sample images from test dataset.

**Response:**
```json
{
  "samples": [
    {
      "image": "data/raw/test/normal/image001.jpg",
      "label": "normal"
    }
  ],
  "total_available": 150,
  "success": true
}
```

### 4. Serve Sample Image
```
GET /sample-image/<image_path>
```

Returns the actual image file for display.

### 5. Model Information
```
GET /model-info
```

Get detailed model information.

**Response:**
```json
{
  "model_name": "EfficientNet-B0",
  "accuracy": "96.7%",
  "training_data": "Chest CT Scan Images",
  "dataset_source": "https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images",
  "classes": [...],
  "optimization": "Blue Whale Optimization + Penguin Optimization",
  "input_size": 224,
  "device": "cuda",
  "success": true
}
```

## Features

### 📤 Upload Tab
- Drag & drop image upload
- File browser selection
- Real-time image preview
- One-click analysis button
- Loading state with spinner

### 📸 Sample Images Tab
- Browse test dataset images
- Click to auto-analyze
- Color-coded by class
- Label display with emoji indicators

### ℹ️ Model Info Tab
- Model architecture details
- Training information
- Supported classes
- Technical specifications
- Clinical disclaimer

### Results Display
- Confidence score with visual bar
- All class probabilities
- Risk level badge (Low/High Risk)
- Clinical recommendations
- Responsive design

## Features Detail

### 1. Smart Image Preprocessing
- Automatic gray-scale conversion to RGB
- Image resizing to 224x224
- Albumentations transforms applied
- Handles various image formats

### 2. Real-time Predictions
- GPU acceleration (CUDA if available)
- Fast inference with EfficientNet-B0
- Confidence score calculation
- All class probabilities

### 3. Beautiful UI
- Modern gradient design
- Smooth animations
- Responsive layout (desktop, tablet, mobile)
- Accessibility considerations

### 4. Error Handling
- Graceful error messages
- Model loading validation
- Input validation
- Network error handling

## Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:5000
FLASK_ENV=development
```

For production, update API_URL to your server URL.

## Performance

- **Model Inference Time**: ~200-300ms (GPU), ~500-800ms (CPU)
- **Accuracy**: 96.7% (after optimization)
- **Input Size**: 224x224 pixels
- **Classes**: 4 (Normal, Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma)

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Troubleshooting

### API not connecting to frontend
- Ensure `REACT_APP_API_URL` is set correctly
- Check CORS is enabled in Flask (it is by default)
- Verify API is running on port 5000

### Model not loading
- Check model file exists at: `models/dl/efficientnet_b0_best.pth`
- Verify PyTorch installation: `pip install torch torchvision`
- Check disk space for model weights

### Port already in use
```bash
# On Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# On Linux/Mac
lsof -i :5000
kill -9 <PID>
```

### CUDA not detected
- Ensure NVIDIA GPU drivers are installed
- Install CUDA-enabled PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## Clinical Disclaimer

⚠️ **IMPORTANT**: This AI model is designed for **research and educational purposes only**. 

The predictions provided are **NOT a substitute for professional medical diagnosis**. Always consult with a qualified radiologist or pulmonologist for proper diagnosis and treatment planning.

## Future Enhancements

- [ ] User authentication
- [ ] Image history/dashboard
- [ ] Batch image processing
- [ ] Model comparison
- [ ] Advanced analytics
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Real-time API monitoring

## Project Modifications

The original project structure remains unchanged. New files added:

- `app.py` - Flask API
- `frontend/` - React application (new folder)
- `requirements-api.txt` - API dependencies
- `Dockerfile` - Container definition
- `docker-compose.yml` - Service orchestration
- `.env.example` - Environment template

All existing scripts, models, and utilities continue to work as before.

## Support & References

- **Dataset Source**: [Kaggle - Chest CT Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
- **Model**: EfficientNet-B0 with custom optimization
- **Optimization**: Blue Whale + Penguin algorithms
- **Framework**: PyTorch + React

## License

See LICENSE file in project root.

---

Built with ❤️ for medical AI research and education.
