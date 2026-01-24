# API Documentation

## Base URL
```
http://localhost:5000
```

## Authentication
Currently, the API does not require authentication. For production, add API key authentication in `app.py`.

## Response Format
All responses are in JSON format.

### Success Response
```json
{
  "success": true,
  "data": {}
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error message here"
}
```

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API and model are ready.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "classes": ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]
}
```

**Status Codes:**
- `200` - API is healthy
- `500` - Model failed to load

---

### 2. Image Prediction

**POST** `/predict`

Analyze a chest CT scan image and get predictions.

**Content-Type:** `multipart/form-data`

**Parameters:**
- `image` (file, required) - CT scan image file (JPEG, PNG, etc.)

**Example Request (cURL):**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@path/to/image.jpg"
```

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

**Response Fields:**
- `predicted_class` (string) - The predicted lung condition class
- `confidence` (float) - Confidence score between 0 and 1
- `class_probabilities` (object) - Probability for each class
- `success` (boolean) - Whether prediction was successful

**Status Codes:**
- `200` - Prediction successful
- `400` - No image provided or invalid file
- `500` - Prediction failed (model error)

**Error Examples:**

No file uploaded:
```json
{
  "error": "No image provided",
  "success": false
}
```

Invalid file type:
```json
{
  "error": "Please drop an image file",
  "success": false
}
```

---

### 3. Get Sample Images

**GET** `/sample-images`

Retrieve list of sample images from test dataset.

**Query Parameters:**
- None

**Response:**
```json
{
  "samples": [
    {
      "image": "data/raw/test/normal/image001.jpg",
      "label": "normal"
    },
    {
      "image": "data/raw/test/adenocarcinoma/image002.jpg",
      "label": "adenocarcinoma"
    }
  ],
  "total_available": 256,
  "success": true
}
```

**Response Fields:**
- `samples` (array) - List of available sample images
- `total_available` (integer) - Total samples in dataset
- `success` (boolean) - Whether request was successful

**Status Codes:**
- `200` - Samples retrieved
- `404` - Test CSV not found
- `500` - Error reading dataset

---

### 4. Serve Sample Image

**GET** `/sample-image/<image_path>`

Get the actual image file for display.

**Parameters:**
- `image_path` (path, required) - Path to image from sample list (URL-encoded)

**Example Request:**
```
GET /sample-image/data/raw/test/normal/image001.jpg
```

**Response:**
- Binary image data with MIME type `image/jpeg`

**Status Codes:**
- `200` - Image found and served
- `403` - Invalid path (directory traversal attempt)
- `404` - Image not found

---

### 5. Model Information

**GET** `/model-info`

Get detailed information about the model.

**Response:**
```json
{
  "model_name": "EfficientNet-B0",
  "accuracy": "96.7%",
  "training_data": "Chest CT Scan Images",
  "dataset_source": "https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images",
  "classes": [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma"
  ],
  "optimization": "Blue Whale Optimization + Penguin Optimization",
  "input_size": 224,
  "device": "cuda",
  "success": true
}
```

**Response Fields:**
- `model_name` (string) - Model architecture name
- `accuracy` (string) - Test accuracy percentage
- `training_data` (string) - Dataset description
- `dataset_source` (string) - Link to dataset source
- `classes` (array) - List of supported disease classes
- `optimization` (string) - Optimization techniques applied
- `input_size` (integer) - Model input image size in pixels
- `device` (string) - Current computation device (cuda/cpu)
- `success` (boolean) - Whether request was successful

**Status Codes:**
- `200` - Model info retrieved

---

## Class Labels

| Index | Class | Description |
|-------|-------|-------------|
| 0 | Adenocarcinoma | Most common type of lung cancer |
| 1 | Large Cell Carcinoma | Less common, poor prognosis |
| 2 | Normal | Healthy lung tissue |
| 3 | Squamous Cell Carcinoma | Common type of lung cancer |

---

## Error Handling

### Common Error Codes

**400 - Bad Request**
```json
{
  "error": "No image provided",
  "success": false
}
```

**404 - Not Found**
```json
{
  "error": "Endpoint not found",
  "success": false
}
```

**500 - Internal Server Error**
```json
{
  "error": "Internal server error",
  "success": false
}
```

---

## Rate Limiting

Currently no rate limiting is implemented. For production:
1. Add Flask-Limiter
2. Implement request throttling
3. Add authentication

---

## CORS Configuration

CORS is enabled for all origins (`*`). For production, restrict to specific domains in `app.py`:

```python
CORS(app, resources={r"/api/*": {"origins": ["https://yourdomain.com"]}})
```

---

## Performance Considerations

### Inference Time
- **GPU (CUDA)**: ~200-300ms per image
- **CPU**: ~500-800ms per image

### Memory Usage
- **Model Size**: ~20MB (EfficientNet-B0)
- **Memory Required**: ~2GB minimum (4GB+ recommended)

### Batch Processing
Currently, the API processes one image at a time. To add batch processing:

1. Modify `/predict` endpoint to accept multiple files
2. Use PyTorch's DataLoader for efficient batching
3. Return array of predictions

---

## Security Notes

⚠️ **Important for Production:**

1. **Add Authentication**
   ```python
   from flask_httpauth import HTTPBasicAuth
   auth = HTTPBasicAuth()
   ```

2. **Input Validation**
   ```python
   MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
   ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
   ```

3. **Rate Limiting**
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app, key_func=get_remote_address)
   ```

4. **HTTPS**
   - Use SSL/TLS certificates
   - Set `SESSION_COOKIE_SECURE = True`

5. **File Upload Security**
   - Validate file type
   - Scan for malware
   - Store files securely

---

## Testing

### Using cURL

**1. Health Check:**
```bash
curl http://localhost:5000/health
```

**2. Predict:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@test_image.jpg"
```

**3. Get Samples:**
```bash
curl http://localhost:5000/sample-images
```

**4. Model Info:**
```bash
curl http://localhost:5000/model-info
```

### Using Python Requests

```python
import requests

# Predict
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post('http://localhost:5000/predict', files=files)
print(response.json())

# Get samples
response = requests.get('http://localhost:5000/sample-images')
print(response.json())

# Model info
response = requests.get('http://localhost:5000/model-info')
print(response.json())
```

### Using JavaScript Fetch

```javascript
// Predict
const formData = new FormData();
formData.append('image', imageFile);

fetch('http://localhost:5000/predict', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data));

// Get samples
fetch('http://localhost:5000/sample-images')
  .then(res => res.json())
  .then(data => console.log(data));
```

---

## Deployment

### Docker Deployment
```bash
docker-compose up --build
```

### Kubernetes Deployment
See `kubernetes/` folder for deployment manifests.

### Cloud Deployment
- **AWS**: Use EC2 or ECS
- **GCP**: Use Cloud Run or AppEngine
- **Azure**: Use App Service or AKS

---

## Support

For issues or questions, refer to [SETUP_GUIDE.md](SETUP_GUIDE.md) or create an issue on GitHub.

---

**Last Updated:** January 2026  
**API Version:** 1.0.0
