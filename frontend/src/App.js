import React, { useState, useEffect } from 'react';
import './App.css';
import ImageUploader from './components/ImageUploader';
import PredictionResult from './components/PredictionResult';
import SampleImages from './components/SampleImages';
import ModelInfo from './components/ModelInfo';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedImage, setSelectedImage] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  useEffect(() => {
    // Fetch model info on mount
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/model-info`);
      const data = await response.json();
      setModelInfo(data);
    } catch (err) {
      console.error('Error fetching model info:', err);
    }
  };

  const handleImageSelect = (imageFile) => {
    setSelectedImage(imageFile);
    setPrediction(null);
    setError(null);
  };

  const handlePredict = async (imageFile) => {
    if (!imageFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.success) {
        setPrediction({
          ...data,
          imagePreview: URL.createObjectURL(imageFile),
        });
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (err) {
      setError(err.message || 'An error occurred during prediction');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSampleImageSelect = (imagePath) => {
    // For sample images, we need to fetch them
    fetch(`${API_BASE_URL}/sample-image/${imagePath}`)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], imagePath.split('/').pop(), { type: 'image/jpeg' });
        handleImageSelect(file);
        // Auto-predict
        setTimeout(() => handlePredict(file), 100);
      })
      .catch(err => {
        setError(`Failed to load sample image: ${err.message}`);
      });
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="header-title">
            <h1>🫁 Lung Cancer Detection AI</h1>
            <p>Advanced CT Scan Analysis using EfficientNet</p>
          </div>
          {modelInfo && (
            <div className="header-badge">
              <span className="accuracy-badge">{modelInfo.accuracy} Accuracy</span>
            </div>
          )}
        </div>
      </header>

      <main className="app-container">
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            📤 Upload Image
          </button>
          <button
            className={`tab ${activeTab === 'samples' ? 'active' : ''}`}
            onClick={() => setActiveTab('samples')}
          >
            📸 Sample Images
          </button>
          <button
            className={`tab ${activeTab === 'info' ? 'active' : ''}`}
            onClick={() => setActiveTab('info')}
          >
            ℹ️ Model Info
          </button>
        </div>

        <div className="content">
          {activeTab === 'upload' && (
            <div className="tab-content">
              <ImageUploader
                onImageSelect={handleImageSelect}
                onPredict={handlePredict}
                loading={loading}
                selectedImage={selectedImage}
              />
              {error && <div className="error-message">{error}</div>}
              {prediction && <PredictionResult prediction={prediction} />}
            </div>
          )}

          {activeTab === 'samples' && (
            <div className="tab-content">
              <SampleImages
                onImageSelect={handleSampleImageSelect}
                loading={loading}
              />
              {error && <div className="error-message">{error}</div>}
              {prediction && <PredictionResult prediction={prediction} />}
            </div>
          )}

          {activeTab === 'info' && (
            <div className="tab-content">
              <ModelInfo modelInfo={modelInfo} />
            </div>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>🔬 Medical AI Research Project | EfficientNet-B0 with Optimization</p>
      </footer>
    </div>
  );
}

export default App;
