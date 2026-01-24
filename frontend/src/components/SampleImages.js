import React, { useState, useEffect } from 'react';
import './SampleImages.css';

function SampleImages({ onImageSelect, loading }) {
  const [samples, setSamples] = useState([]);
  const [sampleLoading, setSampleLoading] = useState(true);
  const [error, setError] = useState(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  useEffect(() => {
    fetchSamples();
  }, []);

  const fetchSamples = async () => {
    try {
      setSampleLoading(true);
      const response = await fetch(`${API_BASE_URL}/sample-images`);
      const data = await response.json();

      if (data.success) {
        setSamples(data.samples.slice(0, 12)); // Show first 12 samples
      } else {
        setError('Failed to load sample images');
      }
    } catch (err) {
      console.error('Error fetching samples:', err);
      setError('Could not fetch sample images');
    } finally {
      setSampleLoading(false);
    }
  };

  const getClassColor = (className) => {
    const colorMap = {
      'adenocarcinoma': '#e74c3c',
      'large.cell.carcinoma': '#f39c12',
      'normal': '#27ae60',
      'squamous.cell.carcinoma': '#f1c40f',
    };
    return colorMap[className.toLowerCase()] || '#95a5a6';
  };

  const getClassEmoji = (className) => {
    const emojiMap = {
      'normal': '✅',
      'adenocarcinoma': '🔴',
      'large.cell.carcinoma': '🟠',
      'squamous.cell.carcinoma': '🟡',
    };
    const key = className.toLowerCase();
    return emojiMap[key] || '🏥';
  };

  if (sampleLoading) {
    return (
      <div className="samples-container">
        <div className="loading-spinner">
          <p>Loading sample images...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="samples-container">
        <div className="error-message">{error}</div>
      </div>
    );
  }

  return (
    <div className="samples-container">
      <div className="samples-header">
        <h2>Test Dataset Samples</h2>
        <p>Click on any image to analyze it</p>
      </div>

      {samples.length === 0 ? (
        <div className="no-samples">
          <p>No sample images available</p>
        </div>
      ) : (
        <div className="samples-grid">
          {samples.map((sample, index) => (
            <div
              key={index}
              className="sample-card"
              onClick={() => onImageSelect(sample.image)}
              style={{ cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.6 : 1 }}
            >
              <div
                className="sample-image"
                style={{
                  backgroundColor: getClassColor(sample.label),
                }}
              >
                <img
                  src={`${API_BASE_URL}/sample-image/${sample.image}`}
                  alt={sample.label}
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
                <div className="sample-overlay">
                  <span className="analyze-text">Click to Analyze</span>
                </div>
              </div>
              <div className="sample-info">
                <p className="sample-class">
                  {getClassEmoji(sample.label)} {sample.label}
                </p>
                <p className="sample-path">{sample.image.split('/').pop()}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="samples-footer">
        <p>💡 Select any image from your test dataset to test the model predictions</p>
      </div>
    </div>
  );
}

export default SampleImages;
