import React from 'react';
import './ModelInfo.css';

function ModelInfo({ modelInfo }) {
  if (!modelInfo) {
    return <div className="loading">Loading model information...</div>;
  }

  return (
    <div className="model-info-container">
      <div className="info-card">
        <h2>🏥 Model Information</h2>

        <div className="info-section">
          <h3>Model Architecture</h3>
          <div className="info-grid">
            <div className="info-item">
              <label>Model Name</label>
              <p>{modelInfo.model_name}</p>
            </div>
            <div className="info-item">
              <label>Accuracy</label>
              <p className="highlight">{modelInfo.accuracy}</p>
            </div>
            <div className="info-item">
              <label>Input Size</label>
              <p>{modelInfo.input_size}x{modelInfo.input_size} pixels</p>
            </div>
            <div className="info-item">
              <label>Device</label>
              <p>{modelInfo.device.toUpperCase()}</p>
            </div>
          </div>
        </div>

        <div className="info-section">
          <h3>Training Details</h3>
          <div className="info-list">
            <div className="list-item">
              <span className="label">Dataset</span>
              <span className="value">{modelInfo.training_data}</span>
            </div>
            <div className="list-item">
              <span className="label">Source</span>
              <a href={modelInfo.dataset_source} target="_blank" rel="noopener noreferrer">
                Kaggle Dataset
              </a>
            </div>
            <div className="list-item">
              <span className="label">Optimization</span>
              <span className="value">{modelInfo.optimization}</span>
            </div>
          </div>
        </div>

        <div className="info-section">
          <h3>Supported Classes</h3>
          <div className="classes-grid">
            {modelInfo.classes.map((className, index) => (
              <div key={index} className="class-badge">
                <div className="class-icon">
                  {className === 'Normal' && '✅'}
                  {className === 'Adenocarcinoma' && '🔴'}
                  {className === 'Large Cell Carcinoma' && '🟠'}
                  {className === 'Squamous Cell Carcinoma' && '🟡'}
                </div>
                <p>{className}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="info-section disclaimer">
          <h3>⚠️ Important Disclaimer</h3>
          <div className="disclaimer-content">
            <p>
              This AI model is designed for research and educational purposes only. 
              The predictions provided are not a substitute for professional medical diagnosis.
            </p>
            <ul>
              <li>Always consult with a qualified radiologist or medical professional</li>
              <li>This model should be used as a second opinion tool only</li>
              <li>Clinical judgment and expertise are essential for diagnosis</li>
              <li>Patient privacy and data security are paramount</li>
            </ul>
          </div>
        </div>

        <div className="info-section">
          <h3>🔬 Technical Specifications</h3>
          <div className="tech-specs">
            <div className="spec-item">
              <span className="spec-label">Framework</span>
              <span className="spec-value">PyTorch</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Pre-training</span>
              <span className="spec-value">ImageNet</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Optimization Methods</span>
              <span className="spec-value">Blue Whale + Penguin Optimization</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Number of Classes</span>
              <span className="spec-value">{modelInfo.classes.length}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ModelInfo;
