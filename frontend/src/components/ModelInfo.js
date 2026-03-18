import React from "react";
import "./ModelInfo.css";

function ModelInfo({ modelInfo }) {
  if (!modelInfo) {
    return <div className="loading">Loading model information...</div>;
  }

  return (
    <div className="model-info-container">
      <div className="info-card">
        <h2>🏥 Model Information</h2>

        {/* Model Architecture */}
        <div className="info-section">
          <h3>🧠 Model Architecture</h3>
          <div className="info-grid">
            <div className="info-item">
              <label>Model Name</label>
              <p>{modelInfo.model_name}</p>
            </div>

            <div className="info-item">
              <label>Accuracy</label>
              <p className="highlight">93.3%</p>
            </div>

            <div className="info-item">
              <label>Input Size</label>
              <p>
                {modelInfo.input_size} 224 x 224 {modelInfo.input_size} pixels
              </p>
            </div>

            <div className="info-item">
              <label>Device</label>
              <p>{modelInfo.device?.toUpperCase()}</p>
            </div>
          </div>
        </div>

        {/* Training Details */}
        <div className="info-section">
          <h3>📊 Training Details</h3>
          <div className="info-list">
            <div className="list-item">
              <span className="label">Dataset</span>
              <span className="value">{modelInfo.training_data}</span>
            </div>

            <div className="list-item">
              <span className="label">Source</span>
              <a
                href="https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images"
                target="_blank"
                rel="noopener noreferrer"
              >
                Kaggle – Chest CT Scan Images
              </a>
            </div>

            <div className="list-item">
              <span className="label">Optimization</span>
              <span className="value">{modelInfo.optimization}</span>
            </div>
          </div>
        </div>

        {/* Supported Classes */}
        <div className="info-section">
          <h3>🧪 Supported Classes</h3>
          <div className="classes-grid">
            {modelInfo.classes?.map((className, index) => (
              <div key={index} className="class-badge">
                <div className="class-icon">
                  {className === "Normal" && "✅"}
                  {className === "Adenocarcinoma" && "🔴"}
                  {className === "Large Cell Carcinoma" && "🟠"}
                  {className === "Squamous Cell Carcinoma" && "🟡"}
                </div>
                <p>{className}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Disclaimer */}
        <div className="info-section disclaimer">
          <h3>⚠️ Important Disclaimer</h3>
          <div className="disclaimer-content">
            <p>
              This AI model is intended for research and educational purposes only.
              The predictions are not a substitute for professional medical diagnosis.
            </p>
            <ul>
              <li>Consult qualified radiologists or medical professionals</li>
              <li>Use as a decision-support tool only</li>
              <li>Clinical judgment remains essential</li>
              <li>Patient data privacy must be maintained</li>
            </ul>
          </div>
        </div>

        {/* Technical Specs */}
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
              <span className="spec-value">
                Blue Whale + Penguin Optimization
              </span>
            </div>

            <div className="spec-item">
              <span className="spec-label">Number of Classes</span>
              <span className="spec-value">
                {modelInfo.classes?.length}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ModelInfo;
