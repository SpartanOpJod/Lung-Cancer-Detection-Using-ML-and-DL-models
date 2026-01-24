import React from 'react';
import './PredictionResult.css';

function PredictionResult({ prediction }) {
  if (!prediction) return null;

  const getColorClass = (confidence) => {
    if (confidence > 0.9) return 'very-high';
    if (confidence > 0.7) return 'high';
    if (confidence > 0.5) return 'medium';
    return 'low';
  };

  const getClassEmoji = (className) => {
    const emojiMap = {
      'Normal': '✅',
      'Adenocarcinoma': '🔴',
      'Large Cell Carcinoma': '🟠',
      'Squamous Cell Carcinoma': '🟡',
    };
    return emojiMap[className] || '🏥';
  };

  const getRiskLevel = (className) => {
    if (className === 'Normal') return { level: 'LOW RISK', color: 'success' };
    if (className === 'Adenocarcinoma') return { level: 'HIGH RISK', color: 'danger' };
    if (className === 'Large Cell Carcinoma') return { level: 'HIGH RISK', color: 'danger' };
    if (className === 'Squamous Cell Carcinoma') return { level: 'HIGH RISK', color: 'danger' };
    return { level: 'UNKNOWN', color: 'warning' };
  };

  const riskInfo = getRiskLevel(prediction.predicted_class);

  return (
    <div className="prediction-container">
      <div className="result-card">
        <div className="result-header">
          <h2>Analysis Results</h2>
          <span className={`risk-badge ${riskInfo.color}`}>
            {riskInfo.level}
          </span>
        </div>

        <div className="result-main">
          <div className="image-section">
            {prediction.imagePreview && (
              <img src={prediction.imagePreview} alt="Analyzed" className="result-image" />
            )}
          </div>

          <div className="diagnosis-section">
            <div className="predicted-class">
              <div className="class-emoji">
                {getClassEmoji(prediction.predicted_class)}
              </div>
              <h3>Predicted Diagnosis</h3>
              <p className="class-name">{prediction.predicted_class}</p>
              <div className="confidence-display">
                <p className="confidence-label">Confidence Score</p>
                <p className="confidence-value">
                  {(prediction.confidence * 100).toFixed(2)}%
                </p>
              </div>
            </div>

            <div className="confidence-bar-section">
              <div className={`confidence-bar ${getColorClass(prediction.confidence)}`}>
                <div
                  className="confidence-fill"
                  style={{ width: `${prediction.confidence * 100}%` }}
                ></div>
              </div>
              <div className="confidence-labels">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="probabilities-section">
          <h4>All Class Probabilities</h4>
          <div className="probability-grid">
            {Object.entries(prediction.class_probabilities).map(([className, prob]) => (
              <div key={className} className="probability-item">
                <div className="prob-label">
                  <span>{getClassEmoji(className)}</span>
                  <span>{className}</span>
                </div>
                <div className="prob-bar-container">
                  <div className="prob-bar">
                    <div
                      className={`prob-fill ${className === prediction.predicted_class ? 'active' : ''}`}
                      style={{ width: `${prob * 100}%` }}
                    ></div>
                  </div>
                  <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {prediction.predicted_class !== 'Normal' && (
          <div className="recommendation-box">
            <h4>⚕️ Clinical Recommendation</h4>
            <p>
              A potential lung abnormality has been detected. This analysis is for
              informational purposes only and should be reviewed by a qualified medical
              professional. Please consult with a radiologist or pulmonologist for proper
              diagnosis and treatment planning.
            </p>
          </div>
        )}

        {prediction.predicted_class === 'Normal' && (
          <div className="success-box">
            <h4>✓ Normal Finding</h4>
            <p>
              The CT scan analysis indicates normal lung tissue with no signs of
              malignancy detected by the AI model.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default PredictionResult;
