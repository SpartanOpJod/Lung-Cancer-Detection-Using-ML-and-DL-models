import React from 'react';
import './PredictionResult.css';

const LOW_CONFIDENCE_THRESHOLD = 0.35;

function PredictionResult({ prediction }) {
  if (!prediction) return null;

  // ---- FORCE CORRECT LOGIC: pick max probability ----
  const probs = prediction.class_probabilities;

  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const finalClass = sorted[0][0];
  const finalConfidence = sorted[0][1];

  const isLowConfidence = finalConfidence < LOW_CONFIDENCE_THRESHOLD;

  const getColorClass = (confidence) => {
    if (confidence > 0.9) return 'very-high';
    if (confidence > 0.7) return 'high';
    if (confidence > 0.5) return 'medium';
    return 'low';
  };

  const getClassEmoji = (className) => {
    const emojiMap = {
      'adenocarcinoma': '🔴',
      'large.cell.carcinoma': '🟠',
      'normal': '✅',
      'squamous.cell.carcinoma': '🟡',
    };
    return emojiMap[className] || '🏥';
  };

  const getRiskLevel = () => {
    if (finalClass === 'normal') {
      return { level: 'LOW RISK', color: 'success' };
    }
    return { level: 'HIGH RISK', color: 'danger' };
  };

  const riskInfo = getRiskLevel();

  return (
    <div className="prediction-container">
      <div className="result-card">
        <div className="result-header">
          <h2>AI Prediction Result</h2>
          <span className={`risk-badge ${riskInfo.color}`}>
            {riskInfo.level}
          </span>
        </div>

        <div className="result-main">
          <div className="image-section">
            {prediction.imagePreview && (
              <img
                src={prediction.imagePreview}
                alt="Analyzed"
                className="result-image"
              />
            )}
          </div>

          <div className="diagnosis-section">
            <div className="predicted-class">
              <div className="class-emoji">
                {getClassEmoji(finalClass)}
              </div>
              <h3>Predicted Disease</h3>
              <p className="class-name">{finalClass}</p>

              <div className="confidence-display">
                <p className="confidence-label">Confidence</p>
                <p className="confidence-value">
                  {(finalConfidence * 100).toFixed(2)}%
                </p>
              </div>

              {isLowConfidence && (
                <p className="low-confidence-warning">
                  ⚠️ Low confidence — prediction is based on limited visual cues.
                </p>
              )}
            </div>

            <div className="confidence-bar-section">
              <div className={`confidence-bar ${getColorClass(finalConfidence)}`}>
                <div
                  className="confidence-fill"
                  style={{ width: `${finalConfidence * 100}%` }}
                />
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
            {sorted.map(([className, prob]) => (
              <div key={className} className="probability-item">
                <div className="prob-label">
                  <span>{getClassEmoji(className)}</span>
                  <span>{className}</span>
                </div>
                <div className="prob-bar-container">
                  <div className="prob-bar">
                    <div
                      className={`prob-fill ${
                        className === finalClass ? 'active' : ''
                      }`}
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                  <span className="prob-value">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {finalClass !== 'normal' && (
          <div className="recommendation-box">
            <h4>⚕️ Clinical Note</h4>
            <p>
              This AI result is for decision support only and should be reviewed
              by a medical professional.
            </p>
          </div>
        )}

        {finalClass === 'normal' && (
          <div className="success-box">
            <h4>✓ Normal Finding</h4>
            <p>
              No malignant patterns were detected by the AI model.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default PredictionResult;
