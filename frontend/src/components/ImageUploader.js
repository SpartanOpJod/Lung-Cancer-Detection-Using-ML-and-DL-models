import React, { useState } from 'react';
import './ImageUploader.css';

function ImageUploader({ onImageSelect, onPredict, loading, selectedImage }) {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onImageSelect(file);
      } else {
        alert('Please drop an image file');
      }
    }
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onImageSelect(e.target.files[0]);
    }
  };

  return (
    <div className="uploader-container">
      <div className="uploader-card">
        <div
          className={`drop-zone ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="drop-content">
            <div className="drop-icon">📁</div>
            <h3>Drop your CT scan image here</h3>
            <p>or click to browse</p>
            <input
              type="file"
              accept="image/*"
              onChange={handleInputChange}
              className="file-input"
              id="image-input"
              disabled={loading}
            />
            <label htmlFor="image-input" className="file-label">
              Choose File
            </label>
          </div>
        </div>

        {selectedImage && (
          <div className="preview-section">
            <h3>Preview</h3>
            <div className="image-preview">
              <img src={URL.createObjectURL(selectedImage)} alt="Preview" />
              <p className="file-name">{selectedImage.name}</p>
            </div>
            <button
              className="predict-button"
              onClick={() => onPredict(selectedImage)}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing...
                </>
              ) : (
                '🔬 Analyze Image'
              )}
            </button>
          </div>
        )}
      </div>

      <div className="info-box">
        <h4>📋 Requirements</h4>
        <ul>
          <li>✓ CT scan chest images</li>
          <li>✓ JPEG, PNG, or other image formats</li>
          <li>✓ Recommended size: 224x224 pixels</li>
          <li>✓ Clear and high-quality images</li>
        </ul>
      </div>
    </div>
  );
}

export default ImageUploader;
