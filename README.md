🫁 Lung Cancer Detection Using Machine Learning & Deep Learning
📌 Project Overview

This project presents an AI-based lung cancer detection system using CT scan images.
Multiple Machine Learning (ML) and Deep Learning (DL) models were trained and evaluated to classify lung CT slices into different cancer types. The best-performing model was further optimized and prepared for deployment.

📊 Dataset

Source: Kaggle – Chest CT Scan Images
https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

Modality: CT scan images (slice-level)

Classes:

Normal

Adenocarcinoma

Large Cell Carcinoma

Squamous Cell Carcinoma

🔧 Data Preprocessing

Images converted to grayscale

Resized to 224 × 224

Intensity normalization applied

Dataset split into train / validation / test

On-the-fly data augmentation used during training to improve generalization

🧠 Models Implemented
Deep Learning Models

ResNet-50

EfficientNet-B0 ✅

Vision Transformer (ViT)

Swin Transformer

Machine Learning Models (on CNN-extracted features)

Logistic Regression

Support Vector Machine (RBF Kernel)

Random Forest

XGBoost

🏆 Best Model

EfficientNet-B0 achieved the best overall performance due to:

Efficient compound scaling

Better generalization

Lower computational cost

Performance:

Base Accuracy: ~93.3%

After Hybrid Optimization: ~96.7%

⚙️ Hybrid Optimization

To further enhance performance, metaheuristic optimization techniques were applied:

Blue Whale Optimization (BWO)

Penguin Optimization Algorithm (POA)

These optimizers fine-tuned model parameters, leading to improved accuracy and stability.

🌐 Deployment Architecture

Backend: Flask (Python)

Frontend: React

Model: EfficientNet-B0 (CPU-friendly)

REST API endpoints for prediction and model information

Designed as a decision-support system, not a diagnostic replacement

⚠️ Important Notes

Accuracy reported is dataset-level test accuracy

Predictions are slice-level, not patient-level

Low confidence on some slices is expected due to tumor invisibility or ambiguity

The system is intended to assist clinicians, not replace medical professionals

📦 Repository Policy

Large datasets and trained model weights are excluded due to size constraints

Only source code, scripts, and metadata are version-controlled

Models can be regenerated using the provided training scripts

✅ Conclusion

This project demonstrates the effective use of deep learning and hybrid optimization techniques for lung cancer detection from CT images, while addressing real-world challenges such as preprocessing, deployment consistency, and ethical AI usage in healthcare.