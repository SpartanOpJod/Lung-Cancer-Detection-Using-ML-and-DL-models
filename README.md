<h1 align="center">🫁 Lung Cancer Detection using Machine Learning & Deep Learning</h1>

<p align="center">
  An end-to-end AI system for lung cancer detection from CT scan images using 
  <b>Deep Learning, Hybrid Optimization</b>, and a <b>Flask + React</b> web application.
</p>

<hr/>

<h2>📌 Project Overview</h2>
<p>
This project focuses on detecting lung cancer from chest CT scan images using multiple 
Machine Learning and Deep Learning models. After extensive evaluation, 
<b>EfficientNet-B0</b> achieved the best performance and was further enhanced using 
<b>hybrid metaheuristic optimization</b>.
</p>

<hr/>

<h2>📊 Dataset</h2>
<ul>
  <li><b>Source:</b> <a href="https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images">Kaggle – Chest CT Scan Images</a></li>
  <li><b>Modality:</b> CT scan slices</li>
  <li><b>Classes:</b>
    <ul>
      <li>Normal</li>
      <li>Adenocarcinoma</li>
      <li>Large Cell Carcinoma</li>
      <li>Squamous Cell Carcinoma</li>
    </ul>
  </li>
</ul>

<p><i>⚠️ Predictions are slice-level, not patient-level.</i></p>

<hr/>

<h2>🔧 Data Preprocessing</h2>
<ul>
  <li>Grayscale conversion</li>
  <li>Resize to <b>224 × 224</b></li>
  <li>Intensity normalization</li>
  <li>Train / Validation / Test split</li>
  <li>On-the-fly data augmentation during training</li>
</ul>

<hr/>

<h2>🧠 Models Implemented</h2>

<h3>Deep Learning</h3>
<ul>
  <li>ResNet-50</li>
  <li><b>EfficientNet-B0 ✅</b></li>
  <li>Vision Transformer (ViT)</li>
  <li>Swin Transformer</li>
</ul>

<h3>Machine Learning (on CNN features)</h3>
<ul>
  <li>Logistic Regression</li>
  <li>SVM (RBF Kernel)</li>
  <li>Random Forest</li>
  <li>XGBoost</li>
</ul>

<hr/>

<h2>🏆 Best Model Performance</h2>
<ul>
  <li><b>Model:</b> EfficientNet-B0</li>
  <li><b>Base Accuracy:</b> ~93.3%</li>
  <li><b>After Optimization:</b> <b>96.7%</b></li>
  <li><b>Input Size:</b> 224 × 224</li>
  <li><b>Device:</b> CUDA / CPU fallback</li>
</ul>

<hr/>

<h2>⚙️ Hybrid Optimization</h2>
<p>
To improve performance and stability, hybrid metaheuristic optimization techniques were applied:
</p>
<ul>
  <li>Blue Whale Optimization (BWO)</li>
  <li>Penguin Optimization Algorithm (POA)</li>
</ul>

<hr/>

<h2>📦 Project Structure</h2>

<pre>
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── meta/
├── models/
├── scripts/
│   ├── train_resnet50.py
│   ├── train_efficientnet_b0.py
│   ├── train_vit.py
│   ├── train_swin.py
│   ├── train_ml_models.py
│   ├── eval_models.py
│   └── run_hybrid.py
├── utils/
│   ├── dataset.py
│   └── transforms.py
├── frontend/        # React UI
├── app.py           # Flask API
└── README.md
</pre>

<hr/>

<h2>🌐 Web Application</h2>

<h3>Backend (Flask API)</h3>
<ul>
  <li><code>/health</code> – API health check</li>
  <li><code>/predict</code> – Image prediction endpoint</li>
  <li><code>/sample-images</code> – List test images</li>
  <li><code>/sample-image/&lt;path&gt;</code> – Serve image</li>
  <li><code>/model-info</code> – Model metadata</li>
</ul>

<h3>Frontend (React)</h3>
<ul>
  <li>Drag & drop CT image upload</li>
  <li>Sample image testing</li>
  <li>Confidence & probability visualization</li>
  <li>Risk-level indicator</li>
  <li>Responsive modern UI</li>
</ul>

<hr/>

<h2>🚀 Quick Start</h2>

<h3>Backend</h3>
<pre>
pip install -r requirements-api.txt
python app.py
</pre>

<h3>Frontend</h3>
<pre>
cd frontend
npm install
npm start
</pre>

<p>Frontend runs at <code>http://localhost:3000</code></p>

<hr/>

<h2>⚠️ Important Notes</h2>
<ul>
  <li>Accuracy is reported on dataset-level test data</li>
  <li>Not a medical diagnostic tool</li>
  <li>Designed as a clinical decision-support system</li>
</ul>

<hr/>

<h2>📄 License</h2>
<p>See the <code>LICENSE</code> file in the project root.</p>

<hr/>

<p align="center">
  <b>Built with ❤️ for Medical AI Research</b><br/>
  <i>Version 1.0.0 • January 2026</i>
</p>
