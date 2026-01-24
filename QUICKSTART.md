# 🚀 Quick Start Guide - 5 Minutes

## The Fastest Way to Get Started

### Step 1: Install Dependencies (2 minutes)

**Backend:**
```bash
pip install -r requirements-api.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### Step 2: Start Both Services

**Terminal 1 - Backend API:**
```bash
python app.py
```
You should see:
```
🫁 Lung Cancer Detection API
✓ Model loaded from models/dl/efficientnet_b0_best.pth
Device: cuda
✓ Ready to serve predictions
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```
Browser opens to `http://localhost:3000`

### Step 3: Use the Application

1. **Upload Tab**: Drag & drop or click to upload a CT scan image
2. **Sample Tab**: Click on test images to analyze them
3. **Info Tab**: View model details and specifications

That's it! Your lung cancer detection UI is live! 🎉

## Using Docker (Even Faster!)

```bash
docker-compose up --build
```

Open browser to:
- **Frontend**: http://localhost:3000
- **API**: http://localhost:5000/health

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| Port 5000 already in use | Change port in `app.py` line 128 |
| Port 3000 already in use | Run `npm start -- --port 3001` |
| Model not found | Verify file at `models/dl/efficientnet_b0_best.pth` |
| No GPU detected | Falls back to CPU automatically |
| Images not loading | Check permissions on `data/` folder |

## Next Steps

- Read [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed documentation
- Customize styling in `frontend/src/components/*.css`
- Modify API endpoints in `app.py`
- Deploy with Docker Compose or Kubernetes

---

Need help? Check the troubleshooting section in SETUP_GUIDE.md!
