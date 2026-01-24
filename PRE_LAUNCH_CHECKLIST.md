# ✅ Pre-Launch Checklist

Use this checklist to ensure everything is ready before launching!

## System Requirements ✓

- [ ] Python 3.8+ installed (`python --version`)
- [ ] Node.js 14+ installed (`node --version`)
- [ ] npm installed (`npm --version`)
- [ ] Git installed (optional, `git --version`)
- [ ] 2GB RAM available
- [ ] 500MB free disk space
- [ ] Internet connection

## File Verification ✓

### Backend Files
- [ ] `app.py` exists and is readable
- [ ] `requirements-api.txt` exists
- [ ] `models/dl/efficientnet_b0_best.pth` exists (~20MB)
- [ ] `utils/transforms.py` exists
- [ ] `utils/dataset.py` exists

### Frontend Files
- [ ] `frontend/package.json` exists
- [ ] `frontend/src/App.js` exists
- [ ] `frontend/public/index.html` exists
- [ ] `frontend/src/components/` folder exists with 4 files

### Configuration Files
- [ ] `docker-compose.yml` exists
- [ ] `Dockerfile` exists
- [ ] `.gitignore` exists
- [ ] `.env.example` exists (optional)

### Documentation
- [ ] `START_HERE.md` exists
- [ ] `QUICKSTART.md` exists
- [ ] `SETUP_GUIDE.md` exists
- [ ] `API_DOCUMENTATION.md` exists
- [ ] `DEPLOYMENT_GUIDE.md` exists
- [ ] `TESTING_GUIDE.md` exists

## Port Availability ✓

- [ ] Port 5000 is free (Flask backend)
  ```bash
  # Check on Windows:
  netstat -ano | findstr :5000
  
  # Check on Linux/Mac:
  lsof -i :5000
  ```

- [ ] Port 3000 is free (React frontend)
  ```bash
  # Check on Windows:
  netstat -ano | findstr :3000
  
  # Check on Linux/Mac:
  lsof -i :3000
  ```

## Environment Setup ✓

- [ ] Python virtual environment ready (optional but recommended)
  ```bash
  python -m venv venv
  # Windows:
  venv\Scripts\activate
  # Linux/Mac:
  source venv/bin/activate
  ```

## Installation Verification ✓

### Backend Dependencies
- [ ] Install backend packages
  ```bash
  pip install -r requirements-api.txt
  ```

- [ ] Verify PyTorch installation
  ```bash
  python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
  ```

- [ ] Verify Flask installation
  ```bash
  python -c "import flask; print('Flask version:', flask.__version__)"
  ```

### Frontend Dependencies
- [ ] Install frontend packages
  ```bash
  cd frontend
  npm install
  cd ..
  ```

- [ ] Verify React installation
  ```bash
  cd frontend
  npm list react
  cd ..
  ```

## Model Verification ✓

- [ ] Model file exists
  ```bash
  # On Windows:
  dir models\dl\efficientnet_b0_best.pth
  
  # On Linux/Mac:
  ls -lh models/dl/efficientnet_b0_best.pth
  ```

- [ ] Model file is readable (not corrupted)
  ```bash
  # Windows:
  python -c "import torch; torch.load('models/dl/efficientnet_b0_best.pth'); print('Model OK')"
  
  # Linux/Mac:
  python -c "import torch; torch.load('models/dl/efficientnet_b0_best.pth'); print('Model OK')"
  ```

- [ ] Model size is ~20MB (not truncated)

## Data Verification ✓

- [ ] `data/meta/test.csv` exists
- [ ] `data/meta/train.csv` exists
- [ ] `data/meta/val.csv` exists
- [ ] At least one sample image exists in test data
- [ ] CSV files are readable

## Configuration ✓

- [ ] Review environment variables in `.env.example`
- [ ] Create `.env` if needed
  ```bash
  cp .env.example .env
  ```

- [ ] Update paths if running on different system
- [ ] Check CORS settings (enabled by default)

## GPU/CUDA (Optional) ✓

- [ ] Check GPU availability
  ```bash
  python -c "import torch; print('GPU:', torch.cuda.is_available())"
  ```

- [ ] If GPU available, verify CUDA
  ```bash
  nvcc --version
  ```

- [ ] If GPU NOT available, CPU will be used automatically

## Docker Verification (If Using Docker) ✓

- [ ] Docker Desktop installed and running
- [ ] Docker Compose installed (`docker-compose --version`)
- [ ] Docker images can be pulled (internet connection)
- [ ] 4GB RAM allocated to Docker
  - Windows: Docker Desktop → Settings → Resources → Memory: 4GB
  - Mac: Docker → Preferences → Resources → Memory: 4GB

## Pre-Launch Tests ✓

### Option 1: Direct Python/Node

Run these commands in sequence:

```bash
# Terminal 1 - Test backend
python app.py
# Should see: ✓ Model loaded, ✓ Ready to serve predictions
# Then: Press Ctrl+C to stop

# Terminal 2 - Test frontend (in separate window)
cd frontend
npm start
# Should see: Compiled successfully, On Your Network: http://...
# Then: Press Ctrl+C to stop
```

### Option 2: Using Docker

```bash
# Test Docker
docker-compose build
docker-compose up

# Should see both services starting
# Then: Press Ctrl+C to stop
```

## Browser & Network ✓

- [ ] Browser installed (Chrome, Firefox, Safari, or Edge)
- [ ] JavaScript enabled in browser
- [ ] Cookies enabled in browser
- [ ] No VPN blocking localhost traffic
- [ ] No firewall blocking ports 5000 and 3000

## Post-Launch Tests ✓

After starting the application:

- [ ] Frontend loads (http://localhost:3000)
- [ ] No console errors (F12 → Console)
- [ ] All three tabs visible
- [ ] Upload area displays correctly
- [ ] Sample images load
- [ ] Model info displays
- [ ] API responds (`curl http://localhost:5000/health`)
- [ ] Can predict with test image

## Troubleshooting Checklist ✓

If something doesn't work:

- [ ] Read error message carefully
- [ ] Check [SETUP_GUIDE.md](SETUP_GUIDE.md) troubleshooting
- [ ] Check logs/console output
- [ ] Try restarting both backend and frontend
- [ ] Check ports are free
- [ ] Verify files exist
- [ ] Clear browser cache
- [ ] Try different browser

## Security Checklist ✓ (Before Deployment)

- [ ] No hardcoded credentials in code
- [ ] CORS properly configured
- [ ] Input validation enabled
- [ ] File upload restrictions set
- [ ] Error messages don't leak information
- [ ] Review API_DOCUMENTATION.md security notes
- [ ] Setup SSL/TLS for production
- [ ] Plan for authentication (if needed)

## Documentation Checklist ✓

- [ ] Read START_HERE.md
- [ ] Read QUICKSTART.md
- [ ] Familiar with SETUP_GUIDE.md
- [ ] Know where API_DOCUMENTATION.md is
- [ ] Know where DEPLOYMENT_GUIDE.md is
- [ ] Know where TESTING_GUIDE.md is

## Performance Baseline ✓

Note these before launching (for comparison later):

- [ ] First inference time: _____ms
- [ ] Backend startup time: _____s
- [ ] Frontend load time: _____s
- [ ] Average prediction time: _____ms
- [ ] UI responsiveness: _____(smooth/acceptable/slow)

## Final Sign-Off ✓

- [ ] All system requirements met
- [ ] All files verified
- [ ] Ports available
- [ ] Dependencies installed
- [ ] Model verified
- [ ] Configuration complete
- [ ] Pre-launch tests passed
- [ ] Documentation reviewed
- [ ] Ready to launch! 🚀

---

## Launch Commands

### Choose Your Method:

#### Windows Users:
```bash
start.bat
```

#### Linux/Mac Users:
```bash
bash start.sh
```

#### Docker Users:
```bash
docker-compose up
```

---

## After Launch

1. ✅ Access http://localhost:3000
2. ✅ Try uploading an image
3. ✅ Try sample images
4. ✅ Check model info
5. ✅ Test API endpoints
6. ✅ Verify predictions work

---

## Success! 🎉

If everything above is checked, your application should be running perfectly!

**Next Steps:**
- Test thoroughly
- Customize colors/styling if desired
- Plan deployment to cloud
- Setup monitoring
- Deploy to production

---

## Emergency Support

| Issue | Quick Fix |
|-------|-----------|
| Port in use | Change port in app.py |
| Model not found | Verify path to model file |
| Can't import torch | `pip install torch torchvision` |
| npm install fails | Delete `node_modules`, try again |
| API 500 error | Check app.py logs, restart backend |
| Images not loading | Check data folder permissions |

---

**You're ready to launch!** 🚀

Mark this checklist as complete when done, then launch the application!
