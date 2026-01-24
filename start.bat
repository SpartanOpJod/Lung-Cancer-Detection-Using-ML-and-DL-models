@echo off
REM 🫁 Lung Cancer Detection Web App - Start Script for Windows

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║       🫁 Lung Cancer Detection AI - Web Application           ║
echo ║            Starting Backend and Frontend Services             ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if Node is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js 14+ from https://nodejs.org/
    pause
    exit /b 1
)

REM Create log directory
if not exist logs mkdir logs

echo 📦 Installing backend dependencies...
pip install -q -r requirements-api.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install backend dependencies
    pause
    exit /b 1
)
echo ✓ Backend dependencies installed
echo.

echo 📦 Installing frontend dependencies...
cd frontend
call npm install --silent
if %errorlevel% neq 0 (
    echo ❌ Failed to install frontend dependencies
    cd ..
    pause
    exit /b 1
)
echo ✓ Frontend dependencies installed
cd ..
echo.

echo 🚀 Starting Flask API server...
start "Flask Backend" cmd /k python app.py
timeout /t 2 /nobreak >nul
echo ✓ API running at http://localhost:5000
echo.

echo 🚀 Starting React development server...
start "React Frontend" cmd /k "cd frontend && npm start"
timeout /t 3 /nobreak >nul
echo ✓ UI running at http://localhost:3000
echo.

echo ╔════════════════════════════════════════════════════════════════╗
echo ║                 ✨ Services Started Successfully ✨            ║
echo ╠════════════════════════════════════════════════════════════════╣
echo ║  🌐 Web UI:      http://localhost:3000                         ║
echo ║  🔌 API Server:  http://localhost:5000                         ║
echo ║  📊 Health:      http://localhost:5000/health                  ║"
echo ║                                                                ║
echo ║  Close the command windows to stop the services               ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Open browser
start http://localhost:3000

pause
