#!/bin/bash

# 🫁 Lung Cancer Detection Web App - Start Script

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       🫁 Lung Cancer Detection AI - Web Application           ║"
echo "║            Starting Backend and Frontend Services             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+"
    exit 1
fi

# Check if Node is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 14+"
    exit 1
fi

# Create log directory
mkdir -p logs

echo "📦 Installing backend dependencies..."
pip install -q -r requirements-api.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install backend dependencies"
    exit 1
fi

echo "✓ Backend dependencies installed"
echo ""

echo "📦 Installing frontend dependencies..."
cd frontend
npm install --silent
if [ $? -ne 0 ]; then
    echo "❌ Failed to install frontend dependencies"
    exit 1
fi

echo "✓ Frontend dependencies installed"
echo ""

cd ..

# Start backend in background
echo "🚀 Starting Flask API server..."
python app.py > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Failed to start backend"
    cat logs/backend.log
    exit 1
fi

echo "✓ API running at http://localhost:5000"
echo ""

# Start frontend in background
echo "🚀 Starting React development server..."
cd frontend
npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"
sleep 3

cd ..

# Check if frontend started
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Failed to start frontend"
    cat logs/frontend.log
    kill $BACKEND_PID
    exit 1
fi

echo "✓ UI running at http://localhost:3000"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                 ✨ Services Started Successfully ✨            ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  🌐 Web UI:      http://localhost:3000                         ║"
echo "║  🔌 API Server:  http://localhost:5000                         ║"
echo "║  📊 Health:      http://localhost:5000/health                  ║"
echo "║                                                                ║"
echo "║  Press Ctrl+C to stop all services                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Wait for user to press Ctrl+C
wait

echo ""
echo "🛑 Stopping services..."
kill $BACKEND_PID 2>/dev/null
kill $FRONTEND_PID 2>/dev/null
echo "✓ All services stopped"
