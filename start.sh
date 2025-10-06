#!/bin/bash

# Multi-Agent Flood Prediction Communication System Startup Script

echo "Starting Multi-Agent Flood Prediction Communication System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is required but not installed. Please install Node.js."
    exit 1
fi

# Start backend
echo "Starting backend server..."
cd backend
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

# Start backend in background
echo "Starting FastAPI server on http://localhost:8000"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Go back to root directory
cd ..

# Start frontend
echo "Starting frontend development server..."
cd frontend
npm install
echo "Starting React development server on http://localhost:3000"
npm start &
FRONTEND_PID=$!

# Wait for user to stop the services
echo ""
echo "Services started successfully!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for processes
wait
