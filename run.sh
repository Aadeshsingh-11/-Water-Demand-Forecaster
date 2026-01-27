#!/bin/bash

# Water Demand Forecaster - Run Script

echo "🚀 Starting Water Demand Forecaster Backend..."
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: .venv not found. Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r BACKEND/requirements.txt
fi

echo ""
echo "📊 Model files status:"
if [ -f "water_usage_model.pkl" ] && [ -f "scaler.pkl" ]; then
    echo "✓ Model files found"
else
    echo "⚠ Warning: Model files not found. Please run main.ipynb first."
    exit 1
fi

echo ""
echo "🌐 Starting Flask server on http://localhost:5000"
echo "   Press CTRL+C to stop the server"
echo ""

cd "$(dirname "$0")"
python BACKEND/app.py
