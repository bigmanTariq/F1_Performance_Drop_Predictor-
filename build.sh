#!/bin/bash

# Render.com build script for F1 Performance Drop Predictor

echo "🏎️ Starting F1 Performance Drop Predictor build..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if data files exist, if not create sample data
echo "📊 Checking data files..."
if [ ! -f "data/f1_performance_drop_clean.csv" ]; then
    echo "⚠️ Data files not found, running data preparation..."
    python src/data_prep.py
else
    echo "✅ Data files found"
fi

# Train models if they don't exist
echo "🤖 Checking for trained models..."
if [ ! -d "models/production" ] || [ -z "$(ls -A models/production 2>/dev/null)" ]; then
    echo "🔧 Training models..."
    python src/train.py
else
    echo "✅ Trained models found"
fi

echo "🎉 Build completed successfully!"