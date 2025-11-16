#!/bin/bash

# Render.com build script for F1 Performance Drop Predictor

echo "🏎️ Starting F1 Performance Drop Predictor build..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Always run data preparation (ensure fresh data)
echo "📊 Running data preparation..."
python src/data_prep.py

# Always train models (ensure they exist)
echo "🤖 Training models..."
python src/train.py

# Verify models were created
echo "✅ Verifying models..."
if [ -d "models/production" ] && [ "$(ls -A models/production)" ]; then
    echo "✅ Models successfully created"
    ls -la models/production/
else
    echo "❌ Model creation failed"
    exit 1
fi

echo "🎉 Build completed successfully!"