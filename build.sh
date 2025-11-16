#!/bin/bash

# Render.com build script for F1 Performance Drop Predictor

echo "🏎️ Starting F1 Performance Drop Predictor build..."

# Set memory limits for Python to avoid OOM on free tier
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Install dependencies
echo "📦 Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Change to project root (build runs in different directory)
if [ -d "/opt/render/project/src" ]; then
    cd /opt/render/project/src
    echo "Working from: $(pwd)"
else
    echo "Working from: $(pwd)"
fi

# Run data preparation first
echo "📊 Preparing data..."
timeout 300 python src/data_prep.py || timeout 300 python data_prep.py || {
    echo "⚠️ Data preparation failed or timed out"
}

# Run model training
echo "🤖 Training models..."
timeout 600 python src/train.py || timeout 600 python train.py || {
    echo "⚠️ Model training failed or timed out"
}

# Verify what was created
echo "🔍 Checking created files..."
echo "Data files:"
ls -la data/ 2>/dev/null || echo "No data directory"

echo "Model files:"
ls -la models/ 2>/dev/null || echo "No models directory"

echo "Production models:"
ls -la models/production/ 2>/dev/null || echo "No production models directory"

# Count actual model files
MODEL_COUNT=$(find models/production -name "*.joblib" 2>/dev/null | wc -l)
echo "Found $MODEL_COUNT model .joblib files"

if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "✅ Models successfully created during build"
else
    echo "⚠️ No model files created - will train on startup"
fi

echo "🎉 Build completed!"