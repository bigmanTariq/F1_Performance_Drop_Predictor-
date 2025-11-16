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

# Run the model deployment fix
echo "🤖 Ensuring models are ready..."
python deploy_models_fix.py || {
    echo "⚠️ Model preparation failed, but continuing..."
    echo "Will attempt to train on startup instead"
}

# Verify final state
echo "✅ Final verification..."
if [ -d "models/production" ] && [ "$(ls -A models/production 2>/dev/null)" ]; then
    echo "✅ Models ready for deployment"
    ls -la models/production/ | head -5
else
    echo "⚠️ No models found - will train on startup"
fi

echo "🎉 Build completed!"