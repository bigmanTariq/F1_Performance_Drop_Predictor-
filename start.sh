#!/bin/bash
echo "🏎️ Starting F1 Performance Drop Predictor..."

# Ensure models exist
if [ ! -d "/opt/render/project/src/models/production" ] || [ -z "$(ls -A /opt/render/project/src/models/production 2>/dev/null)" ]; then
    echo "🔧 Models not found, training now..."
    cd /opt/render/project/src
    python train.py
fi

cd /opt/render/project/src
exec uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8000}