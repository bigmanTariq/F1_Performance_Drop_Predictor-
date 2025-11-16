#!/bin/bash
echo "🏎️ Starting F1 Performance Drop Predictor..."
cd /opt/render/project/src
exec uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8000}