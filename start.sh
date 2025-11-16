#!/bin/bash
echo "🏎️ Starting F1 Performance Drop Predictor..."

# Set memory limits
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1

# Check if we're in Render environment
if [ "$RENDER_DEPLOYMENT" = "true" ]; then
    echo "🌐 Running in Render environment"
    BASE_DIR="/opt/render/project/src"
else
    echo "💻 Running in local environment"
    BASE_DIR="$(pwd)/src"
fi

# Ensure models exist
if [ ! -d "models/production" ] || [ -z "$(ls -A models/production 2>/dev/null)" ]; then
    echo "🔧 Models not found, training now..."
    
    # Check if data exists first
    if [ ! -f "data/f1_features_engineered.csv" ]; then
        echo "📊 Running data preparation..."
        python src/data_prep.py || {
            echo "❌ Data preparation failed"
            exit 1
        }
    fi
    
    echo "🤖 Training models..."
    python src/train.py || {
        echo "❌ Model training failed"
        exit 1
    }
else
    echo "✅ Models found, starting server..."
fi

# Start the server
echo "🚀 Starting server on port ${PORT:-8000}..."
cd src
exec uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1