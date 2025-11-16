#!/bin/bash

# F1 Performance Drop Predictor Setup Script
echo "Setting up F1 Performance Drop Predictor environment..."

# Create conda environment
echo "Creating conda environment 'f1-predictor' with Python 3.9..."
conda env create -f environment.yml

echo ""
echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate f1-predictor"
echo ""
echo "To install dependencies manually (alternative to conda env):"
echo "  pip install -r requirements.txt"
echo ""
echo "Project structure:"
echo "  data/     - F1 CSV datasets"
echo "  src/      - Source code modules"
echo "  models/   - Trained ML models"
echo "  notebooks/ - Jupyter notebooks for exploration"