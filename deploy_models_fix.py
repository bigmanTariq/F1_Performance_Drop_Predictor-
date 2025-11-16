#!/usr/bin/env python3
"""
Deployment model fix script.
Ensures models are properly trained and available for deployment.
"""

import os
import sys
import logging
import subprocess
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_models_exist():
    """Ensure models exist and are properly trained."""
    logger.info("🔍 Checking model availability...")
    
    models_dir = Path("models/production")
    
    # Check if models directory exists and has files
    if models_dir.exists() and list(models_dir.glob("*.joblib")):
        logger.info("✅ Models found in production directory")
        return True
    
    logger.info("❌ No models found, need to train...")
    
    # Ensure data exists first
    data_file = Path("data/f1_features_engineered.csv")
    if not data_file.exists():
        logger.info("📊 Running data preparation...")
        try:
            result = subprocess.run([
                sys.executable, "src/data_prep.py"
            ], check=True, capture_output=True, text=True, timeout=300)
            logger.info("✅ Data preparation completed")
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            return False
    
    # Train models
    logger.info("🤖 Training models...")
    try:
        result = subprocess.run([
            sys.executable, "src/train.py"
        ], check=True, capture_output=True, text=True, timeout=600)
        logger.info("✅ Model training completed")
        return True
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        return False

if __name__ == "__main__":
    success = ensure_models_exist()
    if not success:
        sys.exit(1)
    logger.info("🎉 Models ready for deployment!")