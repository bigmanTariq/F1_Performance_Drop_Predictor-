#!/usr/bin/env python3
"""
Test script to verify deployment fixes work locally.
"""

import requests
import json
import time
import sys

def test_local_api(base_url="http://localhost:8001"):
    """Test the local API endpoints."""
    print(f"🧪 Testing API at {base_url}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        health_data = response.json()
        print(f"Health Status: {health_data['status']}")
        print(f"Models Loaded: {health_data['models_loaded']}")
        
        if not health_data['models_loaded']:
            print("❌ Models not loaded - deployment issue detected")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test model info
    try:
        response = requests.get(f"{base_url}/model_info", timeout=10)
        model_info = response.json()
        print(f"Model Status: {model_info['status']}")
        
        if model_info['status'] != 'Models loaded':
            print("❌ Models not properly loaded")
            return False
            
    except Exception as e:
        print(f"❌ Model info check failed: {e}")
        return False
    
    print("✅ All tests passed!")
    return True

if __name__ == "__main__":
    # Test with different URLs
    urls_to_test = [
        "http://localhost:8001",
        "https://f1-performance-drop-predictor.onrender.com"
    ]
    
    for url in urls_to_test:
        print(f"\n{'='*50}")
        print(f"Testing: {url}")
        print('='*50)
        
        success = test_local_api(url)
        if not success:
            print(f"❌ Tests failed for {url}")
        else:
            print(f"✅ Tests passed for {url}")