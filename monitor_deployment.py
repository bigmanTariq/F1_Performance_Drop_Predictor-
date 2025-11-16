#!/usr/bin/env python3
"""
Monitor Render deployment status.
"""

import requests
import time
import json

def check_deployment_status():
    """Check if the deployment is ready."""
    url = "https://f1-performance-drop-predictor.onrender.com/health"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        print(f"Status: {data['status']}")
        print(f"Models Loaded: {data['models_loaded']}")
        print(f"Uptime: {data['uptime_seconds']:.1f}s")
        print(f"Error Rate: {data['error_rate']:.1f}%")
        
        return data['models_loaded'] and data['status'] == 'healthy'
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def monitor_until_ready(max_wait_minutes=15):
    """Monitor deployment until ready or timeout."""
    print("🔍 Monitoring deployment status...")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while time.time() - start_time < max_wait_seconds:
        print(f"\n⏰ {time.strftime('%H:%M:%S')} - Checking status...")
        
        if check_deployment_status():
            print("✅ Deployment is ready!")
            return True
        
        print("⏳ Waiting 30 seconds before next check...")
        time.sleep(30)
    
    print(f"⏰ Timeout after {max_wait_minutes} minutes")
    return False

if __name__ == "__main__":
    monitor_until_ready()