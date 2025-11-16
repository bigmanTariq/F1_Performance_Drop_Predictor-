#!/usr/bin/env python3
"""
Test API connectivity from different perspectives.
"""

import requests
import json

def test_api_endpoints():
    """Test all API endpoints."""
    base_url = "https://f1-performance-drop-predictor.onrender.com"
    
    endpoints = [
        "/health",
        "/model_info", 
        "/debug",
        "/ui",
        "/"
    ]
    
    print(f"Testing API connectivity to {base_url}")
    print("=" * 60)
    
    for endpoint in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, timeout=10)
            
            if endpoint == "/ui" or endpoint == "/":
                # These return HTML
                status = "✅ OK" if response.status_code == 200 else f"❌ {response.status_code}"
                print(f"{endpoint:15} {status} (HTML content)")
            else:
                # These return JSON
                if response.status_code == 200:
                    data = response.json()
                    if endpoint == "/health":
                        models_loaded = data.get('models_loaded', False)
                        status = data.get('status', 'unknown')
                        print(f"{endpoint:15} ✅ OK - Status: {status}, Models: {models_loaded}")
                    elif endpoint == "/debug":
                        predictor_status = data.get('predictor_status', 'unknown')
                        models_loaded = data.get('models_loaded', False)
                        print(f"{endpoint:15} ✅ OK - Predictor: {predictor_status}, Models: {models_loaded}")
                    else:
                        print(f"{endpoint:15} ✅ OK")
                else:
                    print(f"{endpoint:15} ❌ {response.status_code}")
                    
        except Exception as e:
            print(f"{endpoint:15} ❌ ERROR: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Testing CORS and browser compatibility...")
    
    # Test CORS headers
    try:
        response = requests.options(f"{base_url}/health")
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
        }
        print("CORS Headers:")
        for header, value in cors_headers.items():
            print(f"  {header}: {value or 'Not set'}")
    except Exception as e:
        print(f"CORS test failed: {e}")

if __name__ == "__main__":
    test_api_endpoints()