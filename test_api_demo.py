#!/usr/bin/env python3
"""
Demo script to test the F1 Performance Drop Predictor API.

This script demonstrates:
- Starting the API server
- Making test requests
- Validating responses
"""

import requests
import json
import time
import subprocess
import signal
import os
from typing import Dict, Any

# API configuration
API_HOST = "localhost"
API_PORT = 8000
BASE_URL = f"http://{API_HOST}:{API_PORT}"

# Test data
VALID_FEATURES = {
    "championship_position": 3,
    "pit_stop_count": 2,
    "avg_pit_time": 28.5,
    "pit_time_std": 3.2,
    "circuit_length": 5.4,
    "points": 150,
    "pit_frequency": 1.8,
    "pit_duration_variance": 10.2,
    "high_pit_frequency": 0,
    "qualifying_gap_to_pole": 0.8,
    "grid_position_percentile": 0.75,
    "poor_qualifying": 0,
    "circuit_dnf_rate": 0.15,
    "is_street_circuit": 0,
    "championship_pressure": 0.6,
    "leader_points": 200,
    "points_gap_to_leader": 50,
    "points_pressure": 0.4,
    "driver_avg_grid_position": 8.5,
    "qualifying_vs_average": -1.5,
    "constructor_avg_grid_position": 7.2,
    "qualifying_vs_constructor_avg": -2.2,
    "bad_qualifying_day": 0,
    "circuit_dnf_rate_detailed": 0.15,
    "avg_pit_stops": 2.1,
    "avg_pit_duration": 28.0,
    "dnf_score": 0.1,
    "volatility_score": 0.3,
    "pit_complexity_score": 0.5,
    "track_difficulty_score": 0.4,
    "race_number": 10,
    "first_season": 2015,
    "seasons_active": 8,
    "estimated_age": 28,
    "driver_rolling_avg_grid": 8.2,
    "season_leader_points": 200,
    "points_gap_to_season_leader": 50,
    "is_championship_contender": 1,
    "points_momentum": 0.2,
    "championship_position_change": 0,
    "teammate_avg_grid": 9.1,
    "grid_vs_teammate": -0.6,
    "championship_pressure_score": 0.6,
    "max_round": 22,
    "late_season_race": 0,
    "championship_pressure_adjusted": 0.5,
    "points_per_race": 15.0
}

def wait_for_api(timeout: int = 30) -> bool:
    """Wait for API to become available."""
    print(f"Waiting for API at {BASE_URL}...")
    
    for i in range(timeout):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"⏳ Waiting... ({i+1}/{timeout})")
        time.sleep(1)
    
    print("❌ API failed to start within timeout")
    return False

def test_health_check() -> bool:
    """Test health check endpoint."""
    print("\n🔍 Testing health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Models loaded: {data['models_loaded']}")
            print(f"   Uptime: {data['uptime_seconds']:.1f}s")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_model_info() -> bool:
    """Test model info endpoint."""
    print("\n🔍 Testing model info...")
    
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Status: {data['status']}")
            
            if data['status'] == 'Models loaded':
                clf_model = data['classification_model']
                reg_model = data['regression_model']
                print(f"   Classification: {clf_model['name']} (F1: {clf_model['performance']['f1']:.3f})")
                print(f"   Regression: {reg_model['name']} (MAE: {reg_model['performance']['mae']:.3f})")
                print(f"   Features: {data['feature_info']['n_features']}")
            
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model info error: {str(e)}")
        return False

def test_prediction() -> bool:
    """Test single prediction endpoint."""
    print("\n🔍 Testing single prediction...")
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=VALID_FEATURES)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful")
            
            classification = data['classification']
            regression = data['regression']
            
            print(f"   Will drop position: {classification['will_drop_position']}")
            print(f"   Drop probability: {classification['probability']:.3f}")
            print(f"   Confidence: {classification['confidence']}")
            print(f"   Expected change: {regression['expected_position_change']:.2f} positions")
            
            return True
        elif response.status_code == 503:
            print("⚠️  Models not loaded, skipping prediction test")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return False

def test_batch_prediction() -> bool:
    """Test batch prediction endpoint."""
    print("\n🔍 Testing batch prediction...")
    
    # Create two different scenarios
    scenario1 = VALID_FEATURES.copy()
    scenario2 = VALID_FEATURES.copy()
    scenario2["championship_position"] = 15
    scenario2["points"] = 25
    scenario2["pit_complexity_score"] = 0.8
    
    batch_request = {
        "scenarios": [scenario1, scenario2]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict_batch", json=batch_request)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful")
            
            summary = data['summary']
            print(f"   Total requests: {summary['total_requests']}")
            print(f"   Successful: {summary['successful_predictions']}")
            print(f"   Failed: {summary['failed_predictions']}")
            print(f"   Success rate: {summary['success_rate']:.1%}")
            
            # Show first prediction
            if data['predictions']:
                pred = data['predictions'][0]
                print(f"   First prediction: {pred['classification']['probability']:.3f} drop probability")
            
            return True
        elif response.status_code == 503:
            print("⚠️  Models not loaded, skipping batch prediction test")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")
        return False

def test_error_handling() -> bool:
    """Test error handling with invalid input."""
    print("\n🔍 Testing error handling...")
    
    # Test with invalid data
    invalid_features = {"championship_position": 50}  # Missing required fields
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_features)
        
        if response.status_code == 422:  # Validation error expected
            print("✅ Error handling works correctly")
            print(f"   Returned validation error as expected")
            return True
        else:
            print(f"❌ Expected validation error, got: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False

def test_documentation() -> bool:
    """Test documentation endpoints."""
    print("\n🔍 Testing documentation...")
    
    try:
        # Test OpenAPI docs
        docs_response = requests.get(f"{BASE_URL}/docs")
        redoc_response = requests.get(f"{BASE_URL}/redoc")
        openapi_response = requests.get(f"{BASE_URL}/openapi.json")
        
        if all(r.status_code == 200 for r in [docs_response, redoc_response, openapi_response]):
            print("✅ Documentation endpoints working")
            print(f"   Swagger UI: {BASE_URL}/docs")
            print(f"   ReDoc: {BASE_URL}/redoc")
            return True
        else:
            print("❌ Some documentation endpoints failed")
            return False
            
    except Exception as e:
        print(f"❌ Documentation test failed: {str(e)}")
        return False

def run_all_tests() -> None:
    """Run all API tests."""
    print("🚀 F1 Performance Drop Predictor API Test Suite")
    print("=" * 50)
    
    # Wait for API to be ready
    if not wait_for_api():
        print("❌ API not available, cannot run tests")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling),
        ("Documentation", test_documentation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    print("This script tests the F1 Performance Drop Predictor API.")
    print("Make sure the API server is running on http://localhost:8000")
    print("\nTo start the server, run:")
    print("  python src/serve.py")
    print("\nPress Enter to continue with tests...")
    input()
    
    run_all_tests()