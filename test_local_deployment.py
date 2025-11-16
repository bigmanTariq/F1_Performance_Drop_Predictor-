#!/usr/bin/env python3
"""
Test script to verify local deployment works before pushing to Render.
"""

import requests
import json
import time
import subprocess
import sys
import os

def test_health_endpoint(base_url):
    """Test the health endpoint."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health check status: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"Service status: {health_data.get('status')}")
            print(f"Models loaded: {health_data.get('models_loaded')}")
            return health_data.get('models_loaded', False)
        else:
            print(f"Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def test_prediction_endpoint(base_url):
    """Test the prediction endpoint."""
    # Sample prediction data
    sample_data = {
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
    
    try:
        response = requests.post(f"{base_url}/predict", 
                               json=sample_data, 
                               timeout=30)
        print(f"Prediction status: {response.status_code}")
        
        if response.status_code == 200:
            prediction = response.json()
            print(f"Position drop probability: {prediction['classification']['probability']:.3f}")
            print(f"Expected position change: {prediction['regression']['expected_position_change']:.2f}")
            return True
        else:
            print(f"Prediction failed: {response.text}")
            return False
    except Exception as e:
        print(f"Prediction error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing F1 Performance Drop Predictor Deployment")
    print("=" * 50)
    
    # Check if models exist
    if not os.path.exists("models/production") or not os.listdir("models/production"):
        print("❌ No models found. Running training first...")
        
        # Run data prep if needed
        if not os.path.exists("data/f1_features_engineered.csv"):
            print("📊 Running data preparation...")
            result = subprocess.run([sys.executable, "src/data_prep.py"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Data prep failed: {result.stderr}")
                return False
        
        # Run training
        print("🤖 Training models...")
        result = subprocess.run([sys.executable, "src/train.py"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Training failed: {result.stderr}")
            return False
    
    # Start the server in background
    print("🚀 Starting server...")
    server_process = subprocess.Popen([
        sys.executable, "src/serve.py"
    ], env={**os.environ, "PORT": "8002"})
    
    # Wait for server to start
    time.sleep(10)
    
    try:
        base_url = "http://localhost:8002"
        
        # Test health endpoint
        print("\n🏥 Testing health endpoint...")
        models_loaded = test_health_endpoint(base_url)
        
        if models_loaded:
            print("✅ Health check passed")
            
            # Test prediction endpoint
            print("\n🔮 Testing prediction endpoint...")
            prediction_success = test_prediction_endpoint(base_url)
            
            if prediction_success:
                print("✅ Prediction test passed")
                print("\n🎉 All tests passed! Deployment should work on Render.")
                return True
            else:
                print("❌ Prediction test failed")
                return False
        else:
            print("❌ Health check failed - models not loaded")
            return False
            
    finally:
        # Clean up server process
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)