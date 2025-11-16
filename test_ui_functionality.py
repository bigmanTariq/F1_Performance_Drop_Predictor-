#!/usr/bin/env python3
"""
Test UI Functionality

This script tests that the UI can successfully make predictions through the API.
"""

import requests
import json
import time

def test_ui_prediction():
    """Test that the UI can make a successful prediction"""
    
    print("🧪 Testing F1 UI Prediction Functionality")
    print("=" * 50)
    
    # Test data (same as what the UI would send)
    test_data = {
        "championship_position": 5,
        "pit_stop_count": 2,
        "avg_pit_time": 28.5,
        "pit_time_std": 3.2,
        "circuit_length": 5.4,
        "points": 100,
        "pit_frequency": 1.8,
        "pit_duration_variance": 10.2,
        "high_pit_frequency": 0,
        "qualifying_gap_to_pole": 1.2,
        "grid_position_percentile": 0.6,
        "poor_qualifying": 0,
        "circuit_dnf_rate": 0.15,
        "is_street_circuit": 0,
        "championship_pressure": 0.4,
        "leader_points": 200,
        "points_gap_to_leader": 100,
        "points_pressure": 0.3,
        "driver_avg_grid_position": 10.0,
        "qualifying_vs_average": 0.0,
        "constructor_avg_grid_position": 9.5,
        "qualifying_vs_constructor_avg": 0.5,
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
        "driver_rolling_avg_grid": 10.0,
        "season_leader_points": 200,
        "points_gap_to_season_leader": 100,
        "is_championship_contender": 0,
        "points_momentum": 0.1,
        "championship_position_change": 0,
        "teammate_avg_grid": 11.0,
        "grid_vs_teammate": -1.0,
        "championship_pressure_score": 0.4,
        "max_round": 22,
        "late_season_race": 0,
        "championship_pressure_adjusted": 0.3,
        "points_per_race": 8.0
    }
    
    try:
        # Test API health
        print("1. Testing API health...")
        health_response = requests.get('http://localhost:8000/health', timeout=5)
        if health_response.status_code == 200:
            print("   ✅ API is healthy")
        else:
            print("   ❌ API health check failed")
            return False
        
        # Test UI endpoint
        print("2. Testing UI endpoint...")
        ui_response = requests.get('http://localhost:8000/ui', timeout=5)
        if ui_response.status_code == 200 and 'F1 Performance Drop Predictor' in ui_response.text:
            print("   ✅ UI endpoint is working")
        else:
            print("   ❌ UI endpoint failed")
            return False
        
        # Test static files
        print("3. Testing static files...")
        css_response = requests.get('http://localhost:8000/static/styles.css', timeout=5)
        js_response = requests.get('http://localhost:8000/static/app.js', timeout=5)
        
        if css_response.status_code == 200 and js_response.status_code == 200:
            print("   ✅ Static files are accessible")
        else:
            print("   ❌ Static files failed")
            return False
        
        # Test prediction API
        print("4. Testing prediction API...")
        start_time = time.time()
        
        prediction_response = requests.post(
            'http://localhost:8000/predict',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        if prediction_response.status_code == 200:
            result = prediction_response.json()
            
            # Validate response structure
            if ('classification' in result and 
                'regression' in result and
                'probability' in result['classification'] and
                'expected_position_change' in result['regression']):
                
                print("   ✅ Prediction API is working")
                print(f"   📊 Response time: {response_time:.1f}ms")
                print(f"   🎯 Drop probability: {result['classification']['probability']:.1%}")
                print(f"   📈 Position change: {result['regression']['expected_position_change']:+.2f}")
                
            else:
                print("   ❌ Invalid prediction response format")
                return False
        else:
            print(f"   ❌ Prediction API failed with status {prediction_response.status_code}")
            print(f"   Error: {prediction_response.text}")
            return False
        
        # Test championship leader scenario
        print("5. Testing championship leader scenario...")
        leader_data = test_data.copy()
        leader_data.update({
            "championship_position": 1,
            "points": 250,
            "leader_points": 250,
            "points_gap_to_leader": 0,
            "qualifying_gap_to_pole": 0.0,
            "grid_position_percentile": 1.0,
            "championship_pressure": 0.9,
            "is_championship_contender": 1
        })
        
        leader_response = requests.post(
            'http://localhost:8000/predict',
            json=leader_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if leader_response.status_code == 200:
            leader_result = leader_response.json()
            print("   ✅ Championship leader scenario works")
            print(f"   🏆 Leader drop probability: {leader_result['classification']['probability']:.1%}")
        else:
            print("   ❌ Championship leader scenario failed")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("✨ Your F1 UI is fully functional!")
        print("🌐 Access it at: http://localhost:8000/ui")
        print("📊 API docs at: http://localhost:8000/docs")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("💡 Make sure the server is running: python src/serve.py")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_ui_prediction()
    exit(0 if success else 1)