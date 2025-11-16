#!/usr/bin/env python3
"""
Test script for deployed F1 Performance Drop Predictor API.
Use this to verify your deployment is working correctly.
"""

import requests
import json
import sys
import time
from typing import Dict, Any

def test_deployment(base_url: str) -> Dict[str, Any]:
    """
    Test the deployed API endpoints.
    
    Args:
        base_url: Base URL of the deployed service
        
    Returns:
        Dictionary with test results
    """
    results = {
        'base_url': base_url,
        'tests': {},
        'overall_status': 'UNKNOWN'
    }
    
    print(f"🏎️ Testing F1 Performance Drop Predictor at: {base_url}")
    print("=" * 60)
    
    # Test 1: Health Check
    print("1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            health_data = response.json()
            results['tests']['health'] = {
                'status': 'PASS',
                'response_time': response.elapsed.total_seconds(),
                'data': health_data
            }
            print(f"   ✅ Health check passed ({response.elapsed.total_seconds():.2f}s)")
            print(f"   📊 Status: {health_data.get('status', 'unknown')}")
        else:
            results['tests']['health'] = {
                'status': 'FAIL',
                'error': f"HTTP {response.status_code}"
            }
            print(f"   ❌ Health check failed: HTTP {response.status_code}")
    except Exception as e:
        results['tests']['health'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"   ❌ Health check failed: {str(e)}")
    
    # Test 2: Model Info
    print("\n2️⃣ Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model_info", timeout=30)
        if response.status_code == 200:
            model_data = response.json()
            results['tests']['model_info'] = {
                'status': 'PASS',
                'response_time': response.elapsed.total_seconds(),
                'data': model_data
            }
            print(f"   ✅ Model info retrieved ({response.elapsed.total_seconds():.2f}s)")
            print(f"   🤖 Models loaded: {model_data.get('status', 'unknown')}")
        else:
            results['tests']['model_info'] = {
                'status': 'FAIL',
                'error': f"HTTP {response.status_code}"
            }
            print(f"   ❌ Model info failed: HTTP {response.status_code}")
    except Exception as e:
        results['tests']['model_info'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"   ❌ Model info failed: {str(e)}")
    
    # Test 3: Prediction
    print("\n3️⃣ Testing prediction endpoint...")
    test_data = {
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
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if response.status_code == 200:
            pred_data = response.json()
            results['tests']['prediction'] = {
                'status': 'PASS',
                'response_time': response.elapsed.total_seconds(),
                'data': pred_data
            }
            print(f"   ✅ Prediction successful ({response.elapsed.total_seconds():.2f}s)")
            
            # Extract key prediction results
            classification = pred_data.get('classification', {})
            regression = pred_data.get('regression', {})
            
            print(f"   🎯 Will drop position: {classification.get('will_drop_position', 'unknown')}")
            print(f"   📊 Drop probability: {classification.get('probability', 0):.3f}")
            print(f"   📈 Expected change: {regression.get('expected_position_change', 0):.2f} positions")
            
        else:
            results['tests']['prediction'] = {
                'status': 'FAIL',
                'error': f"HTTP {response.status_code}: {response.text}"
            }
            print(f"   ❌ Prediction failed: HTTP {response.status_code}")
            print(f"   📝 Response: {response.text[:200]}...")
    except Exception as e:
        results['tests']['prediction'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"   ❌ Prediction failed: {str(e)}")
    
    # Test 4: Web Interface
    print("\n4️⃣ Testing web interface...")
    try:
        response = requests.get(f"{base_url}/ui", timeout=30)
        if response.status_code == 200:
            results['tests']['web_ui'] = {
                'status': 'PASS',
                'response_time': response.elapsed.total_seconds()
            }
            print(f"   ✅ Web interface accessible ({response.elapsed.total_seconds():.2f}s)")
            print(f"   🌐 Visit: {base_url}/ui")
        else:
            results['tests']['web_ui'] = {
                'status': 'FAIL',
                'error': f"HTTP {response.status_code}"
            }
            print(f"   ❌ Web interface failed: HTTP {response.status_code}")
    except Exception as e:
        results['tests']['web_ui'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"   ❌ Web interface failed: {str(e)}")
    
    # Calculate overall status
    passed_tests = sum(1 for test in results['tests'].values() if test['status'] == 'PASS')
    total_tests = len(results['tests'])
    
    if passed_tests == total_tests:
        results['overall_status'] = 'ALL_PASS'
        status_emoji = "🎉"
        status_text = "ALL TESTS PASSED"
    elif passed_tests >= total_tests * 0.75:
        results['overall_status'] = 'MOSTLY_PASS'
        status_emoji = "✅"
        status_text = "MOSTLY WORKING"
    else:
        results['overall_status'] = 'FAIL'
        status_emoji = "❌"
        status_text = "NEEDS ATTENTION"
    
    print("\n" + "=" * 60)
    print(f"{status_emoji} DEPLOYMENT TEST RESULTS: {status_text}")
    print(f"📊 Tests passed: {passed_tests}/{total_tests}")
    print(f"🌐 Base URL: {base_url}")
    
    if results['overall_status'] == 'ALL_PASS':
        print("\n🚀 Your F1 Performance Drop Predictor is live and working!")
        print(f"📱 Try the web interface: {base_url}/ui")
        print(f"📖 API documentation: {base_url}/docs")
    
    return results

def main():
    """Main function to run deployment tests."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_deployment.py <base_url>")
        print("Example: python scripts/test_deployment.py https://your-app.onrender.com")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    # Run tests
    results = test_deployment(base_url)
    
    # Save results to file
    with open('deployment_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: deployment_test_results.json")
    
    # Exit with appropriate code
    if results['overall_status'] == 'ALL_PASS':
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()