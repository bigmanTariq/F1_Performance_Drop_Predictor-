#!/usr/bin/env python3
"""
Python API Examples for F1 Performance Drop Predictor

This script demonstrates how to interact with the F1 Performance Drop Predictor API
using Python requests library. It includes examples for all endpoints and various
scenarios.
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

class F1PredictorClient:
    """Client for F1 Performance Drop Predictor API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = timeout
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status information
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Model information including performance metrics
        """
        response = self.session.get(f"{self.base_url}/model_info")
        response.raise_for_status()
        return response.json()
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            features: Race features dictionary
            
        Returns:
            Prediction results
        """
        response = self.session.post(f"{self.base_url}/predict", json=features)
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Make batch predictions.
        
        Args:
            scenarios: List of race feature dictionaries
            
        Returns:
            Batch prediction results
        """
        data = {"scenarios": scenarios}
        response = self.session.post(f"{self.base_url}/predict_batch", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get basic API information.
        
        Returns:
            API information
        """
        response = self.session.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()

def create_sample_features() -> Dict[str, float]:
    """Create sample race features for testing."""
    return {
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

def create_high_stress_scenario() -> Dict[str, float]:
    """Create a high-stress race scenario."""
    return {
        "championship_position": 1,
        "pit_stop_count": 4,
        "avg_pit_time": 35.2,
        "pit_time_std": 8.5,
        "circuit_length": 3.2,
        "points": 250,
        "pit_frequency": 3.5,
        "pit_duration_variance": 25.0,
        "high_pit_frequency": 1,
        "qualifying_gap_to_pole": 0.0,
        "grid_position_percentile": 1.0,
        "poor_qualifying": 0,
        "circuit_dnf_rate": 0.25,
        "is_street_circuit": 1,
        "championship_pressure": 0.9,
        "leader_points": 250,
        "points_gap_to_leader": 0,
        "points_pressure": 0.8,
        "driver_avg_grid_position": 3.2,
        "qualifying_vs_average": -3.2,
        "constructor_avg_grid_position": 2.8,
        "qualifying_vs_constructor_avg": -2.8,
        "bad_qualifying_day": 0,
        "circuit_dnf_rate_detailed": 0.25,
        "avg_pit_stops": 3.8,
        "avg_pit_duration": 34.5,
        "dnf_score": 0.3,
        "volatility_score": 0.7,
        "pit_complexity_score": 0.8,
        "track_difficulty_score": 0.9,
        "race_number": 20,
        "first_season": 2010,
        "seasons_active": 13,
        "estimated_age": 32,
        "driver_rolling_avg_grid": 2.8,
        "season_leader_points": 250,
        "points_gap_to_season_leader": 0,
        "is_championship_contender": 1,
        "points_momentum": 0.8,
        "championship_position_change": 0,
        "teammate_avg_grid": 4.1,
        "grid_vs_teammate": -4.1,
        "championship_pressure_score": 0.9,
        "max_round": 22,
        "late_season_race": 1,
        "championship_pressure_adjusted": 0.95,
        "points_per_race": 12.5
    }

def create_low_stress_scenario() -> Dict[str, float]:
    """Create a low-stress race scenario."""
    return {
        "championship_position": 15,
        "pit_stop_count": 1,
        "avg_pit_time": 22.8,
        "pit_time_std": 1.2,
        "circuit_length": 6.8,
        "points": 5,
        "pit_frequency": 1.0,
        "pit_duration_variance": 2.5,
        "high_pit_frequency": 0,
        "qualifying_gap_to_pole": 3.5,
        "grid_position_percentile": 0.25,
        "poor_qualifying": 1,
        "circuit_dnf_rate": 0.08,
        "is_street_circuit": 0,
        "championship_pressure": 0.1,
        "leader_points": 200,
        "points_gap_to_leader": 195,
        "points_pressure": 0.05,
        "driver_avg_grid_position": 16.2,
        "qualifying_vs_average": 1.8,
        "constructor_avg_grid_position": 15.8,
        "qualifying_vs_constructor_avg": 2.2,
        "bad_qualifying_day": 1,
        "circuit_dnf_rate_detailed": 0.08,
        "avg_pit_stops": 1.2,
        "avg_pit_duration": 23.5,
        "dnf_score": 0.05,
        "volatility_score": 0.2,
        "pit_complexity_score": 0.2,
        "track_difficulty_score": 0.3,
        "race_number": 5,
        "first_season": 2020,
        "seasons_active": 3,
        "estimated_age": 24,
        "driver_rolling_avg_grid": 16.8,
        "season_leader_points": 200,
        "points_gap_to_season_leader": 195,
        "is_championship_contender": 0,
        "points_momentum": -0.1,
        "championship_position_change": 2,
        "teammate_avg_grid": 14.5,
        "grid_vs_teammate": 3.5,
        "championship_pressure_score": 0.1,
        "max_round": 22,
        "late_season_race": 0,
        "championship_pressure_adjusted": 0.05,
        "points_per_race": 1.0
    }

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def print_json(data: Dict[str, Any], title: str = "Response"):
    """Print JSON data in a formatted way."""
    print(f"\n{title}:")
    print(json.dumps(data, indent=2))

def example_health_check(client: F1PredictorClient):
    """Example: Health check."""
    print_section("1. Health Check")
    
    try:
        health = client.health_check()
        print_json(health, "Health Status")
        
        if health.get('status') == 'healthy':
            print("✅ API is healthy and ready!")
        else:
            print("⚠️ API may have issues")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def example_model_info(client: F1PredictorClient):
    """Example: Get model information."""
    print_section("2. Model Information")
    
    try:
        model_info = client.get_model_info()
        print_json(model_info, "Model Information")
        
        if model_info.get('status') == 'Models loaded successfully':
            print("✅ Models are loaded and ready!")
        else:
            print("⚠️ Models may not be loaded properly")
            
    except Exception as e:
        print(f"❌ Model info request failed: {e}")

def example_single_prediction(client: F1PredictorClient):
    """Example: Single prediction."""
    print_section("3. Single Prediction - Typical Scenario")
    
    try:
        features = create_sample_features()
        print("Input features:")
        print(json.dumps(features, indent=2))
        
        result = client.predict_single(features)
        print_json(result, "Prediction Result")
        
        # Extract key results
        classification = result.get('classification', {})
        regression = result.get('regression', {})
        
        print(f"\n📊 Prediction Summary:")
        print(f"   Will drop position: {classification.get('will_drop_position', 'Unknown')}")
        print(f"   Probability: {classification.get('probability', 0):.3f}")
        print(f"   Expected position change: {regression.get('expected_position_change', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")

def example_stress_scenarios(client: F1PredictorClient):
    """Example: Compare high vs low stress scenarios."""
    print_section("4. Stress Scenario Comparison")
    
    scenarios = {
        "High Stress (Championship Leader, Street Circuit)": create_high_stress_scenario(),
        "Low Stress (Backmarker, Easy Circuit)": create_low_stress_scenario()
    }
    
    for scenario_name, features in scenarios.items():
        print(f"\n--- {scenario_name} ---")
        
        try:
            result = client.predict_single(features)
            classification = result.get('classification', {})
            regression = result.get('regression', {})
            
            print(f"Will drop position: {classification.get('will_drop_position', 'Unknown')}")
            print(f"Probability: {classification.get('probability', 0):.3f}")
            print(f"Expected position change: {regression.get('expected_position_change', 0):.2f}")
            
        except Exception as e:
            print(f"❌ Prediction failed for {scenario_name}: {e}")

def example_batch_prediction(client: F1PredictorClient):
    """Example: Batch prediction."""
    print_section("5. Batch Prediction")
    
    try:
        scenarios = [
            create_sample_features(),
            create_high_stress_scenario(),
            create_low_stress_scenario()
        ]
        
        print(f"Predicting {len(scenarios)} scenarios...")
        
        result = client.predict_batch(scenarios)
        print_json(result, "Batch Prediction Result")
        
        # Summarize results
        predictions = result.get('predictions', [])
        print(f"\n📊 Batch Summary:")
        print(f"   Total scenarios: {len(predictions)}")
        
        for i, pred in enumerate(predictions):
            classification = pred.get('classification', {})
            print(f"   Scenario {i+1}: {classification.get('probability', 0):.3f} drop probability")
            
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")

def example_error_handling(client: F1PredictorClient):
    """Example: Error handling with invalid data."""
    print_section("6. Error Handling")
    
    invalid_scenarios = [
        {
            "name": "Invalid championship position",
            "data": {"championship_position": -1, "pit_stop_count": 2}
        },
        {
            "name": "Missing required fields",
            "data": {"championship_position": 5}
        },
        {
            "name": "Out of range values",
            "data": {
                "championship_position": 50,
                "pit_stop_count": 20,
                "avg_pit_time": 300.0,
                "estimated_age": 100
            }
        }
    ]
    
    for scenario in invalid_scenarios:
        print(f"\n--- Testing: {scenario['name']} ---")
        
        try:
            result = client.predict_single(scenario['data'])
            print("⚠️ Unexpected success - validation may be too lenient")
            print_json(result)
            
        except requests.exceptions.HTTPError as e:
            print(f"✅ Expected error caught: {e.response.status_code}")
            try:
                error_detail = e.response.json()
                print(f"   Error message: {error_detail.get('message', 'No message')}")
            except:
                print(f"   Raw error: {e.response.text}")
                
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def example_performance_test(client: F1PredictorClient):
    """Example: Performance testing."""
    print_section("7. Performance Testing")
    
    features = create_sample_features()
    num_requests = 10
    
    print(f"Testing response time with {num_requests} requests...")
    
    response_times = []
    
    for i in range(num_requests):
        start_time = time.time()
        
        try:
            result = client.predict_single(features)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            response_times.append(response_time)
            
            if (i + 1) % 5 == 0:
                print(f"   Completed {i + 1}/{num_requests} requests")
                
        except Exception as e:
            print(f"   Request {i + 1} failed: {e}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\n📊 Performance Results:")
        print(f"   Successful requests: {len(response_times)}/{num_requests}")
        print(f"   Average response time: {avg_time:.1f}ms")
        print(f"   Min response time: {min_time:.1f}ms")
        print(f"   Max response time: {max_time:.1f}ms")
        
        if avg_time < 1000:
            print("   ✅ Performance is good (< 1000ms average)")
        else:
            print("   ⚠️ Performance may need optimization (> 1000ms average)")
    else:
        print("   ❌ No successful requests")

def example_feature_analysis(client: F1PredictorClient):
    """Example: Analyze feature importance."""
    print_section("8. Feature Importance Analysis")
    
    try:
        features = create_sample_features()
        result = client.predict_single(features)
        
        feature_contributions = result.get('feature_contributions', {})
        
        if feature_contributions:
            print("Top 10 most important features for this prediction:")
            
            # Sort features by absolute contribution
            sorted_features = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for i, (feature, contribution) in enumerate(sorted_features[:10]):
                direction = "↑" if contribution > 0 else "↓"
                print(f"   {i+1:2d}. {feature:<30} {direction} {contribution:+.4f}")
        else:
            print("   No feature contributions available")
            
    except Exception as e:
        print(f"❌ Feature analysis failed: {e}")

def main():
    """Main function to run all examples."""
    print("F1 Performance Drop Predictor - Python API Examples")
    print("=" * 60)
    
    # Initialize client
    try:
        client = F1PredictorClient()
        print(f"Connecting to API at: {client.base_url}")
        
        # Run examples
        example_health_check(client)
        example_model_info(client)
        example_single_prediction(client)
        example_stress_scenarios(client)
        example_batch_prediction(client)
        example_error_handling(client)
        example_performance_test(client)
        example_feature_analysis(client)
        
        print_section("Examples Complete!")
        print("✅ All examples completed successfully!")
        print(f"\nFor interactive API documentation, visit: {client.base_url}/docs")
        print(f"For alternative documentation, visit: {client.base_url}/redoc")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the service is running.")
        print("   Start the service with: python src/serve.py")
        print("   Or with Docker: docker run -p 8000:8000 f1-predictor")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()