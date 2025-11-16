#!/usr/bin/env python3
"""
Container Testing Script for F1 Performance Drop Predictor

This script provides comprehensive testing of the Docker container
including API functionality, performance, and error handling.
"""

import requests
import time
import json
import sys
import subprocess
import signal
from typing import Dict, Any, Optional
from datetime import datetime

class ContainerTester:
    """Test suite for F1 Performance Drop Predictor container."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 60):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = 10
        
        # Sample data for testing
        self.sample_features = {
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
    
    def log(self, level: str, message: str):
        """Log message with timestamp and level."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        colors = {
            "INFO": "\033[0;34m",
            "SUCCESS": "\033[0;32m", 
            "WARNING": "\033[1;33m",
            "ERROR": "\033[0;31m"
        }
        color = colors.get(level, "")
        reset = "\033[0m"
        print(f"{color}[{level}]{reset} {timestamp} - {message}")
    
    def wait_for_service(self) -> bool:
        """Wait for the service to become available."""
        self.log("INFO", f"Waiting for service at {self.base_url}...")
        
        for i in range(self.timeout):
            try:
                response = self.session.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    self.log("SUCCESS", "Service is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if i % 10 == 0 and i > 0:
                self.log("INFO", f"Still waiting... ({i}s elapsed)")
            
            time.sleep(1)
        
        self.log("ERROR", f"Service failed to start within {self.timeout} seconds")
        return False
    
    def test_endpoint(self, endpoint: str, method: str = "GET", 
                     data: Optional[Dict] = None, expected_status: int = 200) -> bool:
        """Test a specific API endpoint."""
        url = f"{self.base_url}{endpoint}"
        self.log("INFO", f"Testing {method} {endpoint}")
        
        try:
            if method == "POST":
                response = self.session.post(url, json=data)
            else:
                response = self.session.get(url)
            
            if response.status_code == expected_status:
                self.log("SUCCESS", f"{method} {endpoint} returned {response.status_code}")
                return True
            else:
                self.log("ERROR", f"{method} {endpoint} returned {response.status_code} "
                        f"(expected {expected_status})")
                self.log("ERROR", f"Response: {response.text[:200]}...")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log("ERROR", f"{method} {endpoint} failed: {str(e)}")
            return False
    
    def test_prediction_response(self) -> bool:
        """Test prediction response structure and content."""
        self.log("INFO", "Testing prediction response structure")
        
        try:
            response = self.session.post(f"{self.base_url}/predict", 
                                       json=self.sample_features)
            
            if response.status_code != 200:
                self.log("ERROR", f"Prediction failed with status {response.status_code}")
                return False
            
            data = response.json()
            
            # Check required fields
            required_fields = ["classification", "regression", "feature_contributions", 
                             "model_info", "prediction_timestamp"]
            
            for field in required_fields:
                if field not in data:
                    self.log("ERROR", f"Missing required field: {field}")
                    return False
            
            # Check classification structure
            if "will_drop_position" not in data["classification"]:
                self.log("ERROR", "Missing classification prediction")
                return False
            
            # Check regression structure
            if "expected_position_change" not in data["regression"]:
                self.log("ERROR", "Missing regression prediction")
                return False
            
            self.log("SUCCESS", "Prediction response structure is valid")
            return True
            
        except Exception as e:
            self.log("ERROR", f"Prediction response test failed: {str(e)}")
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction functionality."""
        self.log("INFO", "Testing batch prediction")
        
        batch_data = {
            "scenarios": [self.sample_features, self.sample_features]
        }
        
        try:
            response = self.session.post(f"{self.base_url}/predict_batch", 
                                       json=batch_data)
            
            if response.status_code != 200:
                self.log("ERROR", f"Batch prediction failed with status {response.status_code}")
                return False
            
            data = response.json()
            
            if len(data["predictions"]) != 2:
                self.log("ERROR", f"Expected 2 predictions, got {len(data['predictions'])}")
                return False
            
            self.log("SUCCESS", "Batch prediction test passed")
            return True
            
        except Exception as e:
            self.log("ERROR", f"Batch prediction test failed: {str(e)}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test API error handling with invalid data."""
        self.log("INFO", "Testing error handling")
        
        # Test with invalid data
        invalid_data = {"championship_position": -1}
        
        try:
            response = self.session.post(f"{self.base_url}/predict", 
                                       json=invalid_data)
            
            if response.status_code in [400, 422]:
                self.log("SUCCESS", "Error handling test passed")
                return True
            else:
                self.log("ERROR", f"Expected error status, got {response.status_code}")
                return False
                
        except Exception as e:
            self.log("ERROR", f"Error handling test failed: {str(e)}")
            return False
    
    def test_performance(self, num_requests: int = 5) -> bool:
        """Test API performance with multiple requests."""
        self.log("INFO", f"Testing performance with {num_requests} requests")
        
        response_times = []
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                response = self.session.post(f"{self.base_url}/predict", 
                                           json=self.sample_features)
                
                if response.status_code == 200:
                    response_time = (time.time() - start_time) * 1000
                    response_times.append(response_time)
                else:
                    self.log("WARNING", f"Request {i+1} failed with status {response.status_code}")
                    
            except Exception as e:
                self.log("WARNING", f"Request {i+1} failed: {str(e)}")
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            self.log("INFO", f"Performance results:")
            self.log("INFO", f"  Average: {avg_time:.1f}ms")
            self.log("INFO", f"  Min: {min_time:.1f}ms")
            self.log("INFO", f"  Max: {max_time:.1f}ms")
            
            if avg_time < 1000:
                self.log("SUCCESS", "Performance test passed (< 1000ms average)")
                return True
            else:
                self.log("WARNING", f"Performance test: average {avg_time:.1f}ms (> 1000ms)")
                return True  # Still pass, just warn
        else:
            self.log("ERROR", "No successful requests in performance test")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all container tests."""
        self.log("INFO", "Starting F1 Performance Drop Predictor container tests")
        
        tests = [
            ("Service availability", self.wait_for_service),
            ("Root endpoint", lambda: self.test_endpoint("/")),
            ("Health endpoint", lambda: self.test_endpoint("/health")),
            ("Model info endpoint", lambda: self.test_endpoint("/model_info")),
            ("Prediction endpoint", lambda: self.test_endpoint("/predict", "POST", self.sample_features)),
            ("Prediction response structure", self.test_prediction_response),
            ("Batch prediction", self.test_batch_prediction),
            ("Error handling", self.test_error_handling),
            ("Performance", self.test_performance)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.log("INFO", f"Running test: {test_name}")
            
            try:
                if test_func():
                    passed += 1
                else:
                    self.log("ERROR", f"Test failed: {test_name}")
            except Exception as e:
                self.log("ERROR", f"Test error in {test_name}: {str(e)}")
        
        # Summary
        self.log("INFO", f"Test results: {passed}/{total} tests passed")
        
        if passed == total:
            self.log("SUCCESS", "All container tests passed!")
            return True
        else:
            self.log("ERROR", f"{total - passed} tests failed")
            return False

def main():
    """Main function to run container tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test F1 Performance Drop Predictor container")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL for the API (default: http://localhost:8000)")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Timeout for service startup (default: 60 seconds)")
    
    args = parser.parse_args()
    
    tester = ContainerTester(base_url=args.url, timeout=args.timeout)
    
    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        tester.log("WARNING", "Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        tester.log("ERROR", f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()