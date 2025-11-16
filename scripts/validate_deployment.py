#!/usr/bin/env python3
"""
Deployment Validation Script for F1 Performance Drop Predictor

This script performs comprehensive validation of the deployed container
including functionality, performance, and reliability tests.
"""

import requests
import time
import json
import subprocess
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import concurrent.futures
import statistics

class DeploymentValidator:
    """Comprehensive deployment validation for F1 Performance Drop Predictor."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
        # Test results storage
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': []
        }
        
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
    
    def log(self, level: str, message: str, test_name: str = ""):
        """Log test results with formatting."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "PASS": "\033[0;32m",
            "FAIL": "\033[0;31m", 
            "WARN": "\033[1;33m",
            "INFO": "\033[0;34m"
        }
        color = colors.get(level, "")
        reset = "\033[0m"
        
        formatted_message = f"{color}[{level}]{reset} {timestamp} - {message}"
        print(formatted_message)
        
        # Store result
        result_entry = {
            'timestamp': timestamp,
            'level': level,
            'test': test_name,
            'message': message
        }
        self.test_results['details'].append(result_entry)
        
        if level == "PASS":
            self.test_results['passed'] += 1
        elif level == "FAIL":
            self.test_results['failed'] += 1
        elif level == "WARN":
            self.test_results['warnings'] += 1
    
    def test_connectivity(self) -> bool:
        """Test basic connectivity to the API."""
        test_name = "Connectivity"
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.log("PASS", "API is accessible", test_name)
                return True
            else:
                self.log("FAIL", f"API returned status {response.status_code}", test_name)
                return False
        except requests.exceptions.ConnectionError:
            self.log("FAIL", "Cannot connect to API", test_name)
            return False
        except Exception as e:
            self.log("FAIL", f"Connection test failed: {str(e)}", test_name)
            return False
    
    def test_health_endpoint(self) -> bool:
        """Test health endpoint functionality."""
        test_name = "Health Endpoint"
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code != 200:
                self.log("FAIL", f"Health endpoint returned {response.status_code}", test_name)
                return False
            
            data = response.json()
            
            # Check required fields
            required_fields = ['status', 'timestamp', 'models_loaded']
            for field in required_fields:
                if field not in data:
                    self.log("FAIL", f"Missing field in health response: {field}", test_name)
                    return False
            
            # Check status
            if data['status'] != 'healthy':
                self.log("WARN", f"Health status is '{data['status']}', not 'healthy'", test_name)
            
            # Check models loaded
            if not data['models_loaded']:
                self.log("FAIL", "Models are not loaded", test_name)
                return False
            
            self.log("PASS", "Health endpoint working correctly", test_name)
            return True
            
        except Exception as e:
            self.log("FAIL", f"Health endpoint test failed: {str(e)}", test_name)
            return False
    
    def test_model_info_endpoint(self) -> bool:
        """Test model info endpoint."""
        test_name = "Model Info Endpoint"
        
        try:
            response = self.session.get(f"{self.base_url}/model_info")
            
            if response.status_code != 200:
                self.log("FAIL", f"Model info endpoint returned {response.status_code}", test_name)
                return False
            
            data = response.json()
            
            # Check status
            if data.get('status') != 'Models loaded successfully':
                self.log("WARN", f"Model status: {data.get('status')}", test_name)
            
            # Check model information
            if 'classification_model' not in data or 'regression_model' not in data:
                self.log("FAIL", "Missing model information", test_name)
                return False
            
            self.log("PASS", "Model info endpoint working correctly", test_name)
            return True
            
        except Exception as e:
            self.log("FAIL", f"Model info test failed: {str(e)}", test_name)
            return False
    
    def test_prediction_endpoint(self) -> bool:
        """Test single prediction endpoint."""
        test_name = "Prediction Endpoint"
        
        try:
            response = self.session.post(f"{self.base_url}/predict", 
                                       json=self.sample_features)
            
            if response.status_code != 200:
                self.log("FAIL", f"Prediction endpoint returned {response.status_code}", test_name)
                return False
            
            data = response.json()
            
            # Check required fields
            required_fields = ['classification', 'regression', 'feature_contributions', 
                             'model_info', 'prediction_timestamp']
            for field in required_fields:
                if field not in data:
                    self.log("FAIL", f"Missing field in prediction response: {field}", test_name)
                    return False
            
            # Check classification structure
            classification = data['classification']
            if 'will_drop_position' not in classification or 'probability' not in classification:
                self.log("FAIL", "Invalid classification structure", test_name)
                return False
            
            # Check regression structure
            regression = data['regression']
            if 'expected_position_change' not in regression:
                self.log("FAIL", "Invalid regression structure", test_name)
                return False
            
            # Validate probability range
            prob = classification['probability']
            if not (0 <= prob <= 1):
                self.log("FAIL", f"Probability out of range: {prob}", test_name)
                return False
            
            self.log("PASS", "Prediction endpoint working correctly", test_name)
            return True
            
        except Exception as e:
            self.log("FAIL", f"Prediction test failed: {str(e)}", test_name)
            return False
    
    def test_batch_prediction_endpoint(self) -> bool:
        """Test batch prediction endpoint."""
        test_name = "Batch Prediction Endpoint"
        
        try:
            batch_data = {
                "scenarios": [self.sample_features, self.sample_features]
            }
            
            response = self.session.post(f"{self.base_url}/predict_batch", 
                                       json=batch_data)
            
            if response.status_code != 200:
                self.log("FAIL", f"Batch prediction returned {response.status_code}", test_name)
                return False
            
            data = response.json()
            
            # Check structure
            if 'predictions' not in data or 'summary' not in data:
                self.log("FAIL", "Invalid batch prediction structure", test_name)
                return False
            
            # Check number of predictions
            if len(data['predictions']) != 2:
                self.log("FAIL", f"Expected 2 predictions, got {len(data['predictions'])}", test_name)
                return False
            
            self.log("PASS", "Batch prediction endpoint working correctly", test_name)
            return True
            
        except Exception as e:
            self.log("FAIL", f"Batch prediction test failed: {str(e)}", test_name)
            return False
    
    def test_error_handling(self) -> bool:
        """Test API error handling."""
        test_name = "Error Handling"
        
        error_scenarios = [
            {
                "name": "Invalid data",
                "data": {"championship_position": -1},
                "expected_status": [400, 422]
            },
            {
                "name": "Missing fields",
                "data": {"championship_position": 5},
                "expected_status": [400, 422]
            }
        ]
        
        all_passed = True
        
        for scenario in error_scenarios:
            try:
                response = self.session.post(f"{self.base_url}/predict", 
                                           json=scenario["data"])
                
                if response.status_code in scenario["expected_status"]:
                    self.log("PASS", f"Error handling for {scenario['name']}", test_name)
                else:
                    self.log("FAIL", f"Unexpected status for {scenario['name']}: {response.status_code}", test_name)
                    all_passed = False
                    
            except Exception as e:
                self.log("FAIL", f"Error handling test failed for {scenario['name']}: {str(e)}", test_name)
                all_passed = False
        
        return all_passed
    
    def test_performance(self, num_requests: int = 10) -> bool:
        """Test API performance."""
        test_name = "Performance"
        
        try:
            response_times = []
            
            for i in range(num_requests):
                start_time = time.time()
                
                response = self.session.post(f"{self.base_url}/predict", 
                                           json=self.sample_features)
                
                if response.status_code == 200:
                    response_time = (time.time() - start_time) * 1000
                    response_times.append(response_time)
            
            if not response_times:
                self.log("FAIL", "No successful requests in performance test", test_name)
                return False
            
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            self.log("INFO", f"Performance metrics: avg={avg_time:.1f}ms, median={median_time:.1f}ms, min={min_time:.1f}ms, max={max_time:.1f}ms", test_name)
            
            if avg_time < 1000:
                self.log("PASS", f"Performance test passed (avg: {avg_time:.1f}ms)", test_name)
                return True
            else:
                self.log("WARN", f"Performance may need optimization (avg: {avg_time:.1f}ms)", test_name)
                return True  # Still pass, just warn
                
        except Exception as e:
            self.log("FAIL", f"Performance test failed: {str(e)}", test_name)
            return False
    
    def test_concurrent_requests(self, num_concurrent: int = 5) -> bool:
        """Test concurrent request handling."""
        test_name = "Concurrent Requests"
        
        def make_request():
            try:
                response = self.session.post(f"{self.base_url}/predict", 
                                           json=self.sample_features)
                return response.status_code == 200
            except:
                return False
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(make_request) for _ in range(num_concurrent)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            successful = sum(results)
            
            if successful == num_concurrent:
                self.log("PASS", f"All {num_concurrent} concurrent requests succeeded", test_name)
                return True
            else:
                self.log("WARN", f"Only {successful}/{num_concurrent} concurrent requests succeeded", test_name)
                return successful > 0
                
        except Exception as e:
            self.log("FAIL", f"Concurrent request test failed: {str(e)}", test_name)
            return False
    
    def test_container_health(self) -> bool:
        """Test Docker container health if available."""
        test_name = "Container Health"
        
        try:
            # Try to get container info
            result = subprocess.run(['docker', 'ps', '--filter', 'publish=8000', '--format', '{{.Names}}'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.log("INFO", "Docker not available or container not found", test_name)
                return True  # Not a failure if Docker isn't available
            
            container_names = result.stdout.strip().split('\n')
            if not container_names or not container_names[0]:
                self.log("INFO", "No container found on port 8000", test_name)
                return True
            
            container_name = container_names[0]
            
            # Check container health
            health_result = subprocess.run(['docker', 'inspect', '--format', '{{.State.Health.Status}}', container_name], 
                                         capture_output=True, text=True, timeout=10)
            
            if health_result.returncode == 0:
                health_status = health_result.stdout.strip()
                if health_status == 'healthy':
                    self.log("PASS", f"Container {container_name} is healthy", test_name)
                    return True
                elif health_status == 'no healthcheck':
                    self.log("INFO", f"Container {container_name} has no health check configured", test_name)
                    return True
                else:
                    self.log("WARN", f"Container {container_name} health status: {health_status}", test_name)
                    return True
            else:
                self.log("INFO", "Could not check container health", test_name)
                return True
                
        except Exception as e:
            self.log("INFO", f"Container health check skipped: {str(e)}", test_name)
            return True  # Not a critical failure
    
    def test_data_validation(self) -> bool:
        """Test data validation and edge cases."""
        test_name = "Data Validation"
        
        edge_cases = [
            {
                "name": "Minimum values",
                "data": {**self.sample_features, "championship_position": 1, "points": 0, "estimated_age": 18}
            },
            {
                "name": "Maximum values", 
                "data": {**self.sample_features, "championship_position": 20, "points": 400, "estimated_age": 45}
            },
            {
                "name": "Zero pit stops",
                "data": {**self.sample_features, "pit_stop_count": 0, "pit_frequency": 0}
            }
        ]
        
        all_passed = True
        
        for case in edge_cases:
            try:
                response = self.session.post(f"{self.base_url}/predict", json=case["data"])
                
                if response.status_code == 200:
                    self.log("PASS", f"Edge case handled: {case['name']}", test_name)
                else:
                    self.log("WARN", f"Edge case failed: {case['name']} (status: {response.status_code})", test_name)
                    
            except Exception as e:
                self.log("FAIL", f"Edge case error: {case['name']} - {str(e)}", test_name)
                all_passed = False
        
        return all_passed
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        report_lines = [
            "F1 Performance Drop Predictor - Deployment Validation Report",
            "=" * 65,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"API URL: {self.base_url}",
            "",
            "SUMMARY",
            "-" * 20,
            f"Tests Passed: {self.test_results['passed']}",
            f"Tests Failed: {self.test_results['failed']}",
            f"Warnings: {self.test_results['warnings']}",
            f"Total Tests: {len(self.test_results['details'])}",
            ""
        ]
        
        # Overall status
        if self.test_results['failed'] == 0:
            if self.test_results['warnings'] == 0:
                report_lines.append("✅ OVERALL STATUS: ALL TESTS PASSED")
            else:
                report_lines.append("⚠️ OVERALL STATUS: PASSED WITH WARNINGS")
        else:
            report_lines.append("❌ OVERALL STATUS: SOME TESTS FAILED")
        
        report_lines.extend([
            "",
            "DETAILED RESULTS",
            "-" * 20
        ])
        
        # Group results by test
        test_groups = {}
        for detail in self.test_results['details']:
            test = detail['test'] or 'General'
            if test not in test_groups:
                test_groups[test] = []
            test_groups[test].append(detail)
        
        for test_name, details in test_groups.items():
            report_lines.append(f"\n{test_name}:")
            for detail in details:
                status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "INFO": "ℹ️"}.get(detail['level'], "")
                report_lines.append(f"  {status_icon} [{detail['level']}] {detail['message']}")
        
        return "\n".join(report_lines)
    
    def run_all_tests(self) -> bool:
        """Run all validation tests."""
        self.log("INFO", "Starting deployment validation tests")
        
        tests = [
            ("Connectivity", self.test_connectivity),
            ("Health Endpoint", self.test_health_endpoint),
            ("Model Info", self.test_model_info_endpoint),
            ("Prediction", self.test_prediction_endpoint),
            ("Batch Prediction", self.test_batch_prediction_endpoint),
            ("Error Handling", self.test_error_handling),
            ("Performance", lambda: self.test_performance(10)),
            ("Concurrent Requests", lambda: self.test_concurrent_requests(5)),
            ("Container Health", self.test_container_health),
            ("Data Validation", self.test_data_validation)
        ]
        
        for test_name, test_func in tests:
            self.log("INFO", f"Running {test_name} test...")
            try:
                test_func()
            except Exception as e:
                self.log("FAIL", f"Test {test_name} crashed: {str(e)}", test_name)
        
        # Generate and save report
        report = self.generate_report()
        
        # Save report to file
        report_filename = f"deployment_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(report_filename, 'w') as f:
                f.write(report)
            self.log("INFO", f"Report saved to {report_filename}")
        except Exception as e:
            self.log("WARN", f"Could not save report: {str(e)}")
        
        # Print report
        print("\n" + report)
        
        return self.test_results['failed'] == 0

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate F1 Performance Drop Predictor deployment")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL for the API")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    validator = DeploymentValidator(base_url=args.url, timeout=args.timeout)
    
    try:
        success = validator.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        validator.log("WARN", "Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        validator.log("FAIL", f"Validation failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()