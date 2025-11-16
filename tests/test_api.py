#!/usr/bin/env python3
"""
Unit tests for F1 Performance Drop Predictor API.

This module provides comprehensive tests for:
- API endpoints with valid and invalid inputs
- Error handling and validation
- Model loading and prediction functionality
- Batch processing capabilities
"""

import pytest
import json
import os
import sys
from typing import Dict, Any
from fastapi.testclient import TestClient

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from serve import app

# Create test client
client = TestClient(app)

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

class TestHealthEndpoint:
    """Test cases for health check endpoint."""
    
    def test_health_check_success(self):
        """Test successful health check."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "models_loaded" in data
        assert "uptime_seconds" in data
        assert "request_count" in data
        assert "error_count" in data
        assert "error_rate" in data
        
        # Check data types
        assert isinstance(data["models_loaded"], bool)
        assert isinstance(data["uptime_seconds"], (int, float))
        assert isinstance(data["request_count"], int)
        assert isinstance(data["error_count"], int)
        assert isinstance(data["error_rate"], (int, float))

class TestModelInfoEndpoint:
    """Test cases for model info endpoint."""
    
    def test_model_info_success(self):
        """Test successful model info retrieval."""
        response = client.get("/model_info")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        
        # If models are loaded, check structure
        if data["status"] == "Models loaded":
            assert "classification_model" in data
            assert "regression_model" in data
            assert "feature_info" in data
            
            # Check classification model info
            clf_model = data["classification_model"]
            assert "id" in clf_model
            assert "name" in clf_model
            assert "type" in clf_model
            assert "performance" in clf_model
            
            # Check regression model info
            reg_model = data["regression_model"]
            assert "id" in reg_model
            assert "name" in reg_model
            assert "type" in reg_model
            assert "performance" in reg_model
            
            # Check feature info
            feature_info = data["feature_info"]
            assert "feature_names" in feature_info
            assert "n_features" in feature_info
            assert isinstance(feature_info["feature_names"], list)
            assert isinstance(feature_info["n_features"], int)

class TestRootEndpoint:
    """Test cases for root endpoint."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data
        assert "timestamp" in data
        
        # Check endpoints structure
        endpoints = data["endpoints"]
        expected_endpoints = ["health", "model_info", "predict", "predict_batch", "docs", "redoc"]
        for endpoint in expected_endpoints:
            assert endpoint in endpoints

class TestPredictionEndpoint:
    """Test cases for single prediction endpoint."""
    
    def test_predict_success(self):
        """Test successful prediction with valid input."""
        response = client.post("/predict", json=VALID_FEATURES)
        
        # Skip test if models not loaded
        if response.status_code == 503:
            pytest.skip("Models not loaded")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "classification" in data
        assert "regression" in data
        assert "feature_contributions" in data
        assert "model_info" in data
        assert "prediction_timestamp" in data
        
        # Check classification results
        classification = data["classification"]
        assert "will_drop_position" in classification
        assert "probability" in classification
        assert "confidence" in classification
        assert isinstance(classification["will_drop_position"], bool)
        assert 0 <= classification["probability"] <= 1
        assert classification["confidence"] in ["low", "medium", "high"]
        
        # Check regression results
        regression = data["regression"]
        assert "expected_position_change" in regression
        assert "prediction_interval" in regression
        assert isinstance(regression["expected_position_change"], (int, float))
        assert isinstance(regression["prediction_interval"], list)
        assert len(regression["prediction_interval"]) == 2
    
    def test_predict_missing_field(self):
        """Test prediction with missing required field."""
        invalid_features = VALID_FEATURES.copy()
        del invalid_features["championship_position"]
        
        response = client.post("/predict", json=invalid_features)
        assert response.status_code == 422  # Pydantic validation error
    
    def test_predict_invalid_range(self):
        """Test prediction with out-of-range values."""
        invalid_features = VALID_FEATURES.copy()
        invalid_features["championship_position"] = 50  # Invalid: > 30
        
        response = client.post("/predict", json=invalid_features)
        assert response.status_code == 422  # Pydantic validation error
    
    def test_predict_invalid_type(self):
        """Test prediction with invalid data types."""
        invalid_features = VALID_FEATURES.copy()
        invalid_features["championship_position"] = "invalid"  # Should be numeric
        
        response = client.post("/predict", json=invalid_features)
        assert response.status_code == 422  # Pydantic validation error
    
    def test_predict_negative_values(self):
        """Test prediction with negative values where inappropriate."""
        invalid_features = VALID_FEATURES.copy()
        invalid_features["pit_stop_count"] = -1  # Invalid: < 0
        
        response = client.post("/predict", json=invalid_features)
        assert response.status_code == 422  # Pydantic validation error

class TestBatchPredictionEndpoint:
    """Test cases for batch prediction endpoint."""
    
    def test_batch_predict_success(self):
        """Test successful batch prediction with valid inputs."""
        # Create multiple scenarios
        scenario1 = VALID_FEATURES.copy()
        scenario2 = VALID_FEATURES.copy()
        scenario2["championship_position"] = 10
        scenario2["points"] = 50
        
        batch_request = {
            "scenarios": [scenario1, scenario2]
        }
        
        response = client.post("/predict_batch", json=batch_request)
        
        # Skip test if models not loaded
        if response.status_code == 503:
            pytest.skip("Models not loaded")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "errors" in data
        assert "summary" in data
        
        # Check predictions
        predictions = data["predictions"]
        assert len(predictions) <= 2  # Should have up to 2 predictions
        
        for prediction in predictions:
            assert "classification" in prediction
            assert "regression" in prediction
            assert "feature_contributions" in prediction
        
        # Check summary
        summary = data["summary"]
        assert "total_requests" in summary
        assert "successful_predictions" in summary
        assert "failed_predictions" in summary
        assert "success_rate" in summary
        assert summary["total_requests"] == 2
    
    def test_batch_predict_empty_scenarios(self):
        """Test batch prediction with empty scenarios list."""
        batch_request = {"scenarios": []}
        
        response = client.post("/predict_batch", json=batch_request)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_too_many_scenarios(self):
        """Test batch prediction with too many scenarios."""
        # Create 101 scenarios (exceeds limit of 100)
        scenarios = [VALID_FEATURES.copy() for _ in range(101)]
        batch_request = {"scenarios": scenarios}
        
        response = client.post("/predict_batch", json=batch_request)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_mixed_valid_invalid(self):
        """Test batch prediction with mix of valid and invalid scenarios."""
        valid_scenario = VALID_FEATURES.copy()
        invalid_scenario = VALID_FEATURES.copy()
        invalid_scenario["championship_position"] = 50  # Invalid
        
        batch_request = {
            "scenarios": [valid_scenario, invalid_scenario]
        }
        
        response = client.post("/predict_batch", json=batch_request)
        # Should fail validation before reaching prediction logic
        assert response.status_code == 422

class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_invalid_endpoint(self):
        """Test request to non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_method(self):
        """Test invalid HTTP method on existing endpoint."""
        response = client.delete("/predict")
        assert response.status_code == 405  # Method not allowed
    
    def test_malformed_json(self):
        """Test request with malformed JSON."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

class TestResponseHeaders:
    """Test cases for response headers."""
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check for CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers
    
    def test_request_id_header(self):
        """Test request ID header is added."""
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check for request ID header
        headers = response.headers
        assert "x-request-id" in headers
        assert len(headers["x-request-id"]) == 8  # UUID first 8 chars

class TestDocumentationEndpoints:
    """Test cases for API documentation endpoints."""
    
    def test_openapi_docs(self):
        """Test OpenAPI documentation endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_docs(self):
        """Test ReDoc documentation endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_openapi_json(self):
        """Test OpenAPI JSON schema endpoint."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Validate it's valid JSON
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

# Integration tests
class TestIntegration:
    """Integration test cases."""
    
    def test_full_prediction_workflow(self):
        """Test complete prediction workflow."""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Get model info
        info_response = client.get("/model_info")
        assert info_response.status_code == 200
        
        # 3. Make prediction (if models loaded)
        if info_response.json().get("status") == "Models loaded":
            pred_response = client.post("/predict", json=VALID_FEATURES)
            assert pred_response.status_code == 200
            
            # Verify prediction structure
            pred_data = pred_response.json()
            assert "classification" in pred_data
            assert "regression" in pred_data
    
    def test_error_response_format(self):
        """Test error responses follow consistent format."""
        # Make invalid request
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422
        
        # Check error response structure
        error_data = response.json()
        assert "detail" in error_data  # FastAPI default error format

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])