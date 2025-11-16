# F1 Performance Drop Predictor API Examples

This document provides comprehensive examples of how to use the F1 Performance Drop Predictor API.

## Base URL

When running locally: `http://localhost:8000`

## Authentication

Currently, no authentication is required for the API endpoints.

## Endpoints Overview

- `GET /` - API information
- `GET /health` - Health check
- `GET /model_info` - Model information
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Example Requests and Responses

### 1. Health Check

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-11-16T22:30:00.000Z",
  "models_loaded": true,
  "uptime_seconds": 3600.5,
  "request_count": 42,
  "error_count": 2,
  "error_rate": 4.76,
  "memory_usage": {
    "rss_mb": 256.7,
    "vms_mb": 512.3,
    "percent": 12.5
  }
}
```

### 2. Model Information

**Request:**
```bash
curl -X GET "http://localhost:8000/model_info"
```

**Response:**
```json
{
  "status": "Models loaded",
  "classification_model": {
    "id": "random_forest_classifier_classification_20231116_225020_f58cd65c",
    "name": "random_forest_classifier",
    "type": "RandomForestClassifier",
    "performance": {
      "accuracy": 0.7751,
      "precision": 0.7703,
      "recall": 0.7751,
      "f1": 0.7577,
      "roc_auc": 0.8153
    },
    "overfitting_level": "minimal"
  },
  "regression_model": {
    "id": "random_forest_regressor_regression_20231116_225052_f3139a5a",
    "name": "random_forest_regressor",
    "type": "RandomForestRegressor",
    "performance": {
      "mae": 1.9765,
      "mse": 7.3236,
      "rmse": 2.7062,
      "r2": 0.5360
    },
    "overfitting_level": "minimal"
  },
  "feature_info": {
    "feature_names": [
      "championship_position",
      "pit_stop_count",
      "avg_pit_time",
      "..."
    ],
    "n_features": 47,
    "requires_scaling": true
  }
}
```

### 3. Single Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "classification": {
    "will_drop_position": false,
    "probability": 0.334,
    "confidence": "medium"
  },
  "regression": {
    "expected_position_change": 0.61,
    "prediction_interval": [-3.31, 4.53]
  },
  "feature_contributions": {
    "championship_position": 0.12,
    "qualifying_gap_to_pole": 0.08,
    "pit_frequency": 0.06,
    "driver_avg_grid_position": 0.05,
    "..."
  },
  "model_info": {
    "classification_model": "random_forest_classifier_classification_20231116_225020_f58cd65c",
    "regression_model": "random_forest_regressor_regression_20231116_225052_f3139a5a",
    "prediction_timestamp": "2023-11-16T22:35:00.000Z"
  },
  "prediction_timestamp": "2023-11-16T22:35:00.000Z"
}
```

### 4. Batch Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "scenarios": [
      {
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
      },
      {
        "championship_position": 10,
        "pit_stop_count": 3,
        "avg_pit_time": 32.1,
        "pit_time_std": 4.8,
        "circuit_length": 4.2,
        "points": 45,
        "pit_frequency": 2.3,
        "pit_duration_variance": 15.7,
        "high_pit_frequency": 1,
        "qualifying_gap_to_pole": 1.5,
        "grid_position_percentile": 0.45,
        "poor_qualifying": 1,
        "circuit_dnf_rate": 0.22,
        "is_street_circuit": 1,
        "championship_pressure": 0.3,
        "leader_points": 200,
        "points_gap_to_leader": 155,
        "points_pressure": 0.8,
        "driver_avg_grid_position": 12.3,
        "qualifying_vs_average": 2.2,
        "constructor_avg_grid_position": 11.8,
        "qualifying_vs_constructor_avg": 1.7,
        "bad_qualifying_day": 1,
        "circuit_dnf_rate_detailed": 0.22,
        "avg_pit_stops": 2.8,
        "avg_pit_duration": 31.5,
        "dnf_score": 0.3,
        "volatility_score": 0.7,
        "pit_complexity_score": 0.8,
        "track_difficulty_score": 0.9,
        "race_number": 15,
        "first_season": 2020,
        "seasons_active": 3,
        "estimated_age": 24,
        "driver_rolling_avg_grid": 13.1,
        "season_leader_points": 200,
        "points_gap_to_season_leader": 155,
        "is_championship_contender": 0,
        "points_momentum": -0.1,
        "championship_position_change": 2,
        "teammate_avg_grid": 14.2,
        "grid_vs_teammate": -1.9,
        "championship_pressure_score": 0.3,
        "max_round": 22,
        "late_season_race": 1,
        "championship_pressure_adjusted": 0.2,
        "points_per_race": 3.0
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "classification": {
        "will_drop_position": false,
        "probability": 0.334,
        "confidence": "medium"
      },
      "regression": {
        "expected_position_change": 0.61,
        "prediction_interval": [-3.31, 4.53]
      },
      "feature_contributions": {
        "championship_position": 0.12,
        "qualifying_gap_to_pole": 0.08,
        "..."
      },
      "model_info": {
        "classification_model": "random_forest_classifier_classification_20231116_225020_f58cd65c",
        "regression_model": "random_forest_regressor_regression_20231116_225052_f3139a5a",
        "prediction_timestamp": "2023-11-16T22:40:00.000Z"
      },
      "prediction_timestamp": "2023-11-16T22:40:00.000Z"
    },
    {
      "classification": {
        "will_drop_position": true,
        "probability": 0.782,
        "confidence": "high"
      },
      "regression": {
        "expected_position_change": 3.24,
        "prediction_interval": [-0.68, 7.16]
      },
      "feature_contributions": {
        "pit_complexity_score": 0.15,
        "track_difficulty_score": 0.13,
        "..."
      },
      "model_info": {
        "classification_model": "random_forest_classifier_classification_20231116_225020_f58cd65c",
        "regression_model": "random_forest_regressor_regression_20231116_225052_f3139a5a",
        "prediction_timestamp": "2023-11-16T22:40:00.000Z"
      },
      "prediction_timestamp": "2023-11-16T22:40:00.000Z"
    }
  ],
  "errors": [],
  "summary": {
    "total_requests": 2,
    "successful_predictions": 2,
    "failed_predictions": 0,
    "success_rate": 1.0
  }
}
```

## Error Responses

### Validation Error

**Request with invalid data:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "championship_position": 50,
    "pit_stop_count": "invalid"
  }'
```

**Response:**
```json
{
  "detail": [
    {
      "loc": ["championship_position"],
      "msg": "ensure this value is less than or equal to 30",
      "type": "value_error.number.not_le",
      "ctx": {"limit_value": 30}
    },
    {
      "loc": ["pit_stop_count"],
      "msg": "value is not a valid float",
      "type": "type_error.float"
    }
  ]
}
```

### Model Not Loaded Error

**Response when models are not loaded:**
```json
{
  "error": "ModelNotLoadedError",
  "message": "Machine learning models are not loaded. Service unavailable.",
  "details": {
    "suggestion": "Wait for models to load or contact administrator",
    "path": "http://localhost:8000/predict"
  },
  "timestamp": "2023-11-16T22:45:00.000Z",
  "request_id": "abc12345"
}
```

## Python Client Examples

### Using requests library

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# Example features
features = {
    "championship_position": 3,
    "pit_stop_count": 2,
    "avg_pit_time": 28.5,
    # ... (include all required features)
}

# Make prediction
response = requests.post(f"{BASE_URL}/predict", json=features)

if response.status_code == 200:
    result = response.json()
    print(f"Position drop probability: {result['classification']['probability']:.3f}")
    print(f"Expected position change: {result['regression']['expected_position_change']:.2f}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Using httpx (async)

```python
import httpx
import asyncio

async def make_prediction():
    async with httpx.AsyncClient() as client:
        features = {
            "championship_position": 3,
            "pit_stop_count": 2,
            # ... (include all required features)
        }
        
        response = await client.post("http://localhost:8000/predict", json=features)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            raise Exception(f"Prediction failed: {response.text}")

# Run async function
result = asyncio.run(make_prediction())
print(result)
```

## Feature Descriptions

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| championship_position | float | 1-30 | Driver's current championship position |
| pit_stop_count | float | 0-10 | Number of pit stops in the race |
| avg_pit_time | float | 15-120 | Average pit stop time in seconds |
| pit_time_std | float | 0-50 | Standard deviation of pit times |
| circuit_length | float | 2-8 | Circuit length in kilometers |
| points | float | 0-500 | Driver's championship points |
| qualifying_gap_to_pole | float | 0-10 | Gap to pole position in seconds |
| grid_position_percentile | float | 0-1 | Grid position as percentile |
| circuit_dnf_rate | float | 0-1 | Historical DNF rate at this circuit |
| estimated_age | float | 18-50 | Driver's estimated age |
| seasons_active | float | 1-30 | Number of seasons the driver has been active |
| race_number | float | 1-25 | Race number in the season |

## Rate Limits

Currently, there are no rate limits implemented. In production, consider implementing:
- 100 requests per minute for single predictions
- 10 requests per minute for batch predictions
- Higher limits for authenticated users

## Monitoring and Logging

The API automatically logs:
- All requests with unique request IDs
- Response times and status codes
- Error details and stack traces
- Model performance metrics

Check the `/health` endpoint for service metrics and status information.