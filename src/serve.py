#!/usr/bin/env python3
"""
FastAPI web service for F1 performance drop prediction.

This module provides:
- REST API endpoints for predictions
- Health checks and model information
- Request validation and error handling
- Automatic API documentation
"""

import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import time
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator, ConfigDict
import traceback

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import F1PerformancePredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lifespan manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown (if needed)
    pass

# Initialize FastAPI app
app = FastAPI(
    title="F1 Performance Drop Predictor API",
    description="Machine learning API for predicting Formula 1 finishing position drops",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and responses."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Log request
    logger.info(f"[{request_id}] {request.method} {request.url} - Request started")
    
    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(f"[{request_id}] {request.method} {request.url} - "
                   f"Status: {response.status_code}, Time: {process_time:.3f}s")
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[{request_id}] {request.method} {request.url} - "
                    f"Error: {str(e)}, Time: {process_time:.3f}s")
        raise

# Global predictor instance
predictor = None

# Pydantic models for request/response validation
class RaceFeatures(BaseModel):
    """Input features for F1 race prediction."""
    
    championship_position: float = Field(..., ge=1, le=30, description="Driver's championship position")
    pit_stop_count: float = Field(..., ge=0, le=10, description="Number of pit stops")
    avg_pit_time: float = Field(..., ge=15.0, le=120.0, description="Average pit stop time in seconds")
    pit_time_std: float = Field(..., ge=0.0, le=50.0, description="Standard deviation of pit times")
    circuit_length: float = Field(..., ge=2.0, le=8.0, description="Circuit length in km")
    points: float = Field(..., ge=0, le=500, description="Driver's championship points")
    pit_frequency: float = Field(..., ge=0.0, le=5.0, description="Pit stop frequency")
    pit_duration_variance: float = Field(..., ge=0.0, description="Variance in pit durations")
    high_pit_frequency: float = Field(..., ge=0, le=1, description="High pit frequency flag")
    qualifying_gap_to_pole: float = Field(..., ge=0.0, le=10.0, description="Gap to pole in seconds")
    grid_position_percentile: float = Field(..., ge=0.0, le=1.0, description="Grid position percentile")
    poor_qualifying: float = Field(..., ge=0, le=1, description="Poor qualifying flag")
    circuit_dnf_rate: float = Field(..., ge=0.0, le=1.0, description="Circuit DNF rate")
    is_street_circuit: float = Field(..., ge=0, le=1, description="Street circuit flag")
    championship_pressure: float = Field(..., ge=0.0, le=1.0, description="Championship pressure score")
    leader_points: float = Field(..., ge=0, le=500, description="Championship leader points")
    points_gap_to_leader: float = Field(..., ge=0, description="Points gap to leader")
    points_pressure: float = Field(..., ge=0.0, le=1.0, description="Points pressure score")
    driver_avg_grid_position: float = Field(..., ge=1.0, le=20.0, description="Driver's average grid position")
    qualifying_vs_average: float = Field(..., description="Qualifying vs average performance")
    constructor_avg_grid_position: float = Field(..., ge=1.0, le=20.0, description="Constructor's average grid position")
    qualifying_vs_constructor_avg: float = Field(..., description="Qualifying vs constructor average")
    bad_qualifying_day: float = Field(..., ge=0, le=1, description="Bad qualifying day flag")
    circuit_dnf_rate_detailed: float = Field(..., ge=0.0, le=1.0, description="Detailed circuit DNF rate")
    avg_pit_stops: float = Field(..., ge=0.0, le=10.0, description="Average pit stops")
    avg_pit_duration: float = Field(..., ge=15.0, le=120.0, description="Average pit duration")
    dnf_score: float = Field(..., ge=0.0, le=1.0, description="DNF risk score")
    volatility_score: float = Field(..., ge=0.0, le=1.0, description="Performance volatility score")
    pit_complexity_score: float = Field(..., ge=0.0, le=1.0, description="Pit complexity score")
    track_difficulty_score: float = Field(..., ge=0.0, le=1.0, description="Track difficulty score")
    race_number: float = Field(..., ge=1, le=25, description="Race number in season")
    first_season: float = Field(..., ge=1950, le=2030, description="Driver's first season")
    seasons_active: float = Field(..., ge=1, le=30, description="Seasons active")
    estimated_age: float = Field(..., ge=18, le=50, description="Driver's estimated age")
    driver_rolling_avg_grid: float = Field(..., ge=1.0, le=20.0, description="Driver's rolling average grid position")
    season_leader_points: float = Field(..., ge=0, le=500, description="Season leader points")
    points_gap_to_season_leader: float = Field(..., ge=0, description="Points gap to season leader")
    is_championship_contender: float = Field(..., ge=0, le=1, description="Championship contender flag")
    points_momentum: float = Field(..., description="Points momentum score")
    championship_position_change: float = Field(..., description="Championship position change")
    teammate_avg_grid: float = Field(..., ge=1.0, le=20.0, description="Teammate's average grid position")
    grid_vs_teammate: float = Field(..., description="Grid position vs teammate")
    championship_pressure_score: float = Field(..., ge=0.0, le=1.0, description="Championship pressure score")
    max_round: float = Field(..., ge=1, le=25, description="Maximum rounds in season")
    late_season_race: float = Field(..., ge=0, le=1, description="Late season race flag")
    championship_pressure_adjusted: float = Field(..., ge=0.0, le=1.0, description="Adjusted championship pressure")
    points_per_race: float = Field(..., ge=0.0, description="Points per race average")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    scenarios: List[RaceFeatures] = Field(..., description="List of race scenarios to predict")
    
    @field_validator('scenarios')
    @classmethod
    def validate_scenarios_length(cls, v):
        if len(v) == 0:
            raise ValueError("At least one scenario is required")
        if len(v) > 100:  # Reasonable limit for batch processing
            raise ValueError("Maximum 100 scenarios allowed per batch")
        return v

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    classification: Dict[str, Any] = Field(..., description="Classification prediction results")
    regression: Dict[str, Any] = Field(..., description="Regression prediction results")
    feature_contributions: Dict[str, float] = Field(..., description="Feature importance contributions")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    errors: List[Dict[str, Any]] = Field(..., description="List of prediction errors")
    summary: Dict[str, Any] = Field(..., description="Batch prediction summary")

class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    request_count: int = Field(..., description="Total requests processed")
    error_count: int = Field(..., description="Total errors encountered")
    error_rate: float = Field(..., description="Error rate percentage")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage information")

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    status: str = Field(..., description="Model loading status")
    classification_model: Optional[Dict[str, Any]] = Field(None, description="Classification model info")
    regression_model: Optional[Dict[str, Any]] = Field(None, description="Regression model info")
    feature_info: Optional[Dict[str, Any]] = Field(None, description="Feature information")

# Custom exception classes
class ValidationError(Exception):
    """Custom validation error with detailed information."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)

class ModelNotLoadedError(Exception):
    """Error when models are not loaded."""
    pass

class PredictionError(Exception):
    """Error during prediction process."""
    pass

# Enhanced error response models
class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

# Input validation helpers
def validate_feature_ranges(features: Dict[str, float]) -> List[str]:
    """
    Validate feature values are within reasonable ranges.
    
    Args:
        features: Dictionary of feature values
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Define comprehensive validation rules
    validation_rules = {
        'championship_position': {
            'min': 1, 'max': 30,
            'description': 'Championship position must be between 1 and 30'
        },
        'pit_stop_count': {
            'min': 0, 'max': 10,
            'description': 'Pit stop count must be between 0 and 10'
        },
        'avg_pit_time': {
            'min': 15.0, 'max': 120.0,
            'description': 'Average pit time must be between 15 and 120 seconds'
        },
        'pit_time_std': {
            'min': 0.0, 'max': 50.0,
            'description': 'Pit time standard deviation must be between 0 and 50 seconds'
        },
        'circuit_length': {
            'min': 2.0, 'max': 8.0,
            'description': 'Circuit length must be between 2.0 and 8.0 km'
        },
        'points': {
            'min': 0, 'max': 500,
            'description': 'Championship points must be between 0 and 500'
        },
        'qualifying_gap_to_pole': {
            'min': 0.0, 'max': 10.0,
            'description': 'Qualifying gap to pole must be between 0 and 10 seconds'
        },
        'grid_position_percentile': {
            'min': 0.0, 'max': 1.0,
            'description': 'Grid position percentile must be between 0.0 and 1.0'
        },
        'circuit_dnf_rate': {
            'min': 0.0, 'max': 1.0,
            'description': 'Circuit DNF rate must be between 0.0 and 1.0'
        },
        'estimated_age': {
            'min': 18, 'max': 50,
            'description': 'Driver age must be between 18 and 50 years'
        },
        'seasons_active': {
            'min': 1, 'max': 30,
            'description': 'Seasons active must be between 1 and 30'
        },
        'race_number': {
            'min': 1, 'max': 25,
            'description': 'Race number must be between 1 and 25'
        }
    }
    
    # Check each feature
    for feature, value in features.items():
        if feature in validation_rules:
            rules = validation_rules[feature]
            
            # Check minimum value
            if 'min' in rules and value < rules['min']:
                errors.append(f"{feature}: {rules['description']} (got {value})")
            
            # Check maximum value
            if 'max' in rules and value > rules['max']:
                errors.append(f"{feature}: {rules['description']} (got {value})")
    
    return errors

def validate_feature_consistency(features: Dict[str, float]) -> List[str]:
    """
    Validate logical consistency between features.
    
    Args:
        features: Dictionary of feature values
        
    Returns:
        List of consistency error messages
    """
    errors = []
    
    # Check logical relationships
    try:
        # Points should be reasonable for championship position
        if features.get('championship_position', 0) <= 3 and features.get('points', 0) < 50:
            errors.append("Top 3 championship position should have more than 50 points")
        
        # Pit stop count should match pit frequency roughly
        pit_count = features.get('pit_stop_count', 0)
        pit_freq = features.get('pit_frequency', 0)
        if abs(pit_count - pit_freq) > 3:
            errors.append(f"Pit stop count ({pit_count}) and frequency ({pit_freq}) seem inconsistent")
        
        # Age and seasons active should be consistent
        age = features.get('estimated_age', 25)
        seasons = features.get('seasons_active', 1)
        if age < 18 + seasons:
            errors.append(f"Age ({age}) seems too young for {seasons} seasons active")
        
        # Championship pressure should be higher for contenders
        is_contender = features.get('is_championship_contender', 0)
        pressure = features.get('championship_pressure', 0)
        if is_contender > 0.5 and pressure < 0.3:
            errors.append("Championship contenders should have higher pressure scores")
            
    except (KeyError, TypeError) as e:
        errors.append(f"Error validating feature consistency: {str(e)}")
    
    return errors

# Global variables for tracking
start_time = datetime.now()
request_count = 0
error_count = 0

async def startup_event():
    """Initialize the predictor on startup."""
    global predictor
    try:
        logger.info("Starting F1 Performance Drop Predictor API...")
        
        # Check if models exist, if not create them
        import os
        import subprocess
        from pathlib import Path
        
        models_dir = Path("models/production")
        
        # More robust model checking
        model_files = list(models_dir.glob("*.joblib")) if models_dir.exists() else []
        
        if not model_files:
            logger.info("Models not found, attempting to train...")
            
            # Run the deployment fix script
            try:
                result = subprocess.run([
                    sys.executable, "deploy_models_fix.py"
                ], check=True, capture_output=True, text=True, timeout=900)
                logger.info("Model deployment fix completed successfully")
            except subprocess.TimeoutExpired:
                logger.error("Model deployment fix timed out")
            except subprocess.CalledProcessError as e:
                logger.error(f"Model deployment fix failed: {e.stderr}")
            except Exception as e:
                logger.error(f"Unexpected error in model deployment: {e}")
        
        # Attempt to load the predictor
        try:
            predictor = F1PerformancePredictor()
            logger.info("Predictor instance created, attempting to load models...")
            predictor.load_models()
            logger.info("✅ API startup completed successfully - models loaded")
        except Exception as model_error:
            logger.error(f"❌ Failed to load models: {model_error}")
            logger.error(f"Model error type: {type(model_error).__name__}")
            import traceback
            logger.error(f"Model loading traceback: {traceback.format_exc()}")
            
            # Check what models are actually available
            try:
                import os
                models_dir = "models/production"
                if os.path.exists(models_dir):
                    files = os.listdir(models_dir)
                    logger.info(f"Available model files: {files}")
                else:
                    logger.error("Models directory does not exist")
            except Exception as e:
                logger.error(f"Error checking model files: {e}")
            
            # Create a dummy predictor that will return appropriate errors
            predictor = None
        
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        predictor = None

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors with detailed information."""
    global error_count
    error_count += 1
    
    logger.warning(f"Validation error: {exc.message}")
    
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="ValidationError",
            message=exc.message,
            details={
                "field": exc.field,
                "value": exc.value,
                "path": str(request.url)
            },
            timestamp=datetime.now().isoformat(),
            request_id=request.headers.get("X-Request-ID")
        ).dict()
    )

@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_exception_handler(request: Request, exc: ModelNotLoadedError):
    """Handle model not loaded errors."""
    global error_count
    error_count += 1
    
    logger.error("Model not loaded error")
    
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="ModelNotLoadedError",
            message="Machine learning models are not loaded. Service unavailable.",
            details={
                "suggestion": "Wait for models to load or contact administrator",
                "path": str(request.url)
            },
            timestamp=datetime.now().isoformat(),
            request_id=request.headers.get("X-Request-ID")
        ).dict()
    )

@app.exception_handler(PredictionError)
async def prediction_exception_handler(request: Request, exc: PredictionError):
    """Handle prediction errors."""
    global error_count
    error_count += 1
    
    logger.error(f"Prediction error: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="PredictionError",
            message=f"Prediction failed: {str(exc)}",
            details={
                "suggestion": "Check input data and try again",
                "path": str(request.url)
            },
            timestamp=datetime.now().isoformat(),
            request_id=request.headers.get("X-Request-ID")
        ).dict()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    global error_count
    error_count += 1
    
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={
                "exception_type": type(exc).__name__,
                "path": str(request.url)
            },
            timestamp=datetime.now().isoformat(),
            request_id=request.headers.get("X-Request-ID")
        ).dict()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Service health status and detailed metrics
    """
    global request_count, error_count
    
    uptime = (datetime.now() - start_time).total_seconds()
    error_rate = (error_count / request_count * 100) if request_count > 0 else 0
    
    # Check model status
    models_loaded = predictor and predictor.models_loaded if predictor else False
    
    # Check if models directory exists
    import os
    models_exist = os.path.exists("models/production") and bool(os.listdir("models/production"))
    
    # Determine status
    if models_loaded:
        status = "healthy"
    elif models_exist:
        status = "models_found_but_not_loaded"
    else:
        status = "no_models_found"
    
    # Get memory usage if psutil is available
    memory_info = None
    try:
        import psutil
        process = psutil.Process()
        memory_info = {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    except ImportError:
        pass
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded,
        uptime_seconds=uptime,
        request_count=request_count,
        error_count=error_count,
        error_rate=error_rate,
        memory_usage=memory_info
    )

@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about loaded models.
    
    Returns:
        Model information including performance metrics and feature details
    """
    if not predictor:
        return ModelInfoResponse(status="Predictor not initialized")
    
    model_info = predictor.get_model_info()
    
    return ModelInfoResponse(
        status=model_info['status'],
        classification_model=model_info.get('classification_model'),
        regression_model=model_info.get('regression_model'),
        feature_info=model_info.get('feature_info')
    )

@app.get("/debug")
async def debug_info():
    """
    Debug endpoint to diagnose deployment issues.
    """
    import os
    from pathlib import Path
    
    debug_data = {
        "predictor_status": "initialized" if predictor else "not_initialized",
        "models_loaded": predictor.models_loaded if predictor else False,
        "models_directory_exists": os.path.exists("models/production"),
        "model_files": [],
        "registry_exists": os.path.exists("models/model_registry.json"),
        "data_files": [],
        "environment": {
            "python_path": os.environ.get("PYTHONPATH", "not_set"),
            "current_dir": os.getcwd(),
        }
    }
    
    # Check model files
    models_dir = Path("models/production")
    if models_dir.exists():
        debug_data["model_files"] = [f.name for f in models_dir.iterdir() if f.is_file()]
    
    # Check data files
    data_dir = Path("data")
    if data_dir.exists():
        debug_data["data_files"] = [f.name for f in data_dir.iterdir() if f.is_file()]
    
    # Check registry content
    if os.path.exists("models/model_registry.json"):
        try:
            import json
            with open("models/model_registry.json", 'r') as f:
                registry_data = json.load(f)
                debug_data["registry_models"] = list(registry_data.get("models", {}).keys())
        except Exception as e:
            debug_data["registry_error"] = str(e)
    
    return debug_data

@app.post("/predict", response_model=PredictionResponse)
async def predict_performance_drop(features: RaceFeatures):
    """
    Predict F1 performance drop for a single race scenario.
    
    Args:
        features: Race features and parameters
        
    Returns:
        Prediction results including classification and regression outputs
        
    Raises:
        HTTPException: If prediction fails or models not loaded
    """
    global request_count
    request_count += 1
    
    if not predictor or not predictor.models_loaded:
        raise ModelNotLoadedError("Models not loaded")
    
    try:
        # Convert Pydantic model to dictionary
        input_data = features.dict()
        
        # Additional validation
        range_errors = validate_feature_ranges(input_data)
        if range_errors:
            raise ValidationError(f"Feature range validation failed: {'; '.join(range_errors)}")
        
        consistency_errors = validate_feature_consistency(input_data)
        if consistency_errors:
            logger.warning(f"Feature consistency warnings: {'; '.join(consistency_errors)}")
        
        # Make prediction
        result = predictor.predict_single(input_data)
        
        # Add validation warnings to response if any
        if consistency_errors:
            result['warnings'] = consistency_errors
        
        # Format response
        return PredictionResponse(
            classification=result['classification'],
            regression=result['regression'],
            feature_contributions=result['feature_contributions'],
            model_info=result['model_info'],
            prediction_timestamp=result['model_info']['prediction_timestamp']
        )
        
    except ValidationError:
        raise  # Re-raise validation errors
    except ValueError as e:
        raise ValidationError(f"Input validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise PredictionError(str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch_performance_drop(request: BatchPredictionRequest):
    """
    Predict F1 performance drop for multiple race scenarios.
    
    Args:
        request: Batch prediction request with multiple scenarios
        
    Returns:
        Batch prediction results with summary statistics
        
    Raises:
        HTTPException: If batch prediction fails or models not loaded
    """
    global request_count
    request_count += 1
    
    if not predictor or not predictor.models_loaded:
        raise ModelNotLoadedError("Models not loaded")
    
    try:
        # Convert scenarios to list of dictionaries
        input_list = [scenario.dict() for scenario in request.scenarios]
        
        # Validate each scenario
        validation_errors = []
        for i, scenario in enumerate(input_list):
            range_errors = validate_feature_ranges(scenario)
            if range_errors:
                validation_errors.append(f"Scenario {i}: {'; '.join(range_errors)}")
        
        if validation_errors:
            raise ValidationError(f"Batch validation failed: {'; '.join(validation_errors)}")
        
        # Make batch predictions
        batch_result = predictor.predict_batch(input_list)
        
        # Format predictions
        formatted_predictions = []
        for pred in batch_result['predictions']:
            formatted_predictions.append(PredictionResponse(
                classification=pred['classification'],
                regression=pred['regression'],
                feature_contributions=pred['feature_contributions'],
                model_info=pred['model_info'],
                prediction_timestamp=pred['model_info']['prediction_timestamp']
            ))
        
        return BatchPredictionResponse(
            predictions=formatted_predictions,
            errors=batch_result['errors'],
            summary=batch_result['summary']
        )
        
    except ValidationError:
        raise  # Re-raise validation errors
    except ValueError as e:
        raise ValidationError(f"Batch input validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise PredictionError(str(e))

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Basic API information and links to documentation
    """
    return {
        "message": "F1 Performance Drop Predictor API",
        "version": "1.0.0",
        "description": "Machine learning API for predicting Formula 1 finishing position drops",
        "endpoints": {
            "health": "/health",
            "model_info": "/model_info",
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ui")
async def get_ui():
    """
    Serve the web UI for F1 Performance Drop Predictor.
    
    Returns:
        HTML page with interactive prediction interface
    """
    return FileResponse('static/index.html')

if __name__ == "__main__":
    # Get port from environment variable (for cloud deployment) or default to 8000
    import os
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )