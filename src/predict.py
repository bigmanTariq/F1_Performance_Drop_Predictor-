#!/usr/bin/env python3
"""
Prediction module for F1 performance drop prediction.

This module provides:
- Model loading with error handling
- Single and batch prediction capabilities
- Input validation and preprocessing
- Prediction confidence and interpretation
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import joblib
from model_persistence import ModelRegistry, load_production_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1PerformancePredictor:
    """
    Main predictor class for F1 performance drop predictions.
    """
    
    def __init__(self, model_dir: str = 'models/production') -> None:
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory containing production models
        """
        self.model_dir = model_dir
        self.registry = ModelRegistry()
        self.classification_model = None
        self.regression_model = None
        self.scaler = None
        self.feature_names = None
        self.models_loaded = False
        
    def load_models(self, classification_model_id: Optional[str] = None, 
                   regression_model_id: Optional[str] = None) -> None:
        """
        Load the best available models or specific models by ID.
        
        Args:
            classification_model_id: Specific classification model ID to load
            regression_model_id: Specific regression model ID to load
        """
        logger.info("Loading F1 performance prediction models...")
        
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
            
            # List available model files
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.joblib')]
            logger.info(f"Found {len(model_files)} model files in {self.model_dir}")
            
            # Get best models if not specified
            if classification_model_id is None:
                classification_model_id = self.registry.get_best_model('classification', 'f1')
                logger.info(f"Selected best classification model: {classification_model_id}")
            if regression_model_id is None:
                regression_model_id = self.registry.get_best_model('regression', 'mae')
                logger.info(f"Selected best regression model: {regression_model_id}")
            
            if not classification_model_id or not regression_model_id:
                # List available models for debugging
                available_models = self.registry.list_models()
                logger.error(f"Available models in registry: {available_models}")
                
                # Fallback: try to find models by filename pattern
                logger.info("Attempting fallback model selection...")
                if not classification_model_id:
                    clf_files = [f for f in model_files if 'classification' in f and '_model.joblib' in f]
                    if clf_files:
                        # Extract model ID from filename
                        classification_model_id = clf_files[0].replace('_model.joblib', '')
                        logger.info(f"Fallback classification model: {classification_model_id}")
                
                if not regression_model_id:
                    reg_files = [f for f in model_files if 'regression' in f and '_model.joblib' in f]
                    if reg_files:
                        # Extract model ID from filename
                        regression_model_id = reg_files[0].replace('_model.joblib', '')
                        logger.info(f"Fallback regression model: {regression_model_id}")
                
                if not classification_model_id or not regression_model_id:
                    raise ValueError(f"No suitable models found. Registry: {available_models}, Files: {model_files}")
            
            # Load classification model
            clf_components = load_production_model(classification_model_id, self.model_dir)
            self.classification_model = clf_components['model']
            self.classification_metadata = clf_components['metadata']
            
            # Load regression model
            reg_components = load_production_model(regression_model_id, self.model_dir)
            self.regression_model = reg_components['model']
            self.regression_metadata = reg_components['metadata']
            
            # Use scaler from classification model (they should be the same)
            self.scaler = clf_components['scaler']
            
            # Get feature names (should be consistent across models)
            self.feature_names = clf_components['metadata']['feature_info']['feature_names']
            
            self.models_loaded = True
            
            logger.info(f"Successfully loaded models:")
            logger.info(f"  Classification: {classification_model_id}")
            logger.info(f"  Regression: {regression_model_id}")
            logger.info(f"  Features: {len(self.feature_names)} features")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and preprocess input data.
        
        Args:
            input_data: Dictionary with race parameters
            
        Returns:
            Validated and preprocessed input data
            
        Raises:
            ValueError: If input validation fails
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Check if all required features are provided
        missing_features = []
        for feature in self.feature_names:
            if feature not in input_data:
                missing_features.append(feature)
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Basic validation rules for common feature types
        validation_rules = {
            'championship_position': {'min': 1, 'max': 30},
            'pit_stop_count': {'min': 0, 'max': 10},
            'avg_pit_time': {'min': 15.0, 'max': 120.0},
            'pit_time_std': {'min': 0.0, 'max': 50.0},
            'circuit_length': {'min': 2.0, 'max': 8.0},
            'points': {'min': 0, 'max': 500},
            'pit_frequency': {'min': 0.0, 'max': 5.0},
            'qualifying_gap_to_pole': {'min': 0.0, 'max': 10.0},
            'grid_position_percentile': {'min': 0.0, 'max': 1.0},
            'circuit_dnf_rate': {'min': 0.0, 'max': 1.0},
            'is_street_circuit': {'min': 0, 'max': 1},
            'race_number': {'min': 1, 'max': 25},
            'first_season': {'min': 1950, 'max': 2030},
            'seasons_active': {'min': 1, 'max': 30},
            'estimated_age': {'min': 18, 'max': 50}
        }
        
        validated_data = {}
        errors = []
        
        # Validate each feature
        for feature in self.feature_names:
            value = input_data[feature]
            
            # Type validation - convert to float
            try:
                validated_value = float(value)
            except (ValueError, TypeError):
                errors.append(f"Feature '{feature}' must be numeric, got {type(value)}")
                continue
            
            # Range validation if rules exist
            if feature in validation_rules:
                rules = validation_rules[feature]
                if 'min' in rules and validated_value < rules['min']:
                    errors.append(f"Feature '{feature}' must be >= {rules['min']}, got {validated_value}")
                    continue
                if 'max' in rules and validated_value > rules['max']:
                    errors.append(f"Feature '{feature}' must be <= {rules['max']}, got {validated_value}")
                    continue
            
            validated_data[feature] = validated_value
        
        if errors:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")
        
        return validated_data
    
    def preprocess_input(self, validated_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess validated input data for model prediction.
        
        Args:
            validated_data: Validated input data
            
        Returns:
            Preprocessed feature array
        """
        # Create feature array in the correct order
        feature_array = np.array([validated_data[feature] for feature in self.feature_names])
        feature_array = feature_array.reshape(1, -1)
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            feature_array = self.scaler.transform(feature_array)
        
        return feature_array
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single prediction for race performance drop.
        
        Args:
            input_data: Dictionary with race parameters
            
        Returns:
            Dictionary with prediction results
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        try:
            # Validate and preprocess input
            validated_data = self.validate_input(input_data)
            feature_array = self.preprocess_input(validated_data)
            
            # Make predictions
            clf_prediction = self.classification_model.predict(feature_array)[0]
            clf_probability = self.classification_model.predict_proba(feature_array)[0]
            reg_prediction = self.regression_model.predict(feature_array)[0]
            
            # Calculate confidence based on probability margin
            prob_margin = abs(clf_probability[1] - 0.5)  # Distance from 0.5
            if prob_margin > 0.3:
                confidence = "high"
            elif prob_margin > 0.15:
                confidence = "medium"
            else:
                confidence = "low"
            
            # Get feature importance if available
            feature_contributions = {}
            if hasattr(self.classification_model, 'feature_importances_'):
                importances = self.classification_model.feature_importances_
                for i, feature in enumerate(self.feature_names):
                    feature_contributions[feature] = float(importances[i])
            
            # Create prediction result
            result = {
                'classification': {
                    'will_drop_position': bool(clf_prediction),
                    'probability': float(clf_probability[1]),  # Probability of position drop
                    'confidence': confidence
                },
                'regression': {
                    'expected_position_change': float(reg_prediction),
                    'prediction_interval': [
                        float(reg_prediction - 1.96 * 2.0),  # Approximate 95% CI
                        float(reg_prediction + 1.96 * 2.0)
                    ]
                },
                'feature_contributions': feature_contributions,
                'model_info': {
                    'classification_model': self.classification_metadata['model_id'],
                    'regression_model': self.regression_metadata['model_id'],
                    'prediction_timestamp': datetime.now().isoformat()
                },
                'input_data': validated_data
            }
            
            logger.info(f"Prediction completed: drop_probability={clf_probability[1]:.3f}, "
                       f"expected_change={reg_prediction:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make batch predictions for multiple race scenarios.
        
        Args:
            input_list: List of input dictionaries
            
        Returns:
            List of prediction results
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        logger.info(f"Making batch predictions for {len(input_list)} scenarios...")
        
        results = []
        errors = []
        
        for i, input_data in enumerate(input_list):
            try:
                result = self.predict_single(input_data)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                error_info = {
                    'batch_index': i,
                    'error': str(e),
                    'input_data': input_data
                }
                errors.append(error_info)
                logger.warning(f"Batch prediction {i} failed: {str(e)}")
        
        logger.info(f"Batch prediction completed: {len(results)} successful, {len(errors)} failed")
        
        return {
            'predictions': results,
            'errors': errors,
            'summary': {
                'total_requests': len(input_list),
                'successful_predictions': len(results),
                'failed_predictions': len(errors),
                'success_rate': len(results) / len(input_list) if input_list else 0
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        if not self.models_loaded:
            return {'status': 'Models not loaded'}
        
        return {
            'status': 'Models loaded',
            'classification_model': {
                'id': self.classification_metadata['model_id'],
                'name': self.classification_metadata['model_name'],
                'type': self.classification_metadata['model_type'],
                'performance': self.classification_metadata['performance_metrics'],
                'overfitting_level': self.classification_metadata['validation']['overfitting_analysis']['overfitting_level']
            },
            'regression_model': {
                'id': self.regression_metadata['model_id'],
                'name': self.regression_metadata['model_name'],
                'type': self.regression_metadata['model_type'],
                'performance': self.regression_metadata['performance_metrics'],
                'overfitting_level': self.regression_metadata['validation']['overfitting_analysis']['overfitting_level']
            },
            'feature_info': {
                'feature_names': self.feature_names,
                'n_features': len(self.feature_names),
                'requires_scaling': self.scaler is not None
            }
        }

# Convenience functions for direct usage
_global_predictor = None

def load_models(classification_model_id: Optional[str] = None, 
               regression_model_id: Optional[str] = None) -> None:
    """
    Load models into global predictor instance.
    
    Args:
        classification_model_id: Specific classification model ID
        regression_model_id: Specific regression model ID
    """
    global _global_predictor
    _global_predictor = F1PerformancePredictor()
    _global_predictor.load_models(classification_model_id, regression_model_id)

def predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a single prediction using global predictor.
    
    Args:
        input_data: Race parameters
        
    Returns:
        Prediction results
    """
    if _global_predictor is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    return _global_predictor.predict_single(input_data)

def predict_batch(input_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Make batch predictions using global predictor.
    
    Args:
        input_list: List of race parameters
        
    Returns:
        List of prediction results
    """
    if _global_predictor is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    return _global_predictor.predict_batch(input_list)

def get_model_info() -> Dict[str, Any]:
    """
    Get model information from global predictor.
    
    Returns:
        Model information
    """
    if _global_predictor is None:
        return {'status': 'Models not loaded'}
    return _global_predictor.get_model_info()

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize predictor
        predictor = F1PerformancePredictor()
        predictor.load_models()
        
        # Example prediction with actual feature names
        example_input = {
            'championship_position': 3,
            'pit_stop_count': 2,
            'avg_pit_time': 28.5,
            'pit_time_std': 3.2,
            'circuit_length': 5.4,
            'points': 150,
            'pit_frequency': 1.8,
            'pit_duration_variance': 10.2,
            'high_pit_frequency': 0,
            'qualifying_gap_to_pole': 0.8,
            'grid_position_percentile': 0.75,
            'poor_qualifying': 0,
            'circuit_dnf_rate': 0.15,
            'is_street_circuit': 0,
            'championship_pressure': 0.6,
            'leader_points': 200,
            'points_gap_to_leader': 50,
            'points_pressure': 0.4,
            'driver_avg_grid_position': 8.5,
            'qualifying_vs_average': -1.5,
            'constructor_avg_grid_position': 7.2,
            'qualifying_vs_constructor_avg': -2.2,
            'bad_qualifying_day': 0,
            'circuit_dnf_rate_detailed': 0.15,
            'avg_pit_stops': 2.1,
            'avg_pit_duration': 28.0,
            'dnf_score': 0.1,
            'volatility_score': 0.3,
            'pit_complexity_score': 0.5,
            'track_difficulty_score': 0.4,
            'race_number': 10,
            'first_season': 2015,
            'seasons_active': 8,
            'estimated_age': 28,
            'driver_rolling_avg_grid': 8.2,
            'season_leader_points': 200,
            'points_gap_to_season_leader': 50,
            'is_championship_contender': 1,
            'points_momentum': 0.2,
            'championship_position_change': 0,
            'teammate_avg_grid': 9.1,
            'grid_vs_teammate': -0.6,
            'championship_pressure_score': 0.6,
            'max_round': 22,
            'late_season_race': 0,
            'championship_pressure_adjusted': 0.5,
            'points_per_race': 15.0
        }
        
        result = predictor.predict_single(example_input)
        print("Prediction successful!")
        print(f"Position drop probability: {result['classification']['probability']:.3f}")
        print(f"Expected position change: {result['regression']['expected_position_change']:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")