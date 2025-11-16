#!/usr/bin/env python3
"""
Model Persistence and Registry Module

This module provides model lifecycle management:
- Model registry for tracking trained models
- Production model deployment and versioning
- Overfitting detection and validation
- Model metadata and performance tracking
- Loading utilities for inference

Ensures reproducible model deployment with comprehensive metadata
and performance validation across different data splits.
"""
"""
Model persistence and validation module for F1 performance drop prediction.

This module provides:
- Model persistence with metadata and preprocessing pipeline
- Overfitting detection and validation
- Model versioning and performance tracking
- Model loading and inference utilities
"""

import pandas as pd
import numpy as np
import logging
import json
import os
import joblib
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import hashlib
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Model registry for tracking and managing trained models.
    """
    
    def __init__(self, registry_path: str = 'models/model_registry.json') -> None:
        self.registry_path = registry_path
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load existing model registry or create new one."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading registry: {e}. Creating new registry.")
        
        return {
            'models': {},
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        self.registry['last_updated'] = datetime.now().isoformat()
        
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def register_model(self, model_id: str, model_info: Dict[str, Any]) -> None:
        """Register a new model in the registry."""
        self.registry['models'][model_id] = {
            **model_info,
            'registered_at': datetime.now().isoformat()
        }
        self._save_registry()
        logger.info(f"Registered model: {model_id}")
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered model."""
        return self.registry['models'].get(model_id)
    
    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self.registry['models'].keys())
    
    def get_best_model(self, task_type: str, metric: str = 'f1') -> Optional[str]:
        """Get the best model for a given task type and metric."""
        best_model = None
        best_score = -float('inf') if metric in ['f1', 'accuracy', 'r2'] else float('inf')
        
        for model_id, model_info in self.registry['models'].items():
            if model_info.get('task_type') == task_type:
                score = model_info.get('performance_metrics', {}).get(metric)
                if score is not None:
                    if metric in ['f1', 'accuracy', 'r2']:
                        if score > best_score:
                            best_score = score
                            best_model = model_id
                    else:  # Lower is better (mae, mse, etc.)
                        if score < best_score:
                            best_score = score
                            best_model = model_id
        
        return best_model

def calculate_model_hash(model, feature_names: List[str]) -> str:
    """
    Calculate a hash for the model to track changes.
    
    Args:
        model: Trained model object
        feature_names: List of feature names used
        
    Returns:
        Hash string representing the model
    """
    # Create a string representation of the model
    model_str = str(type(model).__name__)
    
    # Add model parameters if available
    if hasattr(model, 'get_params'):
        params = model.get_params()
        model_str += str(sorted(params.items()))
    
    # Add feature names
    model_str += str(sorted(feature_names))
    
    # Calculate hash
    return hashlib.md5(model_str.encode()).hexdigest()

def detect_overfitting(model, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame, y_val: pd.Series,
                      task_type: str = 'classification') -> Dict[str, Any]:
    """
    Detect overfitting by comparing training and validation performance.
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary with overfitting analysis
    """
    logger.info("Detecting overfitting...")
    
    # Get predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    if task_type == 'classification':
        train_score = f1_score(y_train, train_pred, average='weighted')
        val_score = f1_score(y_val, val_pred, average='weighted')
        metric_name = 'F1 Score'
    else:
        train_score = mean_absolute_error(y_train, train_pred)
        val_score = mean_absolute_error(y_val, val_pred)
        metric_name = 'MAE'
        # For MAE, lower is better, so we need to flip the comparison
        train_score, val_score = -train_score, -val_score
    
    # Calculate overfitting metrics
    performance_gap = train_score - val_score
    relative_gap = performance_gap / abs(train_score) if train_score != 0 else 0
    
    # Determine overfitting severity
    if task_type == 'classification':
        # For classification, gap > 0.1 indicates potential overfitting
        overfitting_threshold = 0.1
        severe_threshold = 0.2
    else:
        # For regression, relative gap > 0.15 indicates potential overfitting
        overfitting_threshold = 0.15
        severe_threshold = 0.3
    
    if relative_gap > severe_threshold:
        overfitting_level = 'severe'
    elif relative_gap > overfitting_threshold:
        overfitting_level = 'moderate'
    else:
        overfitting_level = 'minimal'
    
    analysis = {
        'train_score': float(abs(train_score)) if task_type == 'regression' else float(train_score),
        'validation_score': float(abs(val_score)) if task_type == 'regression' else float(val_score),
        'performance_gap': float(abs(performance_gap)) if task_type == 'regression' else float(performance_gap),
        'relative_gap': float(abs(relative_gap)),
        'overfitting_level': overfitting_level,
        'is_overfitting': overfitting_level != 'minimal',
        'metric_name': metric_name,
        'recommendations': []
    }
    
    # Add recommendations based on overfitting level
    if overfitting_level == 'severe':
        analysis['recommendations'].extend([
            'Consider reducing model complexity',
            'Add regularization',
            'Increase training data',
            'Use cross-validation for model selection'
        ])
    elif overfitting_level == 'moderate':
        analysis['recommendations'].extend([
            'Monitor performance on additional validation sets',
            'Consider slight regularization',
            'Validate feature selection'
        ])
    
    logger.info(f"Overfitting analysis: {overfitting_level} overfitting detected")
    logger.info(f"Performance gap: {analysis['performance_gap']:.3f} ({analysis['relative_gap']:.1%})")
    
    return analysis

def validate_model_performance(model, X: pd.DataFrame, y: pd.Series, 
                             task_type: str = 'classification', cv_folds: int = 5) -> Dict[str, Any]:
    """
    Validate model performance using cross-validation.
    
    Args:
        model: Trained model
        X, y: Features and target
        task_type: 'classification' or 'regression'
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating model performance with {cv_folds}-fold cross-validation...")
    
    # Choose scoring metric
    if task_type == 'classification':
        scoring = 'f1_weighted'
        metric_name = 'F1 Score'
    else:
        scoring = 'neg_mean_absolute_error'
        metric_name = 'MAE'
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
    
    # For MAE, convert back to positive values
    if task_type == 'regression':
        cv_scores = -cv_scores
    
    validation_results = {
        'cv_scores': cv_scores.tolist(),
        'mean_cv_score': float(np.mean(cv_scores)),
        'std_cv_score': float(np.std(cv_scores)),
        'min_cv_score': float(np.min(cv_scores)),
        'max_cv_score': float(np.max(cv_scores)),
        'cv_folds': cv_folds,
        'metric_name': metric_name,
        'scoring': scoring
    }
    
    # Calculate confidence interval (assuming normal distribution)
    confidence_interval = 1.96 * validation_results['std_cv_score'] / np.sqrt(cv_folds)
    validation_results['confidence_interval'] = float(confidence_interval)
    validation_results['ci_lower'] = validation_results['mean_cv_score'] - confidence_interval
    validation_results['ci_upper'] = validation_results['mean_cv_score'] + confidence_interval
    
    logger.info(f"Cross-validation {metric_name}: {validation_results['mean_cv_score']:.3f} ± {validation_results['std_cv_score']:.3f}")
    
    return validation_results

def save_model_with_metadata(model, scaler, feature_names: List[str], 
                           performance_metrics: Dict[str, Any],
                           overfitting_analysis: Dict[str, Any],
                           validation_results: Dict[str, Any],
                           model_name: str, task_type: str,
                           output_dir: str = 'models/production') -> str:
    """
    Save model with comprehensive metadata and validation results.
    
    Args:
        model: Trained model
        scaler: Fitted scaler (can be None)
        feature_names: List of feature names
        performance_metrics: Performance metrics dictionary
        overfitting_analysis: Overfitting analysis results
        validation_results: Cross-validation results
        model_name: Name of the model
        task_type: 'classification' or 'regression'
        output_dir: Directory to save the model
        
    Returns:
        Model ID for the saved model
    """
    logger.info(f"Saving model with metadata: {model_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate model ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_hash = calculate_model_hash(model, feature_names)
    model_id = f"{model_name}_{task_type}_{timestamp}_{model_hash[:8]}"
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_id}_model.joblib")
    joblib.dump(model, model_path)
    
    # Save scaler if provided
    scaler_path = None
    if scaler is not None:
        scaler_path = os.path.join(output_dir, f"{model_id}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
    
    # Create comprehensive metadata
    metadata = {
        'model_id': model_id,
        'model_name': model_name,
        'task_type': task_type,
        'model_type': type(model).__name__,
        'timestamp': timestamp,
        'model_hash': model_hash,
        'files': {
            'model': model_path,
            'scaler': scaler_path,
            'metadata': os.path.join(output_dir, f"{model_id}_metadata.json")
        },
        'feature_info': {
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'requires_scaling': scaler is not None
        },
        'performance_metrics': performance_metrics,
        'validation': {
            'overfitting_analysis': overfitting_analysis,
            'cross_validation': validation_results
        },
        'model_parameters': model.get_params() if hasattr(model, 'get_params') else {},
        'training_info': {
            'framework': 'scikit-learn',
            'python_version': '3.11',  # Could be made dynamic
            'dependencies': {
                'sklearn': '1.3.0',  # Could be made dynamic
                'pandas': '2.0.0',   # Could be made dynamic
                'numpy': '1.24.0'    # Could be made dynamic
            }
        }
    }
    
    # Save metadata
    metadata_path = metadata['files']['metadata']
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Model saved with ID: {model_id}")
    logger.info(f"  Model file: {model_path}")
    logger.info(f"  Scaler file: {scaler_path}")
    logger.info(f"  Metadata file: {metadata_path}")
    
    return model_id

def load_production_model(model_id: str, model_dir: str = 'models/production') -> Dict[str, Any]:
    """
    Load a production model with all its components.
    
    Args:
        model_id: ID of the model to load
        model_dir: Directory containing production models
        
    Returns:
        Dictionary containing model, scaler, and metadata
    """
    logger.info(f"Loading production model: {model_id}")
    
    # Load metadata
    metadata_path = os.path.join(model_dir, f"{model_id}_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model_path = os.path.join(model_dir, f"{model_id}_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load scaler if it exists
    scaler = None
    scaler_path = os.path.join(model_dir, f"{model_id}_scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    logger.info(f"Successfully loaded model: {model_id}")
    
    return {
        'model': model,
        'scaler': scaler,
        'metadata': metadata,
        'model_id': model_id
    }

def create_model_version_tracking(models_info: List[Dict[str, Any]], 
                                output_path: str = 'models/version_history.json') -> None:
    """
    Create version tracking for models with performance comparison.
    
    Args:
        models_info: List of model information dictionaries
        output_path: Path to save version history
    """
    logger.info("Creating model version tracking...")
    
    version_history = {
        'created_at': datetime.now().isoformat(),
        'models': {},
        'performance_trends': {
            'classification': [],
            'regression': []
        }
    }
    
    # Process each model
    for model_info in models_info:
        model_id = model_info['model_id']
        version_history['models'][model_id] = {
            'model_name': model_info['model_name'],
            'task_type': model_info['task_type'],
            'timestamp': model_info['timestamp'],
            'performance_summary': {
                'primary_metric': model_info['performance_metrics'],
                'validation_score': model_info['validation']['cross_validation']['mean_cv_score'],
                'overfitting_level': model_info['validation']['overfitting_analysis']['overfitting_level']
            }
        }
        
        # Add to performance trends
        task_type = model_info['task_type']
        if task_type in version_history['performance_trends']:
            version_history['performance_trends'][task_type].append({
                'model_id': model_id,
                'timestamp': model_info['timestamp'],
                'score': model_info['validation']['cross_validation']['mean_cv_score']
            })
    
    # Sort performance trends by timestamp
    for task_type in version_history['performance_trends']:
        version_history['performance_trends'][task_type].sort(
            key=lambda x: x['timestamp']
        )
    
    # Save version history
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(version_history, f, indent=2, default=str)
    
    logger.info(f"Version history saved to: {output_path}")

def run_model_persistence_pipeline(models_results: Dict[str, Any], 
                                 X_train: pd.DataFrame, y_train_clf: pd.Series, y_train_reg: pd.Series,
                                 X_val: pd.DataFrame, y_val_clf: pd.Series, y_val_reg: pd.Series,
                                 feature_names: List[str]) -> Dict[str, str]:
    """
    Run the complete model persistence pipeline.
    
    Args:
        models_results: Dictionary containing trained models and results
        X_train, y_train_clf, y_train_reg: Training data
        X_val, y_val_clf, y_val_reg: Validation data
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping model names to their saved model IDs
    """
    logger.info("Running model persistence pipeline...")
    
    registry = ModelRegistry()
    saved_models = {}
    models_metadata = []
    
    # Process each model
    for model_name, model_data in models_results.items():
        if model_name == 'scaler' or 'model' not in model_data:
            continue
        
        model = model_data['model']
        scaler = models_results.get('scaler')
        
        # Determine task type
        if 'classifier' in model_name or model_name == 'logistic_regression':
            task_type = 'classification'
            y_train = y_train_clf
            y_val = y_val_clf
        else:
            task_type = 'regression'
            y_train = y_train_reg
            y_val = y_val_reg
        
        try:
            # Detect overfitting
            overfitting_analysis = detect_overfitting(
                model, X_train, y_train, X_val, y_val, task_type
            )
            
            # Validate performance
            validation_results = validate_model_performance(
                model, X_train, y_train, task_type
            )
            
            # Get performance metrics
            performance_metrics = model_data.get('metrics', {})
            
            # Save model with metadata
            model_id = save_model_with_metadata(
                model, scaler, feature_names, performance_metrics,
                overfitting_analysis, validation_results,
                model_name, task_type
            )
            
            # Register model
            registry.register_model(model_id, {
                'model_name': model_name,
                'task_type': task_type,
                'performance_metrics': performance_metrics,
                'validation_score': validation_results['mean_cv_score'],
                'overfitting_level': overfitting_analysis['overfitting_level'],
                'model_path': f'models/production/{model_id}_model.joblib'
            })
            
            saved_models[model_name] = model_id
            
            # Collect metadata for version tracking
            metadata_path = f'models/production/{model_id}_metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    models_metadata.append(json.load(f))
            
            logger.info(f"Successfully processed {model_name} -> {model_id}")
            
        except Exception as e:
            logger.error(f"Error processing {model_name}: {str(e)}")
    
    # Create version tracking
    if models_metadata:
        create_model_version_tracking(models_metadata)
    
    logger.info(f"Model persistence pipeline completed. Saved {len(saved_models)} models.")
    
    return saved_models

if __name__ == "__main__":
    # Example usage - this would typically be called from the training pipeline
    print("Model persistence module loaded successfully.")
    print("Use this module to save and manage trained models with comprehensive metadata.")