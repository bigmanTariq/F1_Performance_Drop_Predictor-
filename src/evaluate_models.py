#!/usr/bin/env python3
"""
Model Evaluation and Analysis Module

This module provides comprehensive model evaluation capabilities:
- Performance metrics calculation for classification and regression
- Cross-validation and statistical significance testing
- Feature importance analysis and interpretation
- Model comparison and selection criteria
- Evaluation report generation

Generates detailed evaluation reports with confidence intervals
and statistical validation of model performance.
"""
"""
Model evaluation script for F1 performance drop prediction.

This script provides comprehensive model evaluation including:
- Detailed classification and regression metrics
- Feature importance analysis
- Model comparison and selection
- Confidence intervals for key metrics
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Any
import joblib
from datetime import datetime

# Import from train module
from train import (
    load_engineered_features, prepare_training_data, time_aware_split,
    evaluate_model_performance, calculate_detailed_classification_metrics,
    calculate_detailed_regression_metrics, analyze_feature_importance
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_saved_models(model_dir: str = 'models') -> Dict[str, Any]:
    """
    Load all saved models from the models directory.
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        Dictionary containing loaded models and metadata
    """
    logger.info(f"Loading saved models from {model_dir}...")
    
    if not os.path.exists(model_dir):
        logger.error(f"Models directory {model_dir} not found")
        return {}
    
    # Find the most recent metadata file
    metadata_files = [f for f in os.listdir(model_dir) if f.startswith('advanced_metadata_') and f.endswith('.json')]
    
    if not metadata_files:
        logger.error("No metadata files found")
        return {}
    
    # Use the most recent metadata file
    latest_metadata = sorted(metadata_files)[-1]
    metadata_path = os.path.join(model_dir, latest_metadata)
    
    logger.info(f"Loading metadata from {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract timestamp from metadata
    timestamp = metadata['timestamp']
    
    # Load models
    loaded_models = {}
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith(f'_{timestamp}.joblib') and 'scaler' not in f]
    
    for model_file in model_files:
        model_name = model_file.replace(f'_{timestamp}.joblib', '')
        model_path = os.path.join(model_dir, model_file)
        
        try:
            model = joblib.load(model_path)
            loaded_models[model_name] = {'model': model}
            logger.info(f"Loaded {model_name}")
        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
    
    # Load scaler if available
    scaler_file = f'advanced_scaler_{timestamp}.joblib'
    scaler_path = os.path.join(model_dir, scaler_file)
    
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            loaded_models['scaler'] = scaler
            logger.info("Loaded scaler")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
    
    return {
        'models': loaded_models,
        'metadata': metadata,
        'timestamp': timestamp
    }

def evaluate_loaded_models(data_path: str = 'data/f1_features_engineered.csv',
                          model_dir: str = 'models') -> Dict[str, Any]:
    """
    Evaluate loaded models on test data.
    
    Args:
        data_path: Path to the engineered features dataset
        model_dir: Directory containing saved models
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info("Starting comprehensive model evaluation...")
    
    # Load data
    df = load_engineered_features(data_path)
    
    # Prepare data with time-aware split
    train_df, test_df = time_aware_split(df)
    test_data = prepare_training_data(test_df)
    
    X_test = test_data['X']
    y_test_clf = test_data['y_classification']
    y_test_reg = test_data['y_regression']
    feature_names = test_data['feature_names']
    
    # Load saved models
    model_data = load_saved_models(model_dir)
    
    if not model_data:
        logger.error("No models loaded, cannot proceed with evaluation")
        return {}
    
    models = model_data['models']
    scaler = models.get('scaler')
    
    # Scale test data if scaler is available
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Generate predictions for all models
    results = {}
    
    for model_name, model_info in models.items():
        if model_name == 'scaler':
            continue
            
        model = model_info['model']
        
        try:
            if 'classifier' in model_name or model_name == 'logistic_regression':
                # Classification model
                if 'tree' in model_name or 'forest' in model_name:
                    # Tree-based models use original features
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    # Linear models use scaled features
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'task_type': 'classification'
                }
                
            elif 'regressor' in model_name or model_name == 'linear_regression':
                # Regression model
                if 'tree' in model_name or 'forest' in model_name:
                    # Tree-based models use original features
                    y_pred = model.predict(X_test)
                else:
                    # Linear models use scaled features
                    y_pred = model.predict(X_test_scaled)
                
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'task_type': 'regression'
                }
            
            logger.info(f"Generated predictions for {model_name}")
            
        except Exception as e:
            logger.error(f"Error generating predictions for {model_name}: {str(e)}")
    
    # Comprehensive evaluation
    evaluation = evaluate_model_performance(results, y_test_clf, y_test_reg, feature_names)
    
    # Add model metadata
    evaluation['model_metadata'] = model_data['metadata']
    evaluation['evaluation_timestamp'] = datetime.now().isoformat()
    
    return evaluation

def generate_evaluation_report(evaluation: Dict[str, Any], output_path: str = 'model_evaluation_report.json') -> None:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        evaluation: Evaluation results dictionary
        output_path: Path to save the report
    """
    logger.info(f"Generating evaluation report: {output_path}")
    
    # Create a formatted report
    report = {
        'evaluation_summary': evaluation.get('summary', {}),
        'model_comparison': evaluation.get('model_comparison', {}),
        'classification_models': {},
        'regression_models': {},
        'feature_importance': evaluation.get('feature_importance', {}),
        'metadata': evaluation.get('model_metadata', {}),
        'evaluation_timestamp': evaluation.get('evaluation_timestamp')
    }
    
    # Summarize classification models
    for model_name, model_data in evaluation.get('classification_models', {}).items():
        report['classification_models'][model_name] = {
            'model_type': model_data['model_type'],
            'accuracy': model_data['detailed_metrics']['accuracy'],
            'f1_score': model_data['detailed_metrics']['f1'],
            'roc_auc': model_data['detailed_metrics']['roc_auc'],
            'accuracy_ci': model_data['detailed_metrics']['accuracy_ci'],
            'f1_ci': model_data['detailed_metrics']['f1_ci'],
            'best_params': model_data.get('best_params', {})
        }
    
    # Summarize regression models
    for model_name, model_data in evaluation.get('regression_models', {}).items():
        report['regression_models'][model_name] = {
            'model_type': model_data['model_type'],
            'mae': model_data['detailed_metrics']['mae'],
            'rmse': model_data['detailed_metrics']['rmse'],
            'r2': model_data['detailed_metrics']['r2'],
            'mae_ci': model_data['detailed_metrics']['mae_ci'],
            'r2_ci': model_data['detailed_metrics']['r2_ci'],
            'median_absolute_error': model_data['detailed_metrics']['median_absolute_error'],
            'best_params': model_data.get('best_params', {})
        }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Evaluation report saved to {output_path}")

def print_evaluation_summary(evaluation: Dict[str, Any]) -> None:
    """
    Print a formatted evaluation summary to console.
    
    Args:
        evaluation: Evaluation results dictionary
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
    print("="*80)
    
    summary = evaluation.get('summary', {})
    comparison = evaluation.get('model_comparison', {})
    
    print(f"\nOverall Performance:")
    print(f"  Total Models Evaluated: {summary.get('total_models_evaluated', 0)}")
    print(f"  Meets Performance Requirements: {summary.get('meets_performance_requirements', False)}")
    
    print(f"\nBest Models:")
    print(f"  Classification: {comparison.get('best_classification_model', 'N/A')} (F1: {comparison.get('best_classification_f1', 0):.3f})")
    print(f"  Regression: {comparison.get('best_regression_model', 'N/A')} (MAE: {comparison.get('best_regression_mae', 0):.3f})")
    
    # Classification model rankings
    if 'classification_ranking' in comparison:
        print(f"\nClassification Model Rankings (by F1 Score):")
        for i, (model_name, f1_score) in enumerate(comparison['classification_ranking'], 1):
            print(f"  {i}. {model_name.replace('_', ' ').title()}: {f1_score:.3f}")
    
    # Regression model rankings
    if 'regression_ranking' in comparison:
        print(f"\nRegression Model Rankings (by MAE):")
        for i, (model_name, mae_score) in enumerate(comparison['regression_ranking'], 1):
            print(f"  {i}. {model_name.replace('_', ' ').title()}: {mae_score:.3f}")
    
    # Feature importance
    feature_importance = evaluation.get('feature_importance', {})
    if 'consensus_top_features' in feature_importance:
        print(f"\nTop Features (across all models):")
        for i, (feature, count) in enumerate(list(feature_importance['consensus_top_features'].items())[:10], 1):
            print(f"  {i}. {feature}: appears in {count} model(s) top 10")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # Run comprehensive model evaluation
    evaluation_results = evaluate_loaded_models()
    
    if evaluation_results:
        # Print summary
        print_evaluation_summary(evaluation_results)
        
        # Generate detailed report
        generate_evaluation_report(evaluation_results)
        
        print(f"\nDetailed evaluation report saved to: model_evaluation_report.json")
    else:
        print("No models found for evaluation. Please run training first.")