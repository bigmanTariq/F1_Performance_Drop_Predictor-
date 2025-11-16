"""
Model Training Module for F1 Performance Drop Prediction

This module implements the complete machine learning training pipeline:
- Loading engineered features and target variables
- Training multiple model types (logistic/linear regression, trees, forests)
- Hyperparameter tuning with cross-validation
- Model evaluation with comprehensive metrics
- Model selection and persistence to production

Supports both classification (position drop prediction) and regression
(position change magnitude) with time-aware validation splits.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None
    XGBRegressor = None

import joblib
import os
from datetime import datetime
import json
from model_persistence import run_model_persistence_pipeline, ModelRegistry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_engineered_features(filepath: str = 'data/f1_features_engineered.csv') -> pd.DataFrame:
    """
    Load the engineered features dataset for model training.
    
    Args:
        filepath: Path to the engineered features CSV file
        
    Returns:
        DataFrame with engineered features
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded engineered features: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading engineered features: {str(e)}")
        raise

def prepare_training_data(df: pd.DataFrame, target_cols: List[str] = None) -> Dict[str, Any]:
    """
    Prepare data for training with proper feature selection and preprocessing.
    
    Args:
        df: DataFrame with engineered features
        target_cols: List of target column names
        
    Returns:
        Dictionary containing prepared training data
    """
    logger.info("Preparing training data...")
    
    if target_cols is None:
        target_cols = ['position_drop_flag', 'position_change_numeric']
    
    # Define feature columns (exclude IDs, names, targets, and leaky features)
    exclude_cols = [
        'driver_id', 'driver_name', 'constructor_id', 'circuit_name',
        'season', 'round', 'position_drop_flag', 'position_change_numeric',
        'position', 'grid_position',  # These are used to create targets, so exclude from features
        # Exclude features that are derived from race results (data leakage)
        'is_dnf', 'is_win', 'is_podium', 'is_points',
        'position_vs_teammate', 'avg_position_change', 'position_change_volatility',
        'driver_rolling_avg_position', 'recent_performance_trend',
        'teammate_avg_position', 'constructor_season_avg_position',
        # Exclude features that use future information
        'career_win_rate', 'career_podium_rate', 'career_points_rate',
        'driver_rolling_dnf_rate', 'driver_reliability_score',
        'constructor_rolling_dnf_rate', 'constructor_reliability_score'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Select only numeric features for baseline models
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle missing values
    X = df[numeric_features].fillna(df[numeric_features].median())
    
    # Prepare targets
    y_classification = df[target_cols[0]]  # position_drop_flag
    y_regression = df[target_cols[1]]      # position_change_numeric
    
    logger.info(f"Selected {len(numeric_features)} numeric features")
    logger.info(f"Classification target distribution: {y_classification.value_counts().to_dict()}")
    logger.info(f"Regression target stats: mean={y_regression.mean():.2f}, std={y_regression.std():.2f}")
    
    return {
        'X': X,
        'y_classification': y_classification,
        'y_regression': y_regression,
        'feature_names': numeric_features,
        'target_cols': target_cols
    }

def time_aware_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform time-aware train/test split to prevent data leakage.
    
    Args:
        df: DataFrame with season and round columns
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info("Performing time-aware train/test split...")
    
    # Sort by season and round
    df_sorted = df.sort_values(['season', 'round'])
    
    # Calculate split point
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    logger.info(f"Train set: {len(train_df)} samples (seasons {train_df['season'].min()}-{train_df['season'].max()})")
    logger.info(f"Test set: {len(test_df)} samples (seasons {test_df['season'].min()}-{test_df['season'].max()})")
    
    return train_df, test_df

def train_baseline_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train_clf: pd.Series, y_test_clf: pd.Series,
                         y_train_reg: pd.Series, y_test_reg: pd.Series) -> Dict[str, Any]:
    """
    Train baseline logistic regression and linear regression models.
    
    Args:
        X_train, X_test: Feature matrices
        y_train_clf, y_test_clf: Classification targets
        y_train_reg, y_test_reg: Regression targets
        
    Returns:
        Dictionary containing trained models and results
    """
    logger.info("Training baseline models...")
    
    results = {}
    
    # Scale features for baseline models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Logistic Regression for Classification
    logger.info("Training logistic regression classifier...")
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train_clf)
    
    # Classification predictions
    y_pred_clf = log_reg.predict(X_test_scaled)
    y_pred_clf_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    
    # Classification metrics
    clf_metrics = {
        'accuracy': accuracy_score(y_test_clf, y_pred_clf),
        'precision': precision_score(y_test_clf, y_pred_clf, average='weighted'),
        'recall': recall_score(y_test_clf, y_pred_clf, average='weighted'),
        'f1': f1_score(y_test_clf, y_pred_clf, average='weighted'),
        'roc_auc': roc_auc_score(y_test_clf, y_pred_clf_proba)
    }
    
    results['logistic_regression'] = {
        'model': log_reg,
        'metrics': clf_metrics,
        'predictions': y_pred_clf,
        'probabilities': y_pred_clf_proba
    }
    
    logger.info(f"Logistic Regression - Accuracy: {clf_metrics['accuracy']:.3f}, F1: {clf_metrics['f1']:.3f}")
    
    # 2. Linear Regression for Regression
    logger.info("Training linear regression model...")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train_reg)
    
    # Regression predictions
    y_pred_reg = lin_reg.predict(X_test_scaled)
    
    # Regression metrics
    reg_metrics = {
        'mae': mean_absolute_error(y_test_reg, y_pred_reg),
        'mse': mean_squared_error(y_test_reg, y_pred_reg),
        'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
        'r2': r2_score(y_test_reg, y_pred_reg)
    }
    
    results['linear_regression'] = {
        'model': lin_reg,
        'metrics': reg_metrics,
        'predictions': y_pred_reg
    }
    
    logger.info(f"Linear Regression - MAE: {reg_metrics['mae']:.3f}, R²: {reg_metrics['r2']:.3f}")
    
    # Store scaler for later use
    results['scaler'] = scaler
    
    return results

def train_advanced_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train_clf: pd.Series, y_test_clf: pd.Series,
                         y_train_reg: pd.Series, y_test_reg: pd.Series) -> Dict[str, Any]:
    """
    Train advanced models with hyperparameter tuning.
    
    Args:
        X_train, X_test: Feature matrices
        y_train_clf, y_test_clf: Classification targets
        y_train_reg, y_test_reg: Regression targets
        
    Returns:
        Dictionary containing trained advanced models and results
    """
    logger.info("Training advanced models with hyperparameter tuning...")
    
    results = {}
    
    # Scale features for tree-based models (optional but can help)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Decision Tree Models
    logger.info("Training decision tree models...")
    
    # Decision Tree Classifier
    dt_clf_params = {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [42]
    }
    
    dt_clf = DecisionTreeClassifier()
    dt_clf_grid = GridSearchCV(
        dt_clf, dt_clf_params, cv=5, scoring='f1_weighted', n_jobs=-1
    )
    dt_clf_grid.fit(X_train, y_train_clf)
    
    # Decision Tree predictions
    y_pred_dt_clf = dt_clf_grid.predict(X_test)
    y_pred_dt_clf_proba = dt_clf_grid.predict_proba(X_test)[:, 1]
    
    dt_clf_metrics = {
        'accuracy': accuracy_score(y_test_clf, y_pred_dt_clf),
        'precision': precision_score(y_test_clf, y_pred_dt_clf, average='weighted'),
        'recall': recall_score(y_test_clf, y_pred_dt_clf, average='weighted'),
        'f1': f1_score(y_test_clf, y_pred_dt_clf, average='weighted'),
        'roc_auc': roc_auc_score(y_test_clf, y_pred_dt_clf_proba)
    }
    
    results['decision_tree_classifier'] = {
        'model': dt_clf_grid.best_estimator_,
        'best_params': dt_clf_grid.best_params_,
        'metrics': dt_clf_metrics,
        'predictions': y_pred_dt_clf,
        'probabilities': y_pred_dt_clf_proba
    }
    
    logger.info(f"Decision Tree Classifier - Accuracy: {dt_clf_metrics['accuracy']:.3f}, F1: {dt_clf_metrics['f1']:.3f}")
    
    # Decision Tree Regressor
    dt_reg_params = {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [42]
    }
    
    dt_reg = DecisionTreeRegressor()
    dt_reg_grid = GridSearchCV(
        dt_reg, dt_reg_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    dt_reg_grid.fit(X_train, y_train_reg)
    
    # Decision Tree regression predictions
    y_pred_dt_reg = dt_reg_grid.predict(X_test)
    
    dt_reg_metrics = {
        'mae': mean_absolute_error(y_test_reg, y_pred_dt_reg),
        'mse': mean_squared_error(y_test_reg, y_pred_dt_reg),
        'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_dt_reg)),
        'r2': r2_score(y_test_reg, y_pred_dt_reg)
    }
    
    results['decision_tree_regressor'] = {
        'model': dt_reg_grid.best_estimator_,
        'best_params': dt_reg_grid.best_params_,
        'metrics': dt_reg_metrics,
        'predictions': y_pred_dt_reg
    }
    
    logger.info(f"Decision Tree Regressor - MAE: {dt_reg_metrics['mae']:.3f}, R²: {dt_reg_metrics['r2']:.3f}")
    
    # 2. Random Forest Models
    logger.info("Training random forest models...")
    
    # Random Forest Classifier
    rf_clf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'random_state': [42]
    }
    
    rf_clf = RandomForestClassifier()
    rf_clf_grid = GridSearchCV(
        rf_clf, rf_clf_params, cv=5, scoring='f1_weighted', n_jobs=-1
    )
    rf_clf_grid.fit(X_train, y_train_clf)
    
    # Random Forest predictions
    y_pred_rf_clf = rf_clf_grid.predict(X_test)
    y_pred_rf_clf_proba = rf_clf_grid.predict_proba(X_test)[:, 1]
    
    rf_clf_metrics = {
        'accuracy': accuracy_score(y_test_clf, y_pred_rf_clf),
        'precision': precision_score(y_test_clf, y_pred_rf_clf, average='weighted'),
        'recall': recall_score(y_test_clf, y_pred_rf_clf, average='weighted'),
        'f1': f1_score(y_test_clf, y_pred_rf_clf, average='weighted'),
        'roc_auc': roc_auc_score(y_test_clf, y_pred_rf_clf_proba)
    }
    
    results['random_forest_classifier'] = {
        'model': rf_clf_grid.best_estimator_,
        'best_params': rf_clf_grid.best_params_,
        'metrics': rf_clf_metrics,
        'predictions': y_pred_rf_clf,
        'probabilities': y_pred_rf_clf_proba
    }
    
    logger.info(f"Random Forest Classifier - Accuracy: {rf_clf_metrics['accuracy']:.3f}, F1: {rf_clf_metrics['f1']:.3f}")
    
    # Random Forest Regressor
    rf_reg_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'random_state': [42]
    }
    
    rf_reg = RandomForestRegressor()
    rf_reg_grid = GridSearchCV(
        rf_reg, rf_reg_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    rf_reg_grid.fit(X_train, y_train_reg)
    
    # Random Forest regression predictions
    y_pred_rf_reg = rf_reg_grid.predict(X_test)
    
    rf_reg_metrics = {
        'mae': mean_absolute_error(y_test_reg, y_pred_rf_reg),
        'mse': mean_squared_error(y_test_reg, y_pred_rf_reg),
        'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_rf_reg)),
        'r2': r2_score(y_test_reg, y_pred_rf_reg)
    }
    
    results['random_forest_regressor'] = {
        'model': rf_reg_grid.best_estimator_,
        'best_params': rf_reg_grid.best_params_,
        'metrics': rf_reg_metrics,
        'predictions': y_pred_rf_reg
    }
    
    logger.info(f"Random Forest Regressor - MAE: {rf_reg_metrics['mae']:.3f}, R²: {rf_reg_metrics['r2']:.3f}")
    
    # 3. XGBoost Models (if available)
    if XGBOOST_AVAILABLE:
        logger.info("Training XGBoost models...")
        
        # XGBoost Classifier
        xgb_clf_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'random_state': [42]
        }
        
        xgb_clf = XGBClassifier(eval_metric='logloss')
        xgb_clf_grid = GridSearchCV(
            xgb_clf, xgb_clf_params, cv=3, scoring='f1_weighted', n_jobs=-1  # Reduced CV for speed
        )
        xgb_clf_grid.fit(X_train_scaled, y_train_clf)
        
        # XGBoost predictions
        y_pred_xgb_clf = xgb_clf_grid.predict(X_test_scaled)
        y_pred_xgb_clf_proba = xgb_clf_grid.predict_proba(X_test_scaled)[:, 1]
        
        xgb_clf_metrics = {
            'accuracy': accuracy_score(y_test_clf, y_pred_xgb_clf),
            'precision': precision_score(y_test_clf, y_pred_xgb_clf, average='weighted'),
            'recall': recall_score(y_test_clf, y_pred_xgb_clf, average='weighted'),
            'f1': f1_score(y_test_clf, y_pred_xgb_clf, average='weighted'),
            'roc_auc': roc_auc_score(y_test_clf, y_pred_xgb_clf_proba)
        }
        
        results['xgboost_classifier'] = {
            'model': xgb_clf_grid.best_estimator_,
            'best_params': xgb_clf_grid.best_params_,
            'metrics': xgb_clf_metrics,
            'predictions': y_pred_xgb_clf,
            'probabilities': y_pred_xgb_clf_proba
        }
        
        logger.info(f"XGBoost Classifier - Accuracy: {xgb_clf_metrics['accuracy']:.3f}, F1: {xgb_clf_metrics['f1']:.3f}")
        
        # XGBoost Regressor
        xgb_reg_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'random_state': [42]
        }
        
        xgb_reg = XGBRegressor()
        xgb_reg_grid = GridSearchCV(
            xgb_reg, xgb_reg_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        xgb_reg_grid.fit(X_train_scaled, y_train_reg)
        
        # XGBoost regression predictions
        y_pred_xgb_reg = xgb_reg_grid.predict(X_test_scaled)
        
        xgb_reg_metrics = {
            'mae': mean_absolute_error(y_test_reg, y_pred_xgb_reg),
            'mse': mean_squared_error(y_test_reg, y_pred_xgb_reg),
            'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_xgb_reg)),
            'r2': r2_score(y_test_reg, y_pred_xgb_reg)
        }
        
        results['xgboost_regressor'] = {
            'model': xgb_reg_grid.best_estimator_,
            'best_params': xgb_reg_grid.best_params_,
            'metrics': xgb_reg_metrics,
            'predictions': y_pred_xgb_reg
        }
        
        logger.info(f"XGBoost Regressor - MAE: {xgb_reg_metrics['mae']:.3f}, R²: {xgb_reg_metrics['r2']:.3f}")
    
    # Store scaler for later use
    results['scaler'] = scaler
    
    return results

def compare_model_performance(baseline_results: Dict[str, Any], 
                            advanced_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare performance between baseline and advanced models.
    
    Args:
        baseline_results: Results from baseline models
        advanced_results: Results from advanced models
        
    Returns:
        Dictionary with model comparison
    """
    logger.info("Comparing model performance...")
    
    comparison = {
        'classification': {},
        'regression': {},
        'best_models': {}
    }
    
    # Classification comparison
    clf_models = {}
    
    # Add baseline
    if 'logistic_regression' in baseline_results:
        clf_models['Logistic Regression'] = baseline_results['logistic_regression']['metrics']
    
    # Add advanced models
    for model_name, model_data in advanced_results.items():
        if 'classifier' in model_name and 'metrics' in model_data:
            display_name = model_name.replace('_', ' ').title()
            clf_models[display_name] = model_data['metrics']
    
    # Find best classification model
    if clf_models:
        best_clf = max(clf_models.items(), key=lambda x: x[1]['f1'])
        comparison['classification'] = {
            'models': clf_models,
            'best_model': best_clf[0],
            'best_f1': best_clf[1]['f1']
        }
    
    # Regression comparison
    reg_models = {}
    
    # Add baseline
    if 'linear_regression' in baseline_results:
        reg_models['Linear Regression'] = baseline_results['linear_regression']['metrics']
    
    # Add advanced models
    for model_name, model_data in advanced_results.items():
        if 'regressor' in model_name and 'metrics' in model_data:
            display_name = model_name.replace('_', ' ').title()
            reg_models[display_name] = model_data['metrics']
    
    # Find best regression model
    if reg_models:
        best_reg = min(reg_models.items(), key=lambda x: x[1]['mae'])
        comparison['regression'] = {
            'models': reg_models,
            'best_model': best_reg[0],
            'best_mae': best_reg[1]['mae']
        }
    
    # Store best models
    comparison['best_models'] = {
        'classification': comparison['classification']['best_model'] if 'classification' in comparison else None,
        'regression': comparison['regression']['best_model'] if 'regression' in comparison else None
    }
    
    logger.info(f"Best classification model: {comparison['best_models']['classification']}")
    logger.info(f"Best regression model: {comparison['best_models']['regression']}")
    
    return comparison

def calculate_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray, 
                                 metric_func: callable, confidence: float = 0.95,
                                 n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calculate confidence intervals for a metric using bootstrap sampling.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metric_func: Function to calculate metric
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    np.random.seed(42)
    bootstrap_scores = []
    
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metric
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)
    
    return lower_bound, upper_bound

def calculate_detailed_classification_metrics(y_true: pd.Series, y_pred: np.ndarray, 
                                            y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate detailed classification metrics with confidence intervals.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary with detailed metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Confidence intervals
    y_true_np = y_true.values
    metrics['accuracy_ci'] = calculate_confidence_intervals(
        y_true_np, y_pred, accuracy_score
    )
    metrics['f1_ci'] = calculate_confidence_intervals(
        y_true_np, y_pred, lambda yt, yp: f1_score(yt, yp, average='weighted')
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Per-class metrics
    class_report = classification_report(y_true, y_pred, output_dict=True)
    metrics['per_class_metrics'] = class_report
    
    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    metrics['roc_curve'] = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }
    
    return metrics

def calculate_detailed_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate detailed regression metrics with confidence intervals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with detailed metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Additional metrics
    residuals = y_true - y_pred
    metrics['mean_residual'] = np.mean(residuals)
    metrics['std_residual'] = np.std(residuals)
    
    # Confidence intervals
    y_true_np = y_true.values
    metrics['mae_ci'] = calculate_confidence_intervals(
        y_true_np, y_pred, mean_absolute_error
    )
    metrics['r2_ci'] = calculate_confidence_intervals(
        y_true_np, y_pred, r2_score
    )
    
    # Percentile-based metrics
    abs_errors = np.abs(residuals)
    metrics['median_absolute_error'] = np.median(abs_errors)
    metrics['90th_percentile_error'] = np.percentile(abs_errors, 90)
    metrics['95th_percentile_error'] = np.percentile(abs_errors, 95)
    
    # Residual analysis
    metrics['residual_stats'] = {
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
        'q25': float(np.percentile(residuals, 25)),
        'q75': float(np.percentile(residuals, 75)),
        'skewness': float(stats.skew(residuals)),
        'kurtosis': float(stats.kurtosis(residuals))
    }
    
    return metrics

def analyze_feature_importance(models: Dict[str, Any], feature_names: List[str]) -> Dict[str, Any]:
    """
    Analyze feature importance across different models.
    
    Args:
        models: Dictionary of trained models
        feature_names: List of feature names
        
    Returns:
        Dictionary with feature importance analysis
    """
    logger.info("Analyzing feature importance...")
    
    importance_analysis = {}
    
    for model_name, model_data in models.items():
        if model_name != 'scaler' and isinstance(model_data, dict) and 'model' in model_data and hasattr(model_data['model'], 'feature_importances_'):
            # Tree-based models
            importances = model_data['model'].feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            importance_analysis[model_name] = {
                'feature_importances': feature_importance_df.to_dict('records'),
                'top_10_features': feature_importance_df.head(10)['feature'].tolist(),
                'importance_sum': float(np.sum(importances))
            }
            
        elif model_name != 'scaler' and isinstance(model_data, dict) and 'model' in model_data and hasattr(model_data['model'], 'coef_'):
            # Linear models
            if len(model_data['model'].coef_.shape) == 1:
                # Regression or binary classification
                coefficients = np.abs(model_data['model'].coef_)
            else:
                # Multi-class classification
                coefficients = np.abs(model_data['model'].coef_).mean(axis=0)
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': coefficients
            }).sort_values('importance', ascending=False)
            
            importance_analysis[model_name] = {
                'feature_importances': feature_importance_df.to_dict('records'),
                'top_10_features': feature_importance_df.head(10)['feature'].tolist(),
                'importance_sum': float(np.sum(coefficients))
            }
    
    # Find most consistently important features
    if importance_analysis:
        all_top_features = []
        for model_analysis in importance_analysis.values():
            all_top_features.extend(model_analysis['top_10_features'])
        
        # Count feature appearances in top 10
        feature_counts = pd.Series(all_top_features).value_counts()
        importance_analysis['consensus_top_features'] = feature_counts.head(10).to_dict()
    
    return importance_analysis

def evaluate_model_performance(results: Dict[str, Any], y_test_clf: pd.Series, y_test_reg: pd.Series,
                             feature_names: List[str] = None) -> Dict[str, Any]:
    """
    Comprehensive model performance evaluation with detailed metrics and feature importance.
    
    Args:
        results: Dictionary containing model results
        y_test_clf: True classification labels
        y_test_reg: True regression values
        feature_names: List of feature names for importance analysis
        
    Returns:
        Dictionary with comprehensive performance evaluation
    """
    logger.info("Evaluating model performance with detailed metrics...")
    
    evaluation = {
        'classification_models': {},
        'regression_models': {},
        'feature_importance': {},
        'model_comparison': {},
        'summary': {}
    }
    
    # Evaluate all classification models
    classification_models = {}
    for model_name, model_data in results.items():
        if 'classifier' in model_name or model_name == 'logistic_regression':
            if 'predictions' in model_data and 'probabilities' in model_data:
                detailed_metrics = calculate_detailed_classification_metrics(
                    y_test_clf, model_data['predictions'], model_data['probabilities']
                )
                
                classification_models[model_name] = {
                    'model_type': model_name.replace('_', ' ').title(),
                    'detailed_metrics': detailed_metrics,
                    'best_params': model_data.get('best_params', {}),
                    'basic_metrics': model_data.get('metrics', {})
                }
    
    evaluation['classification_models'] = classification_models
    
    # Evaluate all regression models
    regression_models = {}
    for model_name, model_data in results.items():
        if 'regressor' in model_name or model_name == 'linear_regression':
            if 'predictions' in model_data:
                detailed_metrics = calculate_detailed_regression_metrics(
                    y_test_reg, model_data['predictions']
                )
                
                regression_models[model_name] = {
                    'model_type': model_name.replace('_', ' ').title(),
                    'detailed_metrics': detailed_metrics,
                    'best_params': model_data.get('best_params', {}),
                    'basic_metrics': model_data.get('metrics', {})
                }
    
    evaluation['regression_models'] = regression_models
    
    # Feature importance analysis
    if feature_names:
        evaluation['feature_importance'] = analyze_feature_importance(results, feature_names)
    
    # Model comparison and selection
    best_classification_model = None
    best_regression_model = None
    best_clf_f1 = 0
    best_reg_mae = float('inf')
    
    # Find best classification model
    for model_name, model_eval in classification_models.items():
        f1_score = model_eval['detailed_metrics']['f1']
        if f1_score > best_clf_f1:
            best_clf_f1 = f1_score
            best_classification_model = model_name
    
    # Find best regression model
    for model_name, model_eval in regression_models.items():
        mae_score = model_eval['detailed_metrics']['mae']
        if mae_score < best_reg_mae:
            best_reg_mae = mae_score
            best_regression_model = model_name
    
    evaluation['model_comparison'] = {
        'best_classification_model': best_classification_model,
        'best_classification_f1': best_clf_f1,
        'best_regression_model': best_regression_model,
        'best_regression_mae': best_reg_mae,
        'classification_ranking': sorted(
            [(name, data['detailed_metrics']['f1']) for name, data in classification_models.items()],
            key=lambda x: x[1], reverse=True
        ),
        'regression_ranking': sorted(
            [(name, data['detailed_metrics']['mae']) for name, data in regression_models.items()],
            key=lambda x: x[1]
        )
    }
    
    # Performance summary
    evaluation['summary'] = {
        'best_classification_accuracy': classification_models[best_classification_model]['detailed_metrics']['accuracy'] if best_classification_model else 0,
        'best_classification_f1': best_clf_f1,
        'best_regression_mae': best_reg_mae,
        'best_regression_r2': regression_models[best_regression_model]['detailed_metrics']['r2'] if best_regression_model else 0,
        'meets_performance_requirements': best_clf_f1 > 0.7 and best_reg_mae < 2.0,
        'total_models_evaluated': len(classification_models) + len(regression_models),
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Comprehensive Performance Summary:")
    logger.info(f"  Best Classification Model: {best_classification_model} (F1: {best_clf_f1:.3f})")
    logger.info(f"  Best Regression Model: {best_regression_model} (MAE: {best_reg_mae:.3f})")
    logger.info(f"  Total Models Evaluated: {evaluation['summary']['total_models_evaluated']}")
    logger.info(f"  Meets Performance Requirements: {evaluation['summary']['meets_performance_requirements']}")
    
    return evaluation

def save_baseline_models(results: Dict[str, Any], evaluation: Dict[str, Any], 
                        feature_names: List[str], output_dir: str = 'models') -> None:
    """
    Save trained baseline models and evaluation results.
    
    Args:
        results: Dictionary containing trained models
        evaluation: Dictionary containing evaluation results
        feature_names: List of feature names
        output_dir: Directory to save models
    """
    logger.info("Saving baseline models...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save models
    if 'logistic_regression' in results:
        model_path = os.path.join(output_dir, f'baseline_classifier_{timestamp}.joblib')
        joblib.dump(results['logistic_regression']['model'], model_path)
        logger.info(f"Saved classification model: {model_path}")
    
    if 'linear_regression' in results:
        model_path = os.path.join(output_dir, f'baseline_regressor_{timestamp}.joblib')
        joblib.dump(results['linear_regression']['model'], model_path)
        logger.info(f"Saved regression model: {model_path}")
    
    # Save scaler
    if 'scaler' in results:
        scaler_path = os.path.join(output_dir, f'baseline_scaler_{timestamp}.joblib')
        joblib.dump(results['scaler'], scaler_path)
        logger.info(f"Saved scaler: {scaler_path}")
    
    # Save evaluation results and metadata
    metadata = {
        'timestamp': timestamp,
        'feature_names': feature_names,
        'evaluation': evaluation,
        'model_info': {
            'classification_model': 'LogisticRegression',
            'regression_model': 'LinearRegression',
            'preprocessing': 'StandardScaler'
        }
    }
    
    metadata_path = os.path.join(output_dir, f'baseline_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved metadata: {metadata_path}")

def run_advanced_training_pipeline(data_path: str = 'data/f1_features_engineered.csv') -> Dict[str, Any]:
    """
    Run the complete advanced model training pipeline.
    
    Args:
        data_path: Path to the engineered features dataset
        
    Returns:
        Dictionary containing all results
    """
    logger.info("Starting advanced model training pipeline...")
    
    try:
        # 1. Load data
        df = load_engineered_features(data_path)
        
        # 2. Prepare training data
        data_dict = prepare_training_data(df)
        
        # 3. Time-aware split
        train_df, test_df = time_aware_split(df)
        
        # Prepare split data
        train_data = prepare_training_data(train_df)
        test_data = prepare_training_data(test_df)
        
        X_train = train_data['X']
        X_test = test_data['X']
        y_train_clf = train_data['y_classification']
        y_test_clf = test_data['y_classification']
        y_train_reg = train_data['y_regression']
        y_test_reg = test_data['y_regression']
        
        # 4. Train baseline models
        baseline_results = train_baseline_models(
            X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg
        )
        
        # 5. Train advanced models
        advanced_results = train_advanced_models(
            X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg
        )
        
        # 6. Compare model performance
        comparison = compare_model_performance(baseline_results, advanced_results)
        
        # 7. Evaluate performance with detailed metrics
        all_results = {**baseline_results, **advanced_results}
        evaluation = evaluate_model_performance(all_results, y_test_clf, y_test_reg, data_dict['feature_names'])
        
        # 8. Save models (traditional approach)
        save_advanced_models(baseline_results, advanced_results, evaluation, 
                           comparison, data_dict['feature_names'])
        
        # 9. Model persistence with validation (production approach)
        # Create validation split from training data for overfitting detection
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_clf_split, y_val_clf_split = train_test_split(
            X_train, y_train_clf, test_size=0.2, random_state=42, stratify=y_train_clf
        )
        _, _, y_train_reg_split, y_val_reg_split = train_test_split(
            X_train, y_train_reg, test_size=0.2, random_state=42
        )
        
        # Run model persistence pipeline
        all_results = {**baseline_results, **advanced_results}
        saved_model_ids = run_model_persistence_pipeline(
            all_results, X_train_split, y_train_clf_split, y_train_reg_split,
            X_val_split, y_val_clf_split, y_val_reg_split, data_dict['feature_names']
        )
        
        logger.info("Advanced model training pipeline completed successfully!")
        
        return {
            'baseline_results': baseline_results,
            'advanced_results': advanced_results,
            'comparison': comparison,
            'evaluation': evaluation,
            'feature_names': data_dict['feature_names'],
            'saved_model_ids': saved_model_ids
        }
        
    except Exception as e:
        logger.error(f"Error in advanced training pipeline: {str(e)}")
        raise

def run_baseline_training_pipeline(data_path: str = 'data/f1_features_engineered.csv') -> Dict[str, Any]:
    """
    Run the complete baseline model training pipeline.
    
    Args:
        data_path: Path to the engineered features dataset
        
    Returns:
        Dictionary containing all results
    """
    logger.info("Starting baseline model training pipeline...")
    
    try:
        # 1. Load data
        df = load_engineered_features(data_path)
        
        # 2. Prepare training data
        data_dict = prepare_training_data(df)
        
        # 3. Time-aware split
        train_df, test_df = time_aware_split(df)
        
        # Prepare split data
        train_data = prepare_training_data(train_df)
        test_data = prepare_training_data(test_df)
        
        X_train = train_data['X']
        X_test = test_data['X']
        y_train_clf = train_data['y_classification']
        y_test_clf = test_data['y_classification']
        y_train_reg = train_data['y_regression']
        y_test_reg = test_data['y_regression']
        
        # 4. Train baseline models
        results = train_baseline_models(
            X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg
        )
        
        # 5. Evaluate performance
        evaluation = evaluate_model_performance(results, y_test_clf, y_test_reg, data_dict['feature_names'])
        
        # 6. Save models
        save_baseline_models(results, evaluation, data_dict['feature_names'])
        
        logger.info("Baseline model training pipeline completed successfully!")
        
        return {
            'results': results,
            'evaluation': evaluation,
            'feature_names': data_dict['feature_names']
        }
        
    except Exception as e:
        logger.error(f"Error in baseline training pipeline: {str(e)}")
        raise

def save_advanced_models(baseline_results: Dict[str, Any], advanced_results: Dict[str, Any],
                        evaluation: Dict[str, Any], comparison: Dict[str, Any],
                        feature_names: List[str], output_dir: str = 'models') -> None:
    """
    Save trained advanced models and evaluation results.
    
    Args:
        baseline_results: Dictionary containing baseline model results
        advanced_results: Dictionary containing advanced model results
        evaluation: Dictionary containing evaluation results
        comparison: Dictionary containing model comparison
        feature_names: List of feature names
        output_dir: Directory to save models
    """
    logger.info("Saving advanced models...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all models
    all_results = {**baseline_results, **advanced_results}
    
    for model_name, model_data in all_results.items():
        if model_name != 'scaler' and isinstance(model_data, dict) and 'model' in model_data:
            model_path = os.path.join(output_dir, f'{model_name}_{timestamp}.joblib')
            joblib.dump(model_data['model'], model_path)
            logger.info(f"Saved {model_name}: {model_path}")
    
    # Save scaler
    if 'scaler' in advanced_results:
        scaler_path = os.path.join(output_dir, f'advanced_scaler_{timestamp}.joblib')
        joblib.dump(advanced_results['scaler'], scaler_path)
        logger.info(f"Saved scaler: {scaler_path}")
    
    # Save comprehensive metadata
    metadata = {
        'timestamp': timestamp,
        'feature_names': feature_names,
        'evaluation': evaluation,
        'comparison': comparison,
        'model_info': {
            'baseline_models': list(baseline_results.keys()),
            'advanced_models': list(advanced_results.keys()),
            'best_classification_model': comparison.get('best_models', {}).get('classification'),
            'best_regression_model': comparison.get('best_models', {}).get('regression'),
            'preprocessing': 'StandardScaler'
        }
    }
    
    metadata_path = os.path.join(output_dir, f'advanced_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved metadata: {metadata_path}")

if __name__ == "__main__":
    import sys
    
    # Check if advanced training is requested
    if len(sys.argv) > 1 and sys.argv[1] == 'advanced':
        # Run advanced training pipeline
        pipeline_results = run_advanced_training_pipeline()
        
        # Print summary
        print("\n" + "="*50)
        print("ADVANCED MODEL TRAINING SUMMARY")
        print("="*50)
        
        comparison = pipeline_results['comparison']
        
        if 'classification' in comparison:
            print(f"\nClassification Models:")
            for model_name, metrics in comparison['classification']['models'].items():
                print(f"  {model_name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
            print(f"  Best: {comparison['classification']['best_model']} (F1={comparison['classification']['best_f1']:.3f})")
        
        if 'regression' in comparison:
            print(f"\nRegression Models:")
            for model_name, metrics in comparison['regression']['models'].items():
                print(f"  {model_name}: MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")
            print(f"  Best: {comparison['regression']['best_model']} (MAE={comparison['regression']['best_mae']:.3f})")
        
        print(f"\nModels saved to: models/")
    
    else:
        # Run baseline training pipeline
        pipeline_results = run_baseline_training_pipeline()
    
    # Print summary
    print("\n" + "="*50)
    print("BASELINE MODEL TRAINING SUMMARY")
    print("="*50)
    
    evaluation = pipeline_results['evaluation']
    
    if 'classification' in evaluation:
        print(f"\nClassification (Position Drop Prediction):")
        print(f"  Model: {evaluation['classification']['model_type']}")
        print(f"  Accuracy: {evaluation['classification']['metrics']['accuracy']:.3f}")
        print(f"  F1 Score: {evaluation['classification']['metrics']['f1']:.3f}")
        print(f"  ROC AUC: {evaluation['classification']['metrics']['roc_auc']:.3f}")
    
    if 'regression' in evaluation:
        print(f"\nRegression (Position Change Prediction):")
        print(f"  Model: {evaluation['regression']['model_type']}")
        print(f"  MAE: {evaluation['regression']['metrics']['mae']:.3f}")
        print(f"  RMSE: {evaluation['regression']['metrics']['rmse']:.3f}")
        print(f"  R²: {evaluation['regression']['metrics']['r2']:.3f}")
    
    print(f"\nPerformance Requirements Met: {evaluation['summary'].get('meets_performance_requirements', 'N/A')}")
    print(f"Models saved to: models/ and models/production/")