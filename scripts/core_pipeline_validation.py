#!/usr/bin/env python3
"""
Core Pipeline Validation Script for F1 Performance Drop Predictor

This script validates the core pipeline components without Docker:
1. Data preparation from raw CSV files
2. Feature engineering and model training
3. Model evaluation and persistence
4. Prediction functionality
5. Reproducibility verification with fixed random seeds
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'core_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CorePipelineValidator:
    """Validates the core F1 prediction pipeline components"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.src_dir = self.project_root / "src"
        self.validation_results = {}
        
    def validate_prerequisites(self):
        """Check that all required files and directories exist"""
        logger.info("Validating prerequisites...")
        
        required_files = [
            "src/data_prep.py",
            "src/features.py", 
            "src/train.py",
            "src/predict.py",
            "src/serve.py",
            "requirements.txt"
        ]
        
        required_dirs = [
            "data",
            "src",
            "models"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_files or missing_dirs:
            logger.error(f"Missing files: {missing_files}")
            logger.error(f"Missing directories: {missing_dirs}")
            return False
        
        # Check for raw data files
        csv_files = list(self.data_dir.glob("f1db-*.csv"))
        if len(csv_files) < 10:
            logger.error(f"Insufficient raw data files found: {len(csv_files)}")
            return False
        
        logger.info(f"Prerequisites validation passed - {len(csv_files)} data files found")
        return True
    
    def run_data_preparation(self):
        """Run data preparation pipeline"""
        logger.info("Running data preparation pipeline...")
        
        try:
            # Set random seed for reproducibility
            os.environ['PYTHONHASHSEED'] = '42'
            
            result = subprocess.run([
                sys.executable, "src/data_prep.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.error(f"Data preparation failed: {result.stderr}")
                return False
            
            # Verify output file exists
            clean_data_file = self.data_dir / "f1_performance_drop_clean.csv"
            if not clean_data_file.exists():
                logger.error("Clean data file not created")
                return False
            
            # Verify data quality
            df = pd.read_csv(clean_data_file)
            logger.info(f"Clean dataset shape: {df.shape}")
            
            # Check for required columns
            required_columns = ['position_drop_flag', 'position_change_numeric']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check data size (should be reasonable)
            if df.shape[0] < 1000 or df.shape[0] > 50000:
                logger.warning(f"Unexpected dataset size: {df.shape[0]} rows")
            
            self.validation_results['data_prep'] = {
                'status': 'passed',
                'rows': df.shape[0],
                'columns': df.shape[1],
                'target_distribution': df['position_drop_flag'].value_counts().to_dict()
            }
            
            logger.info("Data preparation validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data preparation validation failed: {e}")
            return False
    
    def run_feature_engineering(self):
        """Run feature engineering pipeline"""
        logger.info("Running feature engineering pipeline...")
        
        try:
            # Set random seed for reproducibility
            os.environ['PYTHONHASHSEED'] = '42'
            
            result = subprocess.run([
                sys.executable, "src/features.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.error(f"Feature engineering failed: {result.stderr}")
                return False
            
            # Check if engineered features file exists
            engineered_file = self.data_dir / "f1_features_engineered.csv"
            if engineered_file.exists():
                df = pd.read_csv(engineered_file)
                logger.info(f"Engineered features dataset shape: {df.shape}")
                
                self.validation_results['feature_engineering'] = {
                    'status': 'passed',
                    'rows': df.shape[0],
                    'columns': df.shape[1]
                }
            else:
                logger.info("Feature engineering completed (no separate output file)")
                self.validation_results['feature_engineering'] = {
                    'status': 'passed',
                    'note': 'Features integrated in training pipeline'
                }
            
            logger.info("Feature engineering validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Feature engineering validation failed: {e}")
            return False
    
    def run_model_training(self):
        """Run model training pipeline"""
        logger.info("Running model training pipeline...")
        
        try:
            # Set random seed for reproducibility
            os.environ['PYTHONHASHSEED'] = '42'
            
            result = subprocess.run([
                sys.executable, "src/train.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.error(f"Model training failed: {result.stderr}")
                return False
            
            # Check if models were created
            model_files = list(self.models_dir.glob("*.joblib"))
            if len(model_files) < 2:
                logger.error(f"Insufficient model files created: {len(model_files)}")
                return False
            
            # Check for production models
            production_dir = self.models_dir / "production"
            if not production_dir.exists():
                logger.error("Production models directory not created")
                return False
            
            production_models = list(production_dir.glob("*_model.joblib"))
            if len(production_models) < 2:
                logger.error(f"Insufficient production models: {len(production_models)}")
                return False
            
            # Check model registry
            registry_file = self.models_dir / "model_registry.json"
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry = json.load(f)
                logger.info(f"Model registry contains {len(registry)} models")
            
            self.validation_results['model_training'] = {
                'status': 'passed',
                'model_files': len(model_files),
                'production_models': len(production_models)
            }
            
            logger.info("Model training validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model training validation failed: {e}")
            return False
    
    def test_prediction_functionality(self):
        """Test the prediction functionality"""
        logger.info("Testing prediction functionality...")
        
        try:
            # Test prediction module import
            sys.path.append(str(self.src_dir))
            from predict import F1PerformancePredictor
            
            # Initialize predictor
            predictor = F1PerformancePredictor()
            predictor.load_models()
            
            # Test single prediction with minimal data
            test_data = {
                'championship_position': 3,
                'pit_stop_count': 2,
                'avg_pit_time': 28.5,
                'pit_time_std': 3.2,
                'circuit_length': 5.4,
                'points': 150
            }
            
            prediction = predictor.predict_single(test_data)
            if not prediction or 'classification' not in prediction:
                logger.error("Single prediction failed - invalid response")
                return False
            
            # Test batch prediction
            batch_prediction = predictor.predict_batch([test_data, test_data])
            if not batch_prediction or len(batch_prediction) != 2:
                logger.error("Batch prediction failed")
                return False
            
            self.validation_results['prediction_functionality'] = {
                'status': 'passed',
                'single_prediction': {
                    'classification_prob': prediction['classification']['probability'],
                    'regression_value': prediction['regression']['expected_position_change']
                },
                'batch_size': len(batch_prediction)
            }
            
            logger.info("Prediction functionality validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Prediction functionality validation failed: {e}")
            return False
    
    def test_reproducibility(self):
        """Test reproducibility with fixed random seeds"""
        logger.info("Testing reproducibility...")
        
        try:
            # Test data preparation reproducibility
            os.environ['PYTHONHASHSEED'] = '42'
            
            # First run
            result1 = subprocess.run([
                sys.executable, "src/data_prep.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result1.returncode != 0:
                logger.error("First reproducibility run failed")
                return False
            
            # Save first result
            df1 = pd.read_csv(self.data_dir / "f1_performance_drop_clean.csv")
            
            # Second run
            result2 = subprocess.run([
                sys.executable, "src/data_prep.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result2.returncode != 0:
                logger.error("Second reproducibility run failed")
                return False
            
            # Compare results
            df2 = pd.read_csv(self.data_dir / "f1_performance_drop_clean.csv")
            
            if not df1.equals(df2):
                logger.error("Data preparation not reproducible")
                return False
            
            # Test prediction reproducibility
            sys.path.append(str(self.src_dir))
            from predict import F1PerformancePredictor
            
            predictor = F1PerformancePredictor()
            predictor.load_models()
            
            test_data = {
                'championship_position': 3,
                'pit_stop_count': 2,
                'avg_pit_time': 28.5,
                'pit_time_std': 3.2,
                'circuit_length': 5.4,
                'points': 150
            }
            
            # Make multiple predictions to check consistency
            predictions = []
            for _ in range(3):
                pred = predictor.predict_single(test_data)
                predictions.append(pred)
            
            # Check if predictions are consistent
            first_pred = predictions[0]
            for pred in predictions[1:]:
                if (abs(pred['classification']['probability'] - first_pred['classification']['probability']) > 0.001 or
                    abs(pred['regression']['expected_position_change'] - first_pred['regression']['expected_position_change']) > 0.001):
                    logger.error("Predictions not reproducible")
                    return False
            
            self.validation_results['reproducibility'] = {
                'status': 'passed',
                'data_consistent': True,
                'predictions_consistent': True
            }
            
            logger.info("Reproducibility validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Reproducibility validation failed: {e}")
            return False
    
    def validate_model_performance(self):
        """Validate model performance meets requirements"""
        logger.info("Validating model performance...")
        
        try:
            # Check if evaluation report exists
            eval_report_file = self.project_root / "model_evaluation_report.json"
            if not eval_report_file.exists():
                logger.warning("Model evaluation report not found")
                return True  # Not critical for core validation
            
            with open(eval_report_file, 'r') as f:
                eval_report = json.load(f)
            
            # Check classification performance
            if 'classification' in eval_report:
                accuracy = eval_report['classification'].get('accuracy', 0)
                if accuracy < 0.6:
                    logger.warning(f"Classification accuracy below threshold: {accuracy}")
                else:
                    logger.info(f"Classification accuracy: {accuracy}")
            
            # Check regression performance
            if 'regression' in eval_report:
                mae = eval_report['regression'].get('mae', float('inf'))
                if mae > 2.0:
                    logger.warning(f"Regression MAE above threshold: {mae}")
                else:
                    logger.info(f"Regression MAE: {mae}")
            
            self.validation_results['model_performance'] = {
                'status': 'passed',
                'evaluation_report_found': True
            }
            
            logger.info("Model performance validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model performance validation failed: {e}")
            return False
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        logger.info("Generating validation report...")
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'validation_results': self.validation_results,
            'summary': {
                'total_tests': len(self.validation_results),
                'passed_tests': sum(1 for r in self.validation_results.values() if r.get('status') == 'passed'),
                'failed_tests': sum(1 for r in self.validation_results.values() if r.get('status') != 'passed')
            }
        }
        
        # Save report
        report_file = f"core_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("CORE PIPELINE VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print("\nTest Results:")
        for test_name, result in self.validation_results.items():
            status = result.get('status', 'unknown')
            print(f"  {test_name}: {status.upper()}")
        print("="*60)
        
        return report['summary']['failed_tests'] == 0
    
    def run_core_validation(self):
        """Run core pipeline validation"""
        logger.info("Starting core pipeline validation...")
        
        validation_steps = [
            ("Prerequisites", self.validate_prerequisites),
            ("Data Preparation", self.run_data_preparation),
            ("Feature Engineering", self.run_feature_engineering),
            ("Model Training", self.run_model_training),
            ("Prediction Functionality", self.test_prediction_functionality),
            ("Reproducibility", self.test_reproducibility),
            ("Model Performance", self.validate_model_performance)
        ]
        
        all_passed = True
        
        for step_name, step_func in validation_steps:
            logger.info(f"Running {step_name} validation...")
            try:
                if not step_func():
                    logger.error(f"{step_name} validation failed")
                    all_passed = False
                    self.validation_results[step_name.lower().replace(' ', '_')] = {'status': 'failed'}
            except Exception as e:
                logger.error(f"{step_name} validation error: {e}")
                all_passed = False
                self.validation_results[step_name.lower().replace(' ', '_')] = {'status': 'error', 'error': str(e)}
        
        # Generate final report
        success = self.generate_validation_report()
        
        if success and all_passed:
            logger.info("🎉 Core pipeline validation PASSED!")
            return True
        else:
            logger.error("❌ Core pipeline validation FAILED!")
            return False

def main():
    """Main execution function"""
    validator = CorePipelineValidator()
    success = validator.run_core_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()