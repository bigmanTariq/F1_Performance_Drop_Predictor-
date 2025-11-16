#!/usr/bin/env python3
"""
Full Pipeline Validation Script for F1 Performance Drop Predictor

This script validates the complete end-to-end pipeline:
1. Data preparation from raw CSV files
2. Feature engineering and model training
3. Model evaluation and persistence
4. API service deployment and testing
5. Reproducibility verification with fixed random seeds
"""

import os
import sys
import json
import time
import subprocess
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineValidator:
    """Validates the complete F1 prediction pipeline"""
    
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
            "requirements.txt",
            "Dockerfile"
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
        
        logger.info("Prerequisites validation passed")
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
            
            # Check data size (should be approximately 2100 rows)
            if df.shape[0] < 1500 or df.shape[0] > 3000:
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
    
    def test_prediction_module(self):
        """Test the prediction module directly"""
        logger.info("Testing prediction module...")
        
        try:
            # Import and test prediction functions
            sys.path.append(str(self.src_dir))
            from predict import load_models, predict_single, predict_batch
            
            # Load models
            models = load_models()
            if not models:
                logger.error("Failed to load models")
                return False
            
            # Test single prediction
            test_data = {
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
            
            prediction = predict_single(test_data)
            if not prediction or 'classification' not in prediction:
                logger.error("Single prediction failed")
                return False
            
            # Test batch prediction
            batch_prediction = predict_batch([test_data, test_data])
            if not batch_prediction or 'predictions' not in batch_prediction:
                logger.error("Batch prediction failed")
                return False
            
            self.validation_results['prediction_module'] = {
                'status': 'passed',
                'single_prediction': prediction,
                'batch_size': len(batch_prediction['predictions'])
            }
            
            logger.info("Prediction module validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Prediction module validation failed: {e}")
            return False
    
    def test_api_service(self):
        """Test the API service"""
        logger.info("Testing API service...")
        
        try:
            # Start the API service in background
            api_process = subprocess.Popen([
                sys.executable, "src/serve.py"
            ], cwd=self.project_root)
            
            # Wait for service to start
            time.sleep(10)
            
            base_url = "http://localhost:8000"
            
            # Test health endpoint
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.status_code != 200:
                logger.error(f"Health endpoint failed: {response.status_code}")
                api_process.terminate()
                return False
            
            # Test model info endpoint
            response = requests.get(f"{base_url}/model_info", timeout=10)
            if response.status_code != 200:
                logger.error(f"Model info endpoint failed: {response.status_code}")
                api_process.terminate()
                return False
            
            # Test prediction endpoint
            test_data = {
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
            
            response = requests.post(f"{base_url}/predict", json=test_data, timeout=10)
            if response.status_code != 200:
                logger.error(f"Prediction endpoint failed: {response.status_code}")
                api_process.terminate()
                return False
            
            prediction_result = response.json()
            if 'classification' not in prediction_result:
                logger.error("Invalid prediction response format")
                api_process.terminate()
                return False
            
            # Test batch prediction
            batch_data = {"scenarios": [test_data]}
            response = requests.post(f"{base_url}/predict_batch", json=batch_data, timeout=10)
            if response.status_code != 200:
                logger.error(f"Batch prediction endpoint failed: {response.status_code}")
                api_process.terminate()
                return False
            
            # Terminate API service
            api_process.terminate()
            api_process.wait()
            
            self.validation_results['api_service'] = {
                'status': 'passed',
                'prediction_response': prediction_result
            }
            
            logger.info("API service validation passed")
            return True
            
        except Exception as e:
            logger.error(f"API service validation failed: {e}")
            if 'api_process' in locals():
                api_process.terminate()
            return False
    
    def test_reproducibility(self):
        """Test reproducibility with fixed random seeds"""
        logger.info("Testing reproducibility...")
        
        try:
            # Run data preparation twice with same seed
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
            
            # Test model training reproducibility (quick test)
            sys.path.append(str(self.src_dir))
            from predict import load_models, predict_single
            
            test_data = {
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
            
            # Make multiple predictions to check consistency
            predictions = []
            for _ in range(3):
                pred = predict_single(test_data)
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
    
    def run_docker_validation(self):
        """Run Docker container validation"""
        logger.info("Running Docker container validation...")
        
        try:
            # Run the existing e2e deployment test
            result = subprocess.run([
                "bash", "scripts/e2e_deployment_test.sh"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.error(f"Docker validation failed: {result.stderr}")
                return False
            
            self.validation_results['docker'] = {
                'status': 'passed',
                'output': result.stdout[-500:]  # Last 500 chars
            }
            
            logger.info("Docker validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Docker validation failed: {e}")
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
        report_file = f"pipeline_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE VALIDATION SUMMARY")
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
    
    def run_full_validation(self):
        """Run complete pipeline validation"""
        logger.info("Starting full pipeline validation...")
        
        validation_steps = [
            ("Prerequisites", self.validate_prerequisites),
            ("Data Preparation", self.run_data_preparation),
            ("Model Training", self.run_model_training),
            ("Prediction Module", self.test_prediction_module),
            ("API Service", self.test_api_service),
            ("Reproducibility", self.test_reproducibility),
            ("Docker Container", self.run_docker_validation)
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
            logger.info("🎉 Full pipeline validation PASSED!")
            return True
        else:
            logger.error("❌ Full pipeline validation FAILED!")
            return False

def main():
    """Main execution function"""
    validator = PipelineValidator()
    success = validator.run_full_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()