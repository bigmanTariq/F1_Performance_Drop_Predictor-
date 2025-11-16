# Requirements Document

## Introduction

This project builds a machine learning system to predict Formula 1 finishing position drops due to car stress and mechanical factors. Using comprehensive historical F1 data (~2100 race entries), we'll create both classification and regression models to help teams understand when cars are likely to underperform relative to their qualifying position. The system will be deployed as a containerized web service suitable for real-time predictions.

## Requirements

### Requirement 1: Data Pipeline and Feature Engineering

**User Story:** As a data scientist, I want a robust data pipeline that transforms raw F1 historical data into ML-ready features, so that I can build reliable predictive models.

#### Acceptance Criteria

1. WHEN the data preparation script runs THEN the system SHALL merge race results, qualifying positions, pit stops, driver info, and circuit data into a unified dataset
2. WHEN feature engineering executes THEN the system SHALL create stress-related features including pit stop frequency, qualifying-to-finish position delta, driver experience metrics, and team reliability scores
3. WHEN the pipeline completes THEN the system SHALL output a clean CSV with approximately 2100 rows and all necessary features for modeling
4. IF any data quality issues exist THEN the system SHALL log warnings and handle missing values appropriately
5. WHEN the dataset is created THEN the system SHALL include both classification target (position_drop_flag) and regression target (position_change_numeric)

### Requirement 2: Machine Learning Model Development

**User Story:** As an ML engineer, I want to train and evaluate multiple model types for both classification and regression tasks, so that I can select the best performing approach for production.

#### Acceptance Criteria

1. WHEN model training begins THEN the system SHALL implement baseline models (logistic regression for classification, linear regression for regression)
2. WHEN advanced modeling runs THEN the system SHALL train decision trees, random forests, and gradient boosting models
3. WHEN model evaluation executes THEN the system SHALL use appropriate metrics (accuracy, precision, recall, F1 for classification; MAE, MSE, R² for regression)
4. WHEN model selection occurs THEN the system SHALL choose the best model based on cross-validation performance and save it for deployment
5. WHEN training completes THEN the system SHALL generate feature importance analysis and model interpretation outputs

### Requirement 3: Prediction Service and API

**User Story:** As a team strategist, I want a web API that accepts race parameters and returns position drop predictions, so that I can make real-time strategic decisions during race weekends.

#### Acceptance Criteria

1. WHEN the prediction service starts THEN the system SHALL load the trained model and expose a REST API endpoint
2. WHEN a prediction request is received THEN the system SHALL validate input parameters and return both classification and regression predictions
3. WHEN batch predictions are requested THEN the system SHALL handle multiple race scenarios efficiently
4. WHEN invalid input is provided THEN the system SHALL return appropriate error messages with guidance
5. WHEN the service runs THEN the system SHALL be accessible via HTTP POST requests with JSON payloads

### Requirement 4: Containerized Deployment

**User Story:** As a DevOps engineer, I want the entire ML system packaged in a Docker container, so that it can be deployed consistently across different environments.

#### Acceptance Criteria

1. WHEN the Docker container builds THEN the system SHALL include all dependencies, trained models, and the prediction service
2. WHEN the container runs THEN the system SHALL expose the API on a configurable port
3. WHEN using Colima on macOS THEN the system SHALL provide clear instructions for local deployment
4. WHEN the container starts THEN the system SHALL perform health checks and log startup status
5. WHEN deployment completes THEN the system SHALL be testable via curl or Python requests

### Requirement 5: Documentation and Reproducibility

**User Story:** As a course grader, I want comprehensive documentation that allows me to understand, reproduce, and evaluate the entire project, so that I can assess the student's work effectively.

#### Acceptance Criteria

1. WHEN documentation is reviewed THEN the system SHALL include a detailed README with setup instructions for beginners
2. WHEN the grader writeup is read THEN it SHALL provide an engaging technical narrative explaining the problem, approach, and results
3. WHEN reproduction is attempted THEN the system SHALL include exact conda environment specifications and step-by-step commands
4. WHEN peer review occurs THEN the system SHALL include verification instructions with 3 specific checks to validate functionality
5. WHEN the project is evaluated THEN all code SHALL be well-commented and follow Python best practices

### Requirement 6: Model Performance and Validation

**User Story:** As a machine learning practitioner, I want robust model validation that demonstrates the system's reliability and limitations, so that I can trust the predictions for decision-making.

#### Acceptance Criteria

1. WHEN model validation runs THEN the system SHALL use time-aware train/test splits to prevent data leakage
2. WHEN performance is measured THEN the system SHALL achieve reasonable baseline performance (>60% accuracy for classification, <2.0 MAE for regression)
3. WHEN overfitting is checked THEN the system SHALL compare training and validation metrics and report any significant gaps
4. WHEN model interpretation is performed THEN the system SHALL identify the most important features and validate they make domain sense
5. WHEN limitations are documented THEN the system SHALL clearly state assumptions, potential biases, and appropriate use cases