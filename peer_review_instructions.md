# F1 Performance Drop Predictor - Peer Review Instructions

## Overview for Reviewers

You are reviewing a machine learning system that predicts Formula 1 finishing position drops using historical racing data. The system includes data processing, model training, and a containerized web API for real-time predictions.

**Expected Review Time**: 30-45 minutes  
**Technical Level**: Intermediate Python/ML knowledge helpful but not required  
**Prerequisites**: Docker/Colima, Python 3.9+, Terminal access

## 🎯 Three Critical Verification Checks

### ✅ **Check 1: Data Pipeline and Model Training Verification**

**Objective**: Verify that the complete ML pipeline runs successfully and produces valid models.

#### Commands to Execute:
```bash
# 1. Set up environment
conda env create -f environment.yml
conda activate f1-predictor

# 2. Run complete pipeline
python src/data_prep.py
python src/train.py
```

#### Expected Outputs:
- **Data Preparation**: Should create `data/f1_performance_drop_clean.csv` (~2100 rows)
- **Feature Engineering**: Should create `data/f1_features_engineered.csv` with 47+ features
- **Model Training**: Should create files in `models/production/` directory
- **Evaluation Report**: Should create `model_evaluation_report.json`

#### Success Criteria:
```bash
# Verify data files exist and have correct size
ls -la data/f1_performance_drop_clean.csv  # Should exist, ~500KB+
wc -l data/f1_performance_drop_clean.csv   # Should show ~2100 lines

# Verify model files exist
ls models/production/                       # Should contain model files
cat model_evaluation_report.json | grep "best_classification_accuracy"  # Should show >0.60

# Check feature count
python -c "import pandas as pd; df = pd.read_csv('data/f1_features_engineered.csv'); print(f'Features: {df.shape[1]}, Rows: {df.shape[0]}')"
# Expected: Features: 47+, Rows: ~2100
```

#### What to Look For:
- ✅ **PASS**: All files created, accuracy >60%, MAE <2.0, no error messages
- ❌ **FAIL**: Missing files, poor performance, or Python errors
- ⚠️ **INVESTIGATE**: Warnings about data quality (acceptable if pipeline completes)

---

### ✅ **Check 2: API Service and Prediction Functionality**

**Objective**: Verify that the web API starts correctly and returns valid predictions.

#### Commands to Execute:
```bash
# 1. Start the API service (in background)
python src/serve.py &
API_PID=$!

# 2. Wait for service to start
sleep 5

# 3. Test health endpoint
curl -s http://localhost:8000/health

# 4. Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "driver_id": 1,
    "constructor_id": 1,
    "grid_position": 5,
    "circuit_id": 1,
    "season": 2023,
    "driver_experience": 100,
    "recent_reliability": 0.85,
    "qualifying_gap": 0.5
  }'

# 5. Test model info endpoint
curl -s http://localhost:8000/model_info

# 6. Clean up
kill $API_PID
```

#### Expected Outputs:

**Health Check Response**:
```json
{"status": "healthy", "model_loaded": true, "timestamp": "..."}
```

**Prediction Response**:
```json
{
  "classification": {
    "will_drop_position": true,
    "probability": 0.73,
    "confidence": "high"
  },
  "regression": {
    "expected_position_change": 2.1,
    "prediction_interval": [1.2, 3.0]
  },
  "model_version": "..."
}
```

**Model Info Response**:
```json
{
  "model_type": "Random Forest",
  "accuracy": 0.775,
  "mae": 1.976,
  "feature_count": 47,
  "top_features": ["qualifying_gap_to_pole", "..."]
}
```

#### Success Criteria:
- ✅ **PASS**: All endpoints return valid JSON, predictions are reasonable numbers
- ❌ **FAIL**: Service won't start, endpoints return errors, or malformed responses
- ⚠️ **INVESTIGATE**: Slow response times (>5 seconds) or unusual prediction values

---

### ✅ **Check 3: Docker Containerization and Deployment**

**Objective**: Verify that the system can be containerized and deployed successfully.

#### Commands to Execute:
```bash
# 1. Build Docker image
docker build -t f1-predictor .

# 2. Run container
docker run -d -p 8000:8000 --name f1-test f1-predictor

# 3. Wait for container to start
sleep 10

# 4. Test containerized API
curl -s http://localhost:8000/health

# 5. Test prediction in container
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "driver_id": 44,
    "constructor_id": 1,
    "grid_position": 3,
    "circuit_id": 6,
    "season": 2023,
    "driver_experience": 150,
    "recent_reliability": 0.90,
    "qualifying_gap": 0.2
  }'

# 6. Check container logs
docker logs f1-test

# 7. Clean up
docker stop f1-test
docker rm f1-test
```

#### Expected Outputs:

**Docker Build**: Should complete without errors, final image ~500MB-1GB

**Container Health**: Same JSON response as Check 2

**Container Logs**: Should show:
```
INFO: Started server process
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

#### Success Criteria:
- ✅ **PASS**: Container builds, starts, and responds to API calls
- ❌ **FAIL**: Build errors, container crashes, or API unreachable
- ⚠️ **INVESTIGATE**: Very large image size (>2GB) or slow startup (>30 seconds)

---

## 📋 Review Checklist

### Code Quality Assessment
- [ ] **Documentation**: README.md is comprehensive and beginner-friendly
- [ ] **Code Structure**: Source files are well-organized in `src/` directory
- [ ] **Error Handling**: Scripts handle missing files and invalid inputs gracefully
- [ ] **Reproducibility**: Fixed random seeds and clear dependency specifications

### Technical Implementation
- [ ] **Data Processing**: Handles ~2100 race entries with appropriate cleaning
- [ ] **Feature Engineering**: Creates meaningful racing-related features (47+)
- [ ] **Model Performance**: Meets requirements (>60% accuracy, <2.0 MAE)
- [ ] **API Design**: RESTful endpoints with proper JSON input/output

### Deployment and Usability
- [ ] **Environment Setup**: Conda environment works on reviewer's system
- [ ] **Docker Integration**: Container builds and runs successfully
- [ ] **Testing**: Unit tests pass and cover key functionality
- [ ] **Documentation**: Clear instructions for setup and usage

## 🚨 Common Issues and Troubleshooting

### Issue 1: Conda Environment Problems
**Symptoms**: `conda: command not found` or package installation failures
**Solutions**:
```bash
# Install miniconda if needed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Force recreate environment
conda env remove -n f1-predictor
conda env create -f environment.yml --force
```

### Issue 2: Port Already in Use
**Symptoms**: API won't start, "Address already in use" error
**Solutions**:
```bash
# Kill existing processes on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python src/serve.py --port 8001
```

### Issue 3: Docker/Colima Issues (macOS)
**Symptoms**: Docker commands fail or container won't start
**Solutions**:
```bash
# Restart Colima
colima stop
colima start --cpu 2 --memory 4

# Clear Docker cache
docker system prune -f
```

### Issue 4: Missing Data Files
**Symptoms**: FileNotFoundError for CSV files
**Solutions**:
```bash
# Verify data directory
ls data/f1db-*.csv  # Should list multiple CSV files
# If missing, check if data needs to be downloaded separately
```

## 📊 Performance Benchmarks

### Expected Performance Ranges
- **Classification Accuracy**: 70-80% (target: >60%)
- **Regression MAE**: 1.8-2.2 positions (target: <2.0)
- **API Response Time**: <500ms for single predictions
- **Docker Build Time**: 2-5 minutes on modern hardware
- **Training Time**: 3-8 minutes for complete pipeline

### Red Flags (Investigate Further)
- Accuracy <50% or >95% (suggests data leakage or poor model)
- MAE >3.0 positions (poor regression performance)
- API responses >5 seconds (performance issues)
- Container >2GB (inefficient packaging)

## 📝 Review Report Template

```markdown
# F1 Performance Drop Predictor - Review Report

**Reviewer**: [Your Name]
**Date**: [Review Date]
**Review Duration**: [Time Spent]

## Verification Results

### Check 1: Data Pipeline ✅/❌
- Data files created: ✅/❌
- Model performance: Accuracy: X.XX%, MAE: X.XX
- Issues encountered: [None/List issues]

### Check 2: API Service ✅/❌
- Service startup: ✅/❌
- Prediction functionality: ✅/❌
- Response format: ✅/❌
- Issues encountered: [None/List issues]

### Check 3: Docker Deployment ✅/❌
- Container build: ✅/❌
- Container execution: ✅/❌
- API accessibility: ✅/❌
- Issues encountered: [None/List issues]

## Overall Assessment

**Technical Quality**: [Excellent/Good/Satisfactory/Needs Improvement]
**Documentation Quality**: [Excellent/Good/Satisfactory/Needs Improvement]
**Reproducibility**: [Excellent/Good/Satisfactory/Needs Improvement]

## Strengths
- [List 2-3 notable strengths]

## Areas for Improvement
- [List any issues or suggestions]

## Recommendation
- [ ] **Accept**: All checks pass, high quality implementation
- [ ] **Accept with Minor Issues**: Mostly works, small problems noted
- [ ] **Needs Revision**: Major issues that prevent proper functionality
```

## 🎯 Grading Rubric Alignment

This peer review process evaluates:

1. **Technical Implementation (40%)**
   - Data processing pipeline functionality
   - Model training and evaluation
   - Performance meeting requirements

2. **Software Engineering (30%)**
   - Code organization and quality
   - API design and functionality
   - Containerization and deployment

3. **Documentation and Reproducibility (20%)**
   - Clear setup instructions
   - Comprehensive documentation
   - Reproducible results

4. **Innovation and Insights (10%)**
   - Feature engineering creativity
   - Model interpretation and insights
   - Real-world applicability

---

**Happy Reviewing! 🏁**

*Remember: The goal is constructive feedback that helps improve the project while verifying it meets the technical requirements.*