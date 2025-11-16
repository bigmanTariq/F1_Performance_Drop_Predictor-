# 🏎️ F1 Performance Drop Predictor

A machine learning system that predicts Formula 1 finishing position drops due to car stress and mechanical factors. This project analyzes historical F1 data to help teams understand when cars are likely to underperform relative to their qualifying position.

![F1 Predictor](https://img.shields.io/badge/F1-Performance%20Predictor-red?style=for-the-badge&logo=formula1)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Web%20API-green?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)

## ✨ Features

- 🎯 **Accurate Predictions**: ML models trained on 8,200+ race records
- 🌐 **Beautiful Web UI**: Modern, responsive interface with F1 theming  
- 🚀 **Fast API**: Sub-100ms prediction response times
- 📊 **Rich Visualizations**: Probability gauges and feature importance charts
- 🐳 **Docker Ready**: Complete containerization for easy deployment
- 📈 **Multiple Models**: Classification and regression predictions
- 🔧 **Comprehensive Testing**: End-to-end validation and quality assurance

## 🎮 Quick Start

### Option 1: Web UI (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/f1-performance-drop-predictor.git
cd f1-performance-drop-predictor

# 2. Set up environment
python -m venv f1_env
source f1_env/bin/activate  # On Windows: f1_env\Scripts\activate
pip install -r requirements.txt

# 3. Get F1 data (see data/README.md for sources)
# Place CSV files in the data/ directory

# 4. Run the complete pipeline
python src/data_prep.py      # Prepare data
python src/features.py       # Engineer features  
python src/train.py          # Train models

# 5. Start the web interface
python src/serve.py

# 6. Open your browser to http://localhost:8000/ui
```

### Option 2: Docker (One Command)
```bash
docker-compose up --build
# Then visit http://localhost:8000/ui
```

### Option 3: Cloud Deployment (Render.com - Free!)
```bash
# Push to GitHub, then deploy to Render.com
# See DEPLOYMENT.md for detailed instructions
# Live demo: https://your-app-url.onrender.com
```

## 🏎️ Project Overview

This system combines comprehensive F1 datasets, advanced feature engineering, and multiple ML models to predict:
- **Classification**: Will a car drop positions from qualifying to finish?
- **Regression**: How many positions will a car drop/gain?

The trained models are deployed as a containerized web service for real-time predictions.

## 📋 Prerequisites

Before getting started, ensure you have:
- **Python 3.9+** (recommended via Conda)
- **Git** for cloning the repository
- **Docker** and **Colima** (for macOS deployment)
- At least **4GB RAM** and **2GB disk space**

## 🚀 Quick Start

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd f1-performance-drop-predictor
```

### Step 2: Set Up Conda Environment

#### For Beginners: Installing Conda
If you don't have Conda installed:
1. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your operating system
2. Follow the installation instructions
3. Restart your terminal

#### Create and Activate Environment
```bash
# Create the environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate f1-predictor

# Verify installation
python --version  # Should show Python 3.9.x
```

### Step 3: Verify Data Files
Ensure you have the F1 dataset files in the `data/` directory:
```bash
ls data/
# Should show files like f1db-races.csv, f1db-drivers.csv, etc.
```

## 🔧 Running the Pipeline

### Option A: Run Complete Pipeline (Recommended for First Time)
```bash
# Run the complete pipeline step by step
python src/data_prep.py      # Prepare and clean data
python src/train.py          # Train and evaluate models
python src/serve.py          # Start the web service
```

### Option B: Run Individual Components

#### 1. Data Preparation
```bash
python src/data_prep.py
# Output: Creates data/f1_performance_drop_clean.csv and data/f1_features_engineered.csv
```

#### 2. Model Training
```bash
python src/train.py
# Output: Saves best models to models/ directory and creates evaluation report
```

#### 3. Start Web Service
```bash
python src/serve.py
# Service will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

## 🐳 Docker Deployment

### Prerequisites for Docker
1. **Install Docker Desktop** or **Colima** (for macOS)
2. For macOS users, see [COLIMA_SETUP.md](COLIMA_SETUP.md) for detailed instructions

### Build and Run Container
```bash
# Build the Docker image
docker build -t f1-predictor .

# Run the container
docker run -p 8000:8000 f1-predictor
```

### Using Docker Compose (Recommended)
```bash
# Start the service
docker-compose up --build

# Stop the service
docker-compose down
```

## 🧪 Testing the System

### 1. Run Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v
```

### 2. Test the API
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
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
```

### 3. Run End-to-End Tests
```bash
# Test complete deployment
bash scripts/e2e_deployment_test.sh

# Validate deployment
python scripts/validate_deployment.py
```

## 📊 Understanding the Output

### Model Performance Metrics
After training, check `model_evaluation_report.json` for:
- **Classification Accuracy**: >60% target
- **Regression MAE**: <2.0 positions target
- **Feature Importance**: Top predictive factors

### API Response Format
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
  "model_version": "decision_tree_20231116"
}
```

## 📁 Project Structure
```
f1-performance-drop-predictor/
├── data/                    # F1 dataset CSV files
├── src/                     # Source code
│   ├── data_prep.py        # Data preparation pipeline
│   ├── features.py         # Feature engineering
│   ├── train.py            # Model training
│   ├── predict.py          # Prediction functions
│   └── serve.py            # Web service API
├── models/                  # Trained models and metadata
├── tests/                   # Unit and integration tests
├── scripts/                 # Deployment and testing scripts
├── examples/                # Usage examples
├── docs/                    # Additional documentation
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment
├── Dockerfile              # Container configuration
└── docker-compose.yml      # Multi-container setup
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Conda Environment Issues
**Problem**: `conda: command not found`
```bash
# Solution: Add conda to PATH or restart terminal after installation
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc  # or ~/.zshrc
```

**Problem**: Environment creation fails
```bash
# Solution: Update conda and try again
conda update conda
conda env create -f environment.yml --force
```

#### 2. Data Loading Issues
**Problem**: `FileNotFoundError: data/f1db-races.csv`
```bash
# Solution: Ensure all CSV files are in the data/ directory
ls data/f1db-*.csv  # Should list all required files
```

**Problem**: Memory errors during data processing
```bash
# Solution: Increase available memory or use data sampling
# Edit src/data_prep.py and add .sample(frac=0.5) to large DataFrames
```

#### 3. Model Training Issues
**Problem**: Training takes too long (>10 minutes)
```bash
# Solution: Reduce hyperparameter search space
# Edit src/train.py and reduce param_grid size
```

**Problem**: Poor model performance
```bash
# Solution: Check data quality and feature engineering
python src/evaluate_models.py  # Run detailed evaluation
```

#### 4. API Service Issues
**Problem**: `Port 8000 already in use`
```bash
# Solution: Use different port or kill existing process
lsof -ti:8000 | xargs kill -9  # Kill process on port 8000
python src/serve.py --port 8001  # Use different port
```

**Problem**: API returns 500 errors
```bash
# Solution: Check model files exist and are valid
ls models/production/  # Should contain model files
python -c "import src.predict; src.predict.load_model()"  # Test model loading
```

#### 5. Docker Issues
**Problem**: Docker build fails
```bash
# Solution: Clear Docker cache and rebuild
docker system prune -f
docker build --no-cache -t f1-predictor .
```

**Problem**: Container won't start
```bash
# Solution: Check logs and port conflicts
docker logs <container-id>
docker ps -a  # Check container status
```

#### 6. macOS Specific Issues
**Problem**: Colima not working
```bash
# Solution: Restart Colima service
colima stop
colima start --cpu 2 --memory 4
```

**Problem**: Permission denied errors
```bash
# Solution: Fix file permissions
chmod +x scripts/*.sh
```

### Getting Help

1. **Check logs**: Most scripts output detailed logs to help diagnose issues
2. **Run tests**: Use `pytest tests/ -v` to identify specific problems
3. **Validate environment**: Run `python -c "import pandas, sklearn, fastapi; print('All imports successful')"`
4. **Check system resources**: Ensure sufficient RAM and disk space

### Performance Optimization

#### For Faster Training
```bash
# Use fewer hyperparameter combinations
export QUICK_TRAIN=1
python src/train.py
```

#### For Lower Memory Usage
```bash
# Use data sampling
export SAMPLE_DATA=0.5  # Use 50% of data
python src/data_prep.py
```

## 📈 Next Steps

After successful setup:
1. **Explore the data**: Open `notebooks/01_data_pull.ipynb` in Jupyter
2. **Experiment with features**: Modify `src/features.py` to add new predictors
3. **Try different models**: Add new algorithms in `src/train.py`
4. **Customize API**: Extend endpoints in `src/serve.py`

## 🌐 Cloud Deployment (Bonus Points!)

### Quick Deploy to Render.com (Free, No Credit Card)

1. **Prepare for deployment:**
   ```bash
   python deploy_prep.py  # Check if everything is ready
   ```

2. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

3. **Deploy to Render.com:**
   - Go to [render.com](https://render.com) and sign up with GitHub
   - Create new "Web Service" and connect your repository
   - Use these settings:
     - **Build Command**: `./build.sh`
     - **Start Command**: `python src/serve.py`
   - Click "Create Web Service" and wait 5-10 minutes

4. **Test your deployment:**
   ```bash
   python scripts/test_deployment.py https://your-app-url.onrender.com
   ```

**📖 Detailed Instructions**: See [DEPLOYMENT.md](DEPLOYMENT.md) for complete guide

### Live Demo Example
- **API**: https://f1-performance-predictor.onrender.com
- **Web Interface**: https://f1-performance-predictor.onrender.com/ui
- **API Docs**: https://f1-performance-predictor.onrender.com/docs

*Note: First request may take 30 seconds due to free tier cold start.*

## 📝 Additional Resources

- **API Documentation**: http://localhost:8000/docs (when service is running)
- **Example Usage**: See `examples/` directory for curl and Python examples
- **Model Details**: Check `models/production/` for model metadata
- **Test Examples**: Review `tests/` for usage patterns
- **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Deployment Checklist**: [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)

---

**Happy Predicting! 🏁**