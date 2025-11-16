# F1 Performance Drop Predictor - Deployment Troubleshooting

## Common Issues and Solutions

### 1. Models Not Loading on Render

**Symptoms:**
- Health endpoint returns `"models_loaded": false`
- Prediction endpoints return 503 errors
- Server starts but can't make predictions

**Causes & Solutions:**

#### A. Build Timeout
- **Cause:** Model training takes too long during build (>15 minutes on free tier)
- **Solution:** Models will train on first startup instead
- **Check:** Look for "Model training timed out" in build logs

#### B. Memory Limits
- **Cause:** Free tier has 512MB RAM limit, model training uses more
- **Solution:** Reduced model complexity, added memory limits
- **Check:** Look for OOM (Out of Memory) errors in logs

#### C. Missing Data Files
- **Cause:** Data files not included in deployment
- **Solution:** All data files are committed to git
- **Check:** Verify `data/` directory exists with CSV files

### 2. Server Won't Start

**Symptoms:**
- Render shows "Deploy failed" or "Service unavailable"
- No response from health endpoint

**Solutions:**
1. Check Render logs for specific error messages
2. Verify `start.sh` has execute permissions
3. Ensure `PORT` environment variable is set correctly

### 3. Slow Response Times

**Symptoms:**
- Health endpoint takes >30 seconds to respond
- Prediction requests timeout

**Solutions:**
1. Models are training on first request (expected)
2. Subsequent requests should be fast (<2 seconds)
3. Free tier has limited CPU, some slowness expected

### 4. Prediction Errors

**Symptoms:**
- 400 errors on prediction requests
- "Feature validation failed" messages

**Solutions:**
1. Check input data matches expected feature schema
2. Verify all 47 features are provided
3. Check feature value ranges (see API docs)

## Debugging Steps

### 1. Check Health Endpoint
```bash
curl https://f1-performance-drop-predictor.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "uptime_seconds": 120.5
}
```

### 2. Check Model Info
```bash
curl https://f1-performance-drop-predictor.onrender.com/model_info
```

### 3. Test Simple Prediction
```bash
curl -X POST https://f1-performance-drop-predictor.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_prediction.json
```

### 4. Check Render Logs
1. Go to Render dashboard
2. Select your service
3. Click "Logs" tab
4. Look for error messages during startup

## Local Testing

Before deploying, test locally:

```bash
# Run the test script
python test_local_deployment.py

# Or manually test
PORT=8002 python src/serve.py &
curl http://localhost:8002/health
```

## Performance Expectations

### Free Tier Limitations:
- **Memory:** 512MB RAM
- **CPU:** Shared, limited
- **Build Time:** 15 minutes max
- **Cold Starts:** 30+ seconds after inactivity

### Expected Performance:
- **First Request:** 30-60 seconds (model loading)
- **Subsequent Requests:** 1-3 seconds
- **Health Check:** <5 seconds

## Render Configuration

### Environment Variables:
- `PORT`: Set automatically by Render
- `PYTHON_VERSION`: 3.9.18
- `RENDER_DEPLOYMENT`: "true"

### Build Process:
1. Install dependencies (`pip install -r requirements.txt`)
2. Run data preparation (if needed)
3. Train models (if needed, with timeout)
4. Start server with uvicorn

### Startup Process:
1. Check for existing models
2. Train models if missing (with timeout)
3. Load models into memory
4. Start accepting requests

## Getting Help

If issues persist:

1. **Check Render Status:** https://status.render.com/
2. **Review Logs:** Look for specific error messages
3. **Test Locally:** Use `test_local_deployment.py`
4. **Reduce Load:** Try simpler prediction requests first

## Model Training Details

The system uses these models:
- **Classification:** Random Forest (predicts if position will drop)
- **Regression:** Random Forest (predicts position change amount)
- **Features:** 47 engineered features from F1 data
- **Training Time:** 5-10 minutes locally, may timeout on Render

If models fail to train during build, they will train on first API request, which may take 1-2 minutes.