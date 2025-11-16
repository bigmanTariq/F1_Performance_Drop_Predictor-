# 🚀 F1 Performance Drop Predictor - Deployment Checklist

## Pre-Deployment Checklist

### ✅ Code Preparation
- [ ] All code committed to GitHub
- [ ] `render.yaml` configuration file created
- [ ] `build.sh` script created and executable
- [ ] `runtime.txt` specifies Python version
- [ ] `Procfile` defines start command
- [ ] `requirements.txt` includes all dependencies
- [ ] Port configuration updated in `serve.py`

### ✅ Local Testing
- [ ] Run `python src/data_prep.py` successfully
- [ ] Run `python src/train.py` successfully  
- [ ] Run `python src/serve.py` and test locally
- [ ] Test API endpoints with curl/browser
- [ ] Verify web interface works at `/ui`

### ✅ Repository Setup
- [ ] Push all changes to GitHub main branch
- [ ] Repository is public (or Render has access)
- [ ] All deployment files are in root directory

## Deployment Steps

### 1. Render.com Setup
- [ ] Sign up at [render.com](https://render.com) with GitHub
- [ ] Click "New +" → "Web Service"
- [ ] Connect your GitHub repository
- [ ] Select your F1 predictor repo

### 2. Service Configuration
- [ ] **Name**: `f1-performance-predictor`
- [ ] **Runtime**: `Python 3`
- [ ] **Build Command**: `./build.sh`
- [ ] **Start Command**: `python src/serve.py`
- [ ] **Auto-Deploy**: Enabled

### 3. Deploy and Test
- [ ] Click "Create Web Service"
- [ ] Wait for build to complete (5-10 minutes)
- [ ] Test deployment with provided script:
  ```bash
  python scripts/test_deployment.py https://your-app-url.onrender.com
  ```

## Post-Deployment Verification

### ✅ API Endpoints
- [ ] Health check: `GET /health` returns 200
- [ ] Model info: `GET /model_info` returns model details
- [ ] Prediction: `POST /predict` returns valid predictions
- [ ] Web UI: `GET /ui` serves the interface
- [ ] API docs: `GET /docs` shows FastAPI documentation

### ✅ Performance Tests
- [ ] First request completes (may take 30s on free tier)
- [ ] Subsequent requests are fast (<2s)
- [ ] Predictions return reasonable values
- [ ] No error messages in logs

### ✅ Documentation Updates
- [ ] Add live URL to README.md
- [ ] Update project submission with deployment link
- [ ] Take screenshots of working web interface
- [ ] Save example API responses

## Quick Test Commands

Once deployed, run these to verify everything works:

```bash
# Replace YOUR_URL with your actual Render URL
export API_URL="https://your-app-name.onrender.com"

# Test health
curl $API_URL/health

# Test prediction
curl -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "championship_position": 5,
    "pit_stop_count": 2,
    "avg_pit_time": 30.0,
    "pit_time_std": 2.5,
    "circuit_length": 5.5,
    "points": 100,
    "pit_frequency": 2.0,
    "pit_duration_variance": 8.0,
    "high_pit_frequency": 1,
    "qualifying_gap_to_pole": 1.2,
    "grid_position_percentile": 0.6,
    "poor_qualifying": 0,
    "circuit_dnf_rate": 0.12,
    "is_street_circuit": 1,
    "championship_pressure": 0.4,
    "leader_points": 180,
    "points_gap_to_leader": 80,
    "points_pressure": 0.3,
    "driver_avg_grid_position": 9.2,
    "qualifying_vs_average": 1.8,
    "constructor_avg_grid_position": 8.5,
    "qualifying_vs_constructor_avg": 0.5,
    "bad_qualifying_day": 1,
    "circuit_dnf_rate_detailed": 0.12,
    "avg_pit_stops": 2.3,
    "avg_pit_duration": 29.5,
    "dnf_score": 0.15,
    "volatility_score": 0.4,
    "pit_complexity_score": 0.6,
    "track_difficulty_score": 0.5,
    "race_number": 15,
    "first_season": 2018,
    "seasons_active": 5,
    "estimated_age": 26,
    "driver_rolling_avg_grid": 8.8,
    "season_leader_points": 180,
    "points_gap_to_season_leader": 80,
    "is_championship_contender": 0,
    "points_momentum": -0.1,
    "championship_position_change": 2,
    "teammate_avg_grid": 10.1,
    "grid_vs_teammate": -0.9,
    "championship_pressure_score": 0.4,
    "max_round": 22,
    "late_season_race": 1,
    "championship_pressure_adjusted": 0.6,
    "points_per_race": 6.7
  }'

# Test web interface (open in browser)
open $API_URL/ui
```

## Troubleshooting

### Build Fails
- Check build logs in Render dashboard
- Verify all files are committed to GitHub
- Ensure `build.sh` is executable: `chmod +x build.sh`

### Service Won't Start
- Check service logs in Render dashboard
- Verify `PORT` environment variable usage
- Ensure models are created during build

### Slow Performance
- First request after sleep takes ~30 seconds (normal on free tier)
- Keep service warm by pinging health endpoint regularly

### API Errors
- Check that all required features are in prediction request
- Verify model files exist in `models/production/`
- Test locally first to isolate deployment issues

## Success Criteria

Your deployment is successful when:
- ✅ All 4 test endpoints return 200 status
- ✅ Predictions return reasonable probability values (0-1)
- ✅ Web interface loads and displays correctly
- ✅ API documentation is accessible at `/docs`
- ✅ No error messages in service logs

## For Your Submission

Include these in your project submission:
1. **Live URL**: `https://your-app-name.onrender.com`
2. **API Documentation**: `https://your-app-name.onrender.com/docs`
3. **Web Interface**: `https://your-app-name.onrender.com/ui`
4. **Test Results**: Output from `test_deployment.py`
5. **Screenshots**: Working web interface and API responses

---

**🏁 Congratulations! Your F1 ML system is now live on the internet!**