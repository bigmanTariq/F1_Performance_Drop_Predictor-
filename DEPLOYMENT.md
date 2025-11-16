# 🚀 F1 Performance Drop Predictor - Cloud Deployment Guide

## Deploy to Render.com (Free, No Credit Card Required)

### Prerequisites
- GitHub account
- Your F1 project code pushed to a GitHub repository

### Step-by-Step Deployment

#### 1. Prepare Your Repository
Make sure your project is pushed to GitHub with all the deployment files:
```bash
git add .
git commit -m "Add Render.com deployment configuration"
git push origin main
```

#### 2. Sign Up for Render.com
1. Go to [render.com](https://render.com)
2. Click "Get Started for Free"
3. Sign up with your GitHub account (no credit card needed)

#### 3. Create a New Web Service
1. Click "New +" in the Render dashboard
2. Select "Web Service"
3. Connect your GitHub repository
4. Select your F1 predictor repository

#### 4. Configure the Service
Fill in these settings:

**Basic Settings:**
- **Name**: `f1-performance-predictor`
- **Region**: Choose closest to you
- **Branch**: `main`
- **Runtime**: `Python 3`

**Build & Deploy Settings:**
- **Build Command**: `./build.sh`
- **Start Command**: `python src/serve.py`

**Advanced Settings:**
- **Auto-Deploy**: Yes (recommended)

#### 5. Deploy!
1. Click "Create Web Service"
2. Wait 5-10 minutes for the build to complete
3. Your app will be live at: `https://f1-performance-predictor-[random].onrender.com`

### 🎯 Testing Your Deployment

Once deployed, test these endpoints:

**Health Check:**
```bash
curl https://your-app-url.onrender.com/health
```

**Make a Prediction:**
```bash
curl -X POST "https://your-app-url.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "championship_position": 3,
    "pit_stop_count": 2,
    "avg_pit_time": 28.5,
    "pit_time_std": 3.2,
    "circuit_length": 5.4,
    "points": 150,
    "pit_frequency": 1.8,
    "pit_duration_variance": 10.2,
    "high_pit_frequency": 0,
    "qualifying_gap_to_pole": 0.8,
    "grid_position_percentile": 0.75,
    "poor_qualifying": 0,
    "circuit_dnf_rate": 0.15,
    "is_street_circuit": 0,
    "championship_pressure": 0.6,
    "leader_points": 200,
    "points_gap_to_leader": 50,
    "points_pressure": 0.4,
    "driver_avg_grid_position": 8.5,
    "qualifying_vs_average": -1.5,
    "constructor_avg_grid_position": 7.2,
    "qualifying_vs_constructor_avg": -2.2,
    "bad_qualifying_day": 0,
    "circuit_dnf_rate_detailed": 0.15,
    "avg_pit_stops": 2.1,
    "avg_pit_duration": 28.0,
    "dnf_score": 0.1,
    "volatility_score": 0.3,
    "pit_complexity_score": 0.5,
    "track_difficulty_score": 0.4,
    "race_number": 10,
    "first_season": 2015,
    "seasons_active": 8,
    "estimated_age": 28,
    "driver_rolling_avg_grid": 8.2,
    "season_leader_points": 200,
    "points_gap_to_season_leader": 50,
    "is_championship_contender": 1,
    "points_momentum": 0.2,
    "championship_position_change": 0,
    "teammate_avg_grid": 9.1,
    "grid_vs_teammate": -0.6,
    "championship_pressure_score": 0.6,
    "max_round": 22,
    "late_season_race": 0,
    "championship_pressure_adjusted": 0.5,
    "points_per_race": 15.0
  }'
```

**Web Interface:**
Visit: `https://your-app-url.onrender.com/ui`

### 🔧 Troubleshooting

#### Build Fails
- Check the build logs in Render dashboard
- Ensure all files are committed to GitHub
- Verify `requirements.txt` has all dependencies

#### App Won't Start
- Check the service logs in Render dashboard
- Ensure `PORT` environment variable is being used
- Verify models are being created during build

#### Slow Response Times
- First request after inactivity takes ~30 seconds (free tier limitation)
- Subsequent requests should be fast (<1 second)

### 🎁 Free Tier Limitations
- **Sleep after 15 minutes** of inactivity
- **750 hours/month** of runtime (plenty for a demo)
- **Slower cold starts** (~30 seconds)
- **Limited bandwidth** (sufficient for testing)

### 🚀 Alternative Free Platforms

If Render doesn't work, try these (all free, no credit card):

#### Railway.app
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

#### Fly.io
```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch
fly deploy
```

#### Heroku (requires credit card for verification, but free tier)
```bash
# Install Heroku CLI
# Create Heroku app
heroku create f1-performance-predictor

# Deploy
git push heroku main
```

### 📝 Adding to Your README

Add this section to your main README.md:

```markdown
## 🌐 Live Demo

**Live API**: https://your-app-url.onrender.com
**Web Interface**: https://your-app-url.onrender.com/ui
**API Documentation**: https://your-app-url.onrender.com/docs

### Quick Test
```bash
curl https://your-app-url.onrender.com/health
```

*Note: First request may take 30 seconds due to free tier cold start.*
```

### 🎯 For Your Submission

Include this in your project submission:
1. **Live URL** of your deployed API
2. **Screenshot** of the web interface working
3. **API response** from a test prediction
4. **Deployment documentation** (this file)

This demonstrates:
- ✅ **Cloud deployment** (bonus points!)
- ✅ **Production readiness**
- ✅ **Real-world usability**
- ✅ **Professional presentation**

---

**Congratulations! Your F1 ML system is now live on the internet! 🏁**