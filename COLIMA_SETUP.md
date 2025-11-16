# Colima Setup Instructions for macOS

This guide provides step-by-step instructions for setting up and running the F1 Performance Drop Predictor using Colima on macOS.

## Prerequisites

- macOS (Intel or Apple Silicon)
- Homebrew package manager
- Terminal access

## Installation Steps

### 1. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Docker and Colima

```bash
# Install Docker CLI and Colima
brew install docker colima

# Optional: Install Docker Compose (if not included with Docker)
brew install docker-compose
```

### 3. Start Colima

```bash
# Start Colima with recommended settings for this project
colima start --cpu 4 --memory 8 --disk 60

# Verify Colima is running
colima status
```

**Note:** Adjust CPU, memory, and disk settings based on your system capabilities:
- Minimum: `--cpu 2 --memory 4 --disk 30`
- Recommended: `--cpu 4 --memory 8 --disk 60`
- High-performance: `--cpu 8 --memory 16 --disk 100`

### 4. Verify Docker Installation

```bash
# Test Docker connectivity
docker --version
docker info

# Test with a simple container
docker run hello-world
```

## Deployment Options

### Option 1: Simple Docker Run (Recommended for Testing)

```bash
# Navigate to project directory
cd /path/to/f1-performance-drop-predictor

# Build the Docker image
docker build -t f1-predictor .

# Run the container
docker run -d \
  --name f1-predictor \
  -p 8000:8000 \
  --restart unless-stopped \
  f1-predictor

# Check if container is running
docker ps

# View logs
docker logs f1-predictor

# Test the API
curl http://localhost:8000/health
```

### Option 2: Docker Compose (Recommended for Development)

```bash
# Navigate to project directory
cd /path/to/f1-performance-drop-predictor

# Start services with Docker Compose
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f f1-predictor

# Test the API
curl http://localhost:8000/health
```

### Option 3: Docker Compose with Nginx (Production-like Setup)

```bash
# Start with nginx reverse proxy
docker-compose --profile production up -d

# Check all services
docker-compose ps

# Test through nginx (port 80)
curl http://localhost/health

# Test direct API access (port 8000)
curl http://localhost:8000/health
```

## Container Management

### Starting and Stopping

```bash
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart services
docker-compose restart

# Stop Colima when done
colima stop
```

### Monitoring and Debugging

```bash
# View real-time logs
docker-compose logs -f

# Execute commands inside container
docker-compose exec f1-predictor bash

# Check container resource usage
docker stats

# Inspect container details
docker inspect f1-predictor
```

### Updating the Application

```bash
# Rebuild and restart after code changes
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Or use the shorthand
docker-compose up -d --build
```

## Testing the Deployment

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-11-16T...",
  "models_loaded": true,
  "uptime_seconds": 123.45
}
```

### 2. Model Information

```bash
curl http://localhost:8000/model_info
```

### 3. Simple Prediction Test

```bash
curl -X POST http://localhost:8000/predict \
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

## Troubleshooting

### Common Issues

1. **Colima won't start**
   ```bash
   # Check if another Docker runtime is running
   docker context ls
   
   # Reset Colima if needed
   colima delete
   colima start --cpu 4 --memory 8 --disk 60
   ```

2. **Port already in use**
   ```bash
   # Find what's using port 8000
   lsof -i :8000
   
   # Use a different port
   docker run -p 8001:8000 f1-predictor
   ```

3. **Container fails to start**
   ```bash
   # Check container logs
   docker logs f1-predictor
   
   # Run container interactively for debugging
   docker run -it --rm f1-predictor bash
   ```

4. **Models not loading**
   ```bash
   # Check if models directory is properly mounted
   docker exec f1-predictor ls -la models/
   
   # Verify model files exist
   docker exec f1-predictor find models/ -name "*.joblib"
   ```

5. **Memory issues**
   ```bash
   # Increase Colima memory allocation
   colima stop
   colima start --cpu 4 --memory 12 --disk 60
   ```

### Performance Optimization

1. **Faster builds**
   ```bash
   # Use BuildKit for faster builds
   export DOCKER_BUILDKIT=1
   docker build -t f1-predictor .
   ```

2. **Resource monitoring**
   ```bash
   # Monitor resource usage
   docker stats f1-predictor
   
   # Check Colima resource usage
   colima status
   ```

### Cleanup

```bash
# Remove all containers and images
docker-compose down -v --rmi all

# Clean up Docker system
docker system prune -a

# Stop and delete Colima VM
colima stop
colima delete
```

## Security Considerations

- The container runs as a non-root user for security
- Only necessary ports are exposed
- No sensitive data is included in the image
- Health checks ensure service availability
- Logs are properly configured and rotated

## Next Steps

After successful deployment:

1. Access the API documentation at `http://localhost:8000/docs`
2. Use the interactive API explorer at `http://localhost:8000/redoc`
3. Integrate with your applications using the REST API
4. Monitor performance using the `/health` endpoint
5. Scale horizontally by running multiple container instances

For production deployment, consider:
- Using a proper reverse proxy (nginx, traefik)
- Implementing authentication and rate limiting
- Setting up monitoring and alerting
- Using container orchestration (Kubernetes, Docker Swarm)
- Implementing CI/CD pipelines for automated deployments