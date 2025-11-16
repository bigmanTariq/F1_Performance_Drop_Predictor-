#!/bin/bash

# Container Testing Script for F1 Performance Drop Predictor
# This script tests the Docker container functionality

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="f1-predictor-test"
IMAGE_NAME="f1-predictor"
API_PORT="8000"
BASE_URL="http://localhost:${API_PORT}"
TIMEOUT=60

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    log_info "Cleaning up test container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

wait_for_service() {
    local url=$1
    local timeout=$2
    local counter=0
    
    log_info "Waiting for service to be ready at $url..."
    
    while [ $counter -lt $timeout ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            log_success "Service is ready!"
            return 0
        fi
        
        counter=$((counter + 1))
        sleep 1
        
        if [ $((counter % 10)) -eq 0 ]; then
            log_info "Still waiting... (${counter}s elapsed)"
        fi
    done
    
    log_error "Service failed to start within ${timeout} seconds"
    return 1
}

test_endpoint() {
    local endpoint=$1
    local method=${2:-GET}
    local data=${3:-}
    local expected_status=${4:-200}
    
    log_info "Testing $method $endpoint"
    
    if [ "$method" = "POST" ] && [ -n "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint")
    fi
    
    # Extract status code (last line)
    status_code=$(echo "$response" | tail -n1)
    # Extract body (all but last line)
    body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" = "$expected_status" ]; then
        log_success "$method $endpoint returned $status_code"
        return 0
    else
        log_error "$method $endpoint returned $status_code (expected $expected_status)"
        echo "Response body: $body"
        return 1
    fi
}

# Main testing function
main() {
    log_info "Starting F1 Performance Drop Predictor container tests"
    
    # Cleanup any existing test containers
    cleanup
    
    # Build the image
    log_info "Building Docker image..."
    if ! docker build -t $IMAGE_NAME .; then
        log_error "Failed to build Docker image"
        exit 1
    fi
    log_success "Docker image built successfully"
    
    # Run the container
    log_info "Starting container..."
    if ! docker run -d \
        --name $CONTAINER_NAME \
        -p $API_PORT:8000 \
        $IMAGE_NAME; then
        log_error "Failed to start container"
        exit 1
    fi
    log_success "Container started successfully"
    
    # Wait for service to be ready
    if ! wait_for_service "$BASE_URL/health" $TIMEOUT; then
        log_error "Service failed to start"
        docker logs $CONTAINER_NAME
        cleanup
        exit 1
    fi
    
    # Test basic endpoints
    log_info "Running API endpoint tests..."
    
    # Test root endpoint
    test_endpoint "/" || exit 1
    
    # Test health endpoint
    test_endpoint "/health" || exit 1
    
    # Test model info endpoint
    test_endpoint "/model_info" || exit 1
    
    # Test prediction endpoint with sample data
    sample_data='{
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
    
    test_endpoint "/predict" "POST" "$sample_data" || exit 1
    
    # Test batch prediction endpoint
    batch_data='{
        "scenarios": ['"$sample_data"']
    }'
    
    test_endpoint "/predict_batch" "POST" "$batch_data" || exit 1
    
    # Test error handling with invalid data
    log_info "Testing error handling..."
    invalid_data='{"championship_position": -1}'
    test_endpoint "/predict" "POST" "$invalid_data" "422" || exit 1
    
    # Test container health
    log_info "Checking container health..."
    health_status=$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null || echo "no-healthcheck")
    
    if [ "$health_status" = "healthy" ] || [ "$health_status" = "no-healthcheck" ]; then
        log_success "Container health check passed"
    else
        log_warning "Container health status: $health_status"
    fi
    
    # Test performance
    log_info "Running performance test..."
    start_time=$(date +%s%N)
    test_endpoint "/predict" "POST" "$sample_data" >/dev/null
    end_time=$(date +%s%N)
    
    duration_ms=$(( (end_time - start_time) / 1000000 ))
    log_info "Prediction response time: ${duration_ms}ms"
    
    if [ $duration_ms -lt 1000 ]; then
        log_success "Performance test passed (< 1000ms)"
    else
        log_warning "Performance test: response time ${duration_ms}ms (> 1000ms)"
    fi
    
    # Show container stats
    log_info "Container resource usage:"
    docker stats --no-stream $CONTAINER_NAME
    
    # Show container logs (last 20 lines)
    log_info "Recent container logs:"
    docker logs --tail 20 $CONTAINER_NAME
    
    # Cleanup
    cleanup
    
    log_success "All container tests passed!"
    log_info "Container is ready for deployment"
}

# Handle script interruption
trap cleanup EXIT INT TERM

# Run main function
main "$@"