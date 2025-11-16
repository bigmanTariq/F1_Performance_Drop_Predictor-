#!/bin/bash

# End-to-End Deployment Test for F1 Performance Drop Predictor
# This script tests the complete deployment pipeline from build to API testing

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="f1-performance-predictor"
IMAGE_NAME="f1-predictor"
CONTAINER_NAME="f1-predictor-e2e"
API_PORT="8000"
BASE_URL="http://localhost:${API_PORT}"
TIMEOUT=120
TEST_DURATION=30

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

cleanup() {
    log_info "Cleaning up test environment..."
    
    # Stop and remove containers
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    
    # Clean up docker-compose if running
    docker-compose down 2>/dev/null || true
    
    log_info "Cleanup completed"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker/Colima first."
        exit 1
    fi
    
    # Check if required files exist
    required_files=("Dockerfile" "requirements.txt" "docker-compose.yml" "src/serve.py")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    # Check if models directory exists
    if [ ! -d "models" ]; then
        log_error "Models directory not found. Please train models first."
        exit 1
    fi
    
    # Check if production models exist
    if [ ! -d "models/production" ] || [ -z "$(ls -A models/production 2>/dev/null)" ]; then
        log_error "Production models not found. Please train and deploy models first."
        exit 1
    fi
    
    log_success "All prerequisites met"
}

test_docker_build() {
    log_info "Testing Docker build process..."
    
    # Build with no cache to ensure clean build
    if docker build --no-cache -t $IMAGE_NAME .; then
        log_success "Docker build completed successfully"
    else
        log_error "Docker build failed"
        exit 1
    fi
    
    # Check image size
    image_size=$(docker images $IMAGE_NAME --format "table {{.Size}}" | tail -n 1)
    log_info "Built image size: $image_size"
    
    # Check image layers
    layer_count=$(docker history $IMAGE_NAME --quiet | wc -l)
    log_info "Image layers: $layer_count"
}

test_container_startup() {
    log_info "Testing container startup..."
    
    # Start container
    if docker run -d \
        --name $CONTAINER_NAME \
        -p $API_PORT:8000 \
        --health-cmd="curl -f http://localhost:8000/health || exit 1" \
        --health-interval=10s \
        --health-timeout=5s \
        --health-retries=3 \
        --health-start-period=30s \
        $IMAGE_NAME; then
        log_success "Container started successfully"
    else
        log_error "Failed to start container"
        exit 1
    fi
    
    # Wait for container to be healthy
    log_info "Waiting for container to become healthy..."
    for i in $(seq 1 $TIMEOUT); do
        health_status=$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null || echo "starting")
        
        if [ "$health_status" = "healthy" ]; then
            log_success "Container is healthy"
            break
        elif [ "$health_status" = "unhealthy" ]; then
            log_error "Container became unhealthy"
            docker logs $CONTAINER_NAME
            exit 1
        fi
        
        if [ $((i % 10)) -eq 0 ]; then
            log_info "Container status: $health_status (${i}s elapsed)"
        fi
        
        sleep 1
    done
    
    if [ "$health_status" != "healthy" ]; then
        log_error "Container failed to become healthy within ${TIMEOUT} seconds"
        docker logs $CONTAINER_NAME
        exit 1
    fi
}

test_api_endpoints() {
    log_info "Testing API endpoints..."
    
    # Test root endpoint
    if curl -s -f "$BASE_URL/" >/dev/null; then
        log_success "Root endpoint accessible"
    else
        log_error "Root endpoint failed"
        return 1
    fi
    
    # Test health endpoint
    health_response=$(curl -s "$BASE_URL/health")
    if echo "$health_response" | grep -q '"status":"healthy"'; then
        log_success "Health endpoint reports healthy"
    else
        log_error "Health endpoint reports unhealthy: $health_response"
        return 1
    fi
    
    # Test model info endpoint
    if curl -s -f "$BASE_URL/model_info" >/dev/null; then
        log_success "Model info endpoint accessible"
    else
        log_error "Model info endpoint failed"
        return 1
    fi
    
    # Test prediction endpoint
    prediction_data='{
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
    
    prediction_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$prediction_data" \
        "$BASE_URL/predict")
    
    if echo "$prediction_response" | grep -q '"classification"'; then
        log_success "Prediction endpoint working"
    else
        log_error "Prediction endpoint failed: $prediction_response"
        return 1
    fi
    
    # Test batch prediction
    batch_data='{"scenarios": ['"$prediction_data"']}'
    
    batch_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$batch_data" \
        "$BASE_URL/predict_batch")
    
    if echo "$batch_response" | grep -q '"predictions"'; then
        log_success "Batch prediction endpoint working"
    else
        log_error "Batch prediction endpoint failed: $batch_response"
        return 1
    fi
}

test_performance_load() {
    log_info "Running performance and load tests..."
    
    # Test response times
    log_info "Testing response times..."
    total_time=0
    num_requests=10
    
    for i in $(seq 1 $num_requests); do
        start_time=$(date +%s%N)
        
        curl -s -X POST \
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
            }' \
            "$BASE_URL/predict" >/dev/null
        
        end_time=$(date +%s%N)
        request_time=$(( (end_time - start_time) / 1000000 ))
        total_time=$((total_time + request_time))
        
        if [ $((i % 5)) -eq 0 ]; then
            log_info "Completed $i/$num_requests requests"
        fi
    done
    
    avg_time=$((total_time / num_requests))
    log_info "Average response time: ${avg_time}ms"
    
    if [ $avg_time -lt 1000 ]; then
        log_success "Performance test passed (< 1000ms average)"
    else
        log_warning "Performance test: average ${avg_time}ms (> 1000ms)"
    fi
}

test_error_handling() {
    log_info "Testing error handling..."
    
    # Test with invalid data
    invalid_response=$(curl -s -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d '{"championship_position": -1}' \
        "$BASE_URL/predict")
    
    status_code=$(echo "$invalid_response" | tail -c 4)
    
    if [ "$status_code" = "422" ] || [ "$status_code" = "400" ]; then
        log_success "Error handling working correctly"
    else
        log_error "Error handling failed - got status $status_code"
        return 1
    fi
}

test_docker_compose() {
    log_info "Testing Docker Compose deployment..."
    
    # Stop any running containers first
    cleanup
    
    # Start with docker-compose
    if docker-compose up -d; then
        log_success "Docker Compose started successfully"
    else
        log_error "Docker Compose failed to start"
        return 1
    fi
    
    # Wait for service
    log_info "Waiting for Docker Compose service..."
    for i in $(seq 1 60); do
        if curl -s -f "$BASE_URL/health" >/dev/null 2>&1; then
            log_success "Docker Compose service is ready"
            break
        fi
        sleep 1
    done
    
    # Quick API test
    if curl -s -f "$BASE_URL/health" >/dev/null; then
        log_success "Docker Compose API test passed"
    else
        log_error "Docker Compose API test failed"
        return 1
    fi
    
    # Stop docker-compose
    docker-compose down
}

generate_test_report() {
    log_info "Generating test report..."
    
    report_file="deployment_test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "F1 Performance Drop Predictor - Deployment Test Report"
        echo "Generated: $(date)"
        echo "======================================================="
        echo ""
        echo "Container Information:"
        docker inspect $CONTAINER_NAME --format='{{json .}}' | jq -r '
            "Image: " + .Config.Image,
            "Created: " + .Created,
            "Status: " + .State.Status,
            "Health: " + .State.Health.Status,
            "Ports: " + (.NetworkSettings.Ports | tostring)
        ' 2>/dev/null || echo "Container info not available"
        echo ""
        echo "Resource Usage:"
        docker stats --no-stream $CONTAINER_NAME 2>/dev/null || echo "Stats not available"
        echo ""
        echo "Recent Logs:"
        docker logs --tail 20 $CONTAINER_NAME 2>/dev/null || echo "Logs not available"
    } > "$report_file"
    
    log_success "Test report saved to: $report_file"
}

# Main execution
main() {
    log_info "Starting End-to-End Deployment Test for F1 Performance Drop Predictor"
    
    # Set up cleanup trap
    trap cleanup EXIT INT TERM
    
    # Run test phases
    check_prerequisites
    test_docker_build
    test_container_startup
    test_api_endpoints
    test_performance_load
    test_error_handling
    
    # Generate report before cleanup
    generate_test_report
    
    # Test docker-compose (this will cleanup the single container)
    test_docker_compose
    
    log_success "All end-to-end deployment tests passed!"
    log_info "The F1 Performance Drop Predictor is ready for production deployment"
    
    return 0
}

# Run main function
main "$@"