#!/bin/bash

# Comprehensive curl examples for F1 Performance Drop Predictor API
# This script demonstrates all API endpoints with various scenarios

# Configuration
BASE_URL="http://localhost:8000"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}F1 Performance Drop Predictor API - curl Examples${NC}"
echo "=================================================="
echo ""

# Helper function to print section headers
print_section() {
    echo -e "\n${GREEN}$1${NC}"
    echo "$(printf '=%.0s' {1..50})"
}

# Helper function to print commands
print_command() {
    echo -e "\n${YELLOW}Command:${NC}"
    echo "$1"
    echo -e "\n${YELLOW}Response:${NC}"
}

print_section "1. Health Check"
cmd="curl -X GET \"$BASE_URL/health\" -H \"accept: application/json\""
print_command "$cmd"
eval $cmd
echo ""

print_section "2. Root Endpoint"
cmd="curl -X GET \"$BASE_URL/\" -H \"accept: application/json\""
print_command "$cmd"
eval $cmd
echo ""

print_section "3. Model Information"
cmd="curl -X GET \"$BASE_URL/model_info\" -H \"accept: application/json\""
print_command "$cmd"
eval $cmd
echo ""

print_section "4. Single Prediction - Typical Race Scenario"
cmd="curl -X POST \"$BASE_URL/predict\" \\
  -H \"accept: application/json\" \\
  -H \"Content-Type: application/json\" \\
  -d '{
    \"championship_position\": 3,
    \"pit_stop_count\": 2,
    \"avg_pit_time\": 28.5,
    \"pit_time_std\": 3.2,
    \"circuit_length\": 5.4,
    \"points\": 150,
    \"pit_frequency\": 1.8,
    \"pit_duration_variance\": 10.2,
    \"high_pit_frequency\": 0,
    \"qualifying_gap_to_pole\": 0.8,
    \"grid_position_percentile\": 0.75,
    \"poor_qualifying\": 0,
    \"circuit_dnf_rate\": 0.15,
    \"is_street_circuit\": 0,
    \"championship_pressure\": 0.6,
    \"leader_points\": 200,
    \"points_gap_to_leader\": 50,
    \"points_pressure\": 0.4,
    \"driver_avg_grid_position\": 8.5,
    \"qualifying_vs_average\": -1.5,
    \"constructor_avg_grid_position\": 7.2,
    \"qualifying_vs_constructor_avg\": -2.2,
    \"bad_qualifying_day\": 0,
    \"circuit_dnf_rate_detailed\": 0.15,
    \"avg_pit_stops\": 2.1,
    \"avg_pit_duration\": 28.0,
    \"dnf_score\": 0.1,
    \"volatility_score\": 0.3,
    \"pit_complexity_score\": 0.5,
    \"track_difficulty_score\": 0.4,
    \"race_number\": 10,
    \"first_season\": 2015,
    \"seasons_active\": 8,
    \"estimated_age\": 28,
    \"driver_rolling_avg_grid\": 8.2,
    \"season_leader_points\": 200,
    \"points_gap_to_season_leader\": 50,
    \"is_championship_contender\": 1,
    \"points_momentum\": 0.2,
    \"championship_position_change\": 0,
    \"teammate_avg_grid\": 9.1,
    \"grid_vs_teammate\": -0.6,
    \"championship_pressure_score\": 0.6,
    \"max_round\": 22,
    \"late_season_race\": 0,
    \"championship_pressure_adjusted\": 0.5,
    \"points_per_race\": 15.0
  }'"
print_command "$cmd"
eval $cmd
echo ""

print_section "5. Single Prediction - High Stress Scenario"
cmd="curl -X POST \"$BASE_URL/predict\" \\
  -H \"accept: application/json\" \\
  -H \"Content-Type: application/json\" \\
  -d '{
    \"championship_position\": 1,
    \"pit_stop_count\": 4,
    \"avg_pit_time\": 35.2,
    \"pit_time_std\": 8.5,
    \"circuit_length\": 3.2,
    \"points\": 250,
    \"pit_frequency\": 3.5,
    \"pit_duration_variance\": 25.0,
    \"high_pit_frequency\": 1,
    \"qualifying_gap_to_pole\": 0.0,
    \"grid_position_percentile\": 1.0,
    \"poor_qualifying\": 0,
    \"circuit_dnf_rate\": 0.25,
    \"is_street_circuit\": 1,
    \"championship_pressure\": 0.9,
    \"leader_points\": 250,
    \"points_gap_to_leader\": 0,
    \"points_pressure\": 0.8,
    \"driver_avg_grid_position\": 3.2,
    \"qualifying_vs_average\": -3.2,
    \"constructor_avg_grid_position\": 2.8,
    \"qualifying_vs_constructor_avg\": -2.8,
    \"bad_qualifying_day\": 0,
    \"circuit_dnf_rate_detailed\": 0.25,
    \"avg_pit_stops\": 3.8,
    \"avg_pit_duration\": 34.5,
    \"dnf_score\": 0.3,
    \"volatility_score\": 0.7,
    \"pit_complexity_score\": 0.8,
    \"track_difficulty_score\": 0.9,
    \"race_number\": 20,
    \"first_season\": 2010,
    \"seasons_active\": 13,
    \"estimated_age\": 32,
    \"driver_rolling_avg_grid\": 2.8,
    \"season_leader_points\": 250,
    \"points_gap_to_season_leader\": 0,
    \"is_championship_contender\": 1,
    \"points_momentum\": 0.8,
    \"championship_position_change\": 0,
    \"teammate_avg_grid\": 4.1,
    \"grid_vs_teammate\": -4.1,
    \"championship_pressure_score\": 0.9,
    \"max_round\": 22,
    \"late_season_race\": 1,
    \"championship_pressure_adjusted\": 0.95,
    \"points_per_race\": 12.5
  }'"
print_command "$cmd"
eval $cmd
echo ""

print_section "6. Single Prediction - Low Stress Scenario"
cmd="curl -X POST \"$BASE_URL/predict\" \\
  -H \"accept: application/json\" \\
  -H \"Content-Type: application/json\" \\
  -d '{
    \"championship_position\": 15,
    \"pit_stop_count\": 1,
    \"avg_pit_time\": 22.8,
    \"pit_time_std\": 1.2,
    \"circuit_length\": 6.8,
    \"points\": 5,
    \"pit_frequency\": 1.0,
    \"pit_duration_variance\": 2.5,
    \"high_pit_frequency\": 0,
    \"qualifying_gap_to_pole\": 3.5,
    \"grid_position_percentile\": 0.25,
    \"poor_qualifying\": 1,
    \"circuit_dnf_rate\": 0.08,
    \"is_street_circuit\": 0,
    \"championship_pressure\": 0.1,
    \"leader_points\": 200,
    \"points_gap_to_leader\": 195,
    \"points_pressure\": 0.05,
    \"driver_avg_grid_position\": 16.2,
    \"qualifying_vs_average\": 1.8,
    \"constructor_avg_grid_position\": 15.8,
    \"qualifying_vs_constructor_avg\": 2.2,
    \"bad_qualifying_day\": 1,
    \"circuit_dnf_rate_detailed\": 0.08,
    \"avg_pit_stops\": 1.2,
    \"avg_pit_duration\": 23.5,
    \"dnf_score\": 0.05,
    \"volatility_score\": 0.2,
    \"pit_complexity_score\": 0.2,
    \"track_difficulty_score\": 0.3,
    \"race_number\": 5,
    \"first_season\": 2020,
    \"seasons_active\": 3,
    \"estimated_age\": 24,
    \"driver_rolling_avg_grid\": 16.8,
    \"season_leader_points\": 200,
    \"points_gap_to_season_leader\": 195,
    \"is_championship_contender\": 0,
    \"points_momentum\": -0.1,
    \"championship_position_change\": 2,
    \"teammate_avg_grid\": 14.5,
    \"grid_vs_teammate\": 3.5,
    \"championship_pressure_score\": 0.1,
    \"max_round\": 22,
    \"late_season_race\": 0,
    \"championship_pressure_adjusted\": 0.05,
    \"points_per_race\": 1.0
  }'"
print_command "$cmd"
eval $cmd
echo ""

print_section "7. Batch Prediction - Multiple Scenarios"
cmd="curl -X POST \"$BASE_URL/predict_batch\" \\
  -H \"accept: application/json\" \\
  -H \"Content-Type: application/json\" \\
  -d '{
    \"scenarios\": [
      {
        \"championship_position\": 3,
        \"pit_stop_count\": 2,
        \"avg_pit_time\": 28.5,
        \"pit_time_std\": 3.2,
        \"circuit_length\": 5.4,
        \"points\": 150,
        \"pit_frequency\": 1.8,
        \"pit_duration_variance\": 10.2,
        \"high_pit_frequency\": 0,
        \"qualifying_gap_to_pole\": 0.8,
        \"grid_position_percentile\": 0.75,
        \"poor_qualifying\": 0,
        \"circuit_dnf_rate\": 0.15,
        \"is_street_circuit\": 0,
        \"championship_pressure\": 0.6,
        \"leader_points\": 200,
        \"points_gap_to_leader\": 50,
        \"points_pressure\": 0.4,
        \"driver_avg_grid_position\": 8.5,
        \"qualifying_vs_average\": -1.5,
        \"constructor_avg_grid_position\": 7.2,
        \"qualifying_vs_constructor_avg\": -2.2,
        \"bad_qualifying_day\": 0,
        \"circuit_dnf_rate_detailed\": 0.15,
        \"avg_pit_stops\": 2.1,
        \"avg_pit_duration\": 28.0,
        \"dnf_score\": 0.1,
        \"volatility_score\": 0.3,
        \"pit_complexity_score\": 0.5,
        \"track_difficulty_score\": 0.4,
        \"race_number\": 10,
        \"first_season\": 2015,
        \"seasons_active\": 8,
        \"estimated_age\": 28,
        \"driver_rolling_avg_grid\": 8.2,
        \"season_leader_points\": 200,
        \"points_gap_to_season_leader\": 50,
        \"is_championship_contender\": 1,
        \"points_momentum\": 0.2,
        \"championship_position_change\": 0,
        \"teammate_avg_grid\": 9.1,
        \"grid_vs_teammate\": -0.6,
        \"championship_pressure_score\": 0.6,
        \"max_round\": 22,
        \"late_season_race\": 0,
        \"championship_pressure_adjusted\": 0.5,
        \"points_per_race\": 15.0
      },
      {
        \"championship_position\": 8,
        \"pit_stop_count\": 3,
        \"avg_pit_time\": 31.2,
        \"pit_time_std\": 5.1,
        \"circuit_length\": 4.2,
        \"points\": 85,
        \"pit_frequency\": 2.5,
        \"pit_duration_variance\": 18.5,
        \"high_pit_frequency\": 1,
        \"qualifying_gap_to_pole\": 1.8,
        \"grid_position_percentile\": 0.45,
        \"poor_qualifying\": 1,
        \"circuit_dnf_rate\": 0.20,
        \"is_street_circuit\": 1,
        \"championship_pressure\": 0.3,
        \"leader_points\": 200,
        \"points_gap_to_leader\": 115,
        \"points_pressure\": 0.2,
        \"driver_avg_grid_position\": 12.1,
        \"qualifying_vs_average\": 0.9,
        \"constructor_avg_grid_position\": 11.5,
        \"qualifying_vs_constructor_avg\": 1.5,
        \"bad_qualifying_day\": 1,
        \"circuit_dnf_rate_detailed\": 0.20,
        \"avg_pit_stops\": 2.8,
        \"avg_pit_duration\": 30.5,
        \"dnf_score\": 0.2,
        \"volatility_score\": 0.5,
        \"pit_complexity_score\": 0.7,
        \"track_difficulty_score\": 0.6,
        \"race_number\": 15,
        \"first_season\": 2018,
        \"seasons_active\": 5,
        \"estimated_age\": 26,
        \"driver_rolling_avg_grid\": 11.8,
        \"season_leader_points\": 200,
        \"points_gap_to_season_leader\": 115,
        \"is_championship_contender\": 0,
        \"points_momentum\": -0.1,
        \"championship_position_change\": 1,
        \"teammate_avg_grid\": 13.2,
        \"grid_vs_teammate\": -0.1,
        \"championship_pressure_score\": 0.3,
        \"max_round\": 22,
        \"late_season_race\": 1,
        \"championship_pressure_adjusted\": 0.25,
        \"points_per_race\": 5.7
      }
    ]
  }'"
print_command "$cmd"
eval $cmd
echo ""

print_section "8. Error Handling - Invalid Data"
cmd="curl -X POST \"$BASE_URL/predict\" \\
  -H \"accept: application/json\" \\
  -H \"Content-Type: application/json\" \\
  -d '{
    \"championship_position\": -1,
    \"pit_stop_count\": 15,
    \"avg_pit_time\": 200.0
  }'"
print_command "$cmd"
eval $cmd
echo ""

print_section "9. Error Handling - Missing Required Fields"
cmd="curl -X POST \"$BASE_URL/predict\" \\
  -H \"accept: application/json\" \\
  -H \"Content-Type: application/json\" \\
  -d '{
    \"championship_position\": 5
  }'"
print_command "$cmd"
eval $cmd
echo ""

print_section "10. Performance Test - Measure Response Time"
echo "Testing response time for 5 consecutive requests..."
for i in {1..5}; do
    echo "Request $i:"
    time curl -s -X POST "$BASE_URL/predict" \
      -H "accept: application/json" \
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
      }' > /dev/null
    echo ""
done

echo -e "\n${GREEN}All curl examples completed!${NC}"
echo "For interactive API documentation, visit: $BASE_URL/docs"
echo "For alternative documentation, visit: $BASE_URL/redoc"