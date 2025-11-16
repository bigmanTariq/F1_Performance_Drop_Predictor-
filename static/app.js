// F1 Performance Drop Predictor - JavaScript Application

class F1PredictorApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentScenario = 'custom';
        this.allFeatures = {};
        
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.checkApiStatus();
        this.loadScenario('custom');
        this.populateAdvancedFields();
    }

    setupEventListeners() {
        // Scenario buttons
        document.querySelectorAll('.scenario-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const scenario = e.target.closest('.scenario-btn').dataset.scenario;
                this.loadScenario(scenario);
            });
        });

        // Form submission
        document.getElementById('prediction-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.makePrediction();
        });

        // Advanced parameters toggle
        document.getElementById('toggle-advanced').addEventListener('click', () => {
            this.toggleAdvancedParams();
        });

        // Real-time form updates
        document.getElementById('prediction-form').addEventListener('input', () => {
            this.updateFormValues();
        });
    }

    async checkApiStatus() {
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');

        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (response.ok) {
                statusIndicator.className = 'fas fa-circle status-indicator online';
                statusText.textContent = 'Online';
            } else {
                throw new Error('API not responding');
            }
        } catch (error) {
            statusIndicator.className = 'fas fa-circle status-indicator offline';
            statusText.textContent = 'Offline';
            console.error('API Status Check Failed:', error);
        }
    }

    loadScenario(scenarioName) {
        // Update active button
        document.querySelectorAll('.scenario-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-scenario="${scenarioName}"]`).classList.add('active');

        this.currentScenario = scenarioName;

        // Load scenario data
        const scenarios = {
            'custom': {
                championship_position: 5,
                pit_stop_count: 2,
                avg_pit_time: 28.5,
                pit_time_std: 3.2,
                circuit_length: 5.4,
                points: 100,
                qualifying_gap_to_pole: 1.2,
                grid_position_percentile: 0.6,
                poor_qualifying: 0,
                is_street_circuit: 0,
                circuit_dnf_rate: 0.15,
                leader_points: 200,
                estimated_age: 28,
                seasons_active: 8,
                is_championship_contender: 0,
                // Additional fields for completeness
                pit_frequency: 1.8,
                pit_duration_variance: 10.2,
                high_pit_frequency: 0,
                championship_pressure: 0.4,
                points_gap_to_leader: 100,
                points_pressure: 0.3,
                driver_avg_grid_position: 10.0,
                qualifying_vs_average: 0.0,
                constructor_avg_grid_position: 9.5,
                qualifying_vs_constructor_avg: 0.5,
                bad_qualifying_day: 0,
                circuit_dnf_rate_detailed: 0.15,
                avg_pit_stops: 2.1,
                avg_pit_duration: 28.0,
                dnf_score: 0.1,
                volatility_score: 0.3,
                pit_complexity_score: 0.5,
                track_difficulty_score: 0.4,
                race_number: 10,
                first_season: 2015,
                driver_rolling_avg_grid: 10.0,
                season_leader_points: 200,
                points_gap_to_season_leader: 100,
                points_momentum: 0.1,
                championship_position_change: 0,
                teammate_avg_grid: 11.0,
                grid_vs_teammate: -1.0,
                championship_pressure_score: 0.4,
                max_round: 22,
                late_season_race: 0,
                championship_pressure_adjusted: 0.3,
                points_per_race: 8.0
            },
            'championship-leader': {
                championship_position: 1,
                pit_stop_count: 2,
                avg_pit_time: 24.8,
                pit_time_std: 1.5,
                circuit_length: 5.4,
                points: 250,
                qualifying_gap_to_pole: 0.0,
                grid_position_percentile: 1.0,
                poor_qualifying: 0,
                is_street_circuit: 0,
                circuit_dnf_rate: 0.12,
                leader_points: 250,
                estimated_age: 29,
                seasons_active: 10,
                is_championship_contender: 1,
                // Championship leader specific values
                pit_frequency: 1.5,
                pit_duration_variance: 5.0,
                high_pit_frequency: 0,
                championship_pressure: 0.9,
                points_gap_to_leader: 0,
                points_pressure: 0.8,
                driver_avg_grid_position: 2.5,
                qualifying_vs_average: -2.5,
                constructor_avg_grid_position: 2.0,
                qualifying_vs_constructor_avg: -2.0,
                bad_qualifying_day: 0,
                circuit_dnf_rate_detailed: 0.12,
                avg_pit_stops: 1.8,
                avg_pit_duration: 25.0,
                dnf_score: 0.05,
                volatility_score: 0.2,
                pit_complexity_score: 0.3,
                track_difficulty_score: 0.4,
                race_number: 15,
                first_season: 2010,
                driver_rolling_avg_grid: 2.2,
                season_leader_points: 250,
                points_gap_to_season_leader: 0,
                points_momentum: 0.8,
                championship_position_change: 0,
                teammate_avg_grid: 4.0,
                grid_vs_teammate: -4.0,
                championship_pressure_score: 0.9,
                max_round: 22,
                late_season_race: 1,
                championship_pressure_adjusted: 0.85,
                points_per_race: 18.0
            },
            'midfield-battle': {
                championship_position: 8,
                pit_stop_count: 3,
                avg_pit_time: 31.2,
                pit_time_std: 4.8,
                circuit_length: 4.2,
                points: 45,
                qualifying_gap_to_pole: 1.8,
                grid_position_percentile: 0.4,
                poor_qualifying: 1,
                is_street_circuit: 1,
                circuit_dnf_rate: 0.22,
                leader_points: 200,
                estimated_age: 26,
                seasons_active: 4,
                is_championship_contender: 0
            },
            'backmarker': {
                championship_position: 18,
                pit_stop_count: 1,
                avg_pit_time: 35.5,
                pit_time_std: 6.2,
                circuit_length: 6.8,
                points: 2,
                qualifying_gap_to_pole: 4.5,
                grid_position_percentile: 0.1,
                poor_qualifying: 1,
                is_street_circuit: 0,
                circuit_dnf_rate: 0.08,
                leader_points: 200,
                estimated_age: 24,
                seasons_active: 2,
                is_championship_contender: 0
            }
        };

        const scenarioData = scenarios[scenarioName];
        
        // Update form fields
        Object.keys(scenarioData).forEach(key => {
            const field = document.getElementById(key);
            if (field) {
                field.value = scenarioData[key];
            }
        });

        this.updateFormValues();
    }

    populateAdvancedFields() {
        // All 47 required features with default values
        const advancedFeatures = {
            pit_frequency: 1.8,
            pit_duration_variance: 10.2,
            high_pit_frequency: 0,
            championship_pressure: 0.4,
            points_gap_to_leader: 100,
            points_pressure: 0.3,
            driver_avg_grid_position: 10.0,
            qualifying_vs_average: 0.0,
            constructor_avg_grid_position: 9.5,
            qualifying_vs_constructor_avg: 0.5,
            bad_qualifying_day: 0,
            circuit_dnf_rate_detailed: 0.15,
            avg_pit_stops: 2.1,
            avg_pit_duration: 28.0,
            dnf_score: 0.1,
            volatility_score: 0.3,
            pit_complexity_score: 0.5,
            track_difficulty_score: 0.4,
            race_number: 10,
            first_season: 2015,
            driver_rolling_avg_grid: 10.0,
            season_leader_points: 200,
            points_gap_to_season_leader: 100,
            points_momentum: 0.1,
            championship_position_change: 0,
            teammate_avg_grid: 11.0,
            grid_vs_teammate: -1.0,
            championship_pressure_score: 0.4,
            max_round: 22,
            late_season_race: 0,
            championship_pressure_adjusted: 0.3,
            points_per_race: 8.0
        };

        const advancedContainer = document.querySelector('.advanced-params .input-grid');
        
        Object.keys(advancedFeatures).forEach(key => {
            const field = document.getElementById(key);
            if (!field) {
                // Create field if it doesn't exist
                const fieldHtml = `
                    <div class="input-field">
                        <label for="${key}">${this.formatFieldName(key)}</label>
                        <input type="number" id="${key}" name="${key}" 
                               step="0.01" value="${advancedFeatures[key]}" required>
                        <span class="input-help">${this.getFieldDescription(key)}</span>
                    </div>
                `;
                advancedContainer.innerHTML += fieldHtml;
            }
        });
    }

    formatFieldName(key) {
        return key.split('_')
                 .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                 .join(' ');
    }

    getFieldDescription(key) {
        const descriptions = {
            pit_frequency: 'Average pit stops per race',
            pit_duration_variance: 'Pit stop time variability',
            high_pit_frequency: 'Above average pit frequency (0/1)',
            championship_pressure: 'Championship pressure score',
            points_gap_to_leader: 'Points behind leader',
            points_pressure: 'Points pressure indicator',
            driver_avg_grid_position: 'Driver average qualifying position',
            qualifying_vs_average: 'Qualifying vs driver average',
            constructor_avg_grid_position: 'Team average grid position',
            qualifying_vs_constructor_avg: 'Qualifying vs team average',
            bad_qualifying_day: 'Poor qualifying performance (0/1)',
            circuit_dnf_rate_detailed: 'Detailed circuit DNF rate',
            avg_pit_stops: 'Average pit stops at circuit',
            avg_pit_duration: 'Average pit duration at circuit',
            dnf_score: 'DNF probability score',
            volatility_score: 'Performance volatility',
            pit_complexity_score: 'Pit stop complexity',
            track_difficulty_score: 'Track difficulty rating',
            race_number: 'Race number in season',
            first_season: 'Driver first F1 season',
            driver_rolling_avg_grid: 'Rolling average grid position',
            season_leader_points: 'Season leader points',
            points_gap_to_season_leader: 'Gap to season leader',
            points_momentum: 'Recent points momentum',
            championship_position_change: 'Position change from last race',
            teammate_avg_grid: 'Teammate average grid',
            grid_vs_teammate: 'Grid position vs teammate',
            championship_pressure_score: 'Championship pressure metric',
            max_round: 'Total races in season',
            late_season_race: 'Late season race (0/1)',
            championship_pressure_adjusted: 'Adjusted pressure score',
            points_per_race: 'Average points per race'
        };
        return descriptions[key] || 'Advanced parameter';
    }

    updateFormValues() {
        // Auto-calculate derived values
        const championshipPosition = parseInt(document.getElementById('championship_position').value) || 1;
        const points = parseInt(document.getElementById('points').value) || 0;
        const leaderPoints = parseInt(document.getElementById('leader_points').value) || 200;
        
        // Update points gap
        const pointsGap = Math.max(0, leaderPoints - points);
        const pointsGapField = document.getElementById('points_gap_to_leader');
        if (pointsGapField) {
            pointsGapField.value = pointsGap;
        }
        
        // Update season leader points (same as leader points for simplicity)
        const seasonLeaderField = document.getElementById('season_leader_points');
        if (seasonLeaderField) {
            seasonLeaderField.value = leaderPoints;
        }
        
        // Update points gap to season leader
        const seasonGapField = document.getElementById('points_gap_to_season_leader');
        if (seasonGapField) {
            seasonGapField.value = pointsGap;
        }
        
        // Update championship pressure
        const pressure = Math.min(1.0, championshipPosition <= 3 ? 0.8 : 0.2);
        const pressureField = document.getElementById('championship_pressure');
        if (pressureField) {
            pressureField.value = pressure.toFixed(2);
        }
        
        // Update championship pressure score (same as pressure)
        const pressureScoreField = document.getElementById('championship_pressure_score');
        if (pressureScoreField) {
            pressureScoreField.value = pressure.toFixed(2);
        }
        
        // Update championship pressure adjusted
        const pressureAdjustedField = document.getElementById('championship_pressure_adjusted');
        if (pressureAdjustedField) {
            pressureAdjustedField.value = (pressure * 0.8).toFixed(2);
        }
    }

    toggleAdvancedParams() {
        const advancedSection = document.querySelector('.advanced-params');
        const toggleBtn = document.getElementById('toggle-advanced');
        const icon = toggleBtn.querySelector('i');
        
        if (advancedSection.style.display === 'none') {
            advancedSection.style.display = 'block';
            toggleBtn.innerHTML = '<i class="fas fa-chevron-up"></i> Hide Advanced Parameters';
        } else {
            advancedSection.style.display = 'none';
            toggleBtn.innerHTML = '<i class="fas fa-chevron-down"></i> Show Advanced Parameters';
        }
    }

    async makePrediction() {
        const loadingOverlay = document.getElementById('loading-overlay');
        const resultsPlaceholder = document.querySelector('.results-placeholder');
        const predictionResults = document.getElementById('prediction-results');

        try {
            // Show loading
            loadingOverlay.style.display = 'flex';

            // Collect form data
            const formData = new FormData(document.getElementById('prediction-form'));
            const features = {};
            
            for (let [key, value] of formData.entries()) {
                features[key] = parseFloat(value);
            }

            // Ensure all required fields are present with default values
            const requiredFields = {
                championship_position: 5,
                pit_stop_count: 2,
                avg_pit_time: 28.5,
                pit_time_std: 3.2,
                circuit_length: 5.4,
                points: 100,
                pit_frequency: 1.8,
                pit_duration_variance: 10.2,
                high_pit_frequency: 0,
                qualifying_gap_to_pole: 1.2,
                grid_position_percentile: 0.6,
                poor_qualifying: 0,
                circuit_dnf_rate: 0.15,
                is_street_circuit: 0,
                championship_pressure: 0.4,
                leader_points: 200,
                points_gap_to_leader: 100,
                points_pressure: 0.3,
                driver_avg_grid_position: 10.0,
                qualifying_vs_average: 0.0,
                constructor_avg_grid_position: 9.5,
                qualifying_vs_constructor_avg: 0.5,
                bad_qualifying_day: 0,
                circuit_dnf_rate_detailed: 0.15,
                avg_pit_stops: 2.1,
                avg_pit_duration: 28.0,
                dnf_score: 0.1,
                volatility_score: 0.3,
                pit_complexity_score: 0.5,
                track_difficulty_score: 0.4,
                race_number: 10,
                first_season: 2015,
                seasons_active: 8,
                estimated_age: 28,
                driver_rolling_avg_grid: 10.0,
                season_leader_points: 200,
                points_gap_to_season_leader: 100,
                is_championship_contender: 0,
                points_momentum: 0.1,
                championship_position_change: 0,
                teammate_avg_grid: 11.0,
                grid_vs_teammate: -1.0,
                championship_pressure_score: 0.4,
                max_round: 22,
                late_season_race: 0,
                championship_pressure_adjusted: 0.3,
                points_per_race: 8.0
            };

            // Fill in any missing fields with defaults
            Object.keys(requiredFields).forEach(key => {
                if (!(key in features)) {
                    features[key] = requiredFields[key];
                }
            });

            // Make API request
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(features)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`API Error: ${response.status} - ${errorData.detail || 'Unknown error'}`);
            }

            const result = await response.json();
            
            // Hide loading and placeholder
            loadingOverlay.style.display = 'none';
            resultsPlaceholder.style.display = 'none';
            predictionResults.style.display = 'block';

            // Update results
            this.displayResults(result);

        } catch (error) {
            loadingOverlay.style.display = 'none';
            console.error('Prediction Error:', error);
            this.showError(`Failed to make prediction: ${error.message}`);
        }
    }

    displayResults(result) {
        // Classification results
        const dropProbability = (result.classification.probability * 100).toFixed(1);
        const willDrop = result.classification.will_drop_position;
        const confidence = result.classification.confidence;

        document.getElementById('drop-probability').textContent = `${dropProbability}%`;
        document.getElementById('will-drop').textContent = willDrop ? 'Yes' : 'No';
        document.getElementById('will-drop').className = `detail-value ${willDrop ? 'danger' : 'success'}`;
        document.getElementById('confidence').textContent = confidence.charAt(0).toUpperCase() + confidence.slice(1);

        // Update gauge
        this.updateGauge(result.classification.probability);

        // Regression results
        const positionChange = result.regression.expected_position_change.toFixed(2);
        const interval = result.regression.prediction_interval;
        
        document.getElementById('position-change').textContent = positionChange;
        document.getElementById('position-change').className = `change-value ${positionChange >= 0 ? 'positive' : 'negative'}`;
        document.getElementById('prediction-interval').textContent = 
            `[${interval[0].toFixed(1)}, ${interval[1].toFixed(1)}]`;

        // Feature importance
        this.displayFeatureImportance(result.feature_contributions);
    }

    updateGauge(probability) {
        const gauge = document.getElementById('probability-gauge');
        const percentage = probability * 100;
        
        // Create gradient based on probability
        let color1, color2;
        if (percentage < 30) {
            color1 = '#00d084'; // Green
            color2 = '#00a86b';
        } else if (percentage < 70) {
            color1 = '#ff9500'; // Orange
            color2 = '#ff7700';
        } else {
            color1 = '#ff3333'; // Red
            color2 = '#e10600';
        }

        const angle = (percentage / 100) * 360;
        gauge.style.background = `conic-gradient(
            ${color1} 0deg,
            ${color2} ${angle}deg,
            #404040 ${angle}deg
        )`;
    }

    displayFeatureImportance(contributions) {
        const importanceChart = document.getElementById('importance-chart');
        
        // Sort features by importance
        const sortedFeatures = Object.entries(contributions)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 8); // Top 8 features

        const maxImportance = sortedFeatures[0][1];

        importanceChart.innerHTML = sortedFeatures.map(([feature, importance]) => {
            const percentage = (importance / maxImportance) * 100;
            const displayName = this.formatFieldName(feature);
            
            return `
                <div class="importance-item">
                    <span class="importance-name">${displayName}</span>
                    <div class="importance-bar">
                        <div class="importance-fill" style="width: ${percentage}%"></div>
                    </div>
                    <span class="importance-value">${(importance * 100).toFixed(1)}%</span>
                </div>
            `;
        }).join('');
    }

    showError(message) {
        const resultsContainer = document.querySelector('.results-container');
        resultsContainer.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>${message}</p>
            </div>
        `;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new F1PredictorApp();
});

// Add some utility CSS for error messages
const style = document.createElement('style');
style.textContent = `
    .error-message {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 200px;
        color: var(--danger-color);
        text-align: center;
    }
    
    .error-message i {
        font-size: 3rem;
        margin-bottom: 15px;
    }
    
    .error-message p {
        font-size: 1.1rem;
        max-width: 300px;
    }
    
    .detail-value.success {
        color: var(--success-color);
    }
    
    .detail-value.danger {
        color: var(--danger-color);
    }
`;
document.head.appendChild(style);