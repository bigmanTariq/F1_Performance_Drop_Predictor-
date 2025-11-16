"""
Feature Engineering Module for F1 Performance Drop Prediction

This module creates comprehensive engineered features from clean F1 data:
- Stress-related features (pit stop patterns, qualifying gaps)
- Historical performance metrics (driver/constructor reliability)
- Championship pressure indicators
- Track difficulty and circuit characteristics
- Rolling averages and momentum indicators

The module transforms 17 base columns into 80+ engineered features
optimized for predicting finishing position drops.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_clean_dataset(filepath: str = 'data/f1_performance_drop_clean.csv') -> pd.DataFrame:
    """
    Load the clean F1 dataset for feature engineering.
    
    Args:
        filepath: Path to the clean CSV file
        
    Returns:
        DataFrame with clean F1 data
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded clean dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading clean dataset: {str(e)}")
        raise

def calculate_stress_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate stress-related features for F1 performance prediction.
    
    Implements pit stop frequency, duration variance, qualifying gaps, 
    and track difficulty metrics.
    
    Args:
        df: Clean F1 dataset
        
    Returns:
        DataFrame with added stress-related features
    """
    logger.info("Calculating stress-related features...")
    df = df.copy()
    
    # 1. Pit Stop Frequency Features
    # Pit stop frequency (stops per race)
    df['pit_frequency'] = df['pit_stop_count']
    
    # Pit stop duration variance (normalized by average)
    df['pit_duration_variance'] = df['pit_time_std'] / (df['avg_pit_time'] + 1)  # +1 to avoid division by zero
    df['pit_duration_variance'] = df['pit_duration_variance'].fillna(0)
    
    # High pit frequency flag (more than 2 stops)
    df['high_pit_frequency'] = (df['pit_stop_count'] > 2).astype(int)
    
    # 2. Qualifying Gap Features
    # Calculate qualifying gap to pole position for each race
    pole_times = df.groupby(['season', 'round'])['grid_position'].min().reset_index()
    pole_times.columns = ['season', 'round', 'pole_position']
    
    # Qualifying gap as position difference from pole
    df['qualifying_gap_to_pole'] = df['grid_position'] - 1  # Pole is position 1
    
    # Grid position percentile within each race
    df['grid_position_percentile'] = df.groupby(['season', 'round'])['grid_position'].rank(pct=True)
    
    # Poor qualifying flag (started in bottom 25% of grid)
    df['poor_qualifying'] = (df['grid_position_percentile'] > 0.75).astype(int)
    
    # 3. Track Difficulty Metrics
    # Calculate historical DNF rates by circuit
    # First, identify DNFs (positions > 20 or missing positions indicate DNF)
    df['is_dnf'] = (df['position'] > 20).astype(int)
    
    # Calculate circuit DNF rate (historical difficulty)
    circuit_dnf_rates = df.groupby('circuit_name')['is_dnf'].mean().reset_index()
    circuit_dnf_rates.columns = ['circuit_name', 'circuit_dnf_rate']
    
    # Merge back to main dataframe
    df = df.merge(circuit_dnf_rates, on='circuit_name', how='left')
    
    # Track difficulty categories based on DNF rate
    df['track_difficulty'] = pd.cut(
        df['circuit_dnf_rate'], 
        bins=[0, 0.1, 0.2, 1.0], 
        labels=['Easy', 'Medium', 'Hard']
    )
    
    # Circuit type encoding (Street circuits are typically more challenging)
    df['is_street_circuit'] = (df['circuit_type'] == 'STREET').astype(int)
    
    # Circuit length category (longer circuits may have different stress patterns)
    df['circuit_length_category'] = pd.cut(
        df['circuit_length'], 
        bins=[0, 4.0, 5.0, 10.0], 
        labels=['Short', 'Medium', 'Long']
    )
    
    # 4. Additional Stress Indicators
    # Championship pressure (higher for drivers in top positions)
    df['championship_pressure'] = np.where(
        df['championship_position'] <= 5, 
        1 / (df['championship_position'] + 1),  # Higher pressure for better positions
        0.1  # Low pressure for lower positions
    )
    
    # Points gap pressure (if points data available)
    if 'points' in df.columns:
        # Calculate points gap to leader within each race
        race_leader_points = df.groupby(['season', 'round'])['points'].max().reset_index()
        race_leader_points.columns = ['season', 'round', 'leader_points']
        df = df.merge(race_leader_points, on=['season', 'round'], how='left')
        
        df['points_gap_to_leader'] = df['leader_points'] - df['points']
        df['points_pressure'] = np.where(
            df['points_gap_to_leader'] <= 50,  # Close championship fight
            1.0,
            0.5
        )
    else:
        df['points_gap_to_leader'] = 0
        df['points_pressure'] = 0.5
    
    logger.info("Stress-related features calculated successfully")
    return df

def calculate_qualifying_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate more sophisticated qualifying gap features.
    
    Args:
        df: DataFrame with basic qualifying data
        
    Returns:
        DataFrame with enhanced qualifying gap features
    """
    logger.info("Calculating enhanced qualifying gap features...")
    df = df.copy()
    
    # Calculate average grid position by driver (driver skill baseline)
    driver_avg_grid = df.groupby('driver_id')['grid_position'].mean().reset_index()
    driver_avg_grid.columns = ['driver_id', 'driver_avg_grid_position']
    df = df.merge(driver_avg_grid, on='driver_id', how='left')
    
    # Qualifying performance relative to driver's average
    df['qualifying_vs_average'] = df['grid_position'] - df['driver_avg_grid_position']
    
    # Constructor average grid position (car competitiveness)
    constructor_avg_grid = df.groupby('constructor_id')['grid_position'].mean().reset_index()
    constructor_avg_grid.columns = ['constructor_id', 'constructor_avg_grid_position']
    df = df.merge(constructor_avg_grid, on='constructor_id', how='left')
    
    # Qualifying performance relative to constructor average
    df['qualifying_vs_constructor_avg'] = df['grid_position'] - df['constructor_avg_grid_position']
    
    # Bad qualifying day flag (significantly worse than usual)
    df['bad_qualifying_day'] = (df['qualifying_vs_average'] > 3).astype(int)
    
    logger.info("Enhanced qualifying gap features calculated")
    return df

def calculate_track_difficulty_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive track difficulty metrics.
    
    Args:
        df: DataFrame with circuit and race data
        
    Returns:
        DataFrame with track difficulty features
    """
    logger.info("Calculating comprehensive track difficulty features...")
    df = df.copy()
    
    # Calculate various difficulty metrics by circuit
    circuit_stats = df.groupby('circuit_name').agg({
        'is_dnf': 'mean',  # DNF rate
        'position_change_numeric': ['mean', 'std'],  # Position change patterns
        'pit_stop_count': 'mean',  # Average pit stops (strategy complexity)
        'avg_pit_time': 'mean'  # Average pit time (pit lane difficulty)
    }).reset_index()
    
    # Flatten column names
    circuit_stats.columns = [
        'circuit_name', 'circuit_dnf_rate', 'avg_position_change', 
        'position_change_volatility', 'avg_pit_stops', 'avg_pit_duration'
    ]
    
    # Merge back to main dataframe (avoid duplicate column names)
    df = df.merge(circuit_stats, on='circuit_name', how='left', suffixes=('', '_detailed'))
    
    # Use the detailed circuit_dnf_rate if it exists, otherwise use the original
    dnf_rate_col = 'circuit_dnf_rate_detailed' if 'circuit_dnf_rate_detailed' in df.columns else 'circuit_dnf_rate'
    
    # Create composite difficulty score
    # Normalize each component to 0-1 scale
    df['dnf_score'] = (df[dnf_rate_col] - df[dnf_rate_col].min()) / (df[dnf_rate_col].max() - df[dnf_rate_col].min() + 1e-8)
    df['volatility_score'] = (df['position_change_volatility'] - df['position_change_volatility'].min()) / (df['position_change_volatility'].max() - df['position_change_volatility'].min() + 1e-8)
    df['pit_complexity_score'] = (df['avg_pit_stops'] - df['avg_pit_stops'].min()) / (df['avg_pit_stops'].max() - df['avg_pit_stops'].min() + 1e-8)
    
    # Composite difficulty score (weighted average)
    df['track_difficulty_score'] = (
        0.4 * df['dnf_score'] + 
        0.3 * df['volatility_score'] + 
        0.3 * df['pit_complexity_score']
    )
    
    # Track difficulty categories
    df['track_difficulty_level'] = pd.cut(
        df['track_difficulty_score'],
        bins=[0, 0.33, 0.66, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    logger.info("Track difficulty features calculated")
    return df

# Test function for stress features
def test_stress_features():
    """
    Test stress-related feature calculations with sample data.
    """
    logger.info("Testing stress-related features...")
    
    # Create test data
    test_data = pd.DataFrame({
        'season': [2020, 2020, 2020, 2020],
        'round': [1, 1, 1, 1],
        'driver_id': [1, 2, 3, 4],
        'constructor_id': [1, 1, 2, 2],
        'grid_position': [1, 5, 10, 15],
        'position': [3, 2, 12, 8],
        'pit_stop_count': [1, 2, 3, 1],
        'avg_pit_time': [25000, 30000, 35000, 28000],
        'pit_time_std': [0, 1000, 2000, 500],
        'circuit_name': ['Monaco', 'Monaco', 'Monaco', 'Monaco'],
        'circuit_type': ['STREET', 'STREET', 'STREET', 'STREET'],
        'circuit_length': [3.337, 3.337, 3.337, 3.337],
        'championship_position': [1, 5, 10, 15],
        'points': [100, 80, 50, 20],
        'position_drop_flag': [1, 0, 1, 0],
        'position_change_numeric': [2, -3, 2, -7]
    })
    
    # Test stress feature calculation
    result = calculate_stress_features(test_data)
    
    # Verify expected features exist
    expected_features = [
        'pit_frequency', 'pit_duration_variance', 'high_pit_frequency',
        'qualifying_gap_to_pole', 'grid_position_percentile', 'poor_qualifying',
        'is_dnf', 'circuit_dnf_rate', 'track_difficulty',
        'is_street_circuit', 'circuit_length_category',
        'championship_pressure', 'points_gap_to_leader', 'points_pressure'
    ]
    
    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"
    
    # Test specific calculations
    assert result['qualifying_gap_to_pole'].iloc[0] == 0, "Pole position should have 0 gap"
    assert result['is_street_circuit'].iloc[0] == 1, "Monaco should be street circuit"
    assert result['high_pit_frequency'].iloc[2] == 1, "3 pit stops should be high frequency"
    
    logger.info("Stress-related features tests passed!")

def calculate_historical_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical performance features including driver experience,
    rolling reliability scores, and championship standings.
    
    Args:
        df: DataFrame with race data
        
    Returns:
        DataFrame with added historical performance features
    """
    logger.info("Calculating historical performance features...")
    df = df.copy()
    
    # Sort by season and round for time-series calculations
    df = df.sort_values(['season', 'round', 'driver_id'])
    
    # 1. Driver Experience Metrics
    # Calculate career starts for each driver up to each race
    df['race_number'] = df.groupby('driver_id').cumcount() + 1  # Career race number
    
    # Calculate seasons active (approximate)
    driver_first_season = df.groupby('driver_id')['season'].min().reset_index()
    driver_first_season.columns = ['driver_id', 'first_season']
    df = df.merge(driver_first_season, on='driver_id', how='left')
    
    df['seasons_active'] = df['season'] - df['first_season'] + 1
    
    # Driver age approximation (assuming drivers start around age 20-25)
    # This is a rough estimate since we don't have birth dates
    df['estimated_age'] = 22 + df['seasons_active']  # Rough age estimate
    
    # Experience categories
    df['experience_level'] = pd.cut(
        df['race_number'],
        bins=[0, 20, 50, 100, 500],
        labels=['Rookie', 'Junior', 'Experienced', 'Veteran']
    )
    
    # 2. Rolling Reliability Scores
    # Calculate rolling DNF rates for drivers (last 10 races)
    df['is_dnf'] = (df['position'] > 20).astype(int)
    df['driver_rolling_dnf_rate'] = df.groupby('driver_id')['is_dnf'].rolling(
        window=10, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    # Driver reliability score (inverse of DNF rate)
    df['driver_reliability_score'] = 1 - df['driver_rolling_dnf_rate']
    
    # Calculate rolling reliability for constructors (last 10 races)
    df['constructor_rolling_dnf_rate'] = df.groupby('constructor_id')['is_dnf'].rolling(
        window=10, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    df['constructor_reliability_score'] = 1 - df['constructor_rolling_dnf_rate']
    
    # 3. Rolling Performance Metrics
    # Rolling average finishing position (last 5 races)
    df['driver_rolling_avg_position'] = df.groupby('driver_id')['position'].rolling(
        window=5, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    # Rolling average grid position (qualifying performance)
    df['driver_rolling_avg_grid'] = df.groupby('driver_id')['grid_position'].rolling(
        window=5, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    # Performance trend (improving/declining)
    df['recent_performance_trend'] = df['driver_rolling_avg_grid'] - df['driver_rolling_avg_position']
    # Positive trend means finishing better than qualifying (good race pace)
    
    # 4. Championship Standings Features
    # Points gap to championship leader (if points available)
    if 'points' in df.columns:
        # Calculate championship leader points for each race
        season_leader_points = df.groupby(['season', 'round'])['points'].max().reset_index()
        season_leader_points.columns = ['season', 'round', 'season_leader_points']
        df = df.merge(season_leader_points, on=['season', 'round'], how='left')
        
        df['points_gap_to_season_leader'] = df['season_leader_points'] - df['points']
        
        # Championship contender flag (within 50 points of leader)
        df['is_championship_contender'] = (df['points_gap_to_season_leader'] <= 50).astype(int)
        
        # Points momentum (points gained in last 3 races)
        df['points_momentum'] = df.groupby('driver_id')['points'].diff(3).fillna(0)
    else:
        df['points_gap_to_season_leader'] = 0
        df['is_championship_contender'] = 0
        df['points_momentum'] = 0
    
    # Championship position momentum (position change in standings)
    df['championship_position_change'] = df.groupby('driver_id')['championship_position'].diff().fillna(0)
    
    # 5. Team Performance Context
    # Constructor average performance in current season
    constructor_season_avg = df.groupby(['constructor_id', 'season'])['position'].mean().reset_index()
    constructor_season_avg.columns = ['constructor_id', 'season', 'constructor_season_avg_position']
    df = df.merge(constructor_season_avg, on=['constructor_id', 'season'], how='left')
    
    # Driver performance relative to teammate
    # Calculate average position of other drivers in same constructor
    teammate_avg = df.groupby(['constructor_id', 'season', 'round']).agg({
        'position': 'mean',
        'grid_position': 'mean'
    }).reset_index()
    teammate_avg.columns = ['constructor_id', 'season', 'round', 'teammate_avg_position', 'teammate_avg_grid']
    df = df.merge(teammate_avg, on=['constructor_id', 'season', 'round'], how='left')
    
    # Performance relative to teammate
    df['position_vs_teammate'] = df['position'] - df['teammate_avg_position']
    df['grid_vs_teammate'] = df['grid_position'] - df['teammate_avg_grid']
    
    # 6. Historical Success Metrics
    # Career win rate
    df['is_win'] = (df['position'] == 1).astype(int)
    df['career_win_rate'] = df.groupby('driver_id')['is_win'].expanding().mean().reset_index(0, drop=True)
    
    # Career podium rate
    df['is_podium'] = (df['position'] <= 3).astype(int)
    df['career_podium_rate'] = df.groupby('driver_id')['is_podium'].expanding().mean().reset_index(0, drop=True)
    
    # Career points finish rate
    df['is_points'] = (df['position'] <= 10).astype(int)  # Top 10 typically scores points
    df['career_points_rate'] = df.groupby('driver_id')['is_points'].expanding().mean().reset_index(0, drop=True)
    
    logger.info("Historical performance features calculated successfully")
    return df

def calculate_championship_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate detailed championship-related features.
    
    Args:
        df: DataFrame with championship data
        
    Returns:
        DataFrame with enhanced championship features
    """
    logger.info("Calculating championship features...")
    df = df.copy()
    
    # Championship pressure based on position and points gap
    df['championship_pressure_score'] = np.where(
        df['championship_position'] <= 3,
        1.0,  # High pressure for top 3
        np.where(
            df['championship_position'] <= 8,
            0.7,  # Medium pressure for top 8
            0.3   # Low pressure for others
        )
    )
    
    # Late season pressure (races in last quarter of season)
    max_rounds = df.groupby('season')['round'].max().reset_index()
    max_rounds.columns = ['season', 'max_round']
    df = df.merge(max_rounds, on='season', how='left')
    
    df['late_season_race'] = (df['round'] > 0.75 * df['max_round']).astype(int)
    df['championship_pressure_adjusted'] = df['championship_pressure_score'] * (1 + 0.5 * df['late_season_race'])
    
    # Points per race average (efficiency metric)
    if 'points' in df.columns:
        df['points_per_race'] = df['points'] / df['race_number']
    else:
        df['points_per_race'] = 0
    
    logger.info("Championship features calculated")
    return df

# Test function for historical features
def test_historical_features():
    """
    Test historical performance feature calculations.
    """
    logger.info("Testing historical performance features...")
    
    # Create test data with multiple races for same drivers
    test_data = pd.DataFrame({
        'season': [2020, 2020, 2020, 2020, 2020, 2020],
        'round': [1, 2, 1, 2, 1, 2],
        'driver_id': [1, 1, 2, 2, 3, 3],
        'constructor_id': [1, 1, 1, 1, 2, 2],
        'grid_position': [1, 3, 5, 4, 10, 12],
        'position': [1, 2, 3, 5, 8, 10],
        'championship_position': [1, 1, 2, 2, 5, 6],
        'points': [25, 43, 18, 28, 4, 5],
        'pit_stop_count': [1, 2, 2, 1, 3, 2],
        'avg_pit_time': [25000, 30000, 28000, 26000, 35000, 32000],
        'pit_time_std': [0, 1000, 500, 200, 2000, 1500],
        'circuit_name': ['Bahrain', 'Saudi Arabia', 'Bahrain', 'Saudi Arabia', 'Bahrain', 'Saudi Arabia'],
        'circuit_type': ['RACE', 'STREET', 'RACE', 'STREET', 'RACE', 'STREET'],
        'circuit_length': [5.412, 6.174, 5.412, 6.174, 5.412, 6.174]
    })
    
    # Test historical feature calculation
    result = calculate_historical_performance_features(test_data)
    
    # Verify expected features exist
    expected_features = [
        'race_number', 'seasons_active', 'estimated_age', 'experience_level',
        'driver_reliability_score', 'constructor_reliability_score',
        'driver_rolling_avg_position', 'driver_rolling_avg_grid',
        'recent_performance_trend', 'points_gap_to_season_leader',
        'is_championship_contender', 'points_momentum',
        'championship_position_change', 'constructor_season_avg_position',
        'position_vs_teammate', 'grid_vs_teammate',
        'career_win_rate', 'career_podium_rate', 'career_points_rate'
    ]
    
    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"
    
    # Test specific calculations (after sorting, driver 1's second race should have race_number = 2)
    driver_1_races = result[result['driver_id'] == 1].sort_values('round')
    assert driver_1_races['race_number'].iloc[1] == 2, "Driver 1's second race should have race_number = 2"
    assert driver_1_races['career_win_rate'].iloc[0] == 1.0, "Driver with win should have 100% win rate initially"
    assert driver_1_races['is_championship_contender'].iloc[0] == 1, "Leader should be championship contender"
    
    logger.info("Historical performance features tests passed!")

def validate_feature_ranges(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Validate feature ranges and distributions to catch data quality issues.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        Dictionary with validation results and warnings
    """
    logger.info("Validating feature ranges and distributions...")
    
    validation_results = {
        'warnings': [],
        'errors': [],
        'info': []
    }
    
    # Define expected ranges for key features
    feature_ranges = {
        'grid_position': (1, 26),
        'position': (1, 26),
        'pit_stop_count': (0, 10),
        'championship_position': (1, 30),
        'driver_reliability_score': (0, 1),
        'constructor_reliability_score': (0, 1),
        'career_win_rate': (0, 1),
        'career_podium_rate': (0, 1),
        'career_points_rate': (0, 1),
        'track_difficulty_score': (0, 1),
        'championship_pressure_score': (0, 2)  # Can be > 1 with adjustments
    }
    
    # Check ranges
    for feature, (min_val, max_val) in feature_ranges.items():
        if feature in df.columns:
            actual_min = df[feature].min()
            actual_max = df[feature].max()
            
            if actual_min < min_val or actual_max > max_val:
                validation_results['warnings'].append(
                    f"{feature}: Expected range [{min_val}, {max_val}], got [{actual_min:.3f}, {actual_max:.3f}]"
                )
            else:
                validation_results['info'].append(
                    f"{feature}: Range OK [{actual_min:.3f}, {actual_max:.3f}]"
                )
    
    # Check for excessive missing values
    missing_threshold = 0.1  # 10%
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df)
        if missing_pct > missing_threshold:
            validation_results['warnings'].append(
                f"{col}: High missing value rate {missing_pct:.1%}"
            )
    
    # Check for constant features (no variance)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].std() == 0:
            validation_results['warnings'].append(
                f"{col}: Constant feature (no variance)"
            )
    
    # Check for highly correlated features (>0.95)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                    )
        
        if high_corr_pairs:
            validation_results['warnings'].append(
                f"High correlation pairs: {high_corr_pairs[:5]}"  # Show first 5
            )
    
    # Log results
    for warning in validation_results['warnings']:
        logger.warning(warning)
    
    for error in validation_results['errors']:
        logger.error(error)
    
    logger.info(f"Feature validation completed: {len(validation_results['warnings'])} warnings, {len(validation_results['errors'])} errors")
    
    return validation_results

def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features for machine learning.
    
    Args:
        df: DataFrame with categorical features
        
    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    logger.info("Encoding categorical features...")
    
    df_encoded = df.copy()
    encoders = {}
    
    # Identify categorical columns
    categorical_cols = [
        'track_difficulty', 'circuit_length_category', 'experience_level',
        'track_difficulty_level', 'circuit_type'
    ]
    
    # Encode each categorical column
    for col in categorical_cols:
        if col in df_encoded.columns:
            # Convert to string and handle missing values first
            df_encoded[col] = df_encoded[col].astype(str).fillna('Unknown')
            
            # Create and fit encoder
            encoder = LabelEncoder()
            df_encoded[col + '_encoded'] = encoder.fit_transform(df_encoded[col].astype(str))
            encoders[col] = encoder
            
            logger.info(f"Encoded {col}: {len(encoder.classes_)} categories")
    
    # Create binary features for important categorical variables
    if 'circuit_type' in df_encoded.columns:
        df_encoded['is_street_circuit'] = (df_encoded['circuit_type'] == 'STREET').astype(int)
    
    if 'experience_level' in df_encoded.columns:
        df_encoded['is_rookie'] = (df_encoded['experience_level'] == 'Rookie').astype(int)
        df_encoded['is_veteran'] = (df_encoded['experience_level'] == 'Veteran').astype(int)
    
    logger.info("Categorical encoding completed")
    return df_encoded, encoders

def scale_numerical_features(df: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df: DataFrame with numerical features
        feature_cols: List of columns to scale (if None, scale all numerical)
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    logger.info("Scaling numerical features...")
    
    df_scaled = df.copy()
    
    # Select features to scale
    if feature_cols is None:
        # Exclude target variables and IDs from scaling
        exclude_cols = [
            'position_drop_flag', 'position_change_numeric',
            'driver_id', 'constructor_id', 'race_id', 'season', 'round'
        ]
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if not feature_cols:
        logger.warning("No numerical features found to scale")
        return df_scaled, None
    
    # Fit scaler
    scaler = StandardScaler()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    logger.info(f"Scaled {len(feature_cols)} numerical features")
    return df_scaled, scaler

def select_features_by_importance(df: pd.DataFrame, target_col: str, k: int = 20, 
                                task_type: str = 'classification') -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top k features based on statistical importance.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        k: Number of features to select
        task_type: 'classification' or 'regression'
        
    Returns:
        Tuple of (DataFrame with selected features, list of selected feature names)
    """
    logger.info(f"Selecting top {k} features for {task_type}...")
    
    # Prepare feature matrix
    feature_cols = [col for col in df.columns if col not in [
        target_col, 'driver_id', 'constructor_id', 'race_id', 
        'driver_name', 'circuit_name', 'season', 'round'
    ]]
    
    # Remove non-numeric columns
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Select appropriate scoring function
    if task_type == 'classification':
        score_func = f_classif
    else:
        score_func = f_regression
    
    # Perform feature selection
    selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Create output DataFrame
    df_selected = df[['driver_id', 'constructor_id'] + selected_features + [target_col]].copy()
    
    logger.info(f"Selected features: {selected_features}")
    return df_selected, selected_features

def preprocess_features_pipeline(df: pd.DataFrame, target_cols: List[str] = None, 
                                scale_features: bool = True, 
                                select_features: bool = True,
                                n_features: int = 20) -> Dict:
    """
    Complete feature preprocessing pipeline.
    
    Args:
        df: DataFrame with raw features
        target_cols: List of target column names
        scale_features: Whether to scale numerical features
        select_features: Whether to perform feature selection
        n_features: Number of features to select (if select_features=True)
        
    Returns:
        Dictionary with processed data and fitted preprocessors
    """
    logger.info("Running complete feature preprocessing pipeline...")
    
    if target_cols is None:
        target_cols = ['position_drop_flag', 'position_change_numeric']
    
    # 1. Validate features
    validation_results = validate_feature_ranges(df)
    
    # 2. Encode categorical features
    df_encoded, encoders = encode_categorical_features(df)
    
    # 3. Scale numerical features (if requested)
    scaler = None
    if scale_features:
        df_scaled, scaler = scale_numerical_features(df_encoded)
    else:
        df_scaled = df_encoded
    
    # 4. Feature selection (if requested)
    selected_features = {}
    df_selected = {}
    
    if select_features:
        for target_col in target_cols:
            if target_col in df_scaled.columns:
                task_type = 'classification' if 'flag' in target_col else 'regression'
                df_sel, features = select_features_by_importance(
                    df_scaled, target_col, k=n_features, task_type=task_type
                )
                df_selected[target_col] = df_sel
                selected_features[target_col] = features
    else:
        for target_col in target_cols:
            if target_col in df_scaled.columns:
                df_selected[target_col] = df_scaled
                selected_features[target_col] = [col for col in df_scaled.columns 
                                               if col not in target_cols + ['driver_id', 'constructor_id']]
    
    # Package results
    results = {
        'processed_data': df_selected,
        'selected_features': selected_features,
        'encoders': encoders,
        'scaler': scaler,
        'validation_results': validation_results,
        'target_columns': target_cols
    }
    
    logger.info("Feature preprocessing pipeline completed")
    return results

# Test function for preprocessing
def test_preprocessing_features():
    """
    Test feature preprocessing and validation functions.
    """
    logger.info("Testing feature preprocessing...")
    
    # Create test data with various feature types
    test_data = pd.DataFrame({
        'driver_id': [1, 2, 3, 4, 5],
        'constructor_id': [1, 1, 2, 2, 3],
        'grid_position': [1, 5, 10, 15, 20],
        'position': [1, 3, 8, 12, 18],
        'pit_frequency': [1, 2, 3, 1, 2],
        'driver_reliability_score': [0.9, 0.8, 0.7, 0.6, 0.5],
        'track_difficulty': ['Low', 'Medium', 'High', 'Low', 'Medium'],
        'experience_level': ['Veteran', 'Experienced', 'Junior', 'Rookie', 'Experienced'],
        'circuit_type': ['RACE', 'STREET', 'RACE', 'STREET', 'RACE'],
        'position_drop_flag': [0, 1, 1, 1, 1],
        'position_change_numeric': [0, -2, -2, -3, -2]
    })
    
    # Test validation
    validation_results = validate_feature_ranges(test_data)
    assert 'warnings' in validation_results
    assert 'errors' in validation_results
    
    # Test categorical encoding
    df_encoded, encoders = encode_categorical_features(test_data)
    assert 'track_difficulty_encoded' in df_encoded.columns
    assert 'track_difficulty' in encoders
    
    # Test scaling
    df_scaled, scaler = scale_numerical_features(df_encoded)
    assert scaler is not None
    
    # Test feature selection
    df_selected, features = select_features_by_importance(
        df_scaled, 'position_drop_flag', k=5, task_type='classification'
    )
    assert len(features) <= 5
    
    # Test complete pipeline
    results = preprocess_features_pipeline(test_data, scale_features=True, select_features=True, n_features=5)
    assert 'processed_data' in results
    assert 'selected_features' in results
    assert 'encoders' in results
    assert 'scaler' in results
    
    logger.info("Feature preprocessing tests passed!")

if __name__ == "__main__":
    # Run tests
    test_stress_features()
    test_historical_features()
    test_preprocessing_features()
    
    # Load and process real data
    df = load_clean_dataset()
    df_with_features = calculate_stress_features(df)
    df_with_features = calculate_qualifying_gap_features(df_with_features)
    df_with_features = calculate_track_difficulty_features(df_with_features)
    df_with_features = calculate_historical_performance_features(df_with_features)
    df_with_features = calculate_championship_features(df_with_features)
    
    # Run preprocessing pipeline
    preprocessing_results = preprocess_features_pipeline(
        df_with_features, 
        scale_features=True, 
        select_features=True, 
        n_features=20
    )
    
    logger.info(f"Feature engineering completed successfully!")
    logger.info(f"Final dataset shape: {df_with_features.shape}")
    logger.info(f"Selected features for classification: {len(preprocessing_results['selected_features'].get('position_drop_flag', []))}")
    logger.info(f"Selected features for regression: {len(preprocessing_results['selected_features'].get('position_change_numeric', []))}")
    
    # Save processed dataset
    output_path = 'data/f1_features_engineered.csv'
    df_with_features.to_csv(output_path, index=False)
    logger.info(f"Engineered features saved to: {output_path}")