"""
Data Preparation Module for F1 Performance Drop Prediction

This module handles the complete data preparation pipeline:
- Loading and validating raw F1 CSV datasets
- Merging race results with qualifying, pit stops, and metadata
- Computing target variables for classification and regression
- Data cleaning and quality validation
- Exporting clean dataset for feature engineering

The module processes approximately 46 different F1 datasets to create
a unified dataset of ~8,200 race entries with position drop targets.
"""
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_csv_file(filepath: str, required_columns: list) -> bool:
    """
    Validate that a CSV file exists and contains required columns.
    
    Args:
        filepath: Path to the CSV file
        required_columns: List of column names that must be present
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False
    
    try:
        df = pd.read_csv(filepath, nrows=1)  # Read just header
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing columns in {filepath}: {missing_cols}")
            return False
        logger.info(f"Validated {filepath} - {len(df.columns)} columns found")
        return True
    except Exception as e:
        logger.error(f"Error reading {filepath}: {str(e)}")
        return False

def load_csv_with_validation(filepath: str, required_columns: list) -> Optional[pd.DataFrame]:
    """
    Load CSV file with validation and error handling.
    
    Args:
        filepath: Path to the CSV file
        required_columns: List of column names that must be present
        
    Returns:
        DataFrame if successful, None if failed
    """
    if not validate_csv_file(filepath, required_columns):
        return None
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {filepath}: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return None

def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load all relevant F1 datasets with validation.
    
    Returns:
        Dictionary of dataset name to DataFrame
    """
    datasets = {}
    
    # Define required columns for each dataset (using actual column names from CSV files)
    dataset_configs = {
        'grid': {
            'path': 'data/f1db-races-starting-grid-positions.csv',
            'required_cols': ['raceId', 'driverId', 'positionNumber']
        },
        'results': {
            'path': 'data/f1db-races-race-results.csv', 
            'required_cols': ['raceId', 'driverId', 'positionNumber', 'constructorId']
        },
        'drivers': {
            'path': 'data/f1db-drivers.csv',
            'required_cols': ['id', 'lastName']
        },
        'constructors': {
            'path': 'data/f1db-constructors.csv',
            'required_cols': ['id', 'name']
        },
        'races': {
            'path': 'data/f1db-races.csv',
            'required_cols': ['id', 'year', 'round', 'circuitId']
        },
        'pit_stops': {
            'path': 'data/f1db-races-pit-stops.csv',
            'required_cols': ['raceId', 'driverId', 'stop', 'lap', 'timeMillis']
        },
        'circuits': {
            'path': 'data/f1db-circuits.csv',
            'required_cols': ['id', 'name', 'type', 'length']
        },
        'driver_standings': {
            'path': 'data/f1db-races-driver-standings.csv',
            'required_cols': ['raceId', 'driverId', 'positionNumber', 'points']
        }
    }
    
    # Load each dataset
    for name, config in dataset_configs.items():
        df = load_csv_with_validation(config['path'], config['required_cols'])
        if df is not None:
            datasets[name] = df
        else:
            logger.warning(f"Failed to load {name} dataset, continuing without it")
    
    # Check if we have minimum required datasets
    required_datasets = ['grid', 'results', 'drivers', 'constructors', 'races']
    missing_required = set(required_datasets) - set(datasets.keys())
    if missing_required:
        logger.error(f"Missing critical datasets: {missing_required}")
        raise ValueError(f"Cannot proceed without required datasets: {missing_required}")
    
    return datasets

# Load all datasets
logger.info("Starting data loading process...")
datasets = load_all_datasets()

def validate_position_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate position data and filter out invalid entries.
    
    Args:
        df: DataFrame with position data
        
    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)
    
    # Remove rows with invalid positions (should be 1-20 for modern F1)
    df = df[df['grid_position'].between(1, 26)]  # Allow up to 26 for historical races
    df = df[df['position'].between(1, 26)]
    
    # Remove rows where position change is unreasonably large (>20 positions)
    df = df[abs(df['position'] - df['grid_position']) <= 20]
    
    logger.info(f"Position validation: {initial_rows} -> {len(df)} rows ({initial_rows - len(df)} removed)")
    return df

def compute_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute both classification and regression target variables.
    
    Args:
        df: DataFrame with grid_position and position columns
        
    Returns:
        DataFrame with added target variables
    """
    # Classification target: position_drop_flag (1 if finished worse than started)
    df['position_drop_flag'] = (df['position'] > df['grid_position']).astype(int)
    
    # Regression target: position_change_numeric (positive = dropped positions)
    df['position_change_numeric'] = df['position'] - df['grid_position']
    
    # Validation checks
    assert df['position_drop_flag'].isin([0, 1]).all(), "position_drop_flag must be 0 or 1"
    assert df['position_change_numeric'].between(-25, 25).all(), "position_change_numeric out of reasonable range"
    
    # Log target distribution
    drop_rate = df['position_drop_flag'].mean()
    avg_change = df['position_change_numeric'].mean()
    logger.info(f"Target variables computed:")
    logger.info(f"  - Position drop rate: {drop_rate:.3f}")
    logger.info(f"  - Average position change: {avg_change:.3f}")
    logger.info(f"  - Position change range: [{df['position_change_numeric'].min()}, {df['position_change_numeric'].max()}]")
    
    return df

def test_target_computation():
    """
    Unit tests for target computation with known examples.
    """
    # Test data with known outcomes
    test_data = pd.DataFrame({
        'grid_position': [1, 5, 10, 3, 8],
        'position': [3, 2, 15, 3, 1]  # dropped 2, gained 3, dropped 5, same, gained 7
    })
    
    result = compute_target_variables(test_data.copy())
    
    # Expected results
    expected_flags = [1, 0, 1, 0, 0]  # drop, gain, drop, same, gain
    expected_changes = [2, -3, 5, 0, -7]  # positive = dropped
    
    assert result['position_drop_flag'].tolist() == expected_flags, f"Flag mismatch: {result['position_drop_flag'].tolist()}"
    assert result['position_change_numeric'].tolist() == expected_changes, f"Change mismatch: {result['position_change_numeric'].tolist()}"
    
    logger.info("Target computation tests passed!")

# Run unit tests
test_target_computation()

def standardize_column_names(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Standardize column names across datasets for consistent merging.
    
    Args:
        datasets: Dictionary of dataset name to DataFrame
        
    Returns:
        Dictionary with standardized DataFrames
    """
    # Standardize grid positions dataset
    if 'grid' in datasets:
        grid = datasets['grid'].copy()
        grid = grid.rename(columns={
            'raceId': 'race_id',
            'driverId': 'driver_id',
            'positionNumber': 'grid_position'
        })
        datasets['grid'] = grid
    
    # Standardize results dataset
    if 'results' in datasets:
        results = datasets['results'].copy()
        results = results.rename(columns={
            'raceId': 'race_id',
            'driverId': 'driver_id',
            'constructorId': 'constructor_id',
            'positionNumber': 'position'
        })
        datasets['results'] = results
    
    # Standardize drivers dataset
    if 'drivers' in datasets:
        drivers = datasets['drivers'].copy()
        drivers = drivers.rename(columns={
            'id': 'driver_id',
            'lastName': 'surname'
        })
        datasets['drivers'] = drivers
    
    # Standardize constructors dataset
    if 'constructors' in datasets:
        constructors = datasets['constructors'].copy()
        constructors = constructors.rename(columns={
            'id': 'constructor_id'
        })
        datasets['constructors'] = constructors
    
    # Standardize races dataset
    if 'races' in datasets:
        races = datasets['races'].copy()
        races = races.rename(columns={
            'id': 'race_id',
            'year': 'season',
            'circuitId': 'circuit_id'
        })
        datasets['races'] = races
    
    # Standardize pit stops dataset
    if 'pit_stops' in datasets:
        pit_stops = datasets['pit_stops'].copy()
        pit_stops = pit_stops.rename(columns={
            'raceId': 'race_id',
            'driverId': 'driver_id'
        })
        datasets['pit_stops'] = pit_stops
    
    # Standardize driver standings dataset
    if 'driver_standings' in datasets:
        standings = datasets['driver_standings'].copy()
        standings = standings.rename(columns={
            'raceId': 'race_id',
            'driverId': 'driver_id',
            'positionNumber': 'championship_position'
        })
        datasets['driver_standings'] = standings
    
    # Standardize circuits dataset
    if 'circuits' in datasets:
        circuits = datasets['circuits'].copy()
        circuits = circuits.rename(columns={'id': 'circuit_id'})
        datasets['circuits'] = circuits
    
    return datasets

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values with appropriate strategies.
    
    Args:
        df: DataFrame with potential missing values
        
    Returns:
        DataFrame with missing values handled
    """
    initial_rows = len(df)
    
    # Debug: print available columns
    logger.info(f"Available columns: {list(df.columns)}")
    
    # Critical columns that cannot be missing
    critical_cols = ['race_id', 'driver_id', 'grid_position', 'position', 'constructor_id']
    df = df.dropna(subset=critical_cols)
    
    # Forward-fill time series data (season-based)
    if 'season' in df.columns:
        # Check if round column exists, if not use race_id for sorting
        sort_cols = ['season', 'driver_id']
        if 'round' in df.columns:
            sort_cols.insert(1, 'round')
        df = df.sort_values(sort_cols)
        
        # Forward-fill numeric columns within each driver's career
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in critical_cols:
                df[col] = df.groupby('driver_id')[col].ffill()
    
    # Fill remaining missing values with appropriate defaults
    if 'championship_position' in df.columns:
        df['championship_position'] = df['championship_position'].fillna(20)  # Assume low position if missing
    
    if 'points' in df.columns:
        df['points'] = df['points'].fillna(0)  # No points if missing
    
    logger.info(f"Missing value handling: {initial_rows} -> {len(df)} rows ({initial_rows - len(df)} removed)")
    return df

def merge_all_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all datasets on appropriate keys.
    
    Args:
        datasets: Dictionary of dataset name to DataFrame
        
    Returns:
        Merged DataFrame
    """
    # Start with race results as the base
    df = datasets['results'].copy()
    logger.info(f"Starting with results: {len(df)} rows")
    
    # Merge with starting grid positions
    df = df.merge(
        datasets['grid'][['race_id', 'driver_id', 'grid_position']], 
        on=['race_id', 'driver_id'], 
        how='left'
    )
    logger.info(f"After grid merge: {len(df)} rows")
    
    # Merge with driver info
    df = df.merge(
        datasets['drivers'][['driver_id', 'surname']], 
        on='driver_id', 
        how='left'
    )
    logger.info(f"After drivers merge: {len(df)} rows")
    
    # Merge with constructor info
    df = df.merge(
        datasets['constructors'][['constructor_id', 'name']], 
        on='constructor_id', 
        how='left'
    )
    logger.info(f"After constructors merge: {len(df)} rows")
    
    # Merge with race info
    df = df.merge(
        datasets['races'][['race_id', 'season', 'round', 'circuit_id']], 
        on='race_id', 
        how='left'
    )
    logger.info(f"After races merge: {len(df)} rows")
    
    # Merge with circuits info (optional)
    if 'circuits' in datasets:
        df = df.merge(
            datasets['circuits'][['circuit_id', 'name', 'type', 'length']], 
            on='circuit_id', 
            how='left',
            suffixes=('', '_circuit')
        )
        logger.info(f"After circuits merge: {len(df)} rows")
    
    # Merge with driver standings (optional)
    if 'driver_standings' in datasets:
        df = df.merge(
            datasets['driver_standings'][['race_id', 'driver_id', 'championship_position', 'points']], 
            on=['race_id', 'driver_id'], 
            how='left'
        )
        logger.info(f"After driver standings merge: {len(df)} rows")
    
    # Add pit stop aggregations (optional)
    if 'pit_stops' in datasets:
        pit_agg = datasets['pit_stops'].groupby(['race_id', 'driver_id']).agg({
            'stop': 'count',  # Number of pit stops
            'timeMillis': ['mean', 'std']  # Average and std of pit times
        }).reset_index()
        
        # Flatten column names
        pit_agg.columns = ['race_id', 'driver_id', 'pit_stop_count', 'avg_pit_time', 'pit_time_std']
        pit_agg['pit_time_std'] = pit_agg['pit_time_std'].fillna(0)  # Fill NaN std with 0
        
        df = df.merge(pit_agg, on=['race_id', 'driver_id'], how='left')
        logger.info(f"After pit stops merge: {len(df)} rows")
        
        # Fill missing pit stop data with defaults
        df['pit_stop_count'] = df['pit_stop_count'].fillna(0)
        df['avg_pit_time'] = df['avg_pit_time'].fillna(df['avg_pit_time'].median())
        df['pit_time_std'] = df['pit_time_std'].fillna(0)
    
    return df

def create_final_dataset() -> pd.DataFrame:
    """
    Create the final clean dataset for ML.
    
    Returns:
        Clean DataFrame ready for ML
    """
    # Standardize column names
    datasets_std = standardize_column_names(datasets)
    
    # Merge all datasets
    df = merge_all_datasets(datasets_std)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Validate position data
    df = validate_position_data(df)
    
    # Compute target variables
    df = compute_target_variables(df)
    
    # Filter to reasonable time period (modern F1 era) to get ~2100 rows
    # Focus on 2000 onwards for more consistent data
    df = df[df['season'] >= 2000]
    
    # Select final columns for ML (handle duplicate columns from merges)
    ml_columns = [
        'season', 'driver_id', 'surname', 'constructor_id', 'name_circuit',
        'grid_position', 'position', 'championship_position', 
        'pit_stop_count', 'avg_pit_time', 'pit_time_std',
        'type', 'length', 'position_drop_flag', 'position_change_numeric'
    ]
    
    # Handle duplicate columns by selecting the right one
    if 'round_y' in df.columns:
        df['round'] = df['round_y']
        ml_columns.insert(1, 'round')
    elif 'round_x' in df.columns:
        df['round'] = df['round_x'] 
        ml_columns.insert(1, 'round')
    
    if 'points_y' in df.columns:
        df['points'] = df['points_y']
        ml_columns.insert(-2, 'points')
    elif 'points_x' in df.columns:
        df['points'] = df['points_x']
        ml_columns.insert(-2, 'points')
    
    # Only keep columns that exist in the dataframe
    available_columns = [col for col in ml_columns if col in df.columns]
    df_final = df[available_columns].copy()
    
    # Rename columns for clarity
    column_mapping = {
        'surname': 'driver_name', 
        'name_circuit': 'circuit_name',
        'type': 'circuit_type',
        'length': 'circuit_length'
    }
    df_final = df_final.rename(columns=column_mapping)
    
    logger.info(f"Final dataset created: {len(df_final)} rows, {len(df_final.columns)} columns")
    logger.info(f"Columns: {list(df_final.columns)}")
    
    return df_final

# Create the final dataset
logger.info("Creating final merged and cleaned dataset...")
final_df = create_final_dataset()

# Save clean CSV
output_path = 'data/f1_performance_drop_clean.csv'
final_df.to_csv(output_path, index=False)
logger.info(f"Clean dataset saved: {output_path}")

# Display summary statistics
logger.info("Dataset Summary:")
logger.info(f"Shape: {final_df.shape}")
logger.info(f"Date range: {final_df['season'].min()} - {final_df['season'].max()}")
logger.info(f"Unique drivers: {final_df['driver_id'].nunique()}")
logger.info(f"Unique seasons: {final_df['season'].nunique()}")
logger.info(f"Position drop rate: {final_df['position_drop_flag'].mean():.3f}")

print("Data preparation completed successfully!")
print(f"Final dataset: {len(final_df)} rows")
print("\nFirst 5 rows:")
print(final_df.head())