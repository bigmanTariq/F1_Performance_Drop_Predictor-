# F1 Data Directory

This directory contains the Formula 1 datasets used for training the performance drop predictor.

## Required Data Files

The following CSV files are needed to run the complete pipeline:

### Core F1 Data Files
- `f1db-races-starting-grid-positions.csv` - Starting grid positions for all races
- `f1db-races-race-results.csv` - Race results and finishing positions
- `f1db-drivers.csv` - Driver information and metadata
- `f1db-constructors.csv` - Constructor/team information
- `f1db-races.csv` - Race information (circuits, dates, etc.)
- `f1db-races-pit-stops.csv` - Pit stop data and timing
- `f1db-circuits.csv` - Circuit characteristics and information
- `f1db-races-driver-standings.csv` - Championship standings data

### Generated Files
After running the data preparation pipeline:
- `f1_performance_drop_clean.csv` - Cleaned and merged dataset
- `f1_features_engineered.csv` - Dataset with engineered features

## Data Sources

The F1 data comes from the Ergast F1 API and other public F1 databases. 

## Getting the Data

1. **Download from Kaggle**: Search for "Formula 1 World Championship" datasets
2. **Use the F1 API**: Visit http://ergast.com/mrd/ for official data
3. **Run data collection**: Use the notebook `notebooks/01_data_pull.ipynb`

## Data Size

- Total raw data: ~46 CSV files
- Combined size: ~50-100 MB
- Processed data: ~8,200 race records

## Privacy and Usage

All F1 data used is publicly available and follows the terms of use of the respective data sources.