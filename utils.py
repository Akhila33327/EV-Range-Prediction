import pandas as pd
import json
import re
import logging
import numpy as np
from typing import Dict, Tuple

# Set up logging for the project
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = "EV.csv"
CONFIG_PATH = "model_features.json"

def load_config() -> Dict:
    """Loads and returns the model features configuration."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {CONFIG_PATH}.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Configuration file {CONFIG_PATH} contains invalid JSON syntax.")
        raise

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aggressively standardizes column names to lowercase snake_case."""
    
    def clean_name(name: str) -> str:
        name = name.strip().lower().replace(' ', '_').replace('-', '_')
        name = re.sub(r'[^a-z0-9_/%]', '', name)
        # Mapping common variants to standardized keys used in the JSON
        name_mapping = {
            'vehicle_we': 'vehicle_we', 'weather_c': 'weather_c', 'slope_%': 'slope_%',
            'battery_vol': 'battery_vol', 'energy_consumption_kwh': 'energy_consumption_kwh'
        }
        for key, value in name_mapping.items():
            if key in name:
                return value
        return name

    df.columns = [clean_name(col) for col in df.columns]
    logging.info(f"Columns standardized. Cleaned headers: {list(df.columns)}")
    return df

def load_data() -> Tuple[pd.DataFrame, Dict]:
    """Loads CSV, standardizes columns, and converts categorical features to string."""
    config = load_config()
    try:
        df = pd.read_csv(DATA_PATH)
        df = standardize_columns(df) 
        
        # Explicitly cast categorical columns to string type
        categorical_cols = config.get('categorical_features', [])
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('missing')
            
        logging.info(f"Data loaded successfully with shape: {df.shape}")
        return df, config
    except FileNotFoundError:
        logging.error(f"Data file not found at {DATA_PATH}.")
        raise
    except Exception as e:
        logging.error(f"Error during data loading/cleaning: {e}")
        raise

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the engineered features ('e_demand')."""
    df = df.copy() 
    
    # Feature based on available driving load factors (standardized names)
    df['e_demand'] = (
        (df['speed_kmh'] + (df['slope_%'] * 5)) / df['vehicle_we']
    )
    
    # Ensure all engineered features are finite
    df['e_demand'] = df['e_demand'].replace([np.inf, -np.inf], np.nan)
    logging.info("Feature engineering complete: 'e_demand' created.")
    
    return df

def calculate_realistic_range(total_capacity_kwh: float, predicted_consumption_kwh: float, distance_km: float, 
                              battery_health_pct: float, reserve_pct: float) -> float:
    """
    Calculates the estimated remaining range with health and reserve factors.
    Range (km) = (Total Capacity * Health / Consumption per km) - Distance Traveled
    """
    if distance_km <= 0 or predicted_consumption_kwh <= 0:
        return 0.0

    # Apply health factor
    effective_capacity = total_capacity_kwh * (battery_health_pct / 100.0)
    
    consumption_per_km = predicted_consumption_kwh / distance_km
    total_range_possible = effective_capacity / consumption_per_km
    
    # Apply reserve factor
    reserve_factor = 1.0 - (reserve_pct / 100.0)
    
    remaining_range = (total_range_possible * reserve_factor) - distance_km
    
    return max(0.0, remaining_range)

def sanitize_feature_names(names: np.ndarray) -> np.ndarray:
    """Removes special characters introduced by OneHotEncoder for clean plotting."""
    sanitized_names = []
    for name in names:
        # Remove prefixes introduced by ColumnTransformer
        name = name.replace("preprocessor__cat__", "")
        name = name.replace("preprocessor__num__", "")
        name = name.replace("preprocessor__eng__", "")
        # Replace categorical names for readability
        name = name.replace("road_type_", "Road Type = ")
        
        sanitized_names.append(name)
    return np.array(sanitized_names)