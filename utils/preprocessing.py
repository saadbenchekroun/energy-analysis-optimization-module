import pandas as pd

def preprocess_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess energy data for modeling"""
    # Convert timestamp to datetime if needed
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Add working hours feature (8 AM to 6 PM)
    df['is_working_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    
    return df
