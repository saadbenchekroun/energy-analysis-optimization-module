import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
from models.schemas import AnomalyDetectionConfig

# Anomaly detection functions
def detect_anomalies_isolation_forest(
    data: pd.DataFrame,
    config: AnomalyDetectionConfig
) -> Tuple[pd.DataFrame, Dict]:
    """Detect anomalies using Isolation Forest"""
    # Prepare features for anomaly detection
    feature_columns = ['value', 'hour', 'day_of_week', 'is_weekend']
    
    # Add temperature if available
    if 'temperature' in data.columns:
        feature_columns.append('temperature')
    
    # Initialize and fit model
    model = IsolationForest(
        contamination=1.0 - config.sensitivity,
        random_state=42
    )
    
    # Fit model and predict anomalies
    anomaly_scores = model.fit_predict(data[feature_columns])
    
    # Add anomaly score to dataframe (-1 for anomalies, 1 for normal)
    data['anomaly'] = anomaly_scores == -1
    
    # Get anomalies
    anomalies = data[data['anomaly']].copy()
    
    # Calculate metrics
    metrics = {
        'total_points': len(data),
        'anomaly_points': len(anomalies),
        'anomaly_percentage': len(anomalies) / len(data) * 100
    }
    
    return anomalies, metrics

def detect_anomalies_dbscan(
    data: pd.DataFrame,
    config: AnomalyDetectionConfig
) -> Tuple[pd.DataFrame, Dict]:
    """Detect anomalies using DBSCAN clustering"""
    # Prepare features for anomaly detection
    feature_columns = ['value', 'hour', 'day_of_week', 'is_weekend']
    
    # Add temperature if available
    if 'temperature' in data.columns:
        feature_columns.append('temperature')
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[feature_columns])
    
    # Initialize and fit DBSCAN
    eps = 0.5 + (1.0 - config.sensitivity)  # Higher sensitivity = lower eps
    min_samples = max(5, int(len(data) * 0.01))  # At least 1% of data points
    
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(scaled_features)
    
    # Add cluster labels to dataframe (-1 for anomalies)
    data['anomaly'] = clusters == -1
    
    # Get anomalies
    anomalies = data[data['anomaly']].copy()
    
    # Calculate metrics
    metrics = {
        'total_points': len(data),
        'anomaly_points': len(anomalies),
        'anomaly_percentage': len(anomalies) / len(data) * 100
    }
    
    return anomalies, metrics

def detect_anomalies_zscore(
    data: pd.DataFrame,
    config: AnomalyDetectionConfig
) -> Tuple[pd.DataFrame, Dict]:
    """Detect anomalies using Z-score method"""
    # Group by relevant context (e.g., hour of day, day of week)
    grouped = data.groupby(['hour', 'day_of_week'])
    
    # Flag for anomalies
    data['anomaly'] = False
    
    # Calculate z-scores for each group
    for name, group in grouped:
        if len(group) > 1:  # Need at least 2 points for std calculation
            mean = group['value'].mean()
            std = group['value'].std()
            
            if std > 0:
                z_scores = abs((group['value'] - mean) / std)
                threshold = abs(stats.norm.ppf((1 - config.sensitivity) / 2))
                anomaly_indices = group.index[z_scores > threshold]
                data.loc[anomaly_indices, 'anomaly'] = True
    
    
    anomalies = data[data['anomaly']].copy()
    
    metrics = {
        'total_points': len(data),
        'anomaly_points': len(anomalies),
        'anomaly_percentage': len(anomalies) / len(data) * 100
    }
    
    return anomalies, metrics
