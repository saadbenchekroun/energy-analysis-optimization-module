import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from models.schemas import BaselineModelConfig

# Baseline modeling functions
async def create_random_forest_baseline(
    energy_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    config: BaselineModelConfig
) -> Tuple[RandomForestRegressor, Dict]:
    """Create a random forest baseline model"""
    # Merge energy and weather data
    merged_data = pd.merge_asof(
        energy_data.sort_values('timestamp'),
        weather_data.sort_values('timestamp'),
        on='timestamp',
        direction='nearest'
    )
    
    # Prepare features
    feature_columns = []
    for feature in config.features:
        if feature in merged_data.columns:
            feature_columns.append(feature)
    
    X = merged_data[feature_columns]
    y = merged_data['value']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with GridSearchCV for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'best_params': grid_search.best_params_
    }
    
    return best_model, metrics

async def create_arima_baseline(
    energy_data: pd.DataFrame,
    config: BaselineModelConfig
) -> Tuple[ARIMA, Dict]:
    """Create an ARIMA baseline model"""
    # Resample data to hourly intervals if needed
    if len(energy_data) > 1000:  # If the dataset is too large, sample it
        energy_data = energy_data.set_index('timestamp').resample('1H').mean().reset_index()
    
    # Remove any NaN values
    energy_data = energy_data.dropna(subset=['value'])
    
    # Prepare time series data
    ts_data = energy_data.set_index('timestamp')['value']
    
    # Find best ARIMA parameters using auto_arima from pmdarima (if available)
    try:
        from pmdarima import auto_arima
        auto_model = auto_arima(
            ts_data, 
            seasonal=True, 
            m=24,  # Daily seasonality
            suppress_warnings=True,
            error_action='ignore'
        )
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
    except ImportError:
        # Default parameters if auto_arima is not available
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 24)
    
    # Split data for validation
    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]
    
    # Fit SARIMAX model (ARIMA with seasonality)
    model = SARIMAX(
        train_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted_model = model.fit(disp=False)
    
    # Make predictions
    forecast = fitted_model.get_forecast(steps=len(test_data))
    pred_mean = forecast.predicted_mean
    
    # Evaluate model
    metrics = {
        'mae': mean_absolute_error(test_data, pred_mean),
        'rmse': np.sqrt(mean_squared_error(test_data, pred_mean)),
        'order': str(order),
        'seasonal_order': str(seasonal_order)
    }
    
    return fitted_model, metrics

async def create_prophet_baseline(
    energy_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    config: BaselineModelConfig
) -> Tuple[Prophet, Dict]:
    """Create a Prophet baseline model"""
    # Prepare data for Prophet (needs 'ds' for dates and 'y' for values)
    prophet_data = energy_data[['timestamp', 'value']].rename(columns={'timestamp': 'ds', 'value': 'y'})
    
    # Add weather regressors if available
    merged_data = pd.merge_asof(
        prophet_data.sort_values('ds'),
        weather_data.rename(columns={'timestamp': 'ds'}).sort_values('ds'),
        on='ds',
        direction='nearest'
    )
    
    # Select relevant features for regressors
    regressor_features = [f for f in config.features if f in weather_data.columns]
    
    # Initialize and train model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    # Add regressors
    for feature in regressor_features:
        if feature in merged_data.columns:
            model.add_regressor(feature)
    
    # Fit model
    model.fit(merged_data)
    
    # Generate forecast for the existing data to evaluate
    forecast = model.predict(merged_data)
    
    # Evaluate model
    metrics = {
        'mae': mean_absolute_error(merged_data['y'], forecast['yhat']),
        'rmse': np.sqrt(mean_squared_error(merged_data['y'], forecast['yhat'])),
        'regressors_used': regressor_features
    }
    
    return model, metrics