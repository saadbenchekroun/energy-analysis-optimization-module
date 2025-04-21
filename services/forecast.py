import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.schemas import ForecastConfig
from typing import Dict, Tuple

# Forecasting functions
async def create_arima_forecast(
    energy_data: pd.DataFrame,
    config: ForecastConfig
) -> Tuple[pd.DataFrame, Dict]:
    """Create ARIMA forecast"""
    # Resample data to hourly intervals if needed
    if len(energy_data) > 1000:  # If the dataset is too large, sample it
        energy_data = energy_data.set_index('timestamp').resample('1H').mean().reset_index()
    
    # Remove any NaN values
    energy_data = energy_data.dropna(subset=['value'])
    
    # Prepare time series data
    ts_data = energy_data.set_index('timestamp')['value']
    
    # Find best ARIMA parameters
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
    
    # Make predictions for test data
    test_forecast = fitted_model.get_forecast(steps=len(test_data))
    test_pred_mean = test_forecast.predicted_mean
    test_pred_ci = test_forecast.conf_int()
    
    # Evaluate model
    metrics = {
        'mae': mean_absolute_error(test_data, test_pred_mean),
        'rmse': np.sqrt(mean_squared_error(test_data, test_pred_mean)),
        'order': str(order),
        'seasonal_order': str(seasonal_order)
    }
    
    # Make future forecast
    future_forecast = fitted_model.get_forecast(steps=config.horizon)
    future_mean = future_forecast.predicted_mean
    future_ci = future_forecast.conf_int()
    
    # Prepare forecast dataframe
    forecast_df = pd.DataFrame({
        'timestamp': pd.date_range(
            start=energy_data['timestamp'].max() + pd.Timedelta(hours=1),
            periods=config.horizon,
            freq='H'
        ),
        'forecast': future_mean.values,
        'lower_bound': future_ci.iloc[:, 0].values,
        'upper_bound': future_ci.iloc[:, 1].values
    })
    
    return forecast_df, metrics

async def create_prophet_forecast(
    energy_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    config: ForecastConfig
) -> Tuple[pd.DataFrame, Dict]:
    """Create Prophet forecast"""
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
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=config.horizon, freq='H')
    
    # Add regressor values for future dates
    # In a real system, you would use weather forecasts
    # For demonstration, we'll use a simple method
    for feature in regressor_features:
        if feature == 'temperature':
            # Simple temperature model based on time of day
            future['hour'] = future['ds'].dt.hour
            future['temperature'] = 15 + 10 * np.sin((future['hour'] - 6) * np.pi / 12)
        elif feature == 'humidity':
            future['humidity'] = merged_data['humidity'].mean()
        else:
            # Copy the last available value for other features
            future[feature] = merged_data[feature].iloc[-1]
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Evaluate model on historical data
    historical_forecast = forecast[forecast['ds'].isin(merged_data['ds'])]
    metrics = {
        'mae': mean_absolute_error(merged_data['y'], historical_forecast['yhat']),
        'rmse': np.sqrt(mean_squared_error(merged_data['y'], historical_forecast['yhat'])),
        'regressors_used': regressor_features
    }
    
    # Prepare forecast dataframe (only future points)
    future_forecast = forecast[~forecast['ds'].isin(merged_data['ds'])]
    forecast_df = pd.DataFrame({
        'timestamp': future_forecast['ds'],
        'forecast': future_forecast['yhat'],
        'lower_bound': future_forecast['yhat_lower'],
        'upper_bound': future_forecast['yhat_upper']
    })
    
    return forecast_df, metrics
