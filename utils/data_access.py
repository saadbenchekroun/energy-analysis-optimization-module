import pandas as pd
import datetime
from utils.preprocessing import preprocess_energy_data
from typing import Optional
from utils.db import get_db_pool  

async def get_energy_data(
    db_pool, 
    source: str, 
    metric_name: str, 
    source_id: Optional[str] = None,
    start_time: Optional[datetime.datetime] = None,
    end_time: Optional[datetime.datetime] = None
) -> pd.DataFrame:
    """Get energy data from database"""
    if not start_time:
        start_time = datetime.datetime.now() - datetime.timedelta(days=30)
    if not end_time:
        end_time = datetime.datetime.now()
    
    # Build query
    query = """
        SELECT timestamp, source, source_id, metric_name, value, unit
        FROM energy_readings
        WHERE source = $1 
          AND metric_name = $2
          AND timestamp BETWEEN $3 AND $4
    """
    
    params = [source, metric_name, start_time, end_time]
    
    if source_id:
        query += " AND source_id = $5"
        params.append(source_id)
    
    query += " ORDER BY timestamp"
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=['timestamp', 'source', 'source_id', 'metric_name', 'value', 'unit'])
    
    # Preprocess the data
    df = preprocess_energy_data(df)
    
    return df

async def get_weather_data(
    db_pool,
    start_time: datetime.datetime,
    end_time: datetime.datetime
) -> pd.DataFrame:
    """Get weather data from database"""
    query = """
        SELECT timestamp, temperature, humidity, pressure, wind_speed, 
               wind_direction, cloud_coverage, solar_irradiance, description
        FROM weather_data
        WHERE timestamp BETWEEN $1 AND $2
        ORDER BY timestamp
    """
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, start_time, end_time)
    
    # Convert to DataFrame
    df = pd.DataFrame(
        rows, 
        columns=[
            'timestamp', 'temperature', 'humidity', 'pressure', 'wind_speed',
            'wind_direction', 'cloud_coverage', 'solar_irradiance', 'description'
        ]
    )
    
    return df

async def get_price_data(
    db_pool,
    start_time: datetime.datetime,
    end_time: datetime.datetime
) -> pd.DataFrame:
    """Get energy price data"""
    # For demonstration, we'll generate TOU pricing
    # In a real system, this would come from a database or an API
    
    date_range = pd.date_range(start=start_time, end=end_time, freq='H')
    prices = []
    
    for dt in date_range:
        hour = dt.hour
        is_weekend = dt.dayofweek >= 5
        
        # Simple TOU rate structure
        if is_weekend:
            price = 0.08  # Weekend rate
        elif 7 <= hour < 12:  # Morning mid-peak
            price = 0.12
        elif 12 <= hour < 17:  # Afternoon peak
            price = 0.22
        elif 17 <= hour < 22:  # Evening peak
            price = 0.26
        else:  # Night off-peak
            price = 0.07
        
        prices.append({
            'timestamp': dt,
            'price': price,
            'period': (
                'off_peak' if (hour < 7 or hour >= 22 or is_weekend) else
                'peak' if (12 <= hour < 17 or 17 <= hour < 22) else
                'mid_peak'
            )
        })
    
    return pd.DataFrame(prices)

async def get_demand_response_events(
    db_pool,
    start_time: datetime.datetime,
    end_time: datetime.datetime
) -> pd.DataFrame:
    """Get demand response events"""
    query = """
        SELECT event_id, start_time, end_time, type, price_signal, reduction_target, status
        FROM demand_response_events
        WHERE (start_time BETWEEN $1 AND $2) OR (end_time BETWEEN $1 AND $2)
        ORDER BY start_time
    """
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, start_time, end_time)
    
    # Convert to DataFrame
    df = pd.DataFrame(
        rows, 
        columns=[
            'event_id', 'start_time', 'end_time', 'type', 
            'price_signal', 'reduction_target', 'status'
        ]
    )
    
    return df
