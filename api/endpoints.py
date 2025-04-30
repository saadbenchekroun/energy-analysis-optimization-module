from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from models.schemas import (
    BaselineModelConfig, BaselineModelResponse, AnomalyDetectionConfig, AnomalyDetectionResponse,
    ForecastConfig, ForecastResponse, OptimizationConfig, OptimizationResponse,
    RecommendationRequest, RecommendationResponse
)
from utils.data_access import get_energy_data, get_weather_data, get_price_data, get_demand_response_events
from services.baseline import create_random_forest_baseline, create_arima_baseline, create_prophet_baseline
from services.anomaly import detect_anomalies_dbscan, detect_anomalies_zscore, detect_anomalies_isolation_forest
from services.forecast import create_arima_forecast, create_prophet_forecast
from services.optimization import optimize_load_shifting, optimize_demand_response, optimize_energy_storage
from services.recommendation import generate_lighting_recommendations, generate_hvac_recommendations
from utils.storage import save_model
from datetime import datetime, timedelta
from router import router

from utils.db import get_db_pool
import logging
import pandas as pd
import json

# API Endpoints
@router.post("/baseline-model", response_model=BaselineModelResponse)
async def create_baseline_model(
    config: BaselineModelConfig,
    background_tasks: BackgroundTasks,
    db_pool = Depends(get_db_pool)
):
    """Create a baseline model for energy data"""
    try:
        # Get start/end times
        if config.training_start:
            training_start = datetime.datetime.fromisoformat(config.training_start)
        else:
            training_start = datetime.datetime.now() - datetime.timedelta(days=30)
        
        if config.training_end:
            training_end = datetime.datetime.fromisoformat(config.training_end)
        else:
            training_end = datetime.datetime.now()
        
        # Get energy data
        energy_data = await get_energy_data(
            db_pool, 
            config.source, 
            config.metric_name, 
            config.source_id,
            training_start,
            training_end
        )
        
        # Get weather data for the same period
        weather_data = await get_weather_data(db_pool, training_start, training_end)
        
        # Create model based on type
        start_time = datetime.datetime.now()
        model_id = f"{config.source}_{config.metric_name}_{config.model_type}_{start_time.strftime('%Y%m%d%H%M%S')}"
        
        if config.model_type == "random_forest":
            model, metrics = await create_random_forest_baseline(energy_data, weather_data, config)
        elif config.model_type == "arima":
            model, metrics = await create_arima_baseline(energy_data, config)
        elif config.model_type == "prophet":
            model, metrics = await create_prophet_baseline(energy_data, weather_data, config)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {config.model_type}")
        
        # Save model
        model_path = save_model(model, model_id)
        
        # Create response
        response = BaselineModelResponse(
            model_id=model_id,
            source=config.source,
            metric_name=config.metric_name,
            source_id=config.source_id,
            model_type=config.model_type,
            features=config.features,
            training_period={
                "start": training_start.isoformat(),
                "end": training_end.isoformat()
            },
            performance_metrics=metrics,
            created_at=datetime.datetime.now().isoformat()
        )
        
        # Schedule background task to store model metadata in database
        background_tasks.add_task(store_model_metadata, db_pool, response)
        
        return response
    
    except Exception as e:
        logger.error(f"Error creating baseline model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating baseline model: {str(e)}")

@router.post("/detect-anomalies", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    config: AnomalyDetectionConfig,
    db_pool = Depends(get_db_pool)
):
    """Detect anomalies in energy data"""
    try:
        # Calculate time period
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(hours=config.window_size)
        
        # Get energy data
        energy_data = await get_energy_data(
            db_pool, 
            config.source, 
            config.metric_name, 
            config.source_id,
            start_time,
            end_time
        )
        
        # Get weather data for the same period
        # Get weather data for the same period
        weather_data = await get_weather_data(db_pool, start_time, end_time)
        
        # Merge data
        merged_data = pd.merge_asof(
            energy_data.sort_values('timestamp'),
            weather_data.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        # Execute anomaly detection based on method
        start_execution = datetime.datetime.now()
        detection_id = f"{config.source}_{config.metric_name}_{config.detection_method}_{start_execution.strftime('%Y%m%d%H%M%S')}"
        
        if config.detection_method == "isolation_forest":
            anomalies, metrics = detect_anomalies_isolation_forest(merged_data, config)
        elif config.detection_method == "dbscan":
            anomalies, metrics = detect_anomalies_dbscan(merged_data, config)
        elif config.detection_method == "zscore":
            anomalies, metrics = detect_anomalies_zscore(merged_data, config)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported detection method: {config.detection_method}")
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_execution).total_seconds()
        
        # Format anomalies for response
        anomalies_list = []
        for _, row in anomalies.iterrows():
            anomaly_data = {
                "timestamp": row["timestamp"].isoformat(),
                "value": float(row["value"]),
                "expected_range": None  # Would be calculated in a more sophisticated system
            }
            
            # Add weather context if available
            if "temperature" in row:
                anomaly_data["temperature"] = float(row["temperature"])
            
            anomalies_list.append(anomaly_data)
        
        # Create response
        response = AnomalyDetectionResponse(
            detection_id=detection_id,
            anomalies=anomalies_list,
            total_points=metrics["total_points"],
            anomaly_points=metrics["anomaly_points"],
            anomaly_percentage=metrics["anomaly_percentage"],
            detection_method=config.detection_method,
            execution_time=execution_time
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting anomalies: {str(e)}")

@router.post("/forecast", response_model=ForecastResponse)
async def create_forecast(
    config: ForecastConfig,
    db_pool = Depends(get_db_pool)
):
    """Create energy forecast"""
    try:
        # Calculate time period for historical data
        end_time = datetime.datetime.now()
        # Use 5x the forecast horizon as training data
        start_time = end_time - datetime.timedelta(hours=config.horizon * 5)
        
        # Get energy data
        energy_data = await get_energy_data(
            db_pool, 
            config.source, 
            config.metric_name, 
            config.source_id,
            start_time,
            end_time
        )
        
        # Get weather data for the same period
        weather_data = await get_weather_data(db_pool, start_time, end_time)
        
        # Create forecast based on model type
        start_execution = datetime.datetime.now()
        forecast_id = f"{config.source}_{config.metric_name}_{config.model_type}_{start_execution.strftime('%Y%m%d%H%M%S')}"
        
        if config.model_type == "arima":
            forecast_df, metrics = await create_arima_forecast(energy_data, config)
        elif config.model_type == "prophet":
            forecast_df, metrics = await create_prophet_forecast(energy_data, weather_data, config)
        elif config.model_type == "sarimax":
            # Use ARIMA function with seasonal components
            forecast_df, metrics = await create_arima_forecast(energy_data, config)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported forecast model type: {config.model_type}")
        
        # Format forecast for response
        forecast_points = []
        for _, row in forecast_df.iterrows():
            point = {
                "timestamp": row["timestamp"].isoformat(),
                "forecast": float(row["forecast"])
            }
            
            # Add confidence intervals if available
            if "lower_bound" in row and "upper_bound" in row:
                point["lower_bound"] = float(row["lower_bound"])
                point["upper_bound"] = float(row["upper_bound"])
            
            forecast_points.append(point)
        
        # Create response
        response = ForecastResponse(
            forecast_id=forecast_id,
            source=config.source,
            metric_name=config.metric_name,
            source_id=config.source_id,
            model_type=config.model_type,
            horizon=config.horizon,
            forecast_points=forecast_points,
            performance_metrics=metrics
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error creating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating forecast: {str(e)}")

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_energy(
    config: OptimizationConfig,
    db_pool = Depends(get_db_pool)
):
    """Optimize energy usage"""
    try:
        # Parse time period
        start_time = datetime.datetime.fromisoformat(config.start_time)
        end_time = datetime.datetime.fromisoformat(config.end_time)
        
        # Get historical energy data (use 7 days prior to target period)
        historical_start = start_time - datetime.timedelta(days=7)
        historical_end = start_time - datetime.timedelta(hours=1)
        
        energy_data = await get_energy_data(
            db_pool, 
            "main_meter",  # Assuming main meter as default source
            "power",       # Assuming power as default metric
            None,          # No specific source_id
            historical_start,
            historical_end
        )
        
        # Get price data for the optimization period
        price_data = await get_price_data(db_pool, start_time, end_time)
        
        # Get weather data
        weather_data = await get_weather_data(db_pool, historical_start, end_time)
        
        # Start execution timer
        start_execution = datetime.datetime.now()
        optimization_id = f"{config.optimization_type}_{start_execution.strftime('%Y%m%d%H%M%S')}"
        
        # Execute optimization based on type
        if config.optimization_type == "load_shifting":
            result = await optimize_load_shifting(energy_data, price_data, weather_data, config)
        elif config.optimization_type == "demand_response":
            # Get demand response events
            dr_events = await get_demand_response_events(db_pool, start_time, end_time)
            result = await optimize_demand_response(energy_data, price_data, dr_events, config)
        elif config.optimization_type == "energy_storage":
            result = await optimize_energy_storage(energy_data, price_data, config)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported optimization type: {config.optimization_type}")
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_execution).total_seconds()
        result["execution_time"] = execution_time
        
        # Create response
        response = OptimizationResponse(
            optimization_id=optimization_id,
            optimization_type=config.optimization_type,
            schedule=result["schedule"],
            savings=result["savings"],
            constraints_applied=config.constraints,
            execution_time=execution_time
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in optimization: {str(e)}")

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    config: RecommendationRequest,
    db_pool = Depends(get_db_pool)
):
    """Get energy efficiency recommendations"""
    try:
        # Parse time period
        end_time = datetime.datetime.now()
        
        # Parse time period string (e.g., "1 month", "2 weeks")
        period_parts = config.time_period.split()
        if len(period_parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid time period format. Use '1 month', '2 weeks', etc.")
        
        quantity = int(period_parts[0])
        unit = period_parts[1].lower()
        
        # Convert to timedelta
        if unit in ["day", "days"]:
            start_time = end_time - datetime.timedelta(days=quantity)
        elif unit in ["week", "weeks"]:
            start_time = end_time - datetime.timedelta(weeks=quantity)
        elif unit in ["month", "months"]:
            start_time = end_time - datetime.timedelta(days=quantity * 30)  # Approximate
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported time unit: {unit}")
        
        # Get energy data
        source_filter = "main_meter"  # Default
        if config.source_ids and len(config.source_ids) > 0:
            source_filter = config.source_ids[0]  # Just use the first one for simplicity
        
        energy_data = await get_energy_data(
            db_pool, 
            source_filter,
            "power",  # Assuming power as default metric
            None,     # No specific source_id
            start_time,
            end_time
        )
        
        # Get weather data
        weather_data = await get_weather_data(db_pool, start_time, end_time)
        
        # Create recommendation ID
        recommendation_id = f"{config.recommendation_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Generate recommendations based on type
        if config.recommendation_type == "lighting":
            result = await generate_lighting_recommendations(energy_data, config)
        elif config.recommendation_type == "hvac":
            result = await generate_hvac_recommendations(energy_data, weather_data, config)
        elif config.recommendation_type == "production":
            # Placeholder - would implement specific production recommendations
            result = {
                "recommendations": [],
                "potential_savings": {"annual_kwh": 0, "annual_cost": 0}
            }
        elif config.recommendation_type == "general":
            # For general, generate recommendations from multiple categories and combine them
            lighting_results = await generate_lighting_recommendations(energy_data, config)
            hvac_results = await generate_hvac_recommendations(energy_data, weather_data, config)
            
            # Combine recommendations
            result = {
                "recommendations": lighting_results["recommendations"] + hvac_results["recommendations"],
                "potential_savings": {
                    "annual_kwh": lighting_results["potential_savings"]["annual_kwh"] + hvac_results["potential_savings"]["annual_kwh"],
                    "annual_cost": lighting_results["potential_savings"]["annual_cost"] + hvac_results["potential_savings"]["annual_cost"]
                }
            }
            
            # Sort by ROI if available
            if "recommendations" in result and result["recommendations"]:
                result["recommendations"].sort(key=lambda x: x.get("roi", 0), reverse=True)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported recommendation type: {config.recommendation_type}")
        
        # Create response
        total_implementation_cost = 0
        if "implementation_costs" in result and "total" in result["implementation_costs"]:
            total_implementation_cost = result["implementation_costs"]["total"]
        
        annual_cost_savings = result["potential_savings"]["annual_cost"]
        payback_period = None if annual_cost_savings == 0 else total_implementation_cost / annual_cost_savings
        roi = None if total_implementation_cost == 0 else (annual_cost_savings / total_implementation_cost) * 100
        
        response = RecommendationResponse(
            recommendation_id=recommendation_id,
            recommendation_type=config.recommendation_type,
            recommendations=result["recommendations"],
            potential_savings=result["potential_savings"],
            implementation_costs={"total": total_implementation_cost} if total_implementation_cost > 0 else None,
            payback_period=payback_period,
            roi=roi
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Database functions (for background tasks)
async def store_model_metadata(db_pool, model_data: BaselineModelResponse):
    """Store model metadata in database"""
    try:
        query = """
            INSERT INTO model_registry 
            (model_id, source, metric_name, source_id, model_type, features, 
             training_start, training_end, performance_metrics, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        
        async with db_pool.acquire() as conn:
            await conn.execute(
                query,
                model_data.model_id,
                model_data.source,
                model_data.metric_name,
                model_data.source_id,
                model_data.model_type,
                json.dumps(model_data.features),
                model_data.training_period["start"],
                model_data.training_period["end"],
                json.dumps(model_data.performance_metrics),
                model_data.created_at
            )
        
        logger.info(f"Model metadata stored for model {model_data.model_id}")
    
    except Exception as e:
        logger.error(f"Error storing model metadata: {str(e)}")
