from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# Pydantic models for API requests and responses
class BaselineModelConfig(BaseModel):
    source: str
    metric_name: str
    source_id: Optional[str] = None
    model_type: str = "random_forest"  # Options: "random_forest", "arima", "prophet"
    features: List[str] = ["hour", "day_of_week", "month", "temperature"]
    training_start: Optional[str] = None
    training_end: Optional[str] = None

class AnomalyDetectionConfig(BaseModel):
    source: str
    metric_name: str
    source_id: Optional[str] = None
    detection_method: str = "isolation_forest"  # Options: "isolation_forest", "dbscan", "zscore"
    sensitivity: float = 0.95  # Higher = more sensitive
    window_size: int = 24  # Hours of data to consider

class ForecastConfig(BaseModel):
    source: str
    metric_name: str
    source_id: Optional[str] = None
    model_type: str = "prophet"  # Options: "prophet", "arima", "sarimax"
    horizon: int = 24  # Hours to forecast
    features: List[str] = ["hour", "day_of_week", "month", "temperature"]

class OptimizationConfig(BaseModel):
    optimization_type: str  # "load_shifting", "demand_response", "energy_storage"
    start_time: str
    end_time: str
    constraints: Dict = {}
    parameters: Dict = {}

class RecommendationRequest(BaseModel):
    recommendation_type: str  # "lighting", "hvac", "production", "general"
    source_ids: Optional[List[str]] = None
    time_period: Optional[str] = "1 month"  # Period to analyze

# Response models
class BaselineModelResponse(BaseModel):
    model_id: str
    source: str
    metric_name: str
    source_id: Optional[str]
    model_type: str
    features: List[str]
    training_period: Dict[str, str]
    performance_metrics: Dict[str, float]
    created_at: str

class AnomalyDetectionResponse(BaseModel):
    detection_id: str
    anomalies: List[Dict]
    total_points: int
    anomaly_points: int
    anomaly_percentage: float
    detection_method: str
    execution_time: float

class ForecastResponse(BaseModel):
    forecast_id: str
    source: str
    metric_name: str
    source_id: Optional[str]
    model_type: str
    horizon: int
    forecast_points: List[Dict]
    confidence_intervals: Optional[List[Dict]] = None
    performance_metrics: Optional[Dict[str, float]] = None

class OptimizationResponse(BaseModel):
    optimization_id: str
    optimization_type: str
    schedule: List[Dict]
    savings: Dict
    constraints_applied: Dict
    execution_time: float

class RecommendationResponse(BaseModel):
    recommendation_id: str
    recommendation_type: str
    recommendations: List[Dict]
    potential_savings: Dict
    implementation_costs: Optional[Dict] = None
    payback_period: Optional[float] = None
    roi: Optional[float] = None
