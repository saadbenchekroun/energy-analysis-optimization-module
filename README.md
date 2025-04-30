#  Energy Analysis & Optimization API

A modular, production-ready FastAPI backend for analyzing, forecasting, optimizing, and generating energy efficiency recommendations from sensor, weather, and pricing data.

---

##  Features

-  Baseline model generation (`RandomForest`, `ARIMA`, `Prophet`)
-  Anomaly detection (`Isolation Forest`, `DBSCAN`, `Z-score`)
-  Forecasting (short-term energy usage using `ARIMA` or `Prophet`)
-  Optimization:
  - Load Shifting
  - Demand Response
  - Energy Storage
-  Smart Recommendations for:
  - Lighting efficiency
  - HVAC performance
-  Modular file structure for maintainability
-  Model registry with metadata logging

---


##  Database Schema

Ensure the following tables exist in your PostgreSQL (or compatible) database:

- `energy_readings`
- `weather_data`
- `demand_response_events`
- `model_registry`

Use `utils/data_access.py` to manage data retrieval.

---

##  Getting Started

###  Installation

```
git clone https://github.com/your-org/energy-analysis-api.git
cd energy-analysis-api
pip install -r requirements.txt
```

### Environment Variables
Create a .env file or export manually:

```
MODELS_DIR=./models
DATABASE_URL=postgresql://user:pass@localhost:5432/yourdb
```

### Run the API

```
uvicorn main:app --reload
```

## API Endpoints
1. /analysis/baseline-model
Create a baseline prediction model for a specific metric.

2. /analysis/detect-anomalies
Detect anomalies using selected method.

3. /analysis/forecast
Forecast future energy usage.

4. /analysis/optimize
Run an optimization for load shifting, DR, or energy storage.

5. /analysis/recommendations
Generate energy efficiency recommendations.

📦 Dependencies
FastAPI

Uvicorn

scikit-learn

statsmodels

Prophet

pandas, numpy

pulp (linear programming)

asyncpg

## Metrics & Logging
All model metadata is stored in the model_registry table including:

Model type

Features

Training window

Performance metrics (MAE, RMSE, etc.)
