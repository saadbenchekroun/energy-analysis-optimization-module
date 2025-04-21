from fastapi import FastAPI
from api.router import router
import uvicorn

# Main API application setup
def create_analysis_app():
    app = FastAPI(
        title="Energy Analysis and Optimization API",
        description="API for energy data analysis, anomaly detection, forecasting, and optimization",
        version="1.0.0",
    )
    app.include_router(router)
    return app


if __name__ == "__main__":
    import uvicorn
    
    app = create_analysis_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)