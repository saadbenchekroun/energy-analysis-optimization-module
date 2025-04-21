import pandas as pd
import numpy as np
from models.schemas import RecommendationRequest
from typing import Dict

async def generate_lighting_recommendations(
    energy_data: pd.DataFrame,
    config: RecommendationRequest
) -> Dict:
    """Generate lighting efficiency recommendations"""
    # Calculate average lighting energy usage
    # For demonstration, we'll assume lighting is about 20% of total energy
    avg_daily_energy = energy_data.resample('D', on='timestamp')['value'].mean().mean()
    lighting_energy = avg_daily_energy * 0.2  # Estimated lighting portion
    
    # Calculate potential savings for different interventions
    led_savings_pct = 0.65  # 65% savings from switching to LED
    occupancy_savings_pct = 0.35  # 35% savings from occupancy sensors
    natural_light_savings_pct = 0.20  # 20% savings from better using natural light
    dimming_savings_pct = 0.25  # 25% savings from dimming systems
    
    # Electricity price (for calculating financial savings)
    avg_price = 0.15  # $/kWh - would come from price data in a real system
    
    recommendations = [
        {
            "title": "LED Lighting Upgrade",
            "description": "Replace conventional lighting with LED fixtures for significant energy savings.",
            "annual_kwh_savings": lighting_energy * 365 * led_savings_pct,
            "annual_cost_savings": lighting_energy * 365 * led_savings_pct * avg_price,
            "estimated_implementation_cost": lighting_energy * 365 * avg_price * led_savings_pct * 2,  # 2-year payback as estimate
            "payback_period_years": 2,
            "roi": 50,  # 50% annual ROI
            "priority": "High"
        },
        {
            "title": "Occupancy Sensors",
            "description": "Install occupancy sensors in areas with variable usage patterns.",
            "annual_kwh_savings": lighting_energy * 365 * occupancy_savings_pct,
            "annual_cost_savings": lighting_energy * 365 * occupancy_savings_pct * avg_price,
            "estimated_implementation_cost": lighting_energy * 365 * avg_price * occupancy_savings_pct * 1.5,  # 1.5-year payback
            "payback_period_years": 1.5,
            "roi": 67,  # 67% annual ROI
            "priority": "Medium"
        },
        {
            "title": "Natural Light Optimization",
            "description": "Reconfigure workspace to maximize natural light and reduce artificial lighting needs.",
            "annual_kwh_savings": lighting_energy * 365 * natural_light_savings_pct,
            "annual_cost_savings": lighting_energy * 365 * natural_light_savings_pct * avg_price,
            "estimated_implementation_cost": lighting_energy * 365 * avg_price * natural_light_savings_pct * 1,  # 1-year payback
            "payback_period_years": 1,
            "roi": 100,  # 100% annual ROI
            "priority": "Medium"
        },
        {
            "title": "Lighting Control Systems",
            "description": "Implement automated dimming and scheduling for lighting systems.",
            "annual_kwh_savings": lighting_energy * 365 * dimming_savings_pct,
            "annual_cost_savings": lighting_energy * 365 * dimming_savings_pct * avg_price,
            "estimated_implementation_cost": lighting_energy * 365 * avg_price * dimming_savings_pct * 3,  # 3-year payback
            "payback_period_years": 3,
            "roi": 33,  # 33% annual ROI
            "priority": "Low"
        }
    ]
    
    # Calculate total potential savings
    total_kwh_savings = sum(r["annual_kwh_savings"] for r in recommendations)
    total_cost_savings = sum(r["annual_cost_savings"] for r in recommendations)
    total_implementation_cost = sum(r["estimated_implementation_cost"] for r in recommendations)
    
    result = {
        "recommendations": recommendations,
        "potential_savings": {
            "annual_kwh": total_kwh_savings,
            "annual_cost": total_cost_savings,
            "percentage_of_lighting": sum([led_savings_pct, occupancy_savings_pct, natural_light_savings_pct, dimming_savings_pct]),
            "percentage_of_total_energy": sum([led_savings_pct, occupancy_savings_pct, natural_light_savings_pct, dimming_savings_pct]) * 0.2
        },
        "implementation_costs": {
            "total": total_implementation_cost,
            "average_payback_years": total_implementation_cost / total_cost_savings
        }
    }
    
    return result

async def generate_hvac_recommendations(
    energy_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    config: RecommendationRequest
) -> Dict:
    """Generate HVAC efficiency recommendations"""
    # Calculate average HVAC energy usage
    # For demonstration, we'll assume HVAC is about 40% of total energy
    avg_daily_energy = energy_data.resample('D', on='timestamp')['value'].mean().mean()
    hvac_energy = avg_daily_energy * 0.4  # Estimated HVAC portion
    
    # Calculate correlations between temperature and energy use if weather data available
    temperature_correlation = None
    if 'temperature' in weather_data.columns:
        merged_data = pd.merge_asof(
            energy_data.sort_values('timestamp'),
            weather_data.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        temperature_correlation = merged_data['value'].corr(merged_data['temperature'])
    
    # Calculate potential savings for different interventions
    setpoint_savings_pct = 0.08  # 8% savings from optimal setpoints
    maintenance_savings_pct = 0.15  # 15% savings from regular maintenance
    scheduling_savings_pct = 0.12  # 12% savings from better scheduling
    modernization_savings_pct = 0.30  # 30% savings from system modernization
    
    # Electricity price (for calculating financial savings)
    avg_price = 0.15  # $/kWh - would come from price data in a real system
    
    recommendations = [
        {
            "title": "Optimal Temperature Setpoints",
            "description": "Adjust temperature setpoints to 68°F (20°C) in heating season and 76°F (24°C) in cooling season.",
            "annual_kwh_savings": hvac_energy * 365 * setpoint_savings_pct,
            "annual_cost_savings": hvac_energy * 365 * setpoint_savings_pct * avg_price,
            "estimated_implementation_cost": 0,  # No implementation cost
            "payback_period_years": 0,
            "roi": float('inf'),  # Infinite ROI due to zero cost
            "priority": "High"
        },
        {
            "title": "Preventive Maintenance Program",
            "description": "Implement regular HVAC maintenance including filter changes, coil cleaning, and system tuning.",
            "annual_kwh_savings": hvac_energy * 365 * maintenance_savings_pct,
            "annual_cost_savings": hvac_energy * 365 * maintenance_savings_pct * avg_price,
            "estimated_implementation_cost": hvac_energy * 365 * avg_price * maintenance_savings_pct * 0.5,  # 6-month payback
            "payback_period_years": 0.5,
            "roi": 200,  # 200% annual ROI
            "priority": "High"
        },
        {
            "title": "Optimized HVAC Scheduling",
            "description": "Program HVAC systems to match occupancy patterns and pre-heat/pre-cool efficiently.",
            "annual_kwh_savings": hvac_energy * 365 * scheduling_savings_pct,
            "annual_cost_savings": hvac_energy * 365 * scheduling_savings_pct * avg_price,
            "estimated_implementation_cost": hvac_energy * 365 * avg_price * scheduling_savings_pct * 1,  # 1-year payback
            "payback_period_years": 1,
            "roi": 100,  # 100% annual ROI
            "priority": "Medium"
        },
        {
            "title": "HVAC System Modernization",
            "description": "Upgrade to high-efficiency HVAC equipment with variable speed drives and smart controls.",
            "annual_kwh_savings": hvac_energy * 365 * modernization_savings_pct,
            "annual_cost_savings": hvac_energy * 365 * modernization_savings_pct * avg_price,
            "estimated_implementation_cost": hvac_energy * 365 * avg_price * modernization_savings_pct * 4,  # 4-year payback
            "payback_period_years": 4,
            "roi": 25,  # 25% annual ROI
            "priority": "Low"
        }
    ]
    
    # Calculate total potential savings
    total_kwh_savings = sum(r["annual_kwh_savings"] for r in recommendations)
    total_cost_savings = sum(r["annual_cost_savings"] for r in recommendations)
    total_implementation_cost = sum(r["estimated_implementation_cost"] for r in recommendations)
    
    result = {
        "recommendations": recommendations,
        "potential_savings": {
            "annual_kwh": total_kwh_savings,
            "annual_cost": total_cost_savings,
            "percentage_of_hvac": sum([setpoint_savings_pct, maintenance_savings_pct, scheduling_savings_pct, modernization_savings_pct]),
            "percentage_of_total_energy": sum([setpoint_savings_pct, maintenance_savings_pct, scheduling_savings_pct, modernization_savings_pct]) * 0.4
        },
        "implementation_costs": {
            "total": total_implementation_cost,
            "average_payback_years": total_implementation_cost / total_cost_savings
        }
    }
    
    # Add temperature correlation insight if available
    if temperature_correlation is not None:
        result["insights"] = {
            "temperature_correlation": temperature_correlation,
            "interpretation": (
                "Strong positive correlation between temperature and energy usage, suggesting cooling dominates" 
                if temperature_correlation > 0.6 else
                "Strong negative correlation between temperature and energy usage, suggesting heating dominates"
                if temperature_correlation < -0.6 else
                "Moderate correlation between temperature and energy usage, suggesting balanced heating and cooling"
            )
        }
    
    return result