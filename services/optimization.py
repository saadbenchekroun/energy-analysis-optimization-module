import pandas as pd
import numpy as np
import pulp
from datetime import datetime, timedelta
from models.schemas import OptimizationConfig
from typing import Dict, List
import logging

# Optimization functions
async def optimize_load_shifting(
    energy_data: pd.DataFrame,
    price_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    config: OptimizationConfig
) -> Dict:
    """Optimize load shifting"""
    # Parse time period
    start_time = datetime.datetime.fromisoformat(config.start_time)
    end_time = datetime.datetime.fromisoformat(config.end_time)
    
    # Get price data for the period
    period_prices = price_data[
        (price_data['timestamp'] >= start_time) & 
        (price_data['timestamp'] <= end_time)
    ]
    
    # Number of hours in the optimization period
    n_hours = len(period_prices)
    
    # Create PuLP problem
    prob = pulp.LpProblem("Load_Shifting_Optimization", pulp.LpMinimize)
    
    # Define decision variables: how much load to shift from hour i to hour j
    # We'll use a simplified approach where each hour has a fixed load that can be shifted
    base_load = energy_data['value'].mean()  # Use average load as basis
    
    # Default max percentage that can be shifted (from constraints)
    max_shift_pct = config.constraints.get('max_shift_percentage', 20) / 100.0
    
    # Define decision variables for each hour (1 if we run non-essential load, 0 if we defer)
    run_load = {}
    for i in range(n_hours):
        run_load[i] = pulp.LpVariable(f"run_load_{i}", 0, 1, pulp.LpBinary)
    
    # Define variables for each hour's load
    total_load = {}
    for i in range(n_hours):
        total_load[i] = pulp.LpVariable(f"total_load_{i}", 0, None, pulp.LpContinuous)
    
    # Define essential (non-shiftable) load for each hour based on historical pattern
    if 'hour' in energy_data.columns:
        # Group by hour of day to get typical load profile
        hourly_load = energy_data.groupby('hour')['value'].mean().to_dict()
        
        # Calculate essential and shiftable load for each hour
        essential_load = {}
        shiftable_load = {}
        for i in range(n_hours):
            hour_of_day = (start_time + timedelta(hours=i)).hour
            typical_load = hourly_load.get(hour_of_day, base_load)
            essential_load[i] = typical_load * (1 - max_shift_pct)
            shiftable_load[i] = typical_load * max_shift_pct
    else:
        # If hourly pattern not available, use flat load profile
        essential_load = {i: base_load * (1 - max_shift_pct) for i in range(n_hours)}
        shiftable_load = {i: base_load * max_shift_pct for i in range(n_hours)}
    
    # Objective: minimize cost
    prob += pulp.lpSum([
        period_prices.iloc[i]['price'] * total_load[i] for i in range(n_hours)
    ]), "Total_Cost"
    
    # Constraints
    # 1. Total load in each hour is essential load plus shiftable load if we run it
    for i in range(n_hours):
        prob += total_load[i] == essential_load[i] + shiftable_load[i] * run_load[i]
    
    # 2. All shiftable load must be served eventually (conservation of energy)
    prob += pulp.lpSum([shiftable_load[i] * run_load[i] for i in range(n_hours)]) == \
            pulp.lpSum([shiftable_load[i] for i in range(n_hours)])
    
    # 3. Apply any additional constraints from config
    # Maximum consecutive hours with deferred load
    max_defer_consecutive = config.constraints.get('max_defer_consecutive', 3)
    for i in range(n_hours - max_defer_consecutive + 1):
        prob += pulp.lpSum([1 - run_load[i+j] for j in range(max_defer_consecutive)]) <= max_defer_consecutive - 1
    
    # Maximum total load in any hour
    if 'max_total_load' in config.constraints:
        max_load = config.constraints['max_total_load']
        for i in range(n_hours):
            prob += total_load[i] <= max_load
    
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Check if a solution was found
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise ValueError(f"Optimization failed: {pulp.LpStatus[prob.status]}")
    
    # Extract solution
    original_cost = sum(period_prices.iloc[i]['price'] * (essential_load[i] + shiftable_load[i]) for i in range(n_hours))
    optimized_cost = sum(period_prices.iloc[i]['price'] * pulp.value(total_load[i]) for i in range(n_hours))
    
    # Prepare result
    schedule = []
    for i in range(n_hours):
        hour_time = start_time + timedelta(hours=i)
        schedule.append({
            'timestamp': hour_time.isoformat(),
            'original_load': essential_load[i] + shiftable_load[i],
            'optimized_load': pulp.value(total_load[i]),
            'essential_load': essential_load[i],
            'shiftable_load': shiftable_load[i],
            'run_shiftable': pulp.value(run_load[i]) > 0.5,
            'price': period_prices.iloc[i]['price']
        })
    
    result = {
        'schedule': schedule,
        'savings': {
            'original_cost': original_cost,
            'optimized_cost': optimized_cost,
            'savings_amount': original_cost - optimized_cost,
            'savings_percentage': (original_cost - optimized_cost) / original_cost * 100 if original_cost > 0 else 0
        },
        'execution_time': 0  # Will be updated by the API endpoint
    }
    
    return result

async def optimize_demand_response(
    energy_data: pd.DataFrame,
    price_data: pd.DataFrame,
    dr_events: pd.DataFrame,
    config: OptimizationConfig
) -> Dict:
    """Optimize for demand response events"""
    # Parse time period
    start_time = datetime.datetime.fromisoformat(config.start_time)
    end_time = datetime.datetime.fromisoformat(config.end_time)
    
    # Get price data and DR events for the period
    period_prices = price_data[
        (price_data['timestamp'] >= start_time) & 
        (price_data['timestamp'] <= end_time)
    ]
    
    # Filter DR events
    period_events = dr_events[
        ((dr_events['start_time'] >= start_time) & (dr_events['start_time'] <= end_time)) |
        ((dr_events['end_time'] >= start_time) & (dr_events['end_time'] <= end_time))
    ]
    
    # Number of hours in the optimization period
    timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
    n_hours = len(timestamps)
    
    # Create PuLP problem
    prob = pulp.LpProblem("Demand_Response_Optimization", pulp.LpMinimize)
    
    # Calculate base load profile and flexibility
    if 'hour' in energy_data.columns:
        # Group by hour of day to get typical load profile
        hourly_load = energy_data.groupby('hour')['value'].mean().to_dict()
        base_load = [hourly_load.get((start_time + timedelta(hours=i)).hour, energy_data['value'].mean()) for i in range(n_hours)]
    else:
        # If hourly pattern not available, use flat load profile
        avg_load = energy_data['value'].mean()
        base_load = [avg_load for _ in range(n_hours)]
    
    # Define flexibility parameters
    flexibility_pct = config.parameters.get('flexibility_percentage', 20) / 100.0
    
    # Define variables for each hour's load
    load = {}
    for i in range(n_hours):
        # Load can be reduced or increased within flexibility limits
        min_load = base_load[i] * (1 - flexibility_pct)
        max_load = base_load[i] * (1 + flexibility_pct)
        load[i] = pulp.LpVariable(f"load_{i}", min_load, max_load, pulp.LpContinuous)
    
    # Create a dictionary to track which hours fall within DR events
    dr_hours = {}
    for i, timestamp in enumerate(timestamps):
        dr_hours[i] = False
        for _, event in period_events.iterrows():
            if event['start_time'] <= timestamp <= event['end_time']:
                dr_hours[i] = True
                break
    
    # Define objective function: minimize cost plus DR penalties
    regular_cost = pulp.lpSum([
        period_prices.iloc[i]['price'] * load[i] for i in range(n_hours) if not dr_hours[i]
    ])
    
    # Apply DR price signals or penalties during DR events
    dr_incentive_multiplier = config.parameters.get('dr_incentive_multiplier', 3.0)
    dr_cost = pulp.lpSum([
        period_prices.iloc[i]['price'] * dr_incentive_multiplier * load[i] 
        for i in range(n_hours) if dr_hours[i]
    ])
    
    # Total objective
    prob += regular_cost + dr_cost, "Total_Cost"
    
    # Constraints
    # 1. Conservation of energy (total energy used should be approximately the same)
    # Allow a small tolerance for flexibility
    tolerance = 0.05  # 5% tolerance
    total_base_energy = sum(base_load)
    
    prob += pulp.lpSum([load[i] for i in range(n_hours)]) >= total_base_energy * (1 - tolerance)
    prob += pulp.lpSum([load[i] for i in range(n_hours)]) <= total_base_energy * (1 + tolerance)
    
    # 2. Specific DR event constraints
    for _, event in period_events.iterrows():
        # Find hours that fall within this DR event
        event_hours = []
        for i, timestamp in enumerate(timestamps):
            if event['start_time'] <= timestamp <= event['end_time']:
                event_hours.append(i)
        
        if event_hours:
            # Apply DR reduction target
            reduction_target = event['reduction_target'] if 'reduction_target' in event and not pd.isna(event['reduction_target']) else 0.15
            avg_base_load_during_event = sum(base_load[i] for i in event_hours) / len(event_hours)
            
            # Ensure average load during event is reduced by the target percentage
            prob += pulp.lpSum([load[i] for i in event_hours]) <= \
                    sum(base_load[i] for i in event_hours) * (1 - reduction_target)
    
    # 3. Apply any additional constraints from config
    # Maximum rate of change constraint to prevent rapid load fluctuations
    max_change_rate = config.constraints.get('max_change_rate', 0.25)  # Maximum 25% change between consecutive hours
    for i in range(1, n_hours):
        prob += load[i] - load[i-1] <= base_load[i] * max_change_rate
        prob += load[i-1] - load[i] <= base_load[i] * max_change_rate
    
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Check if a solution was found
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise ValueError(f"Optimization failed: {pulp.LpStatus[prob.status]}")
    
    # Extract solution
    original_cost = sum(
        period_prices.iloc[i]['price'] * base_load[i] * (dr_incentive_multiplier if dr_hours[i] else 1) 
        for i in range(n_hours)
    )
    
    optimized_cost = sum(
        period_prices.iloc[i]['price'] * pulp.value(load[i]) * (dr_incentive_multiplier if dr_hours[i] else 1) 
        for i in range(n_hours)
    )
    
    # Prepare result
    schedule = []
    for i in range(n_hours):
        hour_time = timestamps[i]
        schedule.append({
            'timestamp': hour_time.isoformat(),
            'original_load': base_load[i],
            'optimized_load': pulp.value(load[i]),
            'price': period_prices.iloc[i]['price'],
            'in_dr_event': dr_hours[i],
            'price_multiplier': dr_incentive_multiplier if dr_hours[i] else 1.0
        })
    
    result = {
        'schedule': schedule,
        'savings': {
            'original_cost': original_cost,
            'optimized_cost': optimized_cost,
            'savings_amount': original_cost - optimized_cost,
            'savings_percentage': (original_cost - optimized_cost) / original_cost * 100 if original_cost > 0 else 0
        },
        'dr_events': period_events.to_dict(orient='records'),
        'execution_time': 0  # Will be updated by the API endpoint
    }
    
    return result

async def optimize_energy_storage(
    energy_data: pd.DataFrame,
    price_data: pd.DataFrame,
    config: OptimizationConfig
) -> Dict:
    """Optimize energy storage operations"""
    # Parse time period
    start_time = datetime.datetime.fromisoformat(config.start_time)
    end_time = datetime.datetime.fromisoformat(config.end_time)
    
    # Get price data for the period
    period_prices = price_data[
        (price_data['timestamp'] >= start_time) & 
        (price_data['timestamp'] <= end_time)
    ]
    
    # Number of hours in the optimization period
    n_hours = len(period_prices)
    
    # Create PuLP problem
    prob = pulp.LpProblem("Energy_Storage_Optimization", pulp.LpMinimize)
    
    # Storage parameters from config
    battery_capacity_kwh = config.parameters.get('battery_capacity_kwh', 100)
    max_charge_rate_kw = config.parameters.get('max_charge_rate_kw', 25)
    max_discharge_rate_kw = config.parameters.get('max_discharge_rate_kw', 25)
    efficiency = config.parameters.get('round_trip_efficiency', 0.9)
    initial_soc = config.parameters.get('initial_soc', 0.5)  # State of charge (0-1)
    final_soc_min = config.parameters.get('final_soc_min', 0.5)  # Minimum final SOC
    
    # Calculate base load profile
    if 'hour' in energy_data.columns:
        # Group by hour of day to get typical load profile
        hourly_load = energy_data.groupby('hour')['value'].mean().to_dict()
        base_load = [hourly_load.get((start_time + timedelta(hours=i)).hour, energy_data['value'].mean()) for i in range(n_hours)]
    else:
        # If hourly pattern not available, use flat load profile
        avg_load = energy_data['value'].mean()
        base_load = [avg_load for _ in range(n_hours)]
    
    # Define decision variables
    # Grid power (positive = from grid, negative = to grid)
    grid_power = {}
    for i in range(n_hours):
        grid_power[i] = pulp.LpVariable(f"grid_power_{i}", None, None, pulp.LpContinuous)
    
    # Battery charge power (positive = charging, 0 = idle)
    charge_power = {}
    for i in range(n_hours):
        charge_power[i] = pulp.LpVariable(f"charge_power_{i}", 0, max_charge_rate_kw, pulp.LpContinuous)
    
    # Battery discharge power (positive = discharging, 0 = idle)
    discharge_power = {}
    for i in range(n_hours):
        discharge_power[i] = pulp.LpVariable(f"discharge_power_{i}", 0, max_discharge_rate_kw, pulp.LpContinuous)
    
    # Battery state of charge
    soc = {}
    for i in range(n_hours + 1):  # +1 for final state
        soc[i] = pulp.LpVariable(f"soc_{i}", 0, 1, pulp.LpContinuous)
    
    # Set initial SOC
    prob += soc[0] == initial_soc
    
    # Objective: minimize cost
    prob += pulp.lpSum([
        period_prices.iloc[i]['price'] * grid_power[i] for i in range(n_hours)
    ]), "Total_Cost"
    
    # Constraints
    # 1. Power balance at each time step: grid + discharge - charge = load
    for i in range(n_hours):
        prob += grid_power[i] + discharge_power[i] - charge_power[i] == base_load[i]
    
    # 2. Battery SOC evolution
    for i in range(n_hours):
        # SOC(t+1) = SOC(t) + charge_efficiency * charge - discharge / discharge_efficiency
        charge_efficiency = np.sqrt(efficiency)
        discharge_efficiency = np.sqrt(efficiency)
        
        prob += soc[i+1] == soc[i] + (charge_power[i] * charge_efficiency - 
                                       discharge_power[i] / discharge_efficiency) / battery_capacity_kwh
    
    # 3. Final SOC constraint
    prob += soc[n_hours] >= final_soc_min
    
    # 4. Apply any additional constraints from config
    # Peak demand constraint
    if 'max_grid_power' in config.constraints:
        max_grid_power = config.constraints['max_grid_power']
        for i in range(n_hours):
            prob += grid_power[i] <= max_grid_power
    
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Check if a solution was found
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise ValueError(f"Optimization failed: {pulp.LpStatus[prob.status]}")
    
    # Extract solution
    original_cost = sum(period_prices.iloc[i]['price'] * base_load[i] for i in range(n_hours))
    optimized_cost = sum(period_prices.iloc[i]['price'] * pulp.value(grid_power[i]) for i in range(n_hours))
    
    # Prepare result
    schedule = []
    for i in range(n_hours):
        hour_time = start_time + timedelta(hours=i)
        schedule.append({
            'timestamp': hour_time.isoformat(),
            'original_load': base_load[i],
            'grid_power': pulp.value(grid_power[i]),
            'charge_power': pulp.value(charge_power[i]),
            'discharge_power': pulp.value(discharge_power[i]),
            'soc': pulp.value(soc[i]),
            'price': period_prices.iloc[i]['price']
        })
    
    result = {
        'schedule': schedule,
        'savings': {
            'original_cost': original_cost,
            'optimized_cost': optimized_cost,
            'savings_amount': original_cost - optimized_cost,
            'savings_percentage': (original_cost - optimized_cost) / original_cost * 100 if original_cost > 0 else 0
        },
        'final_soc': pulp.value(soc[n_hours]),
        'execution_time': 0  # Will be updated by the API endpoint
    }
    
    return result
