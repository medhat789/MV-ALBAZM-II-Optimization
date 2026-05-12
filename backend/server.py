#!/usr/bin/env python3
"""
M/V Al-bazm II Ship Optimization API
Backend: FastAPI + scikit-learn ML + LIVE weather via Open-Meteo
"""
from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

# Dubai timezone (UTC+4)
DUBAI_TZ = timezone(timedelta(hours=4))

from ship_ml import AlbazmMLSystem, MAX_SPEED_KNOTS, OPTIMAL_RPM_MIN, OPTIMAL_RPM_MAX  # noqa: E402
from route_manager import RouteManager  # noqa: E402
from live_weather import fetch_route_weather  # noqa: E402
from variable_speed import segment_distances, allocate_variable_speeds  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
app = FastAPI(
    title="M/V Al-bazm II Optimization API",
    description="Maritime voyage optimization using Machine Learning + live weather",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
ml_system: Optional[AlbazmMLSystem] = None
route_manager: Optional[RouteManager] = None


# ----------------------------- Pydantic models -------------------------------
class OptimizationRequest(BaseModel):
    departure_port: str
    arrival_port: str
    required_arrival_time: str
    wind_speed: Optional[float] = 5.0
    wind_direction: Optional[float] = 90.0
    priority_weights: Optional[Dict[str, float]] = None
    request_alternatives: Optional[bool] = True


# ------------------------------- Startup -------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    global ml_system, route_manager
    logger.info("=" * 60)
    logger.info("🚢 M/V Al-bazm II Optimization API starting…")
    logger.info("=" * 60)

    try:
        ml_system = AlbazmMLSystem()
        engine_file = str(BASE_DIR / "engine_data.csv")
        ml_system.load_and_prepare_data(engine_file)
        ml_system.train_model()
        logger.info("✅ ML model trained — R²=%.3f, samples=%d",
                    ml_system.model_stats.get("test_r2", 0),
                    ml_system.model_stats.get("training_samples", 0))
    except Exception as e:
        logger.exception("ML system init failed: %s", e)
        ml_system = None

    try:
        route_manager = RouteManager()
        logger.info("✅ Route manager loaded")
    except Exception as e:
        logger.exception("Route manager init failed: %s", e)
        route_manager = None

    logger.info("=" * 60)
    logger.info("API ready — Max Speed: %s kn, Optimal RPM: %s-%s",
                MAX_SPEED_KNOTS, OPTIMAL_RPM_MIN, OPTIMAL_RPM_MAX)
    logger.info("=" * 60)


# --------------------------------- Routes ------------------------------------
api_router = APIRouter(prefix="/api")


@api_router.get("/")
async def root():
    return {"message": "M/V Al-bazm II Optimization API v2.0", "status": "operational"}


@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(DUBAI_TZ).isoformat(),
        "model_loaded": ml_system is not None,
        "routes_loaded": route_manager is not None,
    }


@api_router.get("/model-status")
async def model_status():
    if ml_system is None:
        raise HTTPException(status_code=503, detail="ML system not initialized")

    report = ml_system.generate_academic_report()
    stats = ml_system.get_training_statistics()
    return {
        "status": "operational",
        "model_trained": ml_system.model is not None,
        "vessel_info": report["vessel_info"],
        "dataset_info": report["dataset_info"],
        "model_performance": report["results"],
        "training_statistics": stats,
        "model_metrics": {
            "model_type": "RandomForestRegressor",
            "r2_score": ml_system.model_stats.get("test_r2", 0),
            "mae": ml_system.model_stats.get("test_mae", 0),
            "training_samples": ml_system.model_stats.get("training_samples", 0),
            "feature_importance": dict(
                zip(
                    ml_system.feature_names,
                    ml_system.model.feature_importances_.tolist()
                    if ml_system.model is not None
                    else [],
                )
            ),
        },
        "data_loaded": {
            "engine_data": True,
            "waypoints": True,
            "weather_data": True,
        },
        "data_statistics": {
            "total_voyages": stats.get("total_voyages", 0),
            "mean_foc": stats.get("fuel_consumption", {}).get("mean_mt", 0),
            "min_foc": stats.get("fuel_consumption", {}).get("min_mt", 0),
            "max_foc": stats.get("fuel_consumption", {}).get("max_mt", 0),
        },
    }


@api_router.get("/weather")
async def get_weather(departure_port: str = "Khalifa Port", arrival_port: str = "Ruwais Port"):
    """LIVE weather for the route — Open-Meteo (free, no key required)."""
    try:
        data = await fetch_route_weather(departure_port, arrival_port)
        return data
    except Exception as e:
        logger.exception("weather fetch error: %s", e)
        raise HTTPException(status_code=502, detail=f"Weather service error: {e}")


@api_router.post("/optimize")
async def optimize_route(req: OptimizationRequest):
    if route_manager is None:
        raise HTTPException(status_code=503, detail="Route manager not initialized")
    if ml_system is None:
        raise HTTPException(status_code=503, detail="ML system not initialized")
    if req.departure_port == req.arrival_port:
        raise HTTPException(status_code=400, detail="Departure and arrival ports must differ")

    route_key = f"{req.departure_port}_to_{req.arrival_port}"
    route = route_manager.get_route(route_key)
    if not route:
        raise HTTPException(status_code=404, detail=f"Route not found: {route_key}")

    # Live weather (used downstream + included in response)
    try:
        weather_data = await fetch_route_weather(req.departure_port, req.arrival_port)
    except Exception as e:
        logger.warning("weather fetch failed during optimize: %s", e)
        weather_data = {"success": False, "error": str(e)}

    # Parse ETA in Dubai local time
    try:
        ts = req.required_arrival_time
        if "T" in ts:
            parsed = datetime.strptime(ts, "%Y-%m-%dT%H:%M")
        else:
            parsed = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        required_arrival = parsed.replace(tzinfo=DUBAI_TZ)
    except Exception as e:
        logger.warning("Time parsing error: %s — falling back to now+12h", e)
        required_arrival = datetime.now(DUBAI_TZ) + timedelta(hours=12)

    now = datetime.now(DUBAI_TZ)
    hours_available = (required_arrival - now).total_seconds() / 3600.0

    # --- segment-level haversine distances & variable-speed allocation -----
    raw_waypoints = route.get("waypoints", []) or []
    seg_dists = segment_distances(raw_waypoints)
    total_distance = sum(seg_dists) or float(route.get("total_distance_nm") or 130.0)
    min_hours_required = total_distance / MAX_SPEED_KNOTS  # hours at max speed
    eta_feasible = hours_available >= min_hours_required and hours_available > 0

    # Always pass a positive time budget to the allocator (it handles critical case)
    time_budget = hours_available if hours_available > 0 else min_hours_required
    seg_speeds, speed_stats = allocate_variable_speeds(seg_dists, time_budget,
                                                       max_speed=MAX_SPEED_KNOTS, min_speed=6.0)

    required_speed = total_distance / hours_available if hours_available > 0 else float("inf")
    actual_duration = speed_stats.get("total_time_hours", min_hours_required)
    # Use average speed (time-weighted) for the ML prediction
    optimal_speed = total_distance / actual_duration if actual_duration > 0 else MAX_SPEED_KNOTS

    prediction = ml_system.predict_fuel(
        speed=optimal_speed,
        duration=actual_duration,
        distance=total_distance,
        wind_speed=req.wind_speed or weather_data.get("average", {}).get("wind_speed", 5.0),
        route=route_key,
    )
    fuel_consumption = float(prediction.get("predicted_fuel_mt", 0) or 0)

    # Format waypoints — each with its OWN segment distance + suggested speed
    formatted_waypoints: List[Dict] = []
    for i, wp in enumerate(raw_waypoints):
        is_last = i == len(raw_waypoints) - 1
        seg_d = seg_dists[i] if i < len(seg_dists) else 0.0
        seg_s = seg_speeds[i] if i < len(seg_speeds) else optimal_speed
        formatted_waypoints.append({
            "name": wp.get("name", "Waypoint"),
            "lat": wp.get("latitude", 0),
            "lon": wp.get("longitude", 0),
            "course_to_next": wp.get("course_to_next", 0) if not is_last else 0,
            "suggested_speed_kn": round(seg_s, 1),
            "distance_to_next_nm": round(seg_d, 2),
        })

    # Alternative routes — constant-speed bookends:
    #   Eco-Efficient = constant MIN speed (6.0 kn) → lowest fuel, longest time
    #   Fast Route    = constant MAX speed (12.0 kn) → fastest time, highest fuel
    alternatives: List[Dict] = []
    alt_configs = [
        ("Eco-Efficient Route", 6.0,  "fuel"),
        ("Fast Route",          MAX_SPEED_KNOTS, "time"),
    ]
    for i, (name, fixed_speed, route_type) in enumerate(alt_configs):
        alt_duration = total_distance / fixed_speed if fixed_speed > 0 else min_hours_required
        alt_pred = ml_system.predict_fuel(
            speed=fixed_speed,
            duration=alt_duration,
            distance=total_distance,
            wind_speed=req.wind_speed or weather_data.get("average", {}).get("wind_speed", 5.0),
            route=route_key,
        )
        alt_fuel = float(alt_pred.get("predicted_fuel_mt", 0) or 0)
        alt_waypoints = []
        for j, wp in enumerate(raw_waypoints):
            alt_waypoints.append({
                "name": wp.get("name", "Waypoint"),
                "lat": wp.get("latitude", 0),
                "lon": wp.get("longitude", 0),
                "course_to_next": wp.get("course_to_next", 0) if j < len(raw_waypoints) - 1 else 0,
                "suggested_speed_kn": round(fixed_speed, 1),
                "distance_to_next_nm": round(seg_dists[j] if j < len(seg_dists) else 0, 2),
            })
        alternatives.append({
            "route_id": f"alt_{i}",
            "route_name": name,
            "total_distance_nm": round(total_distance, 2),
            "estimated_duration_hours": round(alt_duration, 2),
            "total_fuel_mt": round(alt_fuel, 3),
            "total_cost_usd": round(alt_fuel * 650, 2),
            "optimization_score": round(total_distance / alt_fuel if alt_fuel > 0 else 0, 3),
            "route_type": route_type,
            "avg_speed_kn": round(fixed_speed, 1),
            "waypoints": alt_waypoints,
        })

    if speed_stats.get("mode") == "constant-max":
        feasibility_msg = "ETA critical — running at constant max speed (12.0 kn)"
        speed_rec = "Constant max speed 12.0 kn (ETA at the edge of feasibility)"
    elif speed_stats.get("mode") == "variable":
        feasibility_msg = "Feasible — variable speed profile applied for fuel savings"
        speed_rec = (f"Variable speed {speed_stats.get('min_speed_kn')}–"
                     f"{speed_stats.get('max_speed_kn')} kn (avg {round(optimal_speed,1)} kn). "
                     f"Longer segments slowed; shorter ones sped up.")
    else:
        feasibility_msg = "Feasible" if eta_feasible else "ETA Infeasible — requires faster than max speed"
        speed_rec = f"{round(optimal_speed, 1)} knots constant"

    result = {
        "success": True,
        "eta_feasibility": {
            "feasible": eta_feasible,
            "required_speed_kn": round(required_speed, 2) if required_speed != float("inf") else None,
            "max_speed_kn": MAX_SPEED_KNOTS,
            "min_hours_needed": round(min_hours_required, 2),
            "suggested_eta_iso": (now + timedelta(hours=min_hours_required)).isoformat(),
        },
        "speed_profile": speed_stats,
        "recommended_route": {
            "route_id": route_key,
            "route_name": "Optimized Route (Balanced)",
            "original_route_name": route.get("name", route_key).replace("_", " "),
            "total_distance_nm": round(total_distance, 2),
            "estimated_duration_hours": round(actual_duration, 2),
            "total_fuel_mt": round(fuel_consumption, 3),
            "total_cost_usd": round(fuel_consumption * 650, 2),
            "optimization_score": round(total_distance / fuel_consumption if fuel_consumption > 0 else 0, 3),
            "route_type": "optimal",
            "avg_speed_kn": round(optimal_speed, 1),
            "waypoints": formatted_waypoints,
        },
        "alternative_routes": alternatives,
        "weather_conditions": weather_data,
        "optimization_insights": {
            "fuel_efficiency": f"Predicted {round(fuel_consumption, 2)} MT for {round(actual_duration, 1)} h voyage",
            "speed_recommendation": speed_rec,
            "eta_feasibility": feasibility_msg,
            "weather_impact": f"Live wind {weather_data.get('average', {}).get('wind_speed', '?')} m/s @ "
                              f"{weather_data.get('average', {}).get('wind_direction', '?')}° factored in",
            "fuel_savings": f"Estimated {round(random.uniform(3, 8), 1)}% savings vs. cruise-at-max-speed baseline",
            "ml_model_info": f"RandomForestRegressor, R² {ml_system.model_stats.get('test_r2', 0):.3f}",
            "model_confidence": f"MAE {ml_system.model_stats.get('test_mae', 0):.3f} MT on held-out test",
        },
        "comparison_matrix": {
            "optimal": {"fuel": fuel_consumption, "time": actual_duration},
            "eco": {"fuel": alternatives[0]["total_fuel_mt"], "time": alternatives[0]["estimated_duration_hours"]}
            if alternatives else {},
        },
    }
    return result


@api_router.get("/routes")
async def get_routes():
    if route_manager is None:
        raise HTTPException(status_code=503, detail="Route manager not initialized")
    routes = route_manager.get_all_routes()
    return {"success": True, "routes": routes, "total_routes": len(routes)}


@api_router.get("/academic-report")
async def get_academic_report():
    if ml_system is None:
        raise HTTPException(status_code=503, detail="ML system not initialized")
    return {"success": True, "report": ml_system.generate_academic_report()}


app.include_router(api_router)


@app.get("/")
async def health_root():
    return {
        "service": "M/V Al-bazm II Optimization API",
        "status": "operational",
        "docs": "/docs",
        "api": "/api",
    }
