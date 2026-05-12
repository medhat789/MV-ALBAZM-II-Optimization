"""
POC test_core.py — Maritime Fuel Optimization System
Tests core workflow before building the app:
 1. Physics fuel/speed/RPM calculation
 2. Open-Meteo Marine API weather fetch (Khalifa + Ruwais ports)
 3. MongoDB connection + seed 372 voyages
 4. sklearn ML model training + accuracy computation
 5. End-to-end optimize_voyage() combining physics + ML + weather
"""
import asyncio
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import httpx
import joblib
import numpy as np
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load env
load_dotenv("/app/backend/.env")

# ============== Constants ==============
PORTS = {
    "Khalifa Port": {"lat": 24.8131, "lon": 54.6552},
    "Ruwais Port":  {"lat": 24.1330, "lon": 52.7437},
}
DISTANCE_NM = 78.0  # Approx distance Khalifa <-> Ruwais
VESSEL = {
    "name": "M/V Al-bazm II",
    "max_speed_kn": 12.0,
    "min_speed_kn": 6.0,
    "rpm_min": 115,
    "rpm_max": 145,
    "fuel_at_ref_speed_tph": 1.10,   # tons/hour at reference speed (10 kn calm)
    "ref_speed_kn": 10.0,
}

DUBAI_TZ = timezone(timedelta(hours=4))


# ============== Physics ==============
def required_speed_kn(distance_nm: float, eta_dt: datetime, now_dt: datetime) -> float:
    hours = (eta_dt - now_dt).total_seconds() / 3600.0
    if hours <= 0:
        return float("inf")
    return distance_nm / hours


def rpm_from_speed(speed_kn: float) -> int:
    # Map speed within [min..max] to RPM band [rpm_min..rpm_max]
    s_lo, s_hi = VESSEL["min_speed_kn"], VESSEL["max_speed_kn"]
    r_lo, r_hi = VESSEL["rpm_min"], VESSEL["rpm_max"]
    s = max(s_lo, min(s_hi, speed_kn))
    frac = (s - s_lo) / (s_hi - s_lo) if s_hi > s_lo else 0
    return int(round(r_lo + frac * (r_hi - r_lo)))


def fuel_rate_tph(speed_kn: float, wind_speed_ms: float = 0.0, rel_wind_deg: float = 0.0) -> float:
    """Fuel consumption in tons/hour.
    Cube-law base + headwind multiplier (rel_wind 0=headwind, 180=tailwind)."""
    if speed_kn <= 0:
        return 0.0
    base = VESSEL["fuel_at_ref_speed_tph"] * (speed_kn / VESSEL["ref_speed_kn"]) ** 3
    # Wind effect: headwind (0deg) increases, tailwind (180) decreases.
    rel = math.radians(rel_wind_deg)
    head_factor = math.cos(rel)  # +1 = headwind, -1 = tailwind
    wind_factor = 1.0 + 0.012 * wind_speed_ms * head_factor
    wind_factor = max(0.85, min(1.35, wind_factor))
    return base * wind_factor


def physics_optimize(distance_nm, hours_available, max_speed, wind_speed=0.0, rel_wind=0.0):
    """Find feasibility and choose lowest fuel speed that meets ETA."""
    req_speed = distance_nm / hours_available if hours_available > 0 else float("inf")
    feasible = req_speed <= max_speed
    chosen_speed = min(max_speed, max(VESSEL["min_speed_kn"], req_speed))
    hours_used = distance_nm / chosen_speed
    fuel_rate = fuel_rate_tph(chosen_speed, wind_speed, rel_wind)
    total_fuel = fuel_rate * hours_used
    return {
        "feasible": feasible,
        "required_speed_kn": round(req_speed, 2),
        "chosen_speed_kn": round(chosen_speed, 2),
        "rpm": rpm_from_speed(chosen_speed),
        "hours": round(hours_used, 2),
        "fuel_rate_tph": round(fuel_rate, 3),
        "total_fuel_t": round(total_fuel, 2),
    }


# ============== Weather ==============
async def fetch_marine_weather(lat: float, lon: float) -> dict:
    """Fetch current marine + wind weather from Open-Meteo (no API key)."""
    async with httpx.AsyncClient(timeout=15) as client:
        # Marine API for wave_height
        marine_url = "https://marine-api.open-meteo.com/v1/marine"
        marine_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "wave_height,wave_direction,wave_period",
            "timezone": "Asia/Dubai",
        }
        # Forecast API for wind
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        forecast_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,relative_humidity_2m",
            "wind_speed_unit": "ms",
            "timezone": "Asia/Dubai",
        }
        marine_resp, forecast_resp = await asyncio.gather(
            client.get(marine_url, params=marine_params),
            client.get(forecast_url, params=forecast_params),
        )
    out = {}
    if forecast_resp.status_code == 200:
        d = forecast_resp.json().get("current", {})
        out["temperature_c"] = d.get("temperature_2m")
        out["wind_speed_ms"] = d.get("wind_speed_10m")
        out["wind_direction_deg"] = d.get("wind_direction_10m")
        out["wind_gusts_ms"] = d.get("wind_gusts_10m")
        out["humidity"] = d.get("relative_humidity_2m")
        out["time"] = d.get("time")
    if marine_resp.status_code == 200:
        m = marine_resp.json().get("current", {})
        out["wave_height_m"] = m.get("wave_height")
        out["wave_direction_deg"] = m.get("wave_direction")
        out["wave_period_s"] = m.get("wave_period")
    return out


async def test_weather():
    print("\n=== TEST 2: Open-Meteo Marine Weather ===")
    results = {}
    for name, coords in PORTS.items():
        w = await fetch_marine_weather(coords["lat"], coords["lon"])
        results[name] = w
        print(f"  {name}: wind={w.get('wind_speed_ms')} m/s @ {w.get('wind_direction_deg')}°, "
              f"temp={w.get('temperature_c')}°C, wave={w.get('wave_height_m')} m")
    assert all(r.get("wind_speed_ms") is not None for r in results.values()), "wind data missing"
    print("  ✅ Weather fetch OK")
    return results


# ============== MongoDB + Seed ==============
async def seed_voyages(db, count=372):
    coll = db["voyages"]
    existing = await coll.count_documents({})
    if existing >= count:
        print(f"  Already has {existing} voyages, skipping seed")
        return existing
    docs = []
    rng = random.Random(42)
    for i in range(count - existing):
        speed = rng.uniform(7.0, 12.0)
        wind = rng.uniform(0.0, 14.0)
        rel = rng.uniform(0, 360)
        # actual fuel uses similar physics + noise
        hours = DISTANCE_NM / speed
        rate = fuel_rate_tph(speed, wind, rel) * rng.uniform(0.92, 1.08)
        fuel = rate * hours
        docs.append({
            "id": f"seed-{i}",
            "departure": "Khalifa Port" if i % 2 == 0 else "Ruwais Port",
            "arrival":   "Ruwais Port" if i % 2 == 0 else "Khalifa Port",
            "distance_nm": DISTANCE_NM,
            "speed_kn": speed,
            "rpm": rpm_from_speed(speed),
            "wind_speed_ms": wind,
            "rel_wind_deg": rel,
            "hours": hours,
            "fuel_tons": fuel,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_seed": True,
        })
    if docs:
        await coll.insert_many(docs)
    total = await coll.count_documents({})
    print(f"  Seeded; total voyages now = {total}")
    return total


async def test_mongo():
    print("\n=== TEST 3: MongoDB + Seed 372 voyages ===")
    mongo_url = os.environ["MONGO_URL"]
    db_name = os.environ["DB_NAME"]
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    # Drop and re-seed for clean POC test
    await db["voyages"].delete_many({"is_seed": True})
    total = await seed_voyages(db, 372)
    assert total >= 372, f"Expected >=372 voyages, got {total}"
    print("  ✅ Mongo seed OK")
    return db


# ============== ML Model ==============
def build_training_matrix(voyages):
    X, y = [], []
    for v in voyages:
        X.append([v["speed_kn"], v["wind_speed_ms"], v["rel_wind_deg"], v["distance_nm"]])
        y.append(v["fuel_tons"])
    return np.array(X), np.array(y)


async def test_ml(db):
    print("\n=== TEST 4: ML Model Training ===")
    voyages = await db["voyages"].find({}, {"_id": 0}).to_list(5000)
    X, y = build_training_matrix(voyages)
    print(f"  Loaded {len(X)} voyages for training")
    if len(X) < 20:
        raise RuntimeError("Not enough training data")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7)
    model = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=7)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    r2 = r2_score(yte, preds)
    print(f"  Model R² accuracy: {r2:.3f}")
    # Save artifact
    os.makedirs("/app/backend/model_store", exist_ok=True)
    joblib.dump(model, "/app/backend/model_store/voyage_model.joblib")
    assert r2 > 0.2, f"R² too low: {r2}"
    print("  ✅ ML training OK")
    return model, float(r2)


# ============== End-to-end optimize ==============
async def test_optimize(db, model, accuracy, weather_data):
    print("\n=== TEST 5: End-to-end optimize_voyage() ===")
    now = datetime.now(DUBAI_TZ)
    eta = now + timedelta(hours=9)  # tight but feasible
    # Use average wind of the route
    winds = [w.get("wind_speed_ms") or 0 for w in weather_data.values()]
    dirs  = [w.get("wind_direction_deg") or 0 for w in weather_data.values()]
    avg_wind = sum(winds)/len(winds)
    avg_dir  = sum(dirs)/len(dirs)
    # Course Khalifa->Ruwais ~ heading West (270°). rel wind = wind_dir - course
    course = 270
    rel = (avg_dir - course) % 360

    hours_available = (eta - now).total_seconds()/3600
    phys = physics_optimize(DISTANCE_NM, hours_available, VESSEL["max_speed_kn"], avg_wind, rel)
    ml_pred = float(model.predict([[phys["chosen_speed_kn"], avg_wind, rel, DISTANCE_NM]])[0])
    # ML refinement: blend 60% physics + 40% ML
    blended_fuel = 0.6 * phys["total_fuel_t"] + 0.4 * ml_pred

    result = {
        "vessel": VESSEL["name"],
        "departure": "Khalifa Port",
        "arrival": "Ruwais Port",
        "now_local": now.isoformat(),
        "eta_local": eta.isoformat(),
        "physics": phys,
        "ml_prediction_fuel_t": round(ml_pred, 2),
        "final_fuel_t": round(blended_fuel, 2),
        "accuracy": round(accuracy*100, 1),
        "weather_avg_wind_ms": round(avg_wind, 2),
        "weather_wind_dir_deg": round(avg_dir, 0),
    }
    # Persist
    voyage_doc = {
        "id": f"poc-{int(now.timestamp())}",
        "departure": result["departure"],
        "arrival": result["arrival"],
        "distance_nm": DISTANCE_NM,
        "speed_kn": phys["chosen_speed_kn"],
        "rpm": phys["rpm"],
        "wind_speed_ms": avg_wind,
        "rel_wind_deg": rel,
        "hours": phys["hours"],
        "fuel_tons": blended_fuel,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_seed": False,
    }
    await db["voyages"].insert_one(voyage_doc)
    new_total = await db["voyages"].count_documents({})
    print(f"  Optimization result: {result}")
    print(f"  Voyage persisted. Total voyages now = {new_total}")
    assert phys["feasible"], "should be feasible"
    assert blended_fuel > 0
    print("  ✅ End-to-end optimize OK")
    return result


# ============== Runner ==============
async def main():
    print("="*60)
    print("MARITIME FUEL OPTIMIZATION — POC TEST")
    print("="*60)

    # Test 1: Physics
    print("\n=== TEST 1: Physics functions ===")
    now = datetime.now(DUBAI_TZ)
    eta = now + timedelta(hours=8)
    p = physics_optimize(DISTANCE_NM, 8, 12.0, 5.0, 30)
    print(f"  optimize(8h, 5m/s wind): {p}")
    p2 = physics_optimize(DISTANCE_NM, 5, 12.0, 5.0, 30)  # infeasible
    print(f"  optimize(5h, infeasible): {p2}")
    assert p["feasible"]
    assert not p2["feasible"]
    print("  ✅ Physics OK")

    # Test 2: Weather
    weather = await test_weather()

    # Test 3: Mongo + seed
    db = await test_mongo()

    # Test 4: ML
    model, r2 = await test_ml(db)

    # Test 5: Full optimize
    final = await test_optimize(db, model, r2, weather)

    print("\n" + "="*60)
    print("✅ ALL POC TESTS PASSED")
    print("="*60)
    return 0


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
        sys.exit(rc)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ POC FAILED: {e}")
        sys.exit(1)
