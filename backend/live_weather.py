"""
Live Weather Integration — Open-Meteo (free, no API key)
Provides real-time atmospheric & marine weather for Khalifa Port and Ruwais Port (UAE).
Falls back to defaults if the API is unreachable.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict

import httpx

logger = logging.getLogger(__name__)

PORTS = {
    "Khalifa Port": {"lat": 24.8131, "lon": 54.6552},
    "Ruwais Port":  {"lat": 24.1330, "lon": 52.7437},
}

OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"

# --- helpers -----------------------------------------------------------------

def estimate_sea_state(wind_mps: float) -> int:
    """Douglas sea-state estimate from wind speed (m/s)."""
    if wind_mps is None:
        return 0
    if wind_mps < 1:  return 0
    if wind_mps < 4:  return 1
    if wind_mps < 7:  return 2
    if wind_mps < 11: return 3
    if wind_mps < 16: return 4
    return 5


def estimate_wave_height(wind_mps: float) -> float:
    """Rough wave-height estimate (m) — only used as fallback if marine API returns null."""
    if wind_mps is None:
        return 0.0
    if wind_mps < 1:  return 0.0
    if wind_mps < 4:  return 0.1
    if wind_mps < 7:  return 0.4
    if wind_mps < 11: return 0.9
    if wind_mps < 16: return 1.8
    return 2.8


def impact_score(wind_mps: float) -> float:
    if wind_mps is None:
        return 1.0
    impact = 1.0
    if wind_mps > 15: impact += 0.30
    elif wind_mps > 10: impact += 0.15
    elif wind_mps > 7:  impact += 0.05
    elif wind_mps < 2:  impact += 0.05
    return round(impact, 2)


# --- fetcher -----------------------------------------------------------------

async def _fetch_port(client: httpx.AsyncClient, name: str, coords: Dict) -> Dict:
    """Fetch live forecast + marine data for a single port."""
    forecast_params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,"
                   "wind_direction_10m,wind_gusts_10m,visibility,weather_code,apparent_temperature",
        "wind_speed_unit": "ms",
        "timezone": "Asia/Dubai",
    }
    marine_params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "current": "wave_height,wave_direction,wave_period",
        "timezone": "Asia/Dubai",
    }
    try:
        forecast_resp, marine_resp = await asyncio.gather(
            client.get(OPENMETEO_FORECAST, params=forecast_params),
            client.get(OPENMETEO_MARINE, params=marine_params),
            return_exceptions=True,
        )
    except Exception as e:
        logger.warning("Open-Meteo gather failure for %s: %s", name, e)
        return _default_port(name)

    out: Dict = {"location_name": name, "lat": coords["lat"], "lon": coords["lon"]}

    # Forecast (wind, temp, humidity, pressure, visibility)
    if isinstance(forecast_resp, httpx.Response) and forecast_resp.status_code == 200:
        current = forecast_resp.json().get("current", {}) or {}
        out["temperature"] = current.get("temperature_2m")
        out["apparent_temperature"] = current.get("apparent_temperature")
        out["wind_speed"] = current.get("wind_speed_10m")           # m/s
        out["wind_direction"] = current.get("wind_direction_10m")    # deg
        out["wind_gusts"] = current.get("wind_gusts_10m")
        out["humidity"] = current.get("relative_humidity_2m")
        out["pressure"] = current.get("pressure_msl")
        visibility_m = current.get("visibility")
        out["visibility"] = round(visibility_m / 1000.0, 1) if isinstance(visibility_m, (int, float)) else None
        out["weather_code"] = current.get("weather_code")
        out["observed_at"] = current.get("time")
    else:
        logger.warning("Open-Meteo forecast missing for %s", name)
        out.update({
            "temperature": 30.0, "wind_speed": 5.0, "wind_direction": 90.0,
            "humidity": 60, "pressure": 1011, "visibility": 10.0,
            "weather_code": None, "observed_at": datetime.utcnow().isoformat(),
        })

    # Marine (wave height/direction/period)
    if isinstance(marine_resp, httpx.Response) and marine_resp.status_code == 200:
        m = marine_resp.json().get("current", {}) or {}
        out["wave_height"] = m.get("wave_height")
        out["wave_direction"] = m.get("wave_direction")
        out["wave_period"] = m.get("wave_period")
    else:
        out["wave_height"] = None
        out["wave_direction"] = None
        out["wave_period"] = None

    if out.get("wave_height") in (None, 0) and out.get("wind_speed") is not None:
        out["wave_height"] = estimate_wave_height(out["wind_speed"])

    out["sea_state"] = estimate_sea_state(out.get("wind_speed") or 0)
    out["impact_score"] = impact_score(out.get("wind_speed") or 0)
    out["source"] = "Open-Meteo"
    return out


def _default_port(name: str) -> Dict:
    return {
        "location_name": name,
        "temperature": 30.0,
        "wind_speed": 5.0,
        "wind_direction": 90.0,
        "wind_gusts": 7.0,
        "humidity": 60,
        "pressure": 1011,
        "visibility": 10.0,
        "wave_height": 0.4,
        "wave_period": 4.0,
        "wave_direction": 90.0,
        "sea_state": 2,
        "impact_score": 1.0,
        "source": "fallback",
        "observed_at": datetime.utcnow().isoformat(),
    }


# --- public API --------------------------------------------------------------

async def fetch_route_weather(departure: str, arrival: str) -> Dict:
    """Get live weather for both ports + a midpoint estimate."""
    dep_coords = PORTS.get(departure, PORTS["Khalifa Port"])
    arr_coords = PORTS.get(arrival, PORTS["Ruwais Port"])

    async with httpx.AsyncClient(timeout=12) as client:
        dep, arr = await asyncio.gather(
            _fetch_port(client, departure, dep_coords),
            _fetch_port(client, arrival, arr_coords),
        )

    # Midpoint = arithmetic average of both ports
    mid = {
        "location_name": "Route Midpoint",
        "lat": round((dep_coords["lat"] + arr_coords["lat"]) / 2, 4),
        "lon": round((dep_coords["lon"] + arr_coords["lon"]) / 2, 4),
        "temperature": _avg(dep.get("temperature"), arr.get("temperature")),
        "apparent_temperature": _avg(dep.get("apparent_temperature"), arr.get("apparent_temperature")),
        "wind_speed": _avg(dep.get("wind_speed"), arr.get("wind_speed")),
        "wind_direction": _avg(dep.get("wind_direction"), arr.get("wind_direction")),
        "wind_gusts": _avg(dep.get("wind_gusts"), arr.get("wind_gusts")),
        "humidity": _avg(dep.get("humidity"), arr.get("humidity")),
        "pressure": _avg(dep.get("pressure"), arr.get("pressure")),
        "visibility": _avg(dep.get("visibility"), arr.get("visibility")),
        "wave_height": _avg(dep.get("wave_height"), arr.get("wave_height")),
        "wave_period": _avg(dep.get("wave_period"), arr.get("wave_period")),
        "wave_direction": _avg(dep.get("wave_direction"), arr.get("wave_direction")),
        "source": "Open-Meteo (interpolated)",
        "observed_at": dep.get("observed_at") or arr.get("observed_at"),
    }
    mid["sea_state"] = estimate_sea_state(mid.get("wind_speed") or 0)
    mid["impact_score"] = impact_score(mid.get("wind_speed") or 0)

    avg_wind = _avg(dep.get("wind_speed"), arr.get("wind_speed")) or 5.0
    avg_dir = _avg(dep.get("wind_direction"), arr.get("wind_direction")) or 90.0
    overall_impact = round(
        ((dep.get("impact_score") or 1.0) + (mid.get("impact_score") or 1.0) + (arr.get("impact_score") or 1.0)) / 3,
        2,
    )

    return {
        "success": True,
        "departure": dep,
        "midpoint": mid,
        "arrival": arr,
        "average": {
            "wind_speed": round(avg_wind, 2),
            "wind_direction": round(avg_dir, 1),
        },
        "overall_impact_score": overall_impact,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }


def _avg(a, b):
    vals = [v for v in (a, b) if isinstance(v, (int, float))]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


if __name__ == "__main__":
    import json
    data = asyncio.run(fetch_route_weather("Khalifa Port", "Ruwais Port"))
    print(json.dumps(data, indent=2, default=str))
