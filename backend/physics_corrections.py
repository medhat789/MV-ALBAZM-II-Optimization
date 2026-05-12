"""Post-ML physics corrections + emissions calculations.

The base ML model takes (speed, duration, distance, wind_speed) as features but
does not know about wind direction or wave height. We apply physics-based
multipliers on top of the ML prediction to capture those effects.
"""
from __future__ import annotations

import math
from typing import Dict


# Heavy Fuel Oil CO₂ emission factor (IMO MEPC.245(66))
HFO_CO2_FACTOR = 3.114  # tonnes CO₂ per tonne fuel
MGO_CO2_FACTOR = 3.206  # tonnes CO₂ per tonne fuel (alt for low-sulfur MGO)


def headwind_component_ms(wind_speed_ms: float, wind_direction_deg: float, course_deg: float) -> float:
    """Component of wind blowing AGAINST the ship's heading (positive = headwind).

    course_deg = ship's heading (0=North).
    wind_direction_deg = where the wind is coming FROM (meteorological convention).
    Headwind is when wind direction ≈ ship course (wind coming straight at bow).
    """
    if wind_speed_ms is None or wind_direction_deg is None or course_deg is None:
        return 0.0
    rel = math.radians((wind_direction_deg - course_deg + 360) % 360)
    return wind_speed_ms * math.cos(rel)


def wind_direction_multiplier(headwind_ms: float, speed_kn: float = 10.0) -> float:
    """Fuel multiplier from headwind/tailwind component.

    Empirical maritime model: headwind ↑ fuel, tailwind ↓ fuel.
    Effect scales weakly with ship speed (more drag at higher speeds).
    """
    # Coefficient tuned so a 10 m/s headwind adds ~12% fuel at 10 kn
    k = 0.012
    speed_factor = (speed_kn / 10.0) ** 0.5  # mild speed dependence
    mult = 1.0 + k * headwind_ms * speed_factor
    return max(0.80, min(1.40, mult))  # clamp ±20% to keep sane


def wave_multiplier(wave_height_m: float) -> float:
    """Fuel multiplier from sea-state / waves.

    Empirical: every 1 m of significant wave height adds ~5% fuel
    (added resistance from pitching/heaving). Saturates at 3 m.
    """
    if wave_height_m is None or wave_height_m <= 0:
        return 1.0
    h = min(3.0, float(wave_height_m))
    return 1.0 + 0.05 * h


def apply_corrections(
    base_fuel_mt: float,
    speed_kn: float,
    wind_speed_ms: float = 0.0,
    wind_direction_deg: float = 90.0,
    course_deg: float = 270.0,
    wave_height_m: float = 0.0,
    co2_factor: float = HFO_CO2_FACTOR,
) -> Dict:
    """Return corrected fuel + CO₂ + breakdown of multipliers."""
    head = headwind_component_ms(wind_speed_ms, wind_direction_deg, course_deg)
    wind_mult = wind_direction_multiplier(head, speed_kn)
    wave_mult = wave_multiplier(wave_height_m)
    corrected = base_fuel_mt * wind_mult * wave_mult
    co2 = corrected * co2_factor
    return {
        "base_fuel_mt": round(base_fuel_mt, 3),
        "corrected_fuel_mt": round(corrected, 3),
        "co2_emissions_mt": round(co2, 3),
        "co2_factor": co2_factor,
        "wind_correction": {
            "headwind_component_ms": round(head, 2),
            "multiplier": round(wind_mult, 3),
            "label": "headwind" if head > 1 else "tailwind" if head < -1 else "crosswind",
        },
        "wave_correction": {
            "wave_height_m": round(float(wave_height_m or 0), 2),
            "multiplier": round(wave_mult, 3),
        },
        "total_multiplier": round(wind_mult * wave_mult, 3),
    }


# Route default headings (approximate constant great-circle course)
ROUTE_COURSES = {
    "Khalifa_to_Ruwais": 270.0,  # heading West
    "Ruwais_to_Khalifa": 90.0,   # heading East
}
