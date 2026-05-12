"""Variable-speed voyage planner.

Given a route's waypoints + required time-available, assign a speed (knots)
to each segment so that:
 • The summed travel time exactly matches the available time (when feasible).
 • Speeds vary by segment — longer segments get slower speeds (better fuel
   economy via cube-law), shorter segments get a bump.
 • No segment exceeds vessel max speed (12 kn). Min speed clamped to 6 kn.
 • If the ETA is critical (required avg ≥ max speed), all segments run at
   constant max speed — flagged as infeasible upstream.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coords in nautical miles."""
    R_km = 6371.0
    lat1r, lon1r, lat2r, lon2r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R_km * c * 0.539957


def segment_distances(waypoints: List[Dict]) -> List[float]:
    """Return distance (NM) from each waypoint to the next. Last value is 0."""
    out: List[float] = []
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        d = haversine_nm(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
        out.append(round(d, 2))
    out.append(0.0)
    return out


def allocate_variable_speeds(
    distances: List[float],
    hours_available: float,
    max_speed: float = 12.0,
    min_speed: float = 6.0,
) -> Tuple[List[float], Dict]:
    """Return per-waypoint speeds + summary stats.

    `distances` has length n (one per waypoint, last is 0).
    Returned speeds list has same length n; speed at index i applies to the
    segment FROM waypoint i TO waypoint i+1 (last is mirrored for display).
    """
    segs = [d for d in distances[:-1] if d > 0]
    n_seg = len(segs)
    total = sum(segs)
    if total <= 0 or n_seg == 0:
        return [max_speed] * len(distances), {
            "mode": "no-distance",
            "required_avg_kn": max_speed,
            "total_distance_nm": 0.0,
            "total_time_hours": 0.0,
        }

    req_avg = total / hours_available if hours_available > 0 else max_speed
    crit_threshold = max_speed - 0.15  # if required avg is within 0.15 kn of max → critical

    # CRITICAL — must run at max speed to even attempt ETA
    if req_avg >= crit_threshold:
        speeds = [max_speed] * n_seg
        total_time = sum(d / s for d, s in zip(segs, speeds))
        full = speeds + [speeds[-1]]
        return full, {
            "mode": "constant-max",
            "required_avg_kn": round(req_avg, 2),
            "total_distance_nm": round(total, 2),
            "total_time_hours": round(total_time, 2),
        }

    # NORMAL — modulate by segment length so longer segments get slower speeds.
    d_mean = total / n_seg
    slack = max_speed - req_avg  # how much headroom we have
    amp = min(1.5, slack * 0.6)  # speed-variation amplitude (knots)

    raw = []
    for d in segs:
        rel = (d - d_mean) / d_mean if d_mean > 0 else 0
        # longer segment → rel > 0 → slower; shorter segment → faster
        s = req_avg - amp * rel
        raw.append(s)

    # Iteratively rebalance so total time exactly matches hours_available
    for _ in range(8):
        clamped = [max(min_speed, min(max_speed, s)) for s in raw]
        total_time = sum(d / s for d, s in zip(segs, clamped))
        if abs(total_time - hours_available) < 0.02:
            raw = clamped
            break
        # If total_time > target → too slow → speed up (factor > 1)
        factor = total_time / hours_available
        raw = [s * factor for s in clamped]
    speeds = [max(min_speed, min(max_speed, s)) for s in raw]

    final_time = sum(d / s for d, s in zip(segs, speeds))
    full = speeds + [speeds[-1]]
    return full, {
        "mode": "variable",
        "required_avg_kn": round(req_avg, 2),
        "amplitude_kn": round(amp, 2),
        "total_distance_nm": round(total, 2),
        "total_time_hours": round(final_time, 2),
        "min_speed_kn": round(min(speeds), 2),
        "max_speed_kn": round(max(speeds), 2),
    }
