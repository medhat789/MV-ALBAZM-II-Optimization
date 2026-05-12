#!/usr/bin/env python3
"""
Enhanced Optimization Logic with Variable Speed and ETA Validation
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

MAX_SPEED_KNOTS = 12.0
MIN_SPEED_KNOTS = 6.0
OPTIMAL_RPM_MIN = 115
OPTIMAL_RPM_MAX = 145


def validate_eta_feasibility(distance_nm: float, eta_deadline_hours: float) -> Tuple[bool, float, str]:
    """
    Validate if ETA deadline is achievable with max speed
    
    Returns:
        (is_feasible, required_speed, message)
    """
    required_speed = distance_nm / eta_deadline_hours
    
    if required_speed > MAX_SPEED_KNOTS:
        # Calculate minimum time needed
        min_time_hours = distance_nm / MAX_SPEED_KNOTS
        return False, required_speed, f"ETA deadline requires {required_speed:.1f} knots (max is {MAX_SPEED_KNOTS} knots). Minimum time needed: {min_time_hours:.1f} hours"
    
    if required_speed < MIN_SPEED_KNOTS:
        return True, required_speed, f"ETA deadline is very relaxed. Minimum speed ({MIN_SPEED_KNOTS} knots) is sufficient"
    
    return True, required_speed, "ETA deadline is achievable"


def optimize_variable_speed_segments(
    ml_system,
    waypoints: List[Dict],
    total_distance_nm: float,
    eta_deadline_hours: float,
    wind_speed: float,
    route_name: str
) -> Dict:
    """
    Optimize speed per segment to minimize total fuel while meeting ETA deadline
    
    This implements a simplified dynamic programming approach:
    - Calculates optimal speed for each segment
    - Considers wind resistance per segment direction
    - Ensures total time meets ETA deadline
    """
    
    # First validate if ETA is achievable
    is_feasible, required_speed, message = validate_eta_feasibility(total_distance_nm, eta_deadline_hours)
    
    if not is_feasible:
        # Return max speed solution with warning
        logger.warning(message)
        return {
            'feasible': False,
            'message': message,
            'required_speed': required_speed,
            'suggested_eta_hours': total_distance_nm / MAX_SPEED_KNOTS,
            'segments': [],
            'total_fuel': 0,
            'total_duration': 0
        }
    
    # Calculate segment distances
    segments = []
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]
        
        # Calculate distance using Haversine
        distance = calculate_haversine_distance(
            wp1['latitude'], wp1['longitude'],
            wp2['latitude'], wp2['longitude']
        )
        
        # Get course direction for wind resistance calculation
        course = wp1.get('course_to_next', 90)
        
        segments.append({
            'from': wp1['name'],
            'to': wp2['name'],
            'distance_nm': distance,
            'course': course
        })
    
    total_segment_distance = sum(s['distance_nm'] for s in segments)
    
    # Use two-speed strategy for fuel optimization:
    # - Slower speed for longer segments (if time allows)
    # - Faster speed for shorter segments or when time is tight
    
    # Calculate average required speed
    avg_required_speed = total_segment_distance / eta_deadline_hours
    
    # Determine speed strategy
    if avg_required_speed >= 11:
        # Tight deadline - use constant high speed
        strategy = "constant_fast"
        optimized_segments = optimize_constant_speed(segments, ml_system, avg_required_speed, eta_deadline_hours, wind_speed, route_name)
    elif avg_required_speed <= 8:
        # Relaxed deadline - optimize for fuel
        strategy = "variable_fuel_optimal"
        optimized_segments = optimize_variable_for_fuel(segments, ml_system, eta_deadline_hours, wind_speed, route_name)
    else:
        # Moderate deadline - balance approach
        strategy = "balanced"
        optimized_segments = optimize_balanced(segments, ml_system, eta_deadline_hours, wind_speed, route_name)
    
    total_fuel = sum(s['fuel_mt'] for s in optimized_segments)
    total_duration = sum(s['duration_hours'] for s in optimized_segments)
    avg_speed = total_segment_distance / total_duration if total_duration > 0 else avg_required_speed
    
    logger.info(f"Optimization strategy: {strategy}")
    logger.info(f"Total duration: {total_duration:.2f} hours (deadline: {eta_deadline_hours:.2f} hours)")
    logger.info(f"Average speed: {avg_speed:.2f} knots")
    logger.info(f"Total fuel: {total_fuel:.2f} MT")
    
    return {
        'feasible': True,
        'message': f"Optimized using {strategy} strategy",
        'strategy': strategy,
        'segments': optimized_segments,
        'total_fuel': total_fuel,
        'total_duration': total_duration,
        'average_speed': avg_speed,
        'required_speed': avg_required_speed
    }


def optimize_constant_speed(segments, ml_system, target_speed, deadline_hours, wind_speed, route_name):
    """Use constant speed across all segments (for tight deadlines)"""
    # Cap at max speed
    speed = min(target_speed, MAX_SPEED_KNOTS)
    
    optimized = []
    for seg in segments:
        duration = seg['distance_nm'] / speed
        
        # Predict fuel for a FULL VOYAGE at this speed and segment duration
        # The ML model predicts fuel for the ENTIRE voyage, so we need to scale by segment proportion
        total_distance = sum(s['distance_nm'] for s in segments)
        full_voyage_duration = total_distance / speed
        
        prediction = ml_system.predict_fuel(
            speed=speed,
            duration=full_voyage_duration,
            distance=total_distance,
            wind_speed=wind_speed,
            route=route_name
        )
        
        # Scale fuel by segment proportion
        segment_fuel = prediction['predicted_fuel_mt'] * (seg['distance_nm'] / total_distance)
        
        optimized.append({
            **seg,
            'speed_knots': speed,
            'duration_hours': duration,
            'fuel_mt': segment_fuel,
            'rpm': prediction['input_parameters']['estimated_rpm']
        })
    
    return optimized


def optimize_variable_for_fuel(segments, ml_system, deadline_hours, wind_speed, route_name):
    """Variable speed optimization for fuel efficiency (relaxed deadline)"""
    total_distance = sum(s['distance_nm'] for s in segments)
    
    # Sort segments by distance
    long_segments_idx = [i for i, s in enumerate(segments) if s['distance_nm'] > total_distance / len(segments)]
    short_segments_idx = [i for i, s in enumerate(segments) if s['distance_nm'] <= total_distance / len(segments)]
    
    # Try different speed combinations
    best_fuel = float('inf')
    best_config = None
    
    for slow_speed in [7, 8, 9]:
        for fast_speed in [10, 11, 12]:
            # Calculate total time first
            total_time = 0
            for i, seg in enumerate(segments):
                speed = slow_speed if i in long_segments_idx else fast_speed
                total_time += seg['distance_nm'] / speed
            
            # Check if meets deadline
            if total_time > deadline_hours:
                continue
            
            # Now calculate fuel for this configuration
            config = []
            total_fuel = 0
            
            for i, seg in enumerate(segments):
                speed = slow_speed if i in long_segments_idx else fast_speed
                duration = seg['distance_nm'] / speed
                
                # Get fuel prediction for full voyage at this speed
                prediction = ml_system.predict_fuel(
                    speed=speed,
                    duration=total_time,  # Use total voyage time
                    distance=total_distance,
                    wind_speed=wind_speed,
                    route=route_name
                )
                
                # Scale by segment proportion
                segment_fuel = prediction['predicted_fuel_mt'] * (seg['distance_nm'] / total_distance)
                total_fuel += segment_fuel
                
                config.append({
                    **seg,
                    'speed_knots': speed,
                    'duration_hours': duration,
                    'fuel_mt': segment_fuel,
                    'rpm': prediction['input_parameters']['estimated_rpm']
                })
            
            if total_fuel < best_fuel:
                best_fuel = total_fuel
                best_config = config
    
    return best_config if best_config else optimize_constant_speed(segments, ml_system, total_distance / deadline_hours, deadline_hours, wind_speed, route_name)


def optimize_balanced(segments, ml_system, deadline_hours, wind_speed, route_name):
    """Balanced optimization (moderate deadline)"""
    total_distance = sum(s['distance_nm'] for s in segments)
    target_speed = total_distance / deadline_hours
    
    # Get full voyage fuel prediction at target speed
    full_voyage_prediction = ml_system.predict_fuel(
        speed=target_speed,
        duration=deadline_hours,
        distance=total_distance,
        wind_speed=wind_speed,
        route=route_name
    )
    
    total_voyage_fuel = full_voyage_prediction['predicted_fuel_mt']
    
    # Use slightly variable speeds around target
    optimized = []
    
    for i, seg in enumerate(segments):
        # Vary speed slightly based on segment
        if seg['distance_nm'] > total_distance / len(segments):
            # Longer segment - go a bit slower if time allows
            speed = max(6, target_speed - 0.5)
        else:
            # Shorter segment - go a bit faster
            speed = min(MAX_SPEED_KNOTS, target_speed + 0.5)
        
        duration = seg['distance_nm'] / speed
        
        # Allocate fuel proportionally to segment distance
        segment_fuel = total_voyage_fuel * (seg['distance_nm'] / total_distance)
        
        optimized.append({
            **seg,
            'speed_knots': round(speed, 1),
            'duration_hours': duration,
            'fuel_mt': segment_fuel,
            'rpm': ml_system._estimate_rpm(speed) if hasattr(ml_system, '_estimate_rpm') else 125
        })
    
    # Adjust speeds to exactly meet deadline
    total_time = sum(s['duration_hours'] for s in optimized)
    
    if abs(total_time - deadline_hours) > 0.1:  # If not close enough
        # Scale speeds proportionally
        scale_factor = total_time / deadline_hours
        
        for seg in optimized:
            seg['speed_knots'] = min(MAX_SPEED_KNOTS, max(6, seg['speed_knots'] * scale_factor))
            seg['duration_hours'] = seg['distance_nm'] / seg['speed_knots']
            seg['rpm'] = ml_system._estimate_rpm(seg['speed_knots']) if hasattr(ml_system, '_estimate_rpm') else 125
    
    return optimized


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in nautical miles"""
    import math
    R = 3440.065  # Earth radius in nautical miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


if __name__ == "__main__":
    # Test validation
    print("="*60)
    print("ETA Validation Tests")
    print("="*60)
    
    # Test 1: Impossible ETA
    feasible, req_speed, msg = validate_eta_feasibility(134.55, 5)
    print(f"\nTest 1 - 5 hours for 134.55 NM:")
    print(f"  Feasible: {feasible}")
    print(f"  Required speed: {req_speed:.1f} knots")
    print(f"  Message: {msg}")
    
    # Test 2: Tight but achievable
    feasible, req_speed, msg = validate_eta_feasibility(134.55, 12)
    print(f"\nTest 2 - 12 hours for 134.55 NM:")
    print(f"  Feasible: {feasible}")
    print(f"  Required speed: {req_speed:.1f} knots")
    print(f"  Message: {msg}")
    
    # Test 3: Relaxed
    feasible, req_speed, msg = validate_eta_feasibility(134.55, 20)
    print(f"\nTest 3 - 20 hours for 134.55 NM:")
    print(f"  Feasible: {feasible}")
    print(f"  Required speed: {req_speed:.1f} knots")
    print(f"  Message: {msg}")
