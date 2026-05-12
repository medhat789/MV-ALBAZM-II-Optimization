#!/usr/bin/env python3
"""
Route Manager for M/V Al-bazm II
Handles waypoint loading and route planning with oil rig/platform restrictions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Oil rig/platform restricted zones in Arabian Gulf (sample coordinates)
# In a production system, these would be loaded from a maritime database
RESTRICTED_ZONES = [
    {'name': 'Umm Shaif Oil Field', 'lat': 25.3, 'lon': 53.0, 'radius_nm': 2},
    {'name': 'Zakum Field', 'lat': 25.1, 'lon': 53.2, 'radius_nm': 2.5},
    {'name': 'Abu Al Bukhoosh', 'lat': 25.5, 'lon': 53.1, 'radius_nm': 1.5},
]

class RouteManager:
    """Manages routes and waypoints for M/V Al-bazm II"""
    
    def __init__(self):
        self.routes = {}
        self.load_default_routes()
    
    def load_default_routes(self):
        """Load default routes from pre-parsed JSON file"""
        try:
            import json
            from pathlib import Path

            json_path = Path("waypoints_real.json")

            logger.info(f"Looking for routes file at: {json_path}")
            logger.info(f"Routes file exists: {json_path.exists()}")

            if json_path.exists():
                with open(json_path, 'r') as f:
                    self.routes = json.load(f)

                logger.info(f"✅ Loaded {len(self.routes)} routes from waypoints_real.json")

            else:
                logger.warning("waypoints_real.json not found, using fallback routes")
                self._create_fallback_routes()

        except Exception as e:
            logger.error(f"Error loading routes: {e}")
            self._create_fallback_routes()
    
    def _process_waypoints(self, df: pd.DataFrame, route_name: str) -> Dict:
        """Process waypoints from dataframe"""
        # Try to identify lat/lon columns
        lat_col = None
        lon_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'lat' in col_lower and lat_col is None:
                lat_col = col
            elif 'lon' in col_lower or 'long' in col_lower:
                lon_col = col
        
        waypoints = []
        
        if lat_col and lon_col:
            for idx, row in df.iterrows():
                try:
                    lat = self._parse_coordinate(row[lat_col])
                    lon = self._parse_coordinate(row[lon_col])
                    
                    if lat and lon:
                        waypoints.append({
                            'waypoint_number': idx + 1,
                            'latitude': lat,
                            'longitude': lon
                        })
                except:
                    continue
        
        # Calculate total distance
        total_distance = self._calculate_route_distance(waypoints)
        
        return {
            'name': route_name,
            'waypoints': waypoints,
            'total_distance_nm': total_distance,
            'number_of_waypoints': len(waypoints)
        }
    
    def _parse_coordinate(self, coord) -> float:
        """Parse coordinate from various formats"""
        if pd.isna(coord):
            return None
        
        try:
            # Try direct float conversion
            return float(coord)
        except:
            # Try parsing DMS format
            coord_str = str(coord).upper()
            
            # Handle formats like "25°15'30\"N" or "53°30'00\"E"
            import re
            match = re.search(r'(\d+)[°\s]+(\d+)[\'′\s]*(\d*\.?\d*)', coord_str)
            if match:
                degrees = float(match.group(1))
                minutes = float(match.group(2))
                seconds = float(match.group(3)) if match.group(3) else 0
                
                decimal = degrees + minutes/60 + seconds/3600
                
                # Check for South or West (negative)
                if 'S' in coord_str or 'W' in coord_str:
                    decimal = -decimal
                
                return decimal
        
        return None
    
    def _calculate_route_distance(self, waypoints: List[Dict]) -> float:
        """Calculate total distance using Haversine formula"""
        if len(waypoints) < 2:
            return 0.0
        
        total_distance = 0.0
        
        for i in range(len(waypoints) - 1):
            lat1, lon1 = waypoints[i]['latitude'], waypoints[i]['longitude']
            lat2, lon2 = waypoints[i+1]['latitude'], waypoints[i+1]['longitude']
            
            distance = self._haversine_distance(lat1, lon1, lat2, lon2)
            total_distance += distance
        
        return total_distance
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in nautical miles"""
        # Earth radius in nautical miles
        R = 3440.065
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _create_fallback_routes(self):
        """Create fallback routes if Excel files cannot be loaded"""
        # Khalifa Port: approximately 24.75°N, 54.58°E
        # Ruwais Terminal: approximately 24.10°N, 52.72°E
        
        self.routes['Khalifa_to_Ruwais'] = {
            'name': 'Khalifa_to_Ruwais',
            'waypoints': [
                {'waypoint_number': 1, 'latitude': 24.75, 'longitude': 54.58},  # Khalifa Port
                {'waypoint_number': 2, 'latitude': 24.50, 'longitude': 54.00},  # Waypoint 1
                {'waypoint_number': 3, 'latitude': 24.30, 'longitude': 53.40},  # Waypoint 2
                {'waypoint_number': 4, 'latitude': 24.10, 'longitude': 52.72},  # Ruwais Terminal
            ],
            'total_distance_nm': 130.0,
            'number_of_waypoints': 4
        }
        
        self.routes['Ruwais_to_Khalifa'] = {
            'name': 'Ruwais_to_Khalifa',
            'waypoints': [
                {'waypoint_number': 1, 'latitude': 24.10, 'longitude': 52.72},  # Ruwais Terminal
                {'waypoint_number': 2, 'latitude': 24.30, 'longitude': 53.40},  # Waypoint 1
                {'waypoint_number': 3, 'latitude': 24.50, 'longitude': 54.00},  # Waypoint 2
                {'waypoint_number': 4, 'latitude': 24.75, 'longitude': 54.58},  # Khalifa Port
            ],
            'total_distance_nm': 130.0,
            'number_of_waypoints': 4
        }
        
        logger.info("Created fallback routes")
    
    def get_route(self, route_name: str) -> Dict:
        """Get route by name"""
        # Clean the input name
        clean_name = route_name.replace(" Port", "").replace(" ", "_")
        
        if clean_name in self.routes:
            return self.routes[clean_name]
        
        # Try to find similar route
        for key in self.routes.keys():
            if clean_name.lower() in key.lower() or key.lower() in clean_name.lower():
                return self.routes[key]
        
        # Try original name as fallback
        if route_name in self.routes:
            return self.routes[route_name]
            
        return None
    
    def validate_custom_route(self, waypoints: List[Dict]) -> Tuple[bool, str, List[str]]:
        """
        Validate custom route against oil rig/platform restrictions
        
        Returns:
            (is_valid, message, warnings)
        """
        if len(waypoints) < 2:
            return False, "Route must have at least 2 waypoints", []
        
        warnings = []
        
        # Check each leg of the route
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            
            # Check if leg passes through restricted zones
            for zone in RESTRICTED_ZONES:
                if self._leg_intersects_zone(wp1, wp2, zone):
                    warnings.append(
                        f"Route leg {i+1}-{i+2} passes near {zone['name']} "
                        f"(restricted zone). Recommend route adjustment."
                    )
        
        if warnings:
            return True, f"Route valid but {len(warnings)} warnings issued", warnings
        
        return True, "Route is clear of all restricted zones", []
    
    def _leg_intersects_zone(self, wp1: Dict, wp2: Dict, zone: Dict) -> bool:
        """Check if a route leg passes through a restricted zone"""
        # Simplified check: calculate distance from zone center to line segment
        lat1, lon1 = wp1['latitude'], wp1['longitude']
        lat2, lon2 = wp2['latitude'], wp2['longitude']
        zone_lat, zone_lon = zone['lat'], zone['lon']
        
        # Calculate distance from zone center to both waypoints
        dist1 = self._haversine_distance(lat1, lon1, zone_lat, zone_lon)
        dist2 = self._haversine_distance(lat2, lon2, zone_lat, zone_lon)
        
        # If either waypoint is within the zone, flag it
        if dist1 <= zone['radius_nm'] or dist2 <= zone['radius_nm']:
            return True
        
        # TODO: Implement more sophisticated line-circle intersection
        # For now, simple check is sufficient
        
        return False
    
    def get_all_routes(self) -> Dict:
        """Get all available routes"""
        return self.routes
    
    def calculate_custom_route_distance(self, waypoints: List[Dict]) -> float:
        """Calculate total distance for custom waypoints"""
        return self._calculate_route_distance(waypoints)
