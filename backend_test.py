#!/usr/bin/env python3
"""
Backend API Testing for M/V Al-bazm II Maritime Fuel Optimization System
Tests all backend endpoints with real API calls
"""
import requests
import sys
import json
from datetime import datetime, timedelta

# Public endpoint from frontend/.env
BASE_URL = "https://invite-platform-2.preview.emergentagent.com/api"

class MaritimeAPITester:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []

    def log_test(self, name, passed, details=""):
        """Log test result"""
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            print(f"✅ PASS: {name}")
            if details:
                print(f"   {details}")
        else:
            self.tests_failed += 1
            self.failures.append({"test": name, "details": details})
            print(f"❌ FAIL: {name}")
            print(f"   {details}")

    def test_health(self):
        """Test GET /api/health"""
        print("\n" + "="*60)
        print("TEST: Health Check")
        print("="*60)
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            data = response.json()
            
            if response.status_code == 200:
                model_loaded = data.get("model_loaded", False)
                routes_loaded = data.get("routes_loaded", False)
                
                if model_loaded and routes_loaded:
                    self.log_test(
                        "Health Check",
                        True,
                        f"Status: {data.get('status')}, model_loaded={model_loaded}, routes_loaded={routes_loaded}"
                    )
                    return True
                else:
                    self.log_test(
                        "Health Check",
                        False,
                        f"model_loaded={model_loaded}, routes_loaded={routes_loaded} (both should be True)"
                    )
                    return False
            else:
                self.log_test("Health Check", False, f"Status {response.status_code}: {data}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False

    def test_model_status(self):
        """Test GET /api/model-status"""
        print("\n" + "="*60)
        print("TEST: Model Status")
        print("="*60)
        try:
            response = requests.get(f"{BASE_URL}/model-status", timeout=10)
            data = response.json()
            
            if response.status_code == 200:
                r2_score = data.get("model_metrics", {}).get("r2_score", 0)
                training_samples = data.get("model_metrics", {}).get("training_samples", 0)
                feature_importance = data.get("model_metrics", {}).get("feature_importance", {})
                total_voyages = data.get("data_statistics", {}).get("total_voyages", 0)
                
                all_good = (
                    r2_score > 0 and
                    training_samples > 0 and
                    len(feature_importance) > 0 and
                    total_voyages > 0
                )
                
                self.log_test(
                    "Model Status",
                    all_good,
                    f"R²={r2_score:.3f}, training_samples={training_samples}, features={len(feature_importance)}, total_voyages={total_voyages}"
                )
                return all_good
            else:
                self.log_test("Model Status", False, f"Status {response.status_code}: {data}")
                return False
        except Exception as e:
            self.log_test("Model Status", False, f"Exception: {str(e)}")
            return False

    def test_weather(self):
        """Test GET /api/weather with live Open-Meteo data"""
        print("\n" + "="*60)
        print("TEST: Live Weather (Open-Meteo)")
        print("="*60)
        try:
            response = requests.get(
                f"{BASE_URL}/weather",
                params={"departure_port": "Khalifa Port", "arrival_port": "Ruwais Port"},
                timeout=15  # Allow more time for external API
            )
            data = response.json()
            
            if response.status_code == 200 and data.get("success"):
                departure = data.get("departure", {})
                midpoint = data.get("midpoint", {})
                arrival = data.get("arrival", {})
                
                # Check all three locations have required fields
                required_fields = ["temperature", "wind_speed", "wind_direction", "humidity", "pressure", "visibility", "wave_height", "sea_state", "impact_score"]
                
                dep_ok = all(field in departure for field in required_fields)
                mid_ok = all(field in midpoint for field in required_fields)
                arr_ok = all(field in arrival for field in required_fields)
                
                # Check source is Open-Meteo
                source_ok = (
                    departure.get("source") == "Open-Meteo" and
                    arrival.get("source") == "Open-Meteo"
                )
                
                all_good = dep_ok and mid_ok and arr_ok and source_ok
                
                self.log_test(
                    "Live Weather",
                    all_good,
                    f"Departure: {departure.get('location_name')} ({departure.get('source')}), "
                    f"Wind: {departure.get('wind_speed')} m/s @ {departure.get('wind_direction')}°, "
                    f"Temp: {departure.get('temperature')}°C, Wave: {departure.get('wave_height')}m"
                )
                
                if all_good:
                    print(f"   Midpoint: Wind {midpoint.get('wind_speed')} m/s, Temp {midpoint.get('temperature')}°C")
                    print(f"   Arrival: {arrival.get('location_name')}, Wind {arrival.get('wind_speed')} m/s")
                
                return all_good
            else:
                self.log_test("Live Weather", False, f"Status {response.status_code} or success=False: {data}")
                return False
        except Exception as e:
            self.log_test("Live Weather", False, f"Exception: {str(e)}")
            return False

    def test_routes(self):
        """Test GET /api/routes"""
        print("\n" + "="*60)
        print("TEST: Routes")
        print("="*60)
        try:
            response = requests.get(f"{BASE_URL}/routes", timeout=10)
            data = response.json()
            
            if response.status_code == 200 and data.get("success"):
                routes = data.get("routes", {})
                total_routes = data.get("total_routes", 0)
                
                # Should have 2 routes: Khalifa->Ruwais and Ruwais->Khalifa
                has_both = (
                    "Khalifa_to_Ruwais" in routes or "Khalifa Port_to_Ruwais Port" in routes
                ) and (
                    "Ruwais_to_Khalifa" in routes or "Ruwais Port_to_Khalifa Port" in routes
                )
                
                all_good = total_routes >= 2 and has_both
                
                self.log_test(
                    "Routes",
                    all_good,
                    f"Total routes: {total_routes}, Routes: {list(routes.keys())}"
                )
                return all_good
            else:
                self.log_test("Routes", False, f"Status {response.status_code}: {data}")
                return False
        except Exception as e:
            self.log_test("Routes", False, f"Exception: {str(e)}")
            return False

    def test_optimize_valid(self):
        """Test POST /api/optimize with valid data"""
        print("\n" + "="*60)
        print("TEST: Optimize Route (Valid)")
        print("="*60)
        try:
            # Create ETA 12 hours from now
            eta = (datetime.utcnow() + timedelta(hours=16)).strftime("%Y-%m-%dT%H:%M")
            
            payload = {
                "departure_port": "Khalifa Port",
                "arrival_port": "Ruwais Port",
                "required_arrival_time": eta,
                "wind_speed": 5.0,
                "wind_direction": 90.0
            }
            
            response = requests.post(f"{BASE_URL}/optimize", json=payload, timeout=20)
            data = response.json()
            
            if response.status_code == 200 and data.get("success"):
                rec_route = data.get("recommended_route", {})
                alt_routes = data.get("alternative_routes", [])
                weather = data.get("weather_conditions", {})
                insights = data.get("optimization_insights", {})
                eta_feas = data.get("eta_feasibility", {})
                
                all_good = (
                    rec_route.get("total_distance_nm") is not None and
                    rec_route.get("estimated_duration_hours") is not None and
                    rec_route.get("total_fuel_mt") is not None and
                    len(alt_routes) > 0 and
                    weather is not None and
                    len(insights) > 0 and
                    eta_feas is not None
                )
                
                self.log_test(
                    "Optimize Valid",
                    all_good,
                    f"Distance: {rec_route.get('total_distance_nm')} NM, "
                    f"Duration: {rec_route.get('estimated_duration_hours')} hrs, "
                    f"Fuel: {rec_route.get('total_fuel_mt')} MT, "
                    f"Alternatives: {len(alt_routes)}, "
                    f"ETA Feasible: {eta_feas.get('feasible')}"
                )
                return all_good
            else:
                self.log_test("Optimize Valid", False, f"Status {response.status_code}: {data}")
                return False
        except Exception as e:
            self.log_test("Optimize Valid", False, f"Exception: {str(e)}")
            return False

    def test_optimize_infeasible_eta(self):
        """Test POST /api/optimize with infeasible ETA (very short time)"""
        print("\n" + "="*60)
        print("TEST: Optimize Route (Infeasible ETA)")
        print("="*60)
        try:
            # Create ETA only 2 hours from now (should be infeasible for 130 NM at max 12 kn)
            eta = (datetime.utcnow() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
            
            payload = {
                "departure_port": "Khalifa Port",
                "arrival_port": "Ruwais Port",
                "required_arrival_time": eta,
                "wind_speed": 5.0,
                "wind_direction": 90.0
            }
            
            response = requests.post(f"{BASE_URL}/optimize", json=payload, timeout=20)
            data = response.json()
            
            if response.status_code == 200:
                eta_feas = data.get("eta_feasibility", {})
                feasible = eta_feas.get("feasible", True)
                required_speed = eta_feas.get("required_speed_kn", 0)
                
                # Should be infeasible and required_speed > 12
                all_good = not feasible and required_speed > 12
                
                self.log_test(
                    "Optimize Infeasible ETA",
                    all_good,
                    f"Feasible: {feasible}, Required Speed: {required_speed} kn (should be > 12)"
                )
                return all_good
            else:
                self.log_test("Optimize Infeasible ETA", False, f"Status {response.status_code}: {data}")
                return False
        except Exception as e:
            self.log_test("Optimize Infeasible ETA", False, f"Exception: {str(e)}")
            return False

    def test_optimize_same_ports(self):
        """Test POST /api/optimize with same departure and arrival (should return 400)"""
        print("\n" + "="*60)
        print("TEST: Optimize Route (Same Ports - Error Case)")
        print("="*60)
        try:
            eta = (datetime.utcnow() + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M")
            
            payload = {
                "departure_port": "Khalifa Port",
                "arrival_port": "Khalifa Port",  # Same as departure
                "required_arrival_time": eta,
                "wind_speed": 5.0,
                "wind_direction": 90.0
            }
            
            response = requests.post(f"{BASE_URL}/optimize", json=payload, timeout=20)
            
            # Should return 400 error
            all_good = response.status_code == 400
            
            self.log_test(
                "Optimize Same Ports",
                all_good,
                f"Status: {response.status_code} (expected 400)"
            )
            return all_good
        except Exception as e:
            self.log_test("Optimize Same Ports", False, f"Exception: {str(e)}")
            return False

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed} ✅")
        print(f"Failed: {self.tests_failed} ❌")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        if self.failures:
            print("\n" + "="*60)
            print("FAILED TESTS:")
            print("="*60)
            for failure in self.failures:
                print(f"❌ {failure['test']}")
                print(f"   {failure['details']}")
        
        return self.tests_failed == 0


def main():
    print("="*60)
    print("M/V AL-BAZM II BACKEND API TESTING")
    print("="*60)
    print(f"Testing endpoint: {BASE_URL}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = MaritimeAPITester()
    
    # Run all tests
    tester.test_health()
    tester.test_model_status()
    tester.test_weather()
    tester.test_routes()
    tester.test_optimize_valid()
    tester.test_optimize_infeasible_eta()
    tester.test_optimize_same_ports()
    
    # Print summary
    success = tester.print_summary()
    
    print("\n" + "="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
