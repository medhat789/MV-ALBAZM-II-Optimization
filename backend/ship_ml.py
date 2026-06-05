#!/usr/bin/env python3
"""
M/V Al-bazm II ML Fuel Prediction System — v2.1
Fixed: numpy type serialization, data loader robustness
BACKWARD COMPATIBILITY: Drop-in replacement for ship_ml.py.
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

try:
    from data_loader import load_all_data
except ImportError:
    load_all_data = None

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_CACHE_DIR / "albazm_model_v2.joblib"
SCALER_PATH = MODEL_CACHE_DIR / "albazm_scaler_v2.joblib"
META_PATH = MODEL_CACHE_DIR / "albazm_meta_v2.json"
MODEL_PATH_V1 = MODEL_CACHE_DIR / "albazm_model.joblib"
SCALER_PATH_V1 = MODEL_CACHE_DIR / "albazm_scaler.joblib"
META_PATH_V1 = MODEL_CACHE_DIR / "albazm_meta.json"

MAX_SPEED_KNOTS = 12.0
OPTIMAL_RPM_MIN = 115
OPTIMAL_RPM_MAX = 145

OPTIMAL_HYPERPARAMS = {
    "n_estimators": 200, "max_depth": 10,
    "min_samples_split": 5, "min_samples_leaf": 2,
    "random_state": 42, "n_jobs": -1,
}


def _to_native(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.bool_, np.bool)):
        return bool(val)
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_native(v) for v in val]
    return val


class AlbazmMLSystem:
    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.training_data: Optional[pd.DataFrame] = None
        self.model_stats: Dict[str, Any] = {}
        self._cached_training_stats: Dict[str, Any] = {}

    def load_and_prepare_data(self, engine_file: str = "engine_data.csv") -> pd.DataFrame:
        logger.info("Loading M/V Al-bazm II data — v2.1")
        df: Optional[pd.DataFrame] = None
        if load_all_data is not None:
            try:
                df = load_all_data()
                if df is not None and not df.empty:
                    logger.info("Using multi-source dataset (CE + ROB + ECDIS)")
                else:
                    df = None
            except Exception as e:
                logger.warning("Multi-source failed (%s: %s) — falling back", type(e).__name__, e)
                df = None
        if df is None or df.empty:
            logger.info("Falling back to legacy engine_data.csv")
            df = self._load_legacy_data(engine_file)
        df = self._engineer_features(df)
        df = self._final_cleaning(df)
        self.training_data = df
        logger.info("Final: %d voyages, %d features", len(df), len(self.feature_names))
        return df

    def _load_legacy_data(self, engine_file: str) -> pd.DataFrame:
        engine_path = Path(engine_file)
        if not engine_path.exists():
            engine_path = BASE_DIR / engine_path.name
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine data not found: {engine_file}")
        logger.info("Loading legacy: %s", engine_path)
        for enc in ["latin1", "iso-8859-1", "cp1252", "utf-8"]:
            try:
                df = pd.read_csv(engine_path, delimiter=";", encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            df = pd.read_csv(engine_path, encoding="utf-8")
        df = df.rename(columns={
            "Date": "date", "Time": "time",
            "Total trip time": "duration", "Place": "place",
            "Slip": "slip", "Total Distance": "distance_nm",
            "Avg speed": "speed_knots", "FOC": "fuel_mt",
            "LOAD ": "load_pct", "RPM": "rpm",
        })
        if "Event" in df.columns:
            df = df[df["Event"] == "EOSP"].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["duration", "distance_nm", "speed_knots", "fuel_mt", "slip"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace("\ufffd", "").str.strip(), errors="coerce")
        if "load_pct" in df.columns:
            df["load_pct"] = pd.to_numeric(df["load_pct"].astype(str).str.replace("%", ""), errors="coerce")
        if "rpm" in df.columns:
            df["rpm"] = pd.to_numeric(df["rpm"], errors="coerce")
        df = df.rename(columns={"fuel_mt": "me_fuel_mt", "load_pct": "load", "rpm": "me_rpm"})
        if "load" not in df.columns or df["load"].isna().all():
            df["load"] = df["speed_knots"].apply(self._estimate_engine_load_pct)
        df["route"] = df["place"].apply(self._classify_route) if "place" in df.columns else "Unknown"
        df = self._add_synthetic_weather(df)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering features...")
        if "me_fuel_mt" not in df.columns:
            if "fuel_mt" in df.columns:
                df = df.rename(columns={"fuel_mt": "me_fuel_mt"})
            else:
                raise ValueError("No fuel column found")
        if "speed_knots" not in df.columns and "distance_nm" in df.columns and "duration" in df.columns:
            mask = df["duration"] > 0
            df.loc[mask, "speed_knots"] = df.loc[mask, "distance_nm"] / df.loc[mask, "duration"]
        df["speed_squared"] = df["speed_knots"] ** 2
        df["speed_cubed"] = df["speed_knots"] ** 3
        if "me_rpm" not in df.columns and "rpm" in df.columns:
            df = df.rename(columns={"rpm": "me_rpm"})
        df["rpm_normalized"] = df["me_rpm"].fillna(125) / 150.0
        df["rpm_optimal"] = df["me_rpm"].fillna(125).apply(
            lambda x: 1 if OPTIMAL_RPM_MIN <= x <= OPTIMAL_RPM_MAX else 0
        )
        if "load" not in df.columns:
            df["load"] = np.nan
        missing = df["load"].isna()
        if missing.any():
            df.loc[missing, "load"] = df.loc[missing, "speed_knots"].apply(self._estimate_engine_load_pct)
            logger.info("  LOAD: actual=%d, estimated=%d", (~missing).sum(), missing.sum())
        else:
            logger.info("  LOAD: actual for all %d", len(df))
        if "slip" not in df.columns:
            df["slip"] = 0.0
        df["speed_rpm_interaction"] = df["speed_knots"] * df["rpm_normalized"]
        df["load_dist_interaction"] = df["load"] * df.get("distance_nm", pd.Series([100] * len(df))) / 100.0
        if "route" not in df.columns:
            df["route"] = "Unknown"
        df["route_encoded"] = df["route"].apply(
            lambda x: 1 if "Ruwais_to_Khalifa" in str(x) else 0
        )
        if "date" in df.columns:
            df["month"] = df["date"].dt.month.fillna(6).astype(int)
            df["hour"] = df["date"].dt.hour.fillna(12).astype(int)
        else:
            df["month"] = 6
            df["hour"] = 12
        df["season"] = (df["month"] % 12 // 3)
        df["hour_bin_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
        df["hour_bin_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
        if "wind_speed" not in df.columns:
            df = self._add_synthetic_weather(df)
        else:
            df["wind_speed"] = df["wind_speed"].fillna(8.5)
        if "wind_resistance" not in df.columns:
            df["wind_resistance"] = df["wind_speed"] * 0.5
        df["wind_resistance"] = df["wind_resistance"].fillna(4.25)
        if "sea_state" not in df.columns:
            df["sea_state"] = 3
        df["sea_state"] = df["sea_state"].fillna(3)
        for col in ["relative_wind_angle", "headwind_component", "stw_sog_diff", "current_avg"]:
            df[col] = df.get(col, pd.Series([0.0] * len(df), index=df.index)).fillna(0.0)
        self.feature_names = [
            "speed_knots", "speed_squared", "speed_cubed",
            "duration", "distance_nm",
            "load", "me_rpm", "rpm_normalized", "rpm_optimal",
            "slip",
            "wind_speed", "wind_resistance", "sea_state",
            "route_encoded", "season",
            "hour_bin_morning", "hour_bin_afternoon",
            "speed_rpm_interaction", "load_dist_interaction",
        ]
        self.feature_names = [c for c in self.feature_names if c in df.columns]
        return df

    def _final_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        req = ["me_fuel_mt"] + [c for c in self.feature_names if c in df.columns]
        df = df.dropna(subset=req)
        df = df[(df["me_fuel_mt"] > 0.1) & (df["me_fuel_mt"] < 15)]
        if "speed_knots" in df.columns:
            df = df[df["speed_knots"].between(3, MAX_SPEED_KNOTS + 2)]
        if "duration" in df.columns:
            df = df[df["duration"].between(0.5, 48)]
        if "distance_nm" in df.columns:
            df = df[df["distance_nm"].between(50, 200)]
        if len(df) > 10:
            q1, q3 = df["me_fuel_mt"].quantile(0.25), df["me_fuel_mt"].quantile(0.75)
            iqr = q3 - q1
            df = df[(df["me_fuel_mt"] >= q1 - 1.5 * iqr) & (df["me_fuel_mt"] <= q3 + 1.5 * iqr)]
        logger.info("Cleaning: %d -> %d (dropped %d)", before, len(df), before - len(df))
        return df.reset_index(drop=True)

    def _add_synthetic_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        np.random.seed(42)
        df["wind_speed"] = np.clip(np.random.normal(8.5, 4.0, n), 0, 25)
        df["wind_direction"] = np.random.normal(300, 45, n) % 360
        df["wind_resistance"] = df["wind_speed"] * 0.5
        df["sea_state"] = np.random.choice([2, 3, 4], n, p=[0.4, 0.4, 0.2])
        return df

    def _estimate_engine_load_pct(self, speed: float) -> float:
        return float(np.clip(5.0 * speed - 10.0, 10.0, 100.0))

    def _estimate_rpm(self, speed: float) -> float:
        min_rpm, max_rpm = 110, 150
        min_speed, max_speed = 6, MAX_SPEED_KNOTS
        if speed <= min_speed:
            return min_rpm
        if speed >= max_speed:
            return max_rpm
        return min_rpm + (speed - min_speed) * (max_rpm - min_rpm) / (max_speed - min_speed)

    def _classify_route(self, place_text) -> str:
        if pd.isna(place_text):
            return "Unknown"
        place = str(place_text).upper()
        if "KHALIFA" in place or "KHL" in place or "KP" in place:
            return "Ruwais_to_Khalifa"
        elif "RUWAIS" in place or "RWS" in place:
            return "Khalifa_to_Ruwais"
        return "Unknown"

    def train_model(self) -> Dict[str, Any]:
        if self.training_data is None:
            raise ValueError("No training data")
        logger.info("Training Random Forest v2.1")
        df = self.training_data
        X = df[self.feature_names].copy()
        y = df["me_fuel_mt"].copy()
        X = X.fillna(X.median())
        logger.info("Features: %s", self.feature_names)
        logger.info("Samples: %d", len(X))
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        self.model = RandomForestRegressor(**OPTIMAL_HYPERPARAMS)
        self.model.fit(X_train_s, y_train)
        y_pred_train = self.model.predict(X_train_s)
        y_pred_test = self.model.predict(X_test_s)
        train_r2 = float(r2_score(y_train, y_pred_train))
        test_r2 = float(r2_score(y_test, y_pred_test))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        test_mae = float(mean_absolute_error(y_test, y_pred_test))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train_s, y_train, cv=kf, scoring="r2")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        np.random.seed(42)
        boot_r2s = []
        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred_test)
        for _ in range(1000):
            idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
            boot_r2s.append(r2_score(y_test_arr[idx], y_pred_arr[idx]))
        ci_lower = float(np.percentile(boot_r2s, 2.5))
        ci_upper = float(np.percentile(boot_r2s, 97.5))
        fi = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        self.model_stats = {
            "train_r2": train_r2, "test_r2": test_r2,
            "test_rmse": test_rmse, "test_mae": test_mae,
            "cv_mean": cv_mean, "cv_std": cv_std,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "training_samples": len(X_train), "test_samples": len(X_test),
            "features_used": len(self.feature_names),
            "feature_importance": fi.to_dict("records"),
            "model_version": "2.1",
            "hyperparams": OPTIMAL_HYPERPARAMS,
        }
        logger.info("Train R2=%.4f | Test R2=%.4f | MAE=%.4f MT", train_r2, test_r2, test_mae)
        logger.info("CV R2=%.4f +/- %.4f | CI=[%.4f, %.4f]", cv_mean, cv_std, ci_lower, ci_upper)
        self.save_model()
        return self.model_stats

    def predict_fuel(self, speed: float, duration: float, distance: Optional[float] = None,
                     wind_speed: float = 8.5, route: str = "Khalifa_to_Ruwais",
                     target_rpm: Optional[float] = None) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "No trained model available"}
        speed = min(speed, MAX_SPEED_KNOTS)
        if distance is None:
            distance = speed * duration
        if target_rpm is None:
            target_rpm = self._estimate_rpm(speed)
        rpm_optimal = 1 if OPTIMAL_RPM_MIN <= target_rpm <= OPTIMAL_RPM_MAX else 0
        row: Dict[str, float] = {
            "speed_knots": speed, "speed_squared": speed ** 2, "speed_cubed": speed ** 3,
            "duration": duration, "distance_nm": distance,
            "load": self._estimate_engine_load_pct(speed),
            "me_rpm": target_rpm, "rpm_normalized": target_rpm / 150.0,
            "rpm_optimal": rpm_optimal, "slip": 0.0,
            "wind_speed": wind_speed, "wind_resistance": wind_speed * 0.5,
            "sea_state": 3 if wind_speed < 10 else 4 if wind_speed < 15 else 5,
            "route_encoded": 1 if "Ruwais_to_Khalifa" in route else 0,
            "season": 1, "hour_bin_morning": 0, "hour_bin_afternoon": 1,
            "speed_rpm_interaction": speed * (target_rpm / 150.0),
            "load_dist_interaction": (self._estimate_engine_load_pct(speed) * distance) / 100.0,
        }
        features = pd.DataFrame({name: [row.get(name, 0)] for name in self.feature_names})
        features_scaled = self.scaler.transform(features)
        prediction = float(self.model.predict(features_scaled)[0])
        prediction = max(1.0, prediction)
        confidence = self.model_stats.get("test_r2", 0.0)
        return {
            "predicted_fuel_mt": round(prediction, 3),
            "model_confidence_r2": confidence,
            "input_parameters": {
                "speed_knots": speed, "duration_hours": duration,
                "distance_nm": distance, "estimated_rpm": target_rpm,
                "rpm_in_optimal_range": bool(rpm_optimal == 1),
                "wind_speed_mps": wind_speed, "route": route,
            },
            "efficiency_metrics": {
                "fuel_per_hour": round(prediction / duration, 3) if duration > 0 else 0,
                "fuel_per_nm": round(prediction / distance, 3) if distance > 0 else 0,
            },
        }

    def save_model(self) -> None:
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        meta = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "version": "2.1",
            "feature_names": list(self.feature_names),
            "model_stats": {k: _to_native(v) for k, v in self.model_stats.items() if k != "feature_importance"},
            "feature_importance": _to_native(self.model_stats.get("feature_importance", [])),
            "training_statistics": self.get_training_statistics(),
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info("Model saved: %s", MODEL_PATH.name)

    def load_model(self) -> bool:
        paths = [(MODEL_PATH, SCALER_PATH, META_PATH),
                 (MODEL_PATH_V1, SCALER_PATH_V1, META_PATH_V1)]
        for model_p, scaler_p, meta_p in paths:
            if not all(p.exists() for p in (model_p, scaler_p, meta_p)):
                continue
            try:
                self.model = joblib.load(model_p)
                self.scaler = joblib.load(scaler_p)
                with open(meta_p) as f:
                    meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])
                self.model_stats = meta.get("model_stats", {})
                self.model_stats["feature_importance"] = meta.get("feature_importance", [])
                self._cached_training_stats = meta.get("training_statistics", {})
                ver = meta.get("version", "1.x")
                logger.info("Loaded cached model v%s from %s", ver, model_p.name)
                return True
            except Exception as e:
                logger.warning("Failed to load %s: %s", model_p, e)
        return False

    def get_training_statistics(self) -> Dict[str, Any]:
        if self.training_data is None:
            return _to_native(getattr(self, "_cached_training_stats", {}) or {})
        df = self.training_data
        stats = {
            "total_voyages": int(len(df)),
            "fuel_consumption": {
                "min_mt": float(df["me_fuel_mt"].min()),
                "max_mt": float(df["me_fuel_mt"].max()),
                "mean_mt": float(df["me_fuel_mt"].mean()),
                "std_mt": float(df["me_fuel_mt"].std()),
            },
            "operational": {
                "mean_speed_knots": float(df["speed_knots"].mean()) if "speed_knots" in df.columns else 0,
                "mean_duration_hours": float(df["duration"].mean()) if "duration" in df.columns else 0,
                "speed_range_knots": f"{df['speed_knots'].min():.1f} - {df['speed_knots'].max():.1f}" if "speed_knots" in df.columns else "N/A",
            },
            "routes": {str(k): int(v) for k, v in df["route"].value_counts().items()} if "route" in df.columns else {},
            "data_sources": {
                "ce_daily_log": True,
                "rob": bool("load" in df.columns and df["load"].notna().any()),
                "ecdis_weather": bool("wind_speed" in df.columns and df["wind_speed"].notna().any()),
            },
        }
        if "date" in df.columns and not df["date"].isna().all():
            stats["date_range"] = {
                "start": df["date"].min().strftime("%Y-%m-%d"),
                "end": df["date"].max().strftime("%Y-%m-%d"),
            }
        return _to_native(stats)

    def generate_academic_report(self) -> Dict[str, Any]:
        if not self.model_stats:
            return {"error": "No model trained yet"}
        report = {
            "vessel_info": {
                "name": "M/V Al-bazm II",
                "max_speed_knots": MAX_SPEED_KNOTS,
                "optimal_rpm_range": f"{OPTIMAL_RPM_MIN}-{OPTIMAL_RPM_MAX}",
            },
            "dataset_info": {
                "data_period": "Jun 2024 - Nov 2025",
                "total_voyages": self.model_stats.get("training_samples", 0) + self.model_stats.get("test_samples", 0),
                "features_used": self.model_stats.get("features_used", 0),
                "training_samples": self.model_stats.get("training_samples", 0),
                "test_samples": self.model_stats.get("test_samples", 0),
                "feature_names": self.feature_names,
            },
            "methodology": {
                "algorithm": "Random Forest Regression",
                "validation_method": "5-fold CV + 70/30 holdout + bootstrap CI",
                "feature_engineering": [
                    "Speed polynomials", "Actual engine LOAD", "Propeller slip",
                    "Weather impact", "Route characteristics", "Temporal features",
                    "Feature interactions",
                ],
                "preprocessing": "StandardScaler + IQR outlier removal",
                "hyperparameters": OPTIMAL_HYPERPARAMS,
            },
            "results": {
                "test_r2_score": self.model_stats.get("test_r2", 0),
                "test_rmse_mt": self.model_stats.get("test_rmse", 0),
                "test_mae_mt": self.model_stats.get("test_mae", 0),
                "cv_mean_r2": self.model_stats.get("cv_mean", 0),
                "cv_std_r2": self.model_stats.get("cv_std", 0),
                "bootstrap_ci_95": [self.model_stats.get("ci_lower", 0),
                                     self.model_stats.get("ci_upper", 0)],
            },
            "feature_importance": self.model_stats.get("feature_importance", []),
            "training_statistics": self.get_training_statistics(),
            "model_version": "2.1",
        }
        return _to_native(report)


if __name__ == "__main__":
    print("=" * 60)
    print("M/V Al-bazm II ML v2.1 — Self Test")
    print("=" * 60)
    ml = AlbazmMLSystem()
    try:
        data = ml.load_and_prepare_data()
        print(f"Loaded {len(data)} voyages")
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        import sys; sys.exit(1)
    stats = ml.train_model()
    print("\nPredictions:")
    for s in [8, 10, 11, 12]:
        p = ml.predict_fuel(speed=s, duration=13.5)
        print(f"  {s} kn: {p['predicted_fuel_mt']:.2f} MT")
    r = ml.generate_academic_report()
    print(f"\nR2: {r['results']['test_r2_score']:.4f}")
    print(f"MAE: {r['results']['test_mae_mt']:.4f} MT")
    print("=" * 60)BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_CACHE_DIR / "albazm_model_v2.joblib"
SCALER_PATH = MODEL_CACHE_DIR / "albazm_scaler_v2.joblib"
META_PATH = MODEL_CACHE_DIR / "albazm_meta_v2.json"
MODEL_PATH_V1 = MODEL_CACHE_DIR / "albazm_model.joblib"
SCALER_PATH_V1 = MODEL_CACHE_DIR / "albazm_scaler.joblib"
META_PATH_V1 = MODEL_CACHE_DIR / "albazm_meta.json"

MAX_SPEED_KNOTS = 12.0
OPTIMAL_RPM_MIN = 115
OPTIMAL_RPM_MAX = 145

OPTIMAL_HYPERPARAMS = {
    "n_estimators": 200, "max_depth": 10,
    "min_samples_split": 5, "min_samples_leaf": 2,
    "random_state": 42, "n_jobs": -1,
}


def _to_native(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.bool_, np.bool)):
        return bool(val)
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_native(v) for v in val]
    return val


class AlbazmMLSystem:
    """ML fuel-prediction system — drop-in replacement for ship_ml.py"""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.training_data: Optional[pd.DataFrame] = None
        self.model_stats: Dict[str, Any] = {}
        self._cached_training_stats: Dict[str, Any] = {}

    def load_and_prepare_data(self, engine_file: str = "engine_data.csv") -> pd.DataFrame:
        logger.info("Loading M/V Al-bazm II data — v2.1")

        df: Optional[pd.DataFrame] = None

        if load_all_data is not None:
            try:
                df = load_all_data()
                if df is not None and not df.empty:
                    logger.info("Using multi-source dataset (CE + ROB + ECDIS)")
                else:
                    logger.info("Multi-source empty — falling back")
                    df = None
            except Exception as e:
                logger.warning("Multi-source failed (%s: %s) — falling back",
                               type(e).__name__, e)
                df = None

        if df is None or df.empty:
            logger.info("Falling back to legacy engine_data.csv")
            df = self._load_legacy_data(engine_file)

        df = self._engineer_features(df)
        df = self._final_cleaning(df)
        self.training_data = df
        logger.info("Final: %d voyages, %d features", len(df), len(self.feature_names))
        return df

    def _load_legacy_data(self, engine_file: str) -> pd.DataFrame:
        engine_path = Path(engine_file)
        if not engine_path.exists():
            engine_path = BASE_DIR / engine_path.name
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine data not found: {engine_file}")

        logger.info("Loading legacy: %s", engine_path)
        for enc in ["latin1", "iso-8859-1", "cp1252", "utf-8"]:
            try:
                df = pd.read_csv(engine_path, delimiter=";", encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            df = pd.read_csv(engine_path, encoding="utf-8")

        df = df.rename(columns={
            "Date": "date", "Time": "time",
            "Total trip time": "duration", "Place": "place",
            "Slip": "slip", "Total Distance": "distance_nm",
            "Avg speed": "speed_knots", "FOC": "fuel_mt",
            "LOAD ": "load_pct", "RPM": "rpm",
        })
        if "Event" in df.columns:
            df = df[df["Event"] == "EOSP"].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["duration", "distance_nm", "speed_knots", "fuel_mt", "slip"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace("\ufffd", "").str.strip(), errors="coerce")
        if "load_pct" in df.columns:
            df["load_pct"] = pd.to_numeric(df["load_pct"].astype(str).str.replace("%", ""), errors="coerce")
        if "rpm" in df.columns:
            df["rpm"] = pd.to_numeric(df["rpm"], errors="coerce")
        df = df.rename(columns={"fuel_mt": "me_fuel_mt", "load_pct": "load", "rpm": "me_rpm"})
        if "load" not in df.columns or df["load"].isna().all():
            df["load"] = df["speed_knots"].apply(self._estimate_engine_load_pct)
        df["route"] = df["place"].apply(self._classify_route) if "place" in df.columns else "Unknown"
        df = self._add_synthetic_weather(df)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering features...")

        if "me_fuel_mt" not in df.columns:
            if "fuel_mt" in df.columns:
                df = df.rename(columns={"fuel_mt": "me_fuel_mt"})
            else:
                raise ValueError("No fuel column found")

        # Speed
        if "speed_knots" not in df.columns and "distance_nm" in df.columns and "duration" in df.columns:
            mask = df["duration"] > 0
            df.loc[mask, "speed_knots"] = df.loc[mask, "distance_nm"] / df.loc[mask, "duration"]

        df["speed_squared"] = df["speed_knots"] ** 2
        df["speed_cubed"] = df["speed_knots"] ** 3

        # RPM
        if "me_rpm" not in df.columns and "rpm" in df.columns:
            df = df.rename(columns={"rpm": "me_rpm"})

        df["rpm_normalized"] = df["me_rpm"].fillna(125) / 150.0
        df["rpm_optimal"] = df["me_rpm"].fillna(125).apply(
            lambda x: 1 if OPTIMAL_RPM_MIN <= x <= OPTIMAL_RPM_MAX else 0
        )

        # LOAD (CRITICAL: actual from ROB, estimate fallback)
        if "load" not in df.columns:
            df["load"] = np.nan
        missing = df["load"].isna()
        if missing.any():
            df.loc[missing, "load"] = df.loc[missing, "speed_knots"].apply(self._estimate_engine_load_pct)
            logger.info("  LOAD: actual=%d, estimated=%d", (~missing).sum(), missing.sum())
        else:
            logger.info("  LOAD: actual for all %d", len(df))

        # Slip
        if "slip" not in df.columns:
            df["slip"] = 0.0

        # Interactions
        df["speed_rpm_interaction"] = df["speed_knots"] * df["rpm_normalized"]
        df["load_dist_interaction"] = df["load"] * df.get("distance_nm", 100) / 100.0

        # Route
        if "route" not in df.columns:
            df["route"] = "Unknown"
        df["route_encoded"] = df["route"].apply(
            lambda x: 1 if "Ruwais_to_Khalifa" in str(x) else 0
        )

        # Temporal
        if "date" in df.columns:
            df["month"] = df["date"].dt.month.fillna(6).astype(int)
            df["hour"] = df["date"].dt.hour.fillna(12).astype(int)
        else:
            df["month"] = 6
            df["hour"] = 12
        df["season"] = (df["month"] % 12 // 3)
        df["hour_bin_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
        df["hour_bin_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)

        # Weather (fill NaN with defaults)
        if "wind_speed" not in df.columns:
            df = self._add_synthetic_weather(df)
        else:
            df["wind_speed"] = df["wind_speed"].fillna(8.5)

        if "wind_resistance" not in df.columns:
            df["wind_resistance"] = df["wind_speed"] * 0.5
        df["wind_resistance"] = df["wind_resistance"].fillna(4.25)

        if "sea_state" not in df.columns:
            df["sea_state"] = 3
        df["sea_state"] = df["sea_state"].fillna(3)

        for col in ["relative_wind_angle", "headwind_component", "stw_sog_diff", "current_avg"]:
            df[col] = df.get(col, pd.Series([0.0] * len(df), index=df.index)).fillna(0.0)

        self.feature_names = [
            "speed_knots", "speed_squared", "speed_cubed",
            "duration", "distance_nm",
            "load", "me_rpm", "rpm_normalized", "rpm_optimal",
            "slip",
            "wind_speed", "wind_resistance", "sea_state",
            "route_encoded", "season",
            "hour_bin_morning", "hour_bin_afternoon",
            "speed_rpm_interaction", "load_dist_interaction",
        ]
        self.feature_names = [c for c in self.feature_names if c in df.columns]
        return df

    def _final_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        req = ["me_fuel_mt"] + [c for c in self.feature_names if c in df.columns]
        df = df.dropna(subset=req)
        df = df[(df["me_fuel_mt"] > 0.1) & (df["me_fuel_mt"] < 15)]
        if "speed_knots" in df.columns:
            df = df[df["speed_knots"].between(3, MAX_SPEED_KNOTS + 2)]
        if "duration" in df.columns:
            df = df[df["duration"].between(0.5, 48)]
        if "distance_nm" in df.columns:
            df = df[df["distance_nm"].between(50, 200)]
        if len(df) > 10:
            q1, q3 = df["me_fuel_mt"].quantile(0.25), df["me_fuel_mt"].quantile(0.75)
            iqr = q3 - q1
            df = df[(df["me_fuel_mt"] >= q1 - 1.5 * iqr) & (df["me_fuel_mt"] <= q3 + 1.5 * iqr)]
        logger.info("Cleaning: %d -> %d (dropped %d)", before, len(df), before - len(df))
        return df.reset_index(drop=True)

    def _add_synthetic_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        np.random.seed(42)
        df["wind_speed"] = np.clip(np.random.normal(8.5, 4.0, n), 0, 25)
        df["wind_direction"] = np.random.normal(300, 45, n) % 360
        df["wind_resistance"] = df["wind_speed"] * 0.5
        df["sea_state"] = np.random.choice([2, 3, 4], n, p=[0.4, 0.4, 0.2])
        return df

    def _estimate_engine_load_pct(self, speed: float) -> float:
        return float(np.clip(5.0 * speed - 10.0, 10.0, 100.0))

    def _estimate_rpm(self, speed: float) -> float:
        min_rpm, max_rpm = 110, 150
        min_speed, max_speed = 6, MAX_SPEED_KNOTS
        if speed <= min_speed:
            return min_rpm
        if speed >= max_speed:
            return max_rpm
        return min_rpm + (speed - min_speed) * (max_rpm - min_rpm) / (max_speed - min_speed)

    def _classify_route(self, place_text) -> str:
        if pd.isna(place_text):
            return "Unknown"
        place = str(place_text).upper()
        if "KHALIFA" in place or "KHL" in place or "KP" in place:
            return "Ruwais_to_Khalifa"
        elif "RUWAIS" in place or "RWS" in place:
            return "Khalifa_to_Ruwais"
        return "Unknown"

    def train_model(self) -> Dict[str, Any]:
        if self.training_data is None:
            raise ValueError("No training data")

        logger.info("Training Random Forest v2.1")
        df = self.training_data
        X = df[self.feature_names].copy()
        y = df["me_fuel_mt"].copy()
        X = X.fillna(X.median())

        logger.info("Features: %s", self.feature_names)
        logger.info("Samples: %d", len(X))

        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = RandomForestRegressor(**OPTIMAL_HYPERPARAMS)
        self.model.fit(X_train_s, y_train)

        y_pred_train = self.model.predict(X_train_s)
        y_pred_test = self.model.predict(X_test_s)

        train_r2 = float(r2_score(y_train, y_pred_train))
        test_r2 = float(r2_score(y_test, y_pred_test))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        test_mae = float(mean_absolute_error(y_test, y_pred_test))

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train_s, y_train, cv=kf, scoring="r2")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        np.random.seed(42)
        boot_r2s = []
        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred_test)
        for _ in range(1000):
            idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
            boot_r2s.append(r2_score(y_test_arr[idx], y_pred_arr[idx]))
        ci_lower = float(np.percentile(boot_r2s, 2.5))
        ci_upper = float(np.percentile(boot_r2s, 97.5))

        fi = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        self.model_stats = {
            "train_r2": train_r2, "test_r2": test_r2,
            "test_rmse": test_rmse, "test_mae": test_mae,
            "cv_mean": cv_mean, "cv_std": cv_std,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "training_samples": len(X_train), "test_samples": len(X_test),
            "features_used": len(self.feature_names),
            "feature_importance": fi.to_dict("records"),
            "model_version": "2.1",
            "hyperparams": OPTIMAL_HYPERPARAMS,
        }

        logger.info("Train R2=%.4f | Test R2=%.4f | MAE=%.4f MT", train_r2, test_r2, test_mae)
        logger.info("CV R2=%.4f +/- %.4f | CI=[%.4f, %.4f]", cv_mean, cv_std, ci_lower, ci_upper)

        self.save_model()
        return self.model_stats

    def predict_fuel(self, speed: float, duration: float, distance: Optional[float] = None,
                     wind_speed: float = 8.5, route: str = "Khalifa_to_Ruwais",
                     target_rpm: Optional[float] = None) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "No trained model available"}

        speed = min(speed, MAX_SPEED_KNOTS)
        if distance is None:
            distance = speed * duration
        if target_rpm is None:
            target_rpm = self._estimate_rpm(speed)

        rpm_optimal = 1 if OPTIMAL_RPM_MIN <= target_rpm <= OPTIMAL_RPM_MAX else 0

        row: Dict[str, float] = {
            "speed_knots": speed, "speed_squared": speed ** 2, "speed_cubed": speed ** 3,
            "duration": duration, "distance_nm": distance,
            "load": self._estimate_engine_load_pct(speed),
            "me_rpm": target_rpm, "rpm_normalized": target_rpm / 150.0,
            "rpm_optimal": rpm_optimal, "slip": 0.0,
            "wind_speed": wind_speed, "wind_resistance": wind_speed * 0.5,
            "sea_state": 3 if wind_speed < 10 else 4 if wind_speed < 15 else 5,
            "route_encoded": 1 if "Ruwais_to_Khalifa" in route else 0,
            "season": 1, "hour_bin_morning": 0, "hour_bin_afternoon": 1,
            "speed_rpm_interaction": speed * (target_rpm / 150.0),
            "load_dist_interaction": (self._estimate_engine_load_pct(speed) * distance) / 100.0,
        }

        features = pd.DataFrame({name: [row.get(name, 0)] for name in self.feature_names})
        features_scaled = self.scaler.transform(features)
        prediction = float(self.model.predict(features_scaled)[0])
        prediction = max(1.0, prediction)
        confidence = self.model_stats.get("test_r2", 0.0)

        return {
            "predicted_fuel_mt": round(prediction, 3),
            "model_confidence_r2": confidence,
            "input_parameters": {
                "speed_knots": speed, "duration_hours": duration,
                "distance_nm": distance, "estimated_rpm": target_rpm,
                "rpm_in_optimal_range": bool(rpm_optimal == 1),
                "wind_speed_mps": wind_speed, "route": route,
            },
            "efficiency_metrics": {
                "fuel_per_hour": round(prediction / duration, 3) if duration > 0 else 0,
                "fuel_per_nm": round(prediction / distance, 3) if distance > 0 else 0,
            },
        }

    def save_model(self) -> None:
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        meta = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "version": "2.1",
            "feature_names": list(self.feature_names),
            "model_stats": {k: _to_native(v) for k, v in self.model_stats.items()
                           if k != "feature_importance"},
            "feature_importance": _to_native(self.model_stats.get("feature_importance", [])),
            "training_statistics": self.get_training_statistics(),
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info("Model saved: %s", MODEL_PATH.name)

    def load_model(self) -> bool:
        paths = [(MODEL_PATH, SCALER_PATH, META_PATH),
                 (MODEL_PATH_V1, SCALER_PATH_V1, META_PATH_V1)]
        for model_p, scaler_p, meta_p in paths:
            if not all(p.exists() for p in (model_p, scaler_p, meta_p)):
                continue
            try:
                self.model = joblib.load(model_p)
                self.scaler = joblib.load(scaler_p)
                with open(meta_p) as f:
                    meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])
                self.model_stats = meta.get("model_stats", {})
                self.model_stats["feature_importance"] = meta.get("feature_importance", [])
                self._cached_training_stats = meta.get("training_statistics", {})
                ver = meta.get("version", "1.x")
                logger.info("Loaded cached model v%s from %s", ver, model_p.name)
                return True
            except Exception as e:
                logger.warning("Failed to load %s: %s", model_p, e)
        return False

    def get_training_statistics(self) -> Dict[str, Any]:
        if self.training_data is None:
            return _to_native(getattr(self, "_cached_training_stats", {}) or {})

        df = self.training_data
        stats = {
            "total_voyages": int(len(df)),
            "fuel_consumption": {
                "min_mt": float(df["me_fuel_mt"].min()),
                "max_mt": float(df["me_fuel_mt"].max()),
                "mean_mt": float(df["me_fuel_mt"].mean()),
                "std_mt": float(df["me_fuel_mt"].std()),
            },
            "operational": {
                "mean_speed_knots": float(df["speed_knots"].mean()) if "speed_knots" in df.columns else 0,
                "mean_duration_hours": float(df["duration"].mean()) if "duration" in df.columns else 0,
                "speed_range_knots": f"{df['speed_knots'].min():.1f} - {df['speed_knots'].max():.1f}" if "speed_knots" in df.columns else "N/A",
            },
            "routes": {str(k): int(v) for k, v in df["route"].value_counts().items()} if "route" in df.columns else {},
            "data_sources": {
                "ce_daily_log": True,
                "rob": bool("load" in df.columns and df["load"].notna().any()),
                "ecdis_weather": bool("wind_speed" in df.columns and df["wind_speed"].notna().any()),
            },
        }
        if "date" in df.columns and not df["date"].isna().all():
            stats["date_range"] = {
                "start": df["date"].min().strftime("%Y-%m-%d"),
                "end": df["date"].max().strftime("%Y-%m-%d"),
            }
        return _to_native(stats)

    def generate_academic_report(self) -> Dict[str, Any]:
        if not self.model_stats:
            return {"error": "No model trained yet"}
        report = {
            "vessel_info": {
                "name": "M/V Al-bazm II",
                "max_speed_knots": MAX_SPEED_KNOTS,
                "optimal_rpm_range": f"{OPTIMAL_RPM_MIN}-{OPTIMAL_RPM_MAX}",
            },
            "dataset_info": {
                "data_period": "Jun 2024 - Nov 2025",
                "total_voyages": self.model_stats.get("training_samples", 0) + self.model_stats.get("test_samples", 0),
                "features_used": self.model_stats.get("features_used", 0),
                "training_samples": self.model_stats.get("training_samples", 0),
                "test_samples": self.model_stats.get("test_samples", 0),
                "feature_names": self.feature_names,
            },
            "methodology": {
                "algorithm": "Random Forest Regression",
                "validation_method": "5-fold CV + 70/30 holdout + bootstrap CI",
                "feature_engineering": [
                    "Speed polynomials", "Actual engine LOAD", "Propeller slip",
                    "Weather impact", "Route characteristics", "Temporal features",
                    "Feature interactions",
                ],
                "preprocessing": "StandardScaler + IQR outlier removal",
                "hyperparameters": OPTIMAL_HYPERPARAMS,
            },
            "results": {
                "test_r2_score": self.model_stats.get("test_r2", 0),
                "test_rmse_mt": self.model_stats.get("test_rmse", 0),
                "test_mae_mt": self.model_stats.get("test_mae", 0),
                "cv_mean_r2": self.model_stats.get("cv_mean", 0),
                "cv_std_r2": self.model_stats.get("cv_std", 0),
                "bootstrap_ci_95": [self.model_stats.get("ci_lower", 0),
                                     self.model_stats.get("ci_upper", 0)],
            },
            "feature_importance": self.model_stats.get("feature_importance", []),
            "training_statistics": self.get_training_statistics(),
            "model_version": "2.1",
        }
        return _to_native(report)


if __name__ == "__main__":
    print("=" * 60)
    print("M/V Al-bazm II ML v2.1 — Self Test")
    print("=" * 60)
    ml = AlbazmMLSystem()
    try:
        data = ml.load_and_prepare_data()
        print(f"Loaded {len(data)} voyages")
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        import sys; sys.exit(1)
    stats = ml.train_model()
    print("\nPredictions:")
    for s in [8, 10, 11, 12]:
        p = ml.predict_fuel(speed=s, duration=13.5)
        print(f"  {s} kn: {p['predicted_fuel_mt']:.2f} MT")
    r = ml.generate_academic_report()
    print(f"\nR2: {r['results']['test_r2_score']:.4f}")
    print(f"MAE: {r['results']['test_mae_mt']:.4f} MT")
    print("=" * 60)
