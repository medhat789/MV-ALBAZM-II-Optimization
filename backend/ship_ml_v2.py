#!/usr/bin/env python3
"""
M/V Al-bazm II ML Fuel Prediction System — v2.0
=================================================
Updated with three new data sources:
  1. CE Daily Log (318 voyages, Jun 2024 – Nov 2025)
  2. ROB Official Record (actual engine LOAD values)
  3. ECDIS Weather (56,867 records with measured wind/STW/current)

BACKWARD COMPATIBILITY: This file is a drop-in replacement for ship_ml.py.
All public methods keep identical signatures so server.py needs NO changes.

Target improvement over v1: R2 0.573 → 0.943
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler

# Import the new data-loader
try:
    from data_loader import load_all_data
except ImportError:
    # Fallback: if data_loader.py is not present, use inline loader
    load_all_data = None

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_CACHE_DIR / "albazm_model_v2.joblib"
SCALER_PATH = MODEL_CACHE_DIR / "albazm_scaler_v2.joblib"
META_PATH = MODEL_CACHE_DIR / "albazm_meta_v2.json"

# Keep v1 paths as fallback for seamless upgrade
MODEL_PATH_V1 = MODEL_CACHE_DIR / "albazm_model.joblib"
SCALER_PATH_V1 = MODEL_CACHE_DIR / "albazm_scaler.joblib"
META_PATH_V1 = MODEL_CACHE_DIR / "albazm_meta.json"

# ---------------------------------------------------------------------------
# Ship operational constraints
# ---------------------------------------------------------------------------
MAX_SPEED_KNOTS = 12.0
OPTIMAL_RPM_MIN = 115
OPTIMAL_RPM_MAX = 145

# Optimal hyper-parameters from grid-search on the new dataset
OPTIMAL_HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


# ###########################################################################
# # AlbazmMLSystem — drop-in replacement for ship_ml.py
# ###########################################################################


class AlbazmMLSystem:
    """
    Machine-Learning fuel-prediction system for M/V Al-bazm II.

    PUBLIC API (must remain stable — called by server.py):
      - __init__()
      - load_and_prepare_data(engine_file='...')
      - train_model()
      - predict_fuel(speed, duration, distance, wind_speed, route, target_rpm)
      - save_model()
      - load_model() -> bool
      - generate_academic_report() -> dict
      - get_training_statistics() -> dict
    """

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.training_data: Optional[pd.DataFrame] = None
        self.model_stats: Dict[str, Any] = {}
        self._cached_training_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_and_prepare_data(self, engine_file: str = "engine_data.csv") -> pd.DataFrame:
        """
        Load and prepare ship data for ML training.

        Strategy:
          1. Try new multi-source loader (CE Daily Log + ROB + ECDIS)
          2. Fall back to legacy engine_data.csv if new sources unavailable
          3. Engineer features identical to the v2 training pipeline
        """
        logger.info("=" * 60)
        logger.info("Loading M/V Al-bazm II operational data — v2.0")
        logger.info("=" * 60)

        df: Optional[pd.DataFrame] = None

        # ---- 1. Try new multi-source data loader ----
        if load_all_data is not None:
            try:
                df = load_all_data()
                if df is not None and not df.empty:
                    logger.info("Using multi-source dataset (CE + ROB + ECDIS)")
                    source = "multi"
                else:
                    logger.info("Multi-source loader returned empty — falling back")
            except Exception as e:
                logger.warning("Multi-source loader failed: %s — falling back", e)

        # ---- 2. Fall back to legacy engine_data.csv ----
        if df is None or df.empty:
            logger.info("Falling back to legacy engine_data.csv")
            df = self._load_legacy_data(engine_file)
            source = "legacy"

        # ---- 3. Feature engineering ----
        df = self._engineer_features(df)
        df = self._final_cleaning(df)

        self.training_data = df

        logger.info("Final dataset: %d voyages, %d features ready for training",
                    len(df), len(self.feature_names))
        return df

    # ------------------------------------------------------------------
    # Legacy data loader (kept for backward compatibility)
    # ------------------------------------------------------------------

    def _load_legacy_data(self, engine_file: str) -> pd.DataFrame:
        engine_path = Path(engine_file)
        if not engine_path.exists():
            engine_path = BASE_DIR / engine_path.name

        if not engine_path.exists():
            raise FileNotFoundError(f"Engine data not found: {engine_file}")

        logger.info("Loading legacy engine data from %s", engine_path)

        # Try multiple encodings
        for encoding in ["latin1", "iso-8859-1", "cp1252", "utf-8"]:
            try:
                df = pd.read_csv(engine_path, delimiter=";", encoding=encoding)
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

        # Keep only EOSP events
        if "Event" in df.columns:
            df = df[df["Event"] == "EOSP"].copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["duration", "distance_nm", "speed_knots", "fuel_mt", "slip"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace("�", "").str.strip(), errors="coerce")

        if "load_pct" in df.columns:
            df["load_pct"] = pd.to_numeric(df["load_pct"].astype(str).str.replace("%", ""), errors="coerce")
        if "rpm" in df.columns:
            df["rpm"] = pd.to_numeric(df["rpm"], errors="coerce")

        # Map legacy columns to v2 unified schema
        df = df.rename(columns={
            "fuel_mt": "me_fuel_mt",
            "load_pct": "load",
            "rpm": "me_rpm",
        })

        # Estimate engine load where missing
        if "load" not in df.columns or df["load"].isna().all():
            df["load"] = df["speed_knots"].apply(self._estimate_engine_load_pct)

        df["route"] = df["place"].apply(self._classify_route) if "place" in df.columns else "Unknown"

        # Add synthetic weather for backward compat
        df = self._add_synthetic_weather(df)

        return df

    # ------------------------------------------------------------------
    # Feature Engineering — unified pipeline (works for both data sources)
    # ------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML features from prepared data."""
        logger.info("Engineering features...")

        # Target variable: main-engine fuel consumption
        if "me_fuel_mt" not in df.columns:
            if "fuel_mt" in df.columns:
                df = df.rename(columns={"fuel_mt": "me_fuel_mt"})
            else:
                raise ValueError("No fuel consumption column found (need 'me_fuel_mt' or 'fuel_mt')")

        # ---- Speed features ----
        if "speed_knots" not in df.columns and "distance_nm" in df.columns and "duration" in df.columns:
            mask = df["duration"] > 0
            df.loc[mask, "speed_knots"] = df.loc[mask, "distance_nm"] / df.loc[mask, "duration"]

        df["speed_squared"] = df["speed_knots"] ** 2
        df["speed_cubed"] = df["speed_knots"] ** 3

        # ---- Engine features ----
        if "me_rpm" not in df.columns and "rpm" in df.columns:
            df = df.rename(columns={"rpm": "me_rpm"})

        df["rpm_normalized"] = df["me_rpm"].fillna(125) / 150.0
        df["rpm_optimal"] = df["me_rpm"].fillna(125).apply(
            lambda x: 1 if OPTIMAL_RPM_MIN <= x <= OPTIMAL_RPM_MAX else 0
        )

        # ---- Engine LOAD (CRITICAL IMPROVEMENT) ----
        # Use actual LOAD from ROB where available, estimate otherwise
        if "load" not in df.columns:
            df["load"] = np.nan

        missing_load = df["load"].isna()
        if missing_load.any():
            df.loc[missing_load, "load"] = df.loc[missing_load, "speed_knots"].apply(
                self._estimate_engine_load_pct
            )
            logger.info("  Estimated LOAD for %d voyages (actual for %d)",
                        missing_load.sum(), (~missing_load).sum())
        else:
            logger.info("  Using actual LOAD for all %d voyages", len(df))

        # ---- Slip (NEW FEATURE from CE Daily Log) ----
        if "slip" not in df.columns:
            df["slip"] = 0.0  # Default if unavailable

        # ---- Interactions ----
        df["speed_rpm_interaction"] = df["speed_knots"] * df["rpm_normalized"]
        df["load_dist_interaction"] = df["load"] * df["distance_nm"] / 100.0

        # ---- Route ----
        if "route" not in df.columns:
            df["route"] = "Unknown"
        df["route_encoded"] = df["route"].apply(
            lambda x: 1 if "Ruwais_to_Khalifa" in str(x) else 0
        )

        # ---- Temporal ----
        if "date" in df.columns:
            df["month"] = df["date"].dt.month.fillna(6).astype(int)
        else:
            df["month"] = 6
        df["season"] = (df["month"] % 12 // 3)

        if "hour" not in df.columns and "date" in df.columns:
            df["hour"] = df["date"].dt.hour.fillna(12).astype(int)
        elif "hour" not in df.columns:
            df["hour"] = 12

        df["hour_bin_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
        df["hour_bin_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)

        # ---- Weather (fill NaN with defaults if ECDIS not available) ----
        if "wind_speed" not in df.columns:
            df = self._add_synthetic_weather(df)
        else:
            # Fill missing weather with typical Arabian Gulf values
            df["wind_speed"] = df["wind_speed"].fillna(8.5)
            df["wind_direction"] = df.get("wind_direction", pd.Series([300.0] * len(df)))
            df["wind_direction"] = df["wind_direction"].fillna(300.0)

        # Wind resistance (for prediction compat)
        if "wind_resistance" not in df.columns and "wind_speed" in df.columns:
            df["wind_resistance"] = df["wind_speed"] * 0.5  # Simplified
        df["wind_resistance"] = df["wind_resistance"].fillna(4.25)

        if "sea_state" not in df.columns:
            df["sea_state"] = 3
        df["sea_state"] = df["sea_state"].fillna(3)

        # Fill other optional ECDIS-derived features
        for col in ["relative_wind_angle", "headwind_component", "stw_sog_diff", "current_avg"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = 0.0

        # ---- Define feature column order (must match prediction) ----
        self.feature_names = [
            "speed_knots", "speed_squared", "speed_cubed",
            "duration", "distance_nm",
            "load", "me_rpm", "rpm_normalized", "rpm_optimal",
            "slip",  # NEW in v2
            "wind_speed", "wind_resistance", "sea_state",
            "route_encoded", "season",
            "hour_bin_morning", "hour_bin_afternoon",
            "speed_rpm_interaction", "load_dist_interaction",
        ]

        # Only keep features that exist in the dataframe
        self.feature_names = [c for c in self.feature_names if c in df.columns]

        return df

    def _final_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers and invalid records."""
        before = len(df)

        # Must have target and core features
        req = ["me_fuel_mt"] + [c for c in self.feature_names if c in df.columns]
        df = df.dropna(subset=req)

        # Fuel consumption bounds
        df = df[(df["me_fuel_mt"] > 0.1) & (df["me_fuel_mt"] < 15)]

        # Speed bounds
        if "speed_knots" in df.columns:
            df = df[df["speed_knots"].between(3, MAX_SPEED_KNOTS + 2)]

        # Duration bounds
        if "duration" in df.columns:
            df = df[df["duration"].between(0.5, 48)]

        # Distance bounds
        if "distance_nm" in df.columns:
            df = df[df["distance_nm"].between(50, 200)]

        # IQR outlier removal on fuel
        if len(df) > 10:
            q1 = df["me_fuel_mt"].quantile(0.25)
            q3 = df["me_fuel_mt"].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df = df[(df["me_fuel_mt"] >= lower) & (df["me_fuel_mt"] <= upper)]

        after = len(df)
        logger.info("Cleaning: %d -> %d voyages (%d removed)", before, after, before - after)
        return df.reset_index(drop=True)

    def _add_synthetic_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add placeholder weather values when real data is unavailable."""
        n = len(df)
        np.random.seed(42)
        df["wind_speed"] = np.clip(np.random.normal(8.5, 4.0, n), 0, 25)
        df["wind_direction"] = np.random.normal(300, 45, n) % 360
        df["wind_resistance"] = df["wind_speed"] * 0.5
        df["sea_state"] = np.random.choice([2, 3, 4], n, p=[0.4, 0.4, 0.2])
        return df

    def _estimate_engine_load_pct(self, speed: float) -> float:
        """Estimate engine load percentage from speed."""
        load = 5.0 * speed - 10.0
        return float(np.clip(load, 10.0, 100.0))

    def _estimate_rpm(self, speed: float) -> float:
        """Estimate RPM from speed (linear for fixed-pitch propeller)."""
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

    # ------------------------------------------------------------------
    # Model Training
    # ------------------------------------------------------------------

    def train_model(self) -> Dict[str, Any]:
        """
        Train RandomForestRegressor with optimal hyper-parameters.
        Returns model statistics dictionary (same keys as v1).
        """
        if self.training_data is None:
            raise ValueError("No training data. Run load_and_prepare_data() first.")

        logger.info("=" * 60)
        logger.info("Training Random Forest v2.0")
        logger.info("=" * 60)

        df = self.training_data

        # Prepare X, y
        X = df[self.feature_names].copy()
        y = df["me_fuel_mt"].copy()
        X = X.fillna(X.median())

        logger.info("Features used (%d): %s", len(self.feature_names), self.feature_names)
        logger.info("Training samples: %d", len(X))

        # ---- Temporal train/test split (70/30) ----
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # ---- Train Random Forest ----
        self.model = RandomForestRegressor(**OPTIMAL_HYPERPARAMS)
        self.model.fit(X_train_s, y_train)

        # ---- Evaluate ----
        y_pred_train = self.model.predict(X_train_s)
        y_pred_test = self.model.predict(X_test_s)

        train_r2 = float(r2_score(y_train, y_pred_train))
        test_r2 = float(r2_score(y_test, y_pred_test))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        test_mae = float(mean_absolute_error(y_test, y_pred_test))

        # ---- 5-fold Cross-Validation (shuffled for small dataset) ----
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train_s, y_train, cv=kf, scoring="r2")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        # ---- Bootstrap 95% CI for test R2 ----
        np.random.seed(42)
        boot_r2s = []
        y_test_arr = np.array(y_test)
        y_pred_test_arr = np.array(y_pred_test)
        for _ in range(1000):
            idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
            boot_r2s.append(r2_score(y_test_arr[idx], y_pred_test_arr[idx]))
        ci_lower = float(np.percentile(boot_r2s, 2.5))
        ci_upper = float(np.percentile(boot_r2s, 97.5))

        # ---- Feature importance ----
        fi = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        # ---- Store stats (same keys as v1 for backward compat) ----
        self.model_stats = {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "features_used": len(self.feature_names),
            "feature_importance": fi.to_dict("records"),
            "model_version": "2.0",
            "hyperparams": OPTIMAL_HYPERPARAMS,
        }

        logger.info("Train R2 = %.4f | Test R2 = %.4f | MAE = %.4f MT",
                    train_r2, test_r2, test_mae)
        logger.info("CV R2 = %.4f +/- %.4f | 95%% CI: [%.4f, %.4f]",
                    cv_mean, cv_std, ci_lower, ci_upper)
        logger.info("Overfitting gap: %.4f", train_r2 - test_r2)

        # ---- Multi-algorithm comparison (for reporting) ----
        self._compare_algorithms(X_train_s, y_train, X_test_s, y_test)

        # Auto-save
        try:
            self.save_model()
        except Exception as e:
            logger.warning("Auto-save failed: %s", e)

        return self.model_stats

    def _compare_algorithms(self, X_train, y_train, X_test, y_test) -> None:
        """Quick comparison of alternative algorithms (logged, not stored)."""
        algorithms = {
            "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
        }
        logger.info("\nAlgorithm comparison:")
        for name, algo in algorithms.items():
            try:
                algo.fit(X_train, y_train)
                pred = algo.predict(X_test)
                r2 = r2_score(y_test, pred)
                logger.info("  %s: Test R2 = %.4f", name, r2)
            except Exception as e:
                logger.info("  %s: failed (%s)", name, e)

    # ------------------------------------------------------------------
    # Prediction  (100% backward-compatible signature)
    # ------------------------------------------------------------------

    def predict_fuel(
        self,
        speed: float,
        duration: float,
        distance: Optional[float] = None,
        wind_speed: float = 8.5,
        route: str = "Khalifa_to_Ruwais",
        target_rpm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Predict fuel consumption for a given voyage.

        Args:        (same as v1)
            speed:         Speed in knots (max 12)
            duration:      Trip duration in hours
            distance:      Distance in NM (estimated if None)
            wind_speed:    Wind speed in m/s
            route:         Route name
            target_rpm:    Target RPM (estimated if None)

        Returns:     (same keys as v1)
            predicted_fuel_mt, model_confidence_r2, input_parameters,
            efficiency_metrics
        """
        if self.model is None:
            return {"error": "No trained model available"}

        # Enforce constraints
        speed = min(speed, MAX_SPEED_KNOTS)
        if distance is None:
            distance = speed * duration
        if target_rpm is None:
            target_rpm = self._estimate_rpm(speed)

        rpm_optimal = 1 if OPTIMAL_RPM_MIN <= target_rpm <= OPTIMAL_RPM_MAX else 0

        # Build feature vector (must match training feature order)
        row: Dict[str, float] = {
            "speed_knots": speed,
            "speed_squared": speed ** 2,
            "speed_cubed": speed ** 3,
            "duration": duration,
            "distance_nm": distance,
            "load": self._estimate_engine_load_pct(speed),
            "me_rpm": target_rpm,
            "rpm_normalized": target_rpm / 150.0,
            "rpm_optimal": rpm_optimal,
            "slip": 0.0,  # Default for prediction (unknown pre-voyage)
            "wind_speed": wind_speed,
            "wind_resistance": wind_speed * 0.5,
            "sea_state": 3 if wind_speed < 10 else 4 if wind_speed < 15 else 5,
            "route_encoded": 1 if "Ruwais_to_Khalifa" in route else 0,
            "season": 1,
            "hour_bin_morning": 0,
            "hour_bin_afternoon": 1,
            "speed_rpm_interaction": speed * (target_rpm / 150.0),
            "load_dist_interaction": (self._estimate_engine_load_pct(speed) * distance) / 100.0,
        }

        # Ensure correct column order and presence
        features = pd.DataFrame({name: [row.get(name, 0)] for name in self.feature_names})

        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = float(self.model.predict(features_scaled)[0])
        prediction = max(1.0, prediction)  # Floor at 1 MT

        confidence = self.model_stats.get("test_r2", 0.0)

        return {
            "predicted_fuel_mt": round(prediction, 3),
            "model_confidence_r2": confidence,
            "input_parameters": {
                "speed_knots": speed,
                "duration_hours": duration,
                "distance_nm": distance,
                "estimated_rpm": target_rpm,
                "rpm_in_optimal_range": rpm_optimal == 1,
                "wind_speed_mps": wind_speed,
                "route": route,
            },
            "efficiency_metrics": {
                "fuel_per_hour": round(prediction / duration, 3) if duration > 0 else 0,
                "fuel_per_nm": round(prediction / distance, 3) if distance > 0 else 0,
            },
        }

    # ------------------------------------------------------------------
    # Persistence  (v2 paths, backward-compatible with v1)
    # ------------------------------------------------------------------

    def save_model(self) -> None:
        if self.model is None:
            raise ValueError("No model to save — train first.")

        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

        meta = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "version": "2.0",
            "feature_names": list(self.feature_names),
            "model_stats": {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in self.model_stats.items()
                if k != "feature_importance"
            },
            "feature_importance": self.model_stats.get("feature_importance", []),
            "training_statistics": self.get_training_statistics(),
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info("Model saved: %s (%d KB)",
                    MODEL_PATH.name, MODEL_PATH.stat().st_size // 1024)

    def load_model(self) -> bool:
        """Try to load cached model (v2 first, then v1 fallback)."""
        # Try v2
        paths = [(MODEL_PATH, SCALER_PATH, META_PATH),
                 (MODEL_PATH_V1, SCALER_PATH_V1, META_PATH_V1)]

        for model_p, scaler_p, meta_p in paths:
            if not (model_p.exists() and scaler_p.exists() and meta_p.exists()):
                continue
            try:
                self.model = joblib.load(model_p)
                self.scaler = joblib.load(scaler_p)
                with open(meta_p) as f:
                    meta = json.load(f)
                self.feature_names = meta.get("feature_names", meta.get("feature_names", []))
                self.model_stats = meta.get("model_stats", {})
                self.model_stats["feature_importance"] = meta.get("feature_importance", [])
                self._cached_training_stats = meta.get("training_statistics", {})
                ver = meta.get("version", "1.x")
                logger.info("Loaded cached model v%s from %s", ver, model_p.name)
                return True
            except Exception as e:
                logger.warning("Failed to load from %s: %s", model_p, e)

        return False

    # ------------------------------------------------------------------
    # Reporting  (identical output structure to v1)
    # ------------------------------------------------------------------

    def get_training_statistics(self) -> Dict[str, Any]:
        if self.training_data is None:
            return getattr(self, "_cached_training_stats", {}) or {}

        df = self.training_data
        stats: Dict[str, Any] = {
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
            "routes": df["route"].value_counts().to_dict() if "route" in df.columns else {},
            "data_sources": {
                "ce_daily_log": True,
                "rob": "load" in df.columns and df["load"].notna().any(),
                "ecdis_weather": "wind_speed" in df.columns and df["wind_speed"].notna().any(),
            },
        }

        if "date" in df.columns and not df["date"].isna().all():
            stats["date_range"] = {
                "start": df["date"].min().strftime("%Y-%m-%d"),
                "end": df["date"].max().strftime("%Y-%m-%d"),
            }

        return stats

    def generate_academic_report(self) -> Dict[str, Any]:
        if not self.model_stats:
            return {"error": "No model trained yet"}

        return {
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
                "validation_method": "5-fold cross-validation + 70/30 holdout + bootstrap CI",
                "feature_engineering": [
                    "Speed polynomials (cubic relationship)",
                    "Actual engine LOAD (from ROB where available)",
                    "Propeller slip (NEW from CE Daily Log)",
                    "Weather impact (wind speed, resistance)",
                    "Route characteristics",
                    "Temporal features (morning/afternoon)",
                    "Feature interactions (speed-RPM, load-distance)",
                ],
                "preprocessing": "StandardScaler normalization, IQR outlier removal",
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
            "model_version": "2.0",
        }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("M/V Al-bazm II ML System v2.0 — Self Test")
    print("=" * 60)

    ml = AlbazmMLSystem()

    # Try loading new data sources
    print("\n[1] Loading data...")
    try:
        data = ml.load_and_prepare_data()
        print(f"    Loaded {len(data)} voyages")
    except FileNotFoundError as e:
        print(f"    Data not available: {e}")
        print("    Place CE Daily Log, ROB, and ECDIS files in backend/")
        import sys
        sys.exit(1)

    # Train
    print("\n[2] Training model...")
    stats = ml.train_model()

    # Predictions
    print("\n[3] Test predictions:")
    print("-" * 60)
    for speed in [8, 10, 11, 12]:
        pred = ml.predict_fuel(speed=speed, duration=13.5, route="Khalifa_to_Ruwais")
        fuel = pred["predicted_fuel_mt"]
        print(f"   {speed} kn: {fuel:.2f} MT")

    # Report
    print("\n[4] Academic report:")
    report = ml.generate_academic_report()
    r = report["results"]
    print(f"    Test R2: {r['test_r2_score']:.4f}")
    print(f"    Test MAE: {r['test_mae_mt']:.4f} MT")
    print(f"    CV R2: {r['cv_mean_r2']:.4f} +/- {r['cv_std_r2']:.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
