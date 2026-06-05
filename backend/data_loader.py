#!/usr/bin/env python3
"""
Data Loader Module for M/V Al-bazm II
Integrates three data sources:
  1. CE Daily Log (Chief Engineer's operational records)
  2. ROB (Official Record of Voyages)
  3. ECDIS Weather (Electronic Chart Display & Information System)

Output: Clean, merged dataset ready for ML feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# File paths — xlsx takes priority (newer, richer data)
CE_DAILY_LOG_XLSX = BASE_DIR / "CE Daily Log 2024 to 2025.xlsx"
CE_DAILY_LOG_CSV = BASE_DIR / "CE_Daily_Log_2024_2025.csv"
ROB_XLSX = BASE_DIR / " - Official ROB 2024 - 2025.xlsx"
ROB_CSV = BASE_DIR / "OfficialROB2024-NOV2025.csv"
ECDIS_XLSX = BASE_DIR / "ECDIS-weather data file.xlsx"
ECDIS_CSV = BASE_DIR / "ecdis_weather_data.csv"


def _find_file(name: str, csv_p: Path, xlsx_p: Path) -> Optional[Path]:
    """Return existing file (xlsx preferred, then csv)."""
    if xlsx_p.exists():
        logger.info("  Using %s", xlsx_p.name)
        return xlsx_p
    if csv_p.exists():
        logger.info("  Using %s", csv_p.name)
        return csv_p
    logger.warning("  %s not found (checked %s, %s)", name, xlsx_p.name, csv_p.name)
    return None


# ===========================================================================
# 1. CE DAILY LOG
# ===========================================================================

def load_ce_daily_log(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load CE Daily Log. Returns cleaned EOSP voyage records."""
    fp = file_path or _find_file("CE Daily Log", CE_DAILY_LOG_CSV, CE_DAILY_LOG_XLSX)
    if fp is None:
        return pd.DataFrame()

    logger.info("Loading CE Daily Log (%s)", fp.name)

    if fp.suffix == ".csv":
        df = pd.read_csv(fp)
    else:
        df = pd.read_excel(fp, engine="calamine")

    logger.info("  Raw records: %d", len(df))
    if len(df) == 0:
        return pd.DataFrame()

    # --- Exact column rename map (verified against actual file) ---
    rename_map = {}
    for c in df.columns:
        c_upper = str(c).upper().strip()
        if c_upper == "DATE":
            rename_map[c] = "date"
        elif c_upper == "TIME":
            rename_map[c] = "time"
        elif c_upper == "EVENT":
            rename_map[c] = "event"
        elif c_upper == "DURATION":
            rename_map[c] = "duration"
        elif c_upper in ("DIST BY OBSERV.", "DIST_BY_OBSERV") or "DIST" in c_upper and "OBSERV" in c_upper:
            rename_map[c] = "distance_nm"
        elif c_upper in ("M.E. MEAN RPM", "MEAN RPM", "ME_RPM") or "MEAN RPM" in c_upper:
            rename_map[c] = "me_rpm"
        elif c_upper == "SLIP":
            rename_map[c] = "slip"
        elif "M/E CONSUM" in c_upper or "MAIN ENGINE FUEL" in c_upper:
            rename_map[c] = "me_fuel_mt"
        elif "G/E CONSUM" in c_upper or "GEN FUEL" in c_upper:
            rename_map[c] = "ge_fuel_mt"
        elif "TOTAL FUEL" in c_upper or "F.O. TOTAL" in c_upper or "FO TOTAL" in c_upper:
            rename_map[c] = "total_fuel_mt"

    df = df.rename(columns=rename_map)
    logger.info("  Mapped columns: %s", [rename_map.get(c, c) for c in df.columns[:10]])

    # Verify critical columns exist
    for critical in ["date", "me_fuel_mt", "me_rpm"]:
        if critical not in df.columns:
            logger.error("  CRITICAL COLUMN MISSING: '%s' — available: %s", critical, list(df.columns))
            return pd.DataFrame()

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Parse numeric columns
    for col in ["duration", "distance_nm", "me_rpm", "slip", "me_fuel_mt", "ge_fuel_mt", "total_fuel_mt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to EOSP events
    if "event" in df.columns:
        before = len(df)
        df = df[df["event"].astype(str).str.strip().str.upper().isin(["EOSP", "EO SP"])].copy()
        logger.info("  EOSP filter: %d -> %d", before, len(df))
    else:
        logger.warning("  No 'event' column — skipping EOSP filter")

    # Drop rows missing critical data
    req = [c for c in ["me_fuel_mt", "me_rpm"] if c in df.columns]
    if req:
        before = len(df)
        df = df.dropna(subset=req)
        logger.info("  Dropna critical: %d (dropped %d)", len(df), before - len(df))

    # Reasonable value filters
    if "me_fuel_mt" in df.columns:
        df = df[(df["me_fuel_mt"] > 0) & (df["me_fuel_mt"] < 15)]
    if "me_rpm" in df.columns:
        df = df[(df["me_rpm"] > 50) & (df["me_rpm"] < 200)]
    if "distance_nm" in df.columns:
        df = df[(df["distance_nm"] > 50) & (df["distance_nm"] < 200)]
    if "slip" in df.columns:
        df = df[df["slip"] >= 0]

    logger.info("  Clean CE voyages: %d", len(df))
    return df.reset_index(drop=True)


# ===========================================================================
# 2. ROB (OFFICIAL RECORD OF VOYAGES)
# ===========================================================================

def load_rob(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load ROB. Returns EOSP records with actual LOAD."""
    fp = file_path or _find_file("ROB", ROB_CSV, ROB_XLSX)
    if fp is None:
        return pd.DataFrame()

    logger.info("Loading ROB (%s)", fp.name)

    if fp.suffix == ".csv":
        df = pd.read_csv(fp)
        # Try to identify columns for CSV format
        if len(df.columns) >= 13:
            df.columns = ([f"col_{i}" for i in range(len(df.columns))])
            df = df.rename(columns={
                "col_3": "event", "col_4": "date", "col_6": "time",
                "col_7": "total_trip_time", "col_9": "place",
                "col_10": "slip", "col_11": "total_distance",
                "col_12": "avg_speed", "col_13": "foc",
                "col_14": "load", "col_15": "rpm",
            })
    else:
        # Excel: 2 header rows + 2 empty rows, data at row 5
        df = pd.read_excel(fp, engine="calamine", header=None, skiprows=4)
        n_cols = len(df.columns)
        if n_cols >= 16:
            df.columns = [f"col_{i}" for i in range(n_cols)]
            df = df.rename(columns={
                "col_3": "event", "col_4": "date", "col_6": "time",
                "col_7": "total_trip_time", "col_9": "place",
                "col_10": "slip", "col_11": "total_distance",
                "col_12": "avg_speed", "col_13": "foc",
                "col_14": "load", "col_15": "rpm",
            })
        else:
            logger.warning("  Unexpected columns: %d", n_cols)
            return pd.DataFrame()

    logger.info("  Raw records: %d", len(df))

    # Filter to FAOP/EOSP
    if "event" in df.columns:
        df["event"] = df["event"].astype(str).str.strip().str.upper()
        df = df[df["event"].isin(["FAOP", "EOSP"])].copy()

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Parse numeric
    for col in ["total_distance", "avg_speed", "foc", "load", "rpm", "slip", "total_trip_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix corrupted avg_speed (>20 kn)
    if "total_trip_time" in df.columns and "total_distance" in df.columns:
        mask = (df["avg_speed"] > 20) | (df["avg_speed"].isna())
        valid = mask & (df["total_trip_time"] > 0)
        df.loc[valid, "avg_speed"] = df.loc[valid, "total_distance"] / df.loc[valid, "total_trip_time"]

    # Keep EOSP only
    df_eosp = df[df["event"] == "EOSP"].copy() if "event" in df.columns else df.copy()

    # Reasonable filters
    if "avg_speed" in df_eosp.columns:
        df_eosp = df_eosp[df_eosp["avg_speed"].between(4, 15)]
    if "load" in df_eosp.columns:
        df_eosp = df_eosp[df_eosp["load"] <= 1.0]

    logger.info("  Clean EOSP: %d", len(df_eosp))

    df_load = df_eosp.dropna(subset=["load"]).copy() if "load" in df_eosp.columns else df_eosp.copy()
    logger.info("  With actual LOAD: %d", len(df_load))
    return df_load.reset_index(drop=True)


# ===========================================================================
# 3. ECDIS WEATHER
# ===========================================================================

def load_ecdis_weather(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load ECDIS weather/navigation data."""
    fp = file_path or _find_file("ECDIS", ECDIS_CSV, ECDIS_XLSX)
    if fp is None:
        return pd.DataFrame()

    logger.info("Loading ECDIS (%s)", fp.name)

    if fp.suffix == ".csv":
        df = pd.read_csv(fp)
    else:
        df = pd.read_excel(fp, engine="calamine", header=None, skiprows=2)
        n_cols = len(df.columns)
        base_cols = ["no", "date", "time", "lat_deg", "lat_min", "lat_ns",
                     "lon_deg", "lon_min", "lon_ew", "cog", "sog", "hdg",
                     "stw", "set_deg", "drift", "wind_dir", "wind_speed", "depth"]
        if n_cols >= len(base_cols):
            df.columns = base_cols + [f"extra_{i}" for i in range(n_cols - len(base_cols))]
        else:
            df.columns = [f"col_{i}" for i in range(n_cols)]

    logger.info("  Raw records: %d", len(df))

    if "no" in df.columns:
        df = df.dropna(subset=["no"])

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce"
    )

    for col in ["sog", "stw", "cog", "hdg", "wind_dir", "wind_speed", "drift", "depth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["wind_dir"] > 360, "wind_dir"] = np.nan
    df.loc[df["wind_speed"] > 50, "wind_speed"] = np.nan
    df.loc[df["sog"] > 30, "sog"] = np.nan

    logger.info("  Valid wind: %d", df["wind_speed"].notna().sum())
    return df.reset_index(drop=True)


# ===========================================================================
# 4. MERGE
# ===========================================================================

def merge_all_sources(
    ce_df: pd.DataFrame,
    rob_df: pd.DataFrame,
    ecdis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge CE (primary) + ROB (LOAD) + ECDIS (weather) by date."""
    logger.info("=== MERGING ===")

    if ce_df.empty:
        logger.error("CE Daily Log empty")
        return pd.DataFrame()

    df = ce_df.copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # Merge ROB (actual LOAD)
    if not rob_df.empty and "load" in rob_df.columns:
        rob_df = rob_df.copy()
        rob_df["date_str"] = rob_df["date"].dt.strftime("%Y-%m-%d")
        rob_by_date = rob_df.groupby("date_str").agg({
            "load": "mean", "rpm": "mean", "slip": "mean",
        }).reset_index()
        df = df.merge(rob_by_date, on="date_str", how="left", suffixes=("", "_rob"))
        n = df["load"].notna().sum()
        logger.info("  ROB: %d/%d voyages have actual LOAD", n, len(df))
    else:
        df["load"] = np.nan
        logger.info("  ROB: skipped")

    # Merge ECDIS (weather)
    if not ecdis_df.empty:
        ecdis_df = ecdis_df.copy()
        ecdis_df["date_str"] = ecdis_df["datetime"].dt.strftime("%Y-%m-%d")
        ecdis_daily = ecdis_df.groupby("date_str").agg({
            "wind_speed": "mean", "wind_dir": "mean", "stw": "mean",
            "sog": "mean", "drift": "mean", "hdg": "mean", "depth": "mean",
        }).reset_index()
        df = df.merge(ecdis_daily, on="date_str", how="left", suffixes=("", "_ecdis"))
        n = df["wind_speed"].notna().sum() if "wind_speed" in df.columns else 0
        logger.info("  ECDIS: %d/%d voyages have weather", n, len(df))
    else:
        logger.info("  ECDIS: skipped")

    # Derived weather features
    if "wind_dir" in df.columns and "hdg" in df.columns:
        raw = np.abs(df["wind_dir"] - df["hdg"])
        df["relative_wind_angle"] = raw.apply(lambda x: min(x, 360 - x) if pd.notna(x) else np.nan)
        df["headwind_component"] = df["wind_speed"] * np.cos(np.radians(df["relative_wind_angle"]))

    if "stw" in df.columns and "sog" in df.columns:
        df["stw_sog_diff"] = df["stw"] - df["sog"]

    # Route classification
    if "place" in df.columns:
        df["route"] = df["place"].apply(_classify_route)
    else:
        df["route"] = "Unknown"

    # Speed
    if "speed_knots" not in df.columns and "distance_nm" in df.columns and "duration" in df.columns:
        mask = df["duration"] > 0
        df.loc[mask, "speed_knots"] = df.loc[mask, "distance_nm"] / df.loc[mask, "duration"]
    if "speed_knots" in df.columns:
        df = df[df["speed_knots"].between(3, 15)]

    logger.info("  Final: %d voyages", len(df))
    return df.reset_index(drop=True)


def _classify_route(place_text) -> str:
    if pd.isna(place_text):
        return "Unknown"
    place = str(place_text).upper()
    if "KHALIFA" in place or "KHL" in place or "KP" in place:
        return "Ruwais_to_Khalifa"
    elif "RUWAIS" in place or "RWS" in place:
        return "Khalifa_to_Ruwais"
    return "Unknown"


def load_all_data() -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("LOADING ALL DATA SOURCES")
    logger.info("=" * 60)
    ce = load_ce_daily_log()
    rob = load_rob()
    ecdis = load_ecdis_weather()
    return merge_all_sources(ce, rob, ecdis)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_all_data()
    print(f"\nFinal: {df.shape}")
    if not df.empty:
        print(f"Cols: {list(df.columns)}")
        preview = [c for c in ["date_str", "me_fuel_mt", "me_rpm", "distance_nm",
                                "load", "slip", "wind_speed"] if c in df.columns]
        print(df[preview].head().to_string())

def _rename_exact(df, old_to_new):
    """Rename columns using exact match, ignoring missing ones."""
    rename_map = {}
    for old, new in old_to_new.items():
        if old in df.columns and new not in df.columns:
            rename_map[old] = new
    return df.rename(columns=rename_map)


# ===========================================================================
# 1. CE DAILY LOG
# ===========================================================================

def load_ce_daily_log(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load CE Daily Log. Returns cleaned EOSP voyage records."""
    fp = file_path or _find_file(CE_DAILY_LOG_PATH, CE_DAILY_LOG_XLSX)
    if fp is None:
        logger.warning("CE Daily Log not found")
        return pd.DataFrame()

    logger.info("Loading CE Daily Log from %s", fp.name)

    if fp.suffix == ".csv":
        df = pd.read_csv(fp)
    else:
        df = pd.read_excel(fp, engine="calamine")

    logger.info("  Raw records: %d", len(df))
    if len(df) == 0:
        return pd.DataFrame()

    # --- Column mapping (exact names from the actual file) ---
    exact_map = {
        "Date": "date",
        "Time": "time",
        "Event": "event",
        "Duration": "duration",
        "Dist by observ.": "distance_nm",
        "M.E. mean RPM": "me_rpm",
        "Slip": "slip",
        "M/E consum.  MT": "me_fuel_mt",
        "G/E consum.   MT": "ge_fuel_mt",
        "F.O. Total consum. Per day   MT": "total_fuel_mt",
        "avg_speed_kn": "speed_knots",
    }
    df = _rename_exact(df, exact_map)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Parse numeric columns
    numeric_cols = ["duration", "distance_nm", "me_rpm", "slip",
                    "me_fuel_mt", "ge_fuel_mt", "total_fuel_mt", "speed_knots"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to EOSP events (complete voyages)
    if "event" in df.columns:
        df = df[df["event"].astype(str).str.strip().str.upper().isin(["EOSP", "EO SP"])].copy()
        logger.info("  EOSP records: %d", len(df))
    else:
        logger.warning("  No 'event' column found - using all records")

    # Clean: remove rows missing critical fields
    req = [c for c in ["me_fuel_mt", "me_rpm", "distance_nm"] if c in df.columns]
    if req:
        before = len(df)
        df = df.dropna(subset=req)
        logger.info("  After dropping missing fields: %d (dropped %d)", len(df), before - len(df))

    # Reasonable value filters
    if "me_fuel_mt" in df.columns:
        df = df[(df["me_fuel_mt"] > 0) & (df["me_fuel_mt"] < 15)]
    if "me_rpm" in df.columns:
        df = df[(df["me_rpm"] > 50) & (df["me_rpm"] < 200)]
    if "distance_nm" in df.columns:
        df = df[(df["distance_nm"] > 50) & (df["distance_nm"] < 200)]
    if "slip" in df.columns:
        df = df[df["slip"] >= 0]

    logger.info("  Clean CE voyages: %d", len(df))
    return df.reset_index(drop=True)


# ===========================================================================
# 2. ROB (OFFICIAL RECORD OF VOYAGES)
# ===========================================================================

def load_rob(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load ROB file. Returns EOSP records with actual LOAD."""
    fp = file_path or _find_file(ROB_PATH, ROB_XLSX)
    if fp is None:
        logger.warning("ROB file not found")
        return pd.DataFrame()

    logger.info("Loading ROB from %s", fp.name)

    if fp.suffix == ".csv":
        df = pd.read_csv(fp)
    else:
        # ROB Excel: 2 header rows + 2 empty rows, data starts at row 5
        df = pd.read_excel(fp, engine="calamine", header=None, skiprows=4)
        n_cols = len(df.columns)
        # Map by column index based on observed file structure
        if n_cols >= 16:
            df.columns = [f"col_{i}" for i in range(n_cols)]
            # Extract key columns by position
            df = df.rename(columns={
                "col_3": "event",
                "col_4": "date",
                "col_6": "time",
                "col_7": "total_trip_time",
                "col_9": "place",
                "col_10": "slip",
                "col_11": "total_distance",
                "col_12": "avg_speed",
                "col_13": "foc",
                "col_14": "load",
                "col_15": "rpm",
            })
        else:
            logger.warning("  Unexpected column count: %d", n_cols)
            return pd.DataFrame()

    logger.info("  Raw records: %d", len(df))

    # Filter to FAOP/EOSP
    if "event" in df.columns:
        df["event"] = df["event"].astype(str).str.strip().str.upper()
        df = df[df["event"].isin(["FAOP", "EOSP"])].copy()

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Parse numeric columns
    for col in ["total_distance", "avg_speed", "foc", "load", "rpm", "slip", "total_trip_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Speed fix: Avg_speed corrupted for ~52% of records (>20 kn)
    if "total_trip_time" in df.columns and "total_distance" in df.columns:
        mask = (df["avg_speed"] > 20) | (df["avg_speed"].isna())
        valid_time = df.loc[mask, "total_trip_time"] > 0
        df.loc[mask & valid_time, "avg_speed"] = (
            df.loc[mask & valid_time, "total_distance"] /
            df.loc[mask & valid_time, "total_trip_time"]
        )

    # Keep EOSP only (complete voyages)
    df_eosp = df[df["event"] == "EOSP"].copy()

    # Reasonable filters
    df_eosp = df_eosp[df_eosp["avg_speed"].between(4, 15)]
    df_eosp = df_eosp[df_eosp["load"] <= 1.0]

    logger.info("  Clean EOSP: %d", len(df_eosp))

    df_load = df_eosp.dropna(subset=["load"]).copy()
    logger.info("  With actual LOAD: %d", len(df_load))

    return df_load.reset_index(drop=True)


# ===========================================================================
# 3. ECDIS WEATHER
# ===========================================================================

def load_ecdis_weather(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load ECDIS weather/navigation data."""
    fp = file_path or _find_file(ECDIS_PATH, ECDIS_XLSX)
    if fp is None:
        logger.warning("ECDIS file not found")
        return pd.DataFrame()

    logger.info("Loading ECDIS from %s", fp.name)

    if fp.suffix == ".csv":
        df = pd.read_csv(fp)
    else:
        df = pd.read_excel(fp, engine="calamine", header=None, skiprows=2)
        n_cols = len(df.columns)
        base_cols = ["no", "date", "time", "lat_deg", "lat_min", "lat_ns",
                     "lon_deg", "lon_min", "lon_ew", "cog", "sog", "hdg",
                     "stw", "set_deg", "drift", "wind_dir", "wind_speed", "depth"]
        if n_cols >= len(base_cols):
            df.columns = (base_cols + [f"extra_{i}" for i in range(n_cols - len(base_cols))])
        else:
            df.columns = [f"col_{i}" for i in range(n_cols)]

    logger.info("  Raw records: %d", len(df))

    # Remove empty rows
    if "no" in df.columns:
        df = df.dropna(subset=["no"])

    # Parse datetime
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )

    # Parse numeric
    for col in ["sog", "stw", "cog", "hdg", "wind_dir", "wind_speed", "drift", "depth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean impossible values
    df.loc[df["wind_dir"] > 360, "wind_dir"] = np.nan
    df.loc[df["wind_speed"] > 50, "wind_speed"] = np.nan
    df.loc[df["sog"] > 30, "sog"] = np.nan

    logger.info("  Valid wind: %d records", df["wind_speed"].notna().sum())
    return df.reset_index(drop=True)


# ===========================================================================
# 4. MERGE
# ===========================================================================

def merge_all_sources(
    ce_df: pd.DataFrame,
    rob_df: pd.DataFrame,
    ecdis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge CE (primary) + ROB (LOAD) + ECDIS (weather) by date."""
    logger.info("\n=== MERGING DATA SOURCES ===")

    if ce_df.empty:
        logger.error("CE Daily Log is empty")
        return pd.DataFrame()

    df = ce_df.copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # Merge ROB (actual LOAD)
    if not rob_df.empty:
        rob_df = rob_df.copy()
        rob_df["date_str"] = rob_df["date"].dt.strftime("%Y-%m-%d")
        rob_by_date = rob_df.groupby("date_str").agg({
            "load": "mean", "rpm": "mean", "slip": "mean",
        }).reset_index()
        df = df.merge(rob_by_date, on="date_str", how="left", suffixes=("", "_rob"))
        n_actual = df["load"].notna().sum()
        logger.info("  ROB merge: %d/%d voyages have actual LOAD", n_actual, len(df))
    else:
        df["load"] = np.nan
        logger.info("  ROB: skipped")

    # Merge ECDIS (weather)
    if not ecdis_df.empty:
        ecdis_df = ecdis_df.copy()
        ecdis_df["date_str"] = ecdis_df["datetime"].dt.strftime("%Y-%m-%d")
        ecdis_daily = ecdis_df.groupby("date_str").agg({
            "wind_speed": "mean", "wind_dir": "mean", "stw": "mean",
            "sog": "mean", "drift": "mean", "hdg": "mean", "depth": "mean",
        }).reset_index()
        df = df.merge(ecdis_daily, on="date_str", how="left", suffixes=("", "_ecdis"))
        n_wx = df["wind_speed"].notna().sum() if "wind_speed" in df.columns else 0
        logger.info("  ECDIS merge: %d/%d voyages have weather", n_wx, len(df))
    else:
        logger.info("  ECDIS: skipped")

    # Derived weather features
    if "wind_dir" in df.columns and "hdg" in df.columns:
        raw_angle = np.abs(df["wind_dir"] - df["hdg"])
        df["relative_wind_angle"] = raw_angle.apply(
            lambda x: min(x, 360 - x) if pd.notna(x) else np.nan
        )
        df["headwind_component"] = df["wind_speed"] * np.cos(
            np.radians(df["relative_wind_angle"])
        )

    if "stw" in df.columns and "sog" in df.columns:
        df["stw_sog_diff"] = df["stw"] - df["sog"]

    # Route classification
    if "place" in df.columns:
        df["route"] = df["place"].apply(_classify_route)
    else:
        df["route"] = "Unknown"

    # Speed computation
    if "speed_knots" not in df.columns and "distance_nm" in df.columns and "duration" in df.columns:
        mask = df["duration"] > 0
        df.loc[mask, "speed_knots"] = df.loc[mask, "distance_nm"] / df.loc[mask, "duration"]
    if "speed_knots" in df.columns:
        df = df[df["speed_knots"].between(3, 15)]

    logger.info("  Final merged: %d voyages", len(df))
    return df.reset_index(drop=True)


def _classify_route(place_text) -> str:
    if pd.isna(place_text):
        return "Unknown"
    place = str(place_text).upper()
    if "KHALIFA" in place or "KHL" in place or "KP" in place:
        return "Ruwais_to_Khalifa"
    elif "RUWAIS" in place or "RWS" in place:
        return "Khalifa_to_Ruwais"
    return "Unknown"


def load_all_data() -> pd.DataFrame:
    """Load and merge all available data sources."""
    logger.info("=" * 60)
    logger.info("LOADING ALL DATA SOURCES")
    logger.info("=" * 60)
    ce = load_ce_daily_log()
    rob = load_rob()
    ecdis = load_ecdis_weather()
    return merge_all_sources(ce, rob, ecdis)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_all_data()
    print(f"\nFinal: {df.shape}")
    if not df.empty:
        print(f"Cols: {list(df.columns)}")
        preview = [c for c in ["date_str", "me_fuel_mt", "me_rpm", "distance_nm",
                                "load", "slip", "wind_speed"] if c in df.columns]
        print(df[preview].head().to_string())
