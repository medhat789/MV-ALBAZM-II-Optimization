#!/usr/bin/env python3
"""
M/V Atlas — Voyage Data Pipeline (v3)
=============================================
Validated data loading + feature engineering pipeline matching the corrected
thesis/manuscript methodology. Replaces the old date-merge approach in
data_loader.py (which produced ~375 loosely-matched rows and never used
real propeller slip) with precise FAOP/EOSP voyage-window pairing.

Bug fixes vs. the original deployed pipeline (see train_model_bugfixed.py):
  1. Open-Meteo wind: km/h -> m/s (divide by 3.6)
  2. wind_dir: actual compass direction for all voyages (not a boolean flag)
  3. distance_nm: excluded from Stage 1 features (used only in Stage 2 scaling)
  4. trip_time_hrs: computed from FAOP/EOSP timestamps (ROB column was corrupted)

Produces 155 quality-assured voyages with 14 Stage 1 features (11 base + 3
physically-motivated interaction terms), matching the published manuscript.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "voyage_data"

# Full 14-feature set (matches manuscript Table 1 / Section 3.3)
STAGE1_FEATURES_FULL = [
    'load_pct', 'rpm', 'slip_pct', 'trip_time_hrs',
    'voyage_sequence', 'days_from_start',
    'slip_load_interaction', 'rpm_load_interaction', 'rpm_slip_interaction',
    'direction_kp_to_rws',
    'wind_speed', 'wind_dir', 'max_wind', 'avg_speed',
]

# Deployed 11-feature set — no interaction terms (matches manuscript Section 4.5)
STAGE1_FEATURES_DEPLOYED = [f for f in STAGE1_FEATURES_FULL if 'interaction' not in f]

PROPELLER_PITCH_M = 3.276  # nominal fixed propeller pitch, from vessel particulars


def _find(*candidates: str) -> Optional[Path]:
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            return p
        p2 = BASE_DIR / name
        if p2.exists():
            return p2
    return None


def _read_table(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    return pd.read_excel(path, engine="calamine", **kwargs)


def load_voyage_dataset() -> pd.DataFrame:
    """Load and feature-engineer the full validated voyage dataset.

    Returns a DataFrame of 155 voyages with all 14 Stage 1 features plus
    the target columns (me_foc_mt, foc_per_nm, rob_distance_nm) and
    reporting-friendly aliases (me_fuel_mt, speed_knots, duration, route, date).
    """
    logger.info("Loading voyage dataset (v3 validated pipeline)...")

    cedl_path = _find("CE Daily Log 2024 to 2025.csv", "CE Daily Log 2024 to 2025.xlsx")
    rob_path = _find(" - Official ROB 2024 - 2025.csv", " - Official ROB 2024 - 2025.xlsx")
    ecdis_path = _find("ECDIS-weather data file.csv", "ECDIS-weather data file.xlsx")
    om_path = _find("openmeteo_weather.csv")

    if not all([cedl_path, rob_path, ecdis_path, om_path]):
        missing = [n for n, p in [("CE Daily Log", cedl_path), ("ROB", rob_path),
                                   ("ECDIS", ecdis_path), ("Open-Meteo", om_path)] if p is None]
        raise FileNotFoundError(f"Missing required data file(s): {missing}")

    # ---- CE Daily Log ----
    cedl = _read_table(cedl_path)

    def decimal_time_to_str(t):
        if pd.isna(t):
            return None
        try:
            t = float(t)
            hours = int(t)
            mins = int(round((t - hours) * 100))
            if mins >= 60:
                hours += 1
                mins -= 60
            return f"{hours:02d}:{mins:02d}:00"
        except Exception:
            return None

    cedl_clean = cedl[cedl['Date'].astype(str) != '12/31/1899'].copy()
    cedl_clean['time_str'] = cedl_clean['Time'].apply(decimal_time_to_str)
    cedl_clean['datetime'] = pd.to_datetime(
        cedl_clean['Date'].astype(str) + ' ' + cedl_clean['time_str'].astype(str), errors='coerce')
    cedl_ts = cedl_clean.set_index('datetime').sort_index()

    # ---- ROB (voyage pairing via FAOP/EOSP) ----
    rob = _read_table(rob_path, header=1)
    rob = rob.dropna(subset=['Event']).copy()
    faop_all = rob[rob['Event'].astype(str).str.contains('FAOP', na=False)].copy()
    eosp_all = rob[rob['Event'].astype(str).str.contains('EOSP', na=False)].copy()
    for df_ in [faop_all, eosp_all]:
        df_['datetime'] = pd.to_datetime(df_['Date'].astype(str) + ' ' + df_['Time'].astype(str), errors='coerce')

    voyage_data = []
    for i in range(min(len(faop_all), len(eosp_all))):
        faop_date = faop_all.iloc[i]['datetime']
        eosp_date = eosp_all.iloc[i]['datetime']
        if pd.isna(faop_date) or pd.isna(eosp_date):
            continue
        mask = (cedl_ts.index >= faop_date) & (cedl_ts.index <= eosp_date)
        me_foc = pd.to_numeric(cedl_ts.loc[mask, 'M/E consum.  MT'], errors='coerce').sum()
        rob_distance = pd.to_numeric(eosp_all.iloc[i]['Total Distance'], errors='coerce')
        rob_slip = pd.to_numeric(eosp_all.iloc[i]['Slip'], errors='coerce')
        rob_load = str(eosp_all.iloc[i]['LOAD ']).replace('%', '').replace('\xa0', '').strip()
        rob_load = pd.to_numeric(rob_load, errors='coerce') if rob_load else np.nan
        rob_rpm = pd.to_numeric(eosp_all.iloc[i]['RPM'], errors='coerce')
        place = str(eosp_all.iloc[i]['Place']).upper()
        direction = 'KP_to_RWS' if 'RWS' in place else 'RWS_to_KP' if ('KP' in place or 'KHL' in place) else 'unknown'
        voyage_data.append({
            'voyage_num': i + 1, 'faop_time': faop_date, 'eosp_time': eosp_date,
            'me_foc_mt': me_foc, 'rob_distance_nm': rob_distance, 'rob_slip_pct': rob_slip,
            'rob_load_pct': rob_load, 'rob_rpm': rob_rpm, 'direction': direction,
        })
    vdf = pd.DataFrame(voyage_data)
    vdf = vdf[vdf['me_foc_mt'] > 0].copy()
    vdf['foc_per_nm'] = vdf['me_foc_mt'] / vdf['rob_distance_nm']

    # ---- ECDIS weather ----
    ecl = _read_table(ecdis_path).iloc[2:].reset_index(drop=True).copy()
    cols = {'Unnamed: 0': 'No', 'Unnamed: 1': 'Date', 'Unnamed: 2': 'Time',
            'Unnamed: 9': 'COG', 'Unnamed: 10': 'SOG', 'Unnamed: 11': 'HDG',
            'Unnamed: 12': 'STW', 'Unnamed: 15': 'Wind_Dir', 'Unnamed: 16': 'Wind_Speed'}
    ecl = ecl.rename(columns=cols)
    ecl['datetime'] = pd.to_datetime(ecl['Date'].astype(str) + ' ' + ecl['Time'].astype(str), errors='coerce')
    ecl['Wind_Speed'] = pd.to_numeric(ecl['Wind_Speed'], errors='coerce')
    ecl.loc[ecl['Wind_Speed'] > 100, 'Wind_Speed'] = np.nan
    ecl['Wind_Dir'] = pd.to_numeric(ecl['Wind_Dir'], errors='coerce')
    ecl = ecl[ecl['datetime'].notna()].copy()
    ecdis_start, ecdis_end = ecl['datetime'].min(), ecl['datetime'].max()

    ecdis_weather = []
    for _, row in vdf.iterrows():
        if row['faop_time'] < ecdis_start or row['eosp_time'] > ecdis_end:
            continue
        mask = (ecl['datetime'] >= row['faop_time']) & (ecl['datetime'] <= row['eosp_time'])
        voyage_ecl = ecl.loc[mask]
        if len(voyage_ecl) == 0:
            continue
        ecdis_weather.append({
            'voyage_num': row['voyage_num'],
            'avg_wind_speed_ecdis': voyage_ecl['Wind_Speed'].mean(),
            'avg_wind_dir_ecdis': voyage_ecl['Wind_Dir'].mean(),
            'max_wind_speed_ecdis': voyage_ecl['Wind_Speed'].max(),
        })
    ecdis_wdf = pd.DataFrame(ecdis_weather)

    # ---- Open-Meteo (gap-fill, with km/h -> m/s fix) ----
    om_df = _read_table(om_path)
    om_df['datetime'] = pd.to_datetime(om_df['datetime'])
    om_df['wind_speed_ms'] = om_df['wind_speed'] / 3.6

    om_weather = []
    for _, row in vdf.iterrows():
        if row['faop_time'] >= ecdis_start and row['eosp_time'] <= ecdis_end:
            continue
        mask = (om_df['datetime'] >= row['faop_time']) & (om_df['datetime'] <= row['eosp_time'])
        voyage_om = om_df.loc[mask]
        if len(voyage_om) == 0:
            continue
        om_weather.append({
            'voyage_num': row['voyage_num'],
            'avg_wind_speed_om': voyage_om['wind_speed_ms'].mean(),
            'avg_wind_dir_om': voyage_om['wind_dir'].mean(),
            'max_wind_speed_om': voyage_om['wind_speed_ms'].max(),
        })
    om_wdf = pd.DataFrame(om_weather)

    # ---- Merge + feature engineering ----
    vdf = vdf.merge(ecdis_wdf, on='voyage_num', how='left')
    vdf = vdf.merge(om_wdf, on='voyage_num', how='left')

    vdf['wind_speed'] = vdf['avg_wind_speed_ecdis'].fillna(vdf['avg_wind_speed_om'])
    vdf['wind_dir'] = vdf['avg_wind_dir_ecdis'].fillna(vdf['avg_wind_dir_om']).fillna(0)
    vdf['max_wind'] = vdf['max_wind_speed_ecdis'].fillna(vdf['max_wind_speed_om'])

    vdf['load_pct'] = vdf['rob_load_pct']
    vdf['rpm'] = vdf['rob_rpm']
    vdf['slip_pct'] = vdf['rob_slip_pct']

    vdf['trip_time_hrs'] = (vdf['eosp_time'] - vdf['faop_time']).dt.total_seconds() / 3600
    median_trip = vdf['trip_time_hrs'].median()
    vdf.loc[vdf['trip_time_hrs'] < 5, 'trip_time_hrs'] = median_trip
    vdf.loc[vdf['trip_time_hrs'] > 25, 'trip_time_hrs'] = median_trip

    vdf['voyage_sequence'] = np.arange(len(vdf))
    vdf['days_from_start'] = (vdf['faop_time'] - vdf['faop_time'].min()).dt.total_seconds() / (24 * 3600)
    vdf['slip_load_interaction'] = vdf['slip_pct'] * vdf['load_pct'] / 100
    vdf['rpm_load_interaction'] = vdf['rpm'] * vdf['load_pct'] / 100
    vdf['rpm_slip_interaction'] = vdf['rpm'] * vdf['slip_pct'] / 100
    vdf['direction_kp_to_rws'] = (vdf['direction'] == 'KP_to_RWS').astype(int)
    vdf['avg_speed'] = vdf['rob_distance_nm'] / vdf['trip_time_hrs']

    clean_df = vdf.dropna(subset=STAGE1_FEATURES_FULL + ['foc_per_nm']).copy()
    clean_df.loc[clean_df['load_pct'] > 100, 'load_pct'] = np.nan
    clean_df = clean_df.dropna(subset=STAGE1_FEATURES_FULL + ['foc_per_nm']).copy()
    q1, q3 = clean_df['foc_per_nm'].quantile([0.25, 0.75])
    iqr = q3 - q1
    clean_df = clean_df[(clean_df['foc_per_nm'] >= q1 - 1.5 * iqr) & (clean_df['foc_per_nm'] <= q3 + 1.5 * iqr)].copy()
    clean_df = clean_df.reset_index(drop=True)

    # Reporting-friendly aliases (kept for backward compatibility with
    # get_training_statistics() / generate_academic_report())
    clean_df['me_fuel_mt'] = clean_df['me_foc_mt']
    clean_df['speed_knots'] = clean_df['avg_speed']
    clean_df['duration'] = clean_df['trip_time_hrs']
    clean_df['route'] = clean_df['direction']
    clean_df['date'] = clean_df['faop_time']
    clean_df['load'] = clean_df['load_pct']

    logger.info("Voyage pipeline: %d voyages, %d Stage 1 features", len(clean_df), len(STAGE1_FEATURES_FULL))
    return clean_df


def compute_slip_pct(rpm: float, speed_through_water_kn: float) -> float:
    """Physics-based propeller slip, matching manuscript Eq. 1-2.

    V_t = RPM * pitch * 60 / 1852   (theoretical speed, knots)
    slip = 1 - (STW / V_t)
    """
    if rpm is None or rpm <= 0:
        return 0.0
    v_t = (rpm * PROPELLER_PITCH_M * 60) / 1852
    if v_t <= 0:
        return 0.0
    slip = 1 - (speed_through_water_kn / v_t)
    return float(np.clip(slip * 100, -10, 60))  # as a percentage, clipped to a sane range
