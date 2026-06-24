# What changed — v2.1 → v3.0 ML pipeline rebuild

## Root cause of the mismatch
`backend/ship_ml.py`'s old `_engineer_features()` hardcoded `slip = 0.0`, invented
`wind_resistance = wind_speed * 0.5`, fell back to **randomly generated synthetic
weather**, and estimated engine load from a fake formula when real data was missing.
`backend/data_loader.py`'s date-based merge (rather than precise FAOP/EOSP voyage
windows) also inflated the voyage count to ~375 loosely-matched rows. This is why
the deployed app's feature importance panel showed `wind_resistance`, `speed_cubed`,
`season`, `sea_state` — none of which exist in the corrected thesis/manuscript.

## What was rebuilt

**New file: `backend/voyage_pipeline.py`**
Validated FAOP/EOSP voyage-pairing pipeline (matches manuscript Section 3.1-3.3
exactly): 155 quality-assured voyages, 14 Stage 1 features (11 base + 3 interaction
terms), physics-based propeller slip (`compute_slip_pct()`, Eq. 1-2 from the
manuscript). Reproduces the manuscript's Table 1 numbers to 3 decimal places.

**New file: `backend/voyage_data/`**
Bundles the validated CSV exports of your 3 source files (CE Daily Log, ROB, ECDIS)
plus the historical Open-Meteo weather data and a precomputed `model_diagnostics_v3.json`
(bootstrap CIs, full 5-algorithm comparison, permutation feature importance, ablation
study — computed once offline rather than recomputed on every server boot, for
speed and stability).

**Rewritten: `backend/ship_ml.py`** (old version kept as `ship_ml_v2_backup.py`)
- Deploys **Random Forest, 11 features, no interaction terms** (manuscript Section 4.5),
  not the old ad-hoc feature set
- `predict_fuel()` now computes slip physically from RPM + propeller pitch + speed,
  instead of hardcoding it to zero
- Reports real cross-validated R²/MAE/bootstrap CI instead of a single noisy 70/30 holdout
- Same public method signatures as before — **`server.py` and the entire frontend
  required zero changes**, since both already consume the API generically rather
  than hardcoding feature names

## Verified end-to-end
- `python3 ship_ml.py` self-test: Stage 2 R² = 0.564, matching the manuscript's
  deployed-model number almost exactly
- Full `/optimize`-equivalent simulation (route lookup → ML prediction → physics
  corrections) produces a 6.5 MT prediction for a ~134 NM voyage — consistent with
  the manuscript's reported mean of 6.89 MT
- Simulated `/model-status` response: 155 voyages, mean fuel 6.89 MT (exact match
  to the manuscript), real feature importance led by `direction_kp_to_rws` and
  `load_pct` — the synthetic features are gone

## Before deploying
1. Push this updated repo (or apply the diff) to your GitHub repo
2. Railway will reinstall `requirements.txt` — `python-calamine` is already listed
   there, so no dependency changes needed
3. The trained model artifacts are pre-bundled in `backend/model_cache/`, so the
   app will load them instantly on startup rather than retraining from scratch
4. Once redeployed, re-export the screenshots for manuscript Figures 6-7 from the
   live app — they'll now show the correct model
