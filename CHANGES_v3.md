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

---

# v3.1 — Variable-speed validation + fuel-savings fix + anonymization

## What was validated
Tested whether per-segment speed variation (slowing on long segments, speeding
on short ones, at a fixed overall average/ETA) actually reduces predicted fuel,
since this was flagged as the thesis's main original idea.

**Result: it doesn't.** Confirmed two ways:
- **Analytically**: for any smooth speed-to-fuel relationship (cube law or otherwise),
  minimizing fuel for a fixed total voyage time is solved by *constant* speed
  across all segments -- a clean Lagrange-multiplier result, independent of how
  distances are distributed.
- **Empirically**: ran the actual deployed model against the real route at a 9 kn
  required average -- variable speed came within 0.1% of constant speed (not a
  real difference). Also found that evaluating the model at per-segment
  granularity is itself invalid -- segment durations (minutes) fall far outside the
  7-22 hour range the model was trained on, and naive per-segment summation
  inflated total fuel ~2x versus the valid whole-voyage prediction.

**What *is* real and validated**: reducing the *overall average speed* when
schedule allows it (classic slow steaming). Sweeping 8-12 kn on the live route
showed up to 17-18% predicted fuel savings at lower average speeds versus cruising
at max speed -- a genuine, model-supported finding.

## What changed in the app (backend/server.py)
- Removed the fuel_savings line that used `random.uniform(3, 8)` -- this was a
  literally random number, not computed from anything. Replaced with a real
  comparison: achieved average speed vs. the constant-max-speed ("Fast Route")
  alternative, both evaluated through the same model.
- Reworded the speed_recommendation / eta_feasibility text for variable-speed
  mode to stop claiming a per-segment fuel benefit, and to point to the real
  lever (overall average speed) instead.
- Removed the now-unused `import random`.

## What changed in the frontend (frontend/src/App.js)
- The Speed Profile banner no longer says "longer segments slowed to save fuel"
  (false claim) -- it now explains that the fuel total reflects the overall
  average speed, and points to the Fuel Savings insight for the real number.
- No other frontend changes needed -- Pareto-Efficient Alternatives and the
  Performance Data / Feature Importance panels already pull live from the API,
  so they automatically reflect the corrected backend.

## What changed in the manuscript
Added new Section 4.6 ("Operational Speed Optimization: Isolating the Genuine
Fuel-Saving Lever") with the full analytical + empirical validation and a new
Table 6 (fuel vs. average speed). Updated the Abstract and Discussion to
reference this finding. This is a legitimate, additional contribution --
arguably stronger than the original per-segment claim would have been, since
it's actually validated.

## Anonymization
Replaced every instance of the original vessel name (the original vessel name and its casing variants)
across the entire repo -- code, comments, docstrings, config files
(render.yaml, fly.toml, DEPLOY.md), and tests -- with the generic placeholder
"Atlas" (class AtlasMLSystem, model files vessel_model_v3.joblib, etc.). Also
removed several now-unused legacy files that still contained the real name and
were no longer imported by anything (data_loader.py, enhanced_data_processor.py,
ml_models.py, old ship_ml backups, and a legacy CSV with the name embedded in
its data). Port names (Khalifa Port, Ruwais Port) were left as-is since they're
public geography already in the manuscript, not vessel-identifying.

## Verified
- `python3 ship_ml.py` self-test: trains cleanly, Stage 2 R-squared = 0.564
- Full /optimize simulation across multiple ETA scenarios: real savings figures
  scale sensibly with schedule slack (15-17% with slack, ~1% when ETA is tight)
- Repo-wide case-insensitive search for the old vessel name: zero remaining hits
