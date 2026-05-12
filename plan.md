# plan.md — Maritime Fuel Optimization System (M/V Al-bazm II)

## 1. Objectives
- Deliver a working V1 clone of the reference UI/flow (tabs: Optimization, Results, Weather, ML Model) with open access.
- Core workflow: user inputs voyage + ETA + wind → system computes feasibility + optimal speed/RPM + fuel estimate using **physics + ML refinement**, persists voyage, updates ML stats.
- Use real weather via **Open-Meteo Marine API** and persist voyage history in **MongoDB**.

---

## 2. Implementation Steps

### Phase 1 — Core POC (isolation; must pass before UI build)
**Goal:** prove the end-to-end optimize pipeline works with real weather + ML training + persistence.

1) Web research / integration playbook
- Confirm best-practice usage for Open-Meteo Marine API (endpoints/params, timezone handling).
- Confirm sklearn model persistence approach (joblib) + incremental retrain strategy for small datasets.

2) Create `/app/test_core.py` (single runnable script)
- Define constants: ports (Khalifa, Ruwais) + lat/lon + distance_nm (~78nm, adjustable).
- Physics functions:
  - `required_speed_kn(distance_nm, eta_dt, now_dt)`
  - `fuel_rate(speed_kn, wind_speed, rel_wind_angle)` with cubic speed term + wind drag factor
  - `rpm_from_speed(speed_kn)` mapping to optimal band 115–145
- Weather fetch:
  - Use `httpx` to call Open-Meteo marine forecast for both ports (wind speed/direction + waves if available).
  - Derive a simple route wind estimate (average of ports for ETA window).
- MongoDB:
  - Connect, create `voyages` collection; insert optimization runs.
- Seed + ML:
  - If `voyages` empty, generate 372 synthetic voyages (matching reference count) and store.
  - Train sklearn regressor (e.g., RandomForestRegressor) to predict fuel adjustment or final fuel.
  - Compute accuracy metric (R² or MAPE) on holdout split; output `ml_accuracy`.
- `optimize_voyage()`:
  - Combine physics baseline + ML correction.
  - Output: feasibility banner inputs (required vs max speed, min time, suggested ETA), optimal speed/RPM, fuel estimate, and weather snapshot.

3) POC acceptance loop (no app build until green)
- Run script locally; verify:
  - Open-Meteo calls succeed and parse reliably.
  - DB seed count == 372 on first run.
  - Optimize returns stable numeric outputs.
  - Accuracy metric computed and non-null.
- Iterate until consistently passing.

**Phase 1 user stories**
1. As a developer, I can run one script and see a complete optimization result printed.
2. As a developer, I can fetch real marine weather for both ports without API keys.
3. As a developer, I can seed 372 voyages and confirm Mongo persistence.
4. As a developer, I can train an ML model and view an accuracy metric.
5. As a developer, I can run optimize twice and see training voyages increment.

---

### Phase 2 — V1 App Development (build around proven core)
**Goal:** ship a working MVP with reference-like UI and real backend.

1) Backend (FastAPI)
- Create modules:
  - `fuel_physics.py`, `weather_service.py`, `ml_model.py`, `db.py`, `seed.py`.
- Startup:
  - Connect MongoDB, seed 372 voyages if empty, train/load model, compute stats.
- API endpoints (`/api`):
  - `POST /optimize` → runs optimization, persists voyage, retrains (or schedules retrain) + returns results payload.
  - `GET /weather` → returns latest forecast snapshot for both ports.
  - `GET /ml/info` → accuracy + training_voyages + feature importance.
  - `GET /voyages` → recent voyage history.
  - `GET /vessel` → vessel specs.
- Robustness:
  - Validate inputs, timezone-safe datetime parsing, graceful fallback if weather unavailable.

2) Frontend (React)
- Visual: dark navy theme + cyan accents; match reference layout.
- Structure:
  - Header: ship icon/title/subtitle + Dubai UTC+4 live clock.
  - Tabs: Optimization | Results | Weather | ML Model.
- Optimization tab:
  - Left form (ports dropdowns, datetime, wind speed/direction) + big OPTIMIZE ROUTE.
  - Right stats card: vessel, max speed 12 kn, RPM band, ML accuracy, training voyages.
- Results tab:
  - Feasibility banner (warning when ETA cannot be met), required/max speed, min time, suggested ETA.
  - Optimization details (speed, rpm, fuel, distance, weather used).
- Weather tab:
  - Cards for both ports with key marine fields (wind speed/dir, wave height if present, timestamp).
- ML Model tab:
  - Accuracy + training count + feature importance chart + recent voyages table.

3) End-to-end V1 testing
- Run one full flow: optimize → view Results → Weather → ML Model → voyage history.
- Call testing agent once and fix all issues found.

**Phase 2 user stories**
1. As an officer, I open the app and immediately see vessel info and live Dubai time.
2. As an officer, I enter voyage inputs and get an optimization result after clicking OPTIMIZE ROUTE.
3. As an officer, I can clearly see if my ETA is infeasible and what speed would be required.
4. As an officer, I can view real marine weather for both ports on the Weather tab.
5. As an officer, I can view ML accuracy and training voyage count on the ML Model tab.

---

### Phase 3 — Productionizing the MVP (stability + quality)
1) Model lifecycle improvements
- Persist model artifact + metadata (joblib) and reload on boot.
- Retrain policy: retrain every N new voyages or via background task to avoid blocking requests.

2) Better domain outputs
- Add route distance configurability + clearer units (kn, nm, hours, tons).
- Add “why” breakdown: physics baseline vs ML adjustment.

3) Data + UI enhancements
- Voyage history page/section (filter by date, export JSON/CSV).
- Chart: fuel vs speed and historical fuel distribution.

4) Testing
- Call testing agent once; verify all tabs, error states (weather down, invalid ETA).

**Phase 3 user stories**
1. As an officer, I can see how much of the fuel estimate comes from physics vs ML.
2. As an officer, I can browse past optimizations and compare outcomes.
3. As an operator, I can retrain the model without slowing down optimize requests.
4. As an officer, I can export voyage history for reporting.
5. As an officer, I still get a usable estimate even if weather API is temporarily unavailable.

---

## 3. Next Actions
1) Implement and run `/app/test_core.py` until it passes consistently (weather + seed + train + optimize + persist).
2) Lock the output schema from POC and mirror it in `POST /api/optimize` response.
3) Build backend modules + endpoints in one pass, then build React UI tabs in one pass.
4) Run end-to-end test + testing agent after Phase 2.

---

## 4. Success Criteria
- POC: weather fetch works, seeds 372 voyages, trains ML model, returns optimization payload, persists to Mongo.
- V1: user can optimize a voyage from UI and see results; Weather and ML tabs populate with real data.
- Reliability: graceful handling of infeasible ETA and weather API failures.
- Persistence: every optimization creates a DB record and training voyage count increases accordingly.
- Testing agent passes core flows across all tabs with no blocking bugs.