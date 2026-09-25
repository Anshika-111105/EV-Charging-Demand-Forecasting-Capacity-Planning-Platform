# Production Model Card: EV Charging Demand Forecaster

## 1. Model Details
- **Model Identifier**: `ev_charging_demand_forecaster`
- **Model Type**: Seasonal Autoregressive Integrated Moving Average (`SARIMA`)
- **Model Version**: `v20260922_152647`
- **Stage**: `Production`
- **Hyperparameters**:
  - Non-seasonal Order $(p, d, q)$: `(1, 1, 1)`
  - Seasonal Order $(P, D, Q, s)$: `(1, 0, 1, 24)` (24-hour diurnal cycle)
  - Optimization: Maximum Likelihood Estimation with bounded iterations and stationary fallback
- **Date Promoted**: `2026-09-22T15:26:47 UTC`
- **MLflow Tracking Run ID**: `f43720560ac54181919c99da5377eda5`
- **Target Variable**: Hourly energy demand in kilowatt-hours (`energy_demand_kwh`)

---

## 2. Intended Use & Scope
- **Primary Objective**: Forecast hourly electric vehicle (EV) charging energy consumption (kWh) across multi-station charging networks up to 168 hours (7 days) in advance.
- **Decision Support Application**: 
  - Automated capacity risk alerting (Overload, Critical, High, Medium, Low).
  - Peak demand shaved scheduling and transformer headroom monitoring.
  - What-if infrastructure expansion planning (+10%, +20%, +50% capacity expansion).
- **Out of Scope**: Real-time sub-second frequency regulation or individual driver session billing.

---

## 3. Training & Validation Lineage
- **Source Dataset**: Caltech Adaptive Charging Network (ACN-Data) benchmark.
- **Sites Monitored**:
  - `caltech` (California Garage 01)
  - `jpl` (Arroyo Garage 01)
  - `office_01` (Parking Lot 01)
- **Data Integrity**:
  - Raw session data SHA-256 hash: `ba37156dd9d3dd49a96db65b2189aa85bf30140623bc9d4e6e6dd797a7b7952b`
  - Total regularized station-hours: 31,968 continuous hourly observations.
  - Quality gates: 100% session retention, zero unhandled negative values, complete hourly grid reconstruction with zero-demand filling.
- **Validation Protocol**:
  - Chronological 3-fold expanding-window walk-forward backtesting (minimum 720 hours training history, expanding by 168-hour steps).
  - Zero temporal data leakage (strict past-only feature alignment).

---

## 4. Benchmark Performance & Leaderboard

### Primary Horizon (168 Hours / 7 Days Ahead)
| Model | Model Type | RMSE (kWh) | MAE (kWh) | Zero-Safe MAPE (%) | sMAPE (%) | Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SARIMA (1,1,1)(1,0,1,24)** | **Statistical** | **0.0845** | **0.0486** | **4.86%** | **44.44%** | **1 (Winner)** |
| ARIMA (2,1,1) | Statistical | 0.1022 | 0.0589 | 5.89% | 66.65% | 2 |
| Seasonal Naive (24h) | Baseline | 0.1669 | 0.0552 | 5.52% | 7.41% | 3 |
| Prophet (Daily+Weekly) | Prophet | 0.1768 | 0.0683 | 6.83% | 32.41% | 4 |
| Seasonal Naive (168h) | Baseline | 0.2070 | 0.0614 | 6.14% | 9.39% | 5 |
| XGBoost Global | Gradient Boosting | 1.0748 | 0.6843 | 68.43% | 187.82% | 6 |

### Multi-Horizon Summary
- **24-Hour Ahead**: Prophet RMSE: `0.0635`, Seasonal Naive 168 RMSE: `0.0709`, SARIMA RMSE: `0.0767`.
- **48-Hour Ahead**: SARIMA RMSE: `0.0812`, ARIMA RMSE: `0.0996`, Prophet RMSE: `0.1431`.
- **168-Hour Ahead**: SARIMA dominates all competitors with lowest RMSE (`0.0845`) and MAE (`0.0486`).

---

## 5. Promotion Quality Gate Verification
- [x] **Primary Metric Superiority**: SARIMA achieved lowest composite error score across expanding-window backtesting folds.
- [x] **Beats Baseline Verification**: Outperforms Seasonal Naive 168 baseline (`0.0845` vs `0.2070` RMSE, a 59.2% error reduction).
- [x] **Zero-Safe MAPE Gate**: 4.86% (Strictly below the 30.0% maximum error threshold).
- [x] **Artifact Integrity**: Serialization verified at `models/checkpoints/production_model.pkl` and registered in `models/checkpoints/registry_metadata.json`.

---

## 6. Operational Risk, Drift & Monitoring Policy
- **Statistical Drift Checks**: Continuous monitoring via Kolmogorov-Smirnov test ($\alpha = 0.05$) and Population Stability Index (PSI).
- **Drift Baseline Status**: `HEALTHY` (Zero drifted features on test set).
- **Retraining Trigger Policy**:
  - Retrain automatically if statistical drift exceeds PSI threshold (`> 0.25`) or if rolling 7-day forecast error degrades by $> 15\%$.
  - Requires minimum of 168 new verified ground-truth hours.
- **Audit Logging**: Every single inference is logged with unique `forecast_id`, `batch_id`, input hash, and timestamp in SQLite/PostgreSQL audit tables.
