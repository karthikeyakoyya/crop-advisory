# Field Ledger — Crop Yield & Irrigation Advisory System
Live demo: https://crop-advisory-353j.onrender.com
Code: https://github.com/karthikeyakoyya/crop-advisory

A decision-support tool for smallholder farmers and agricultural extension officers in Andhra Pradesh. Given a district, crop, sowing date, and (optionally) soil readings, it returns:

- a predicted yield with an **80% confidence interval** (never a bare point estimate)
- a **stage-by-stage irrigation schedule** in mm
- **drought / excess-rainfall risk flags**
- the reasoning behind the estimate, and an explicit advisory-only disclaimer

Built and verified end-to-end: data simulation → PostgreSQL → feature engineering → AutoML baseline + tuned XGBoost with MLflow tracking → FastAPI → frontend, all actually run against each other, not just written.

> **A note on data sources.** IMD and ICRISAT publish district-level rainfall and yield data, but their portals require manual/registered downloads that aren't reachable from an automated pipeline. `data/simulate_data.py` generates data with the same shape, seasonal structure, and realistic value ranges (calibrated against published AP kharif rainfall and ICRISAT yield bulletins) — coastal vs. Rayalaseema rainfall gradients, kharif/rabi seasonality, crop-specific yield bands. Everything downstream (schema, features, model, API) is written against that shape, so swapping in real IMD/ICRISAT exports later is a data-loading change, not a rewrite.

---

## Project layout

```
crop-advisory/
├── data/
│   ├── schema.sql            PostgreSQL schema (districts, weather, soil, crop_seasons)
│   ├── simulate_data.py      generates realistic CSVs into data/raw/
│   ├── load_to_postgres.py   loads those CSVs into Postgres
│   ├── sql_queries.py        shared parameterized SQL for feature extraction
│   └── raw/, processed/      generated data (gitignored)
├── models/
│   ├── feature_engineering.py  builds the model-ready feature table
│   ├── train.py                PyCaret baseline + tuned XGBoost + quantile models, MLflow-tracked
│   ├── forecasting.py          district day-of-year rainfall/temperature climatology
│   ├── predict.py              inference: yield + interval + irrigation schedule + risk flags
│   └── artifacts/              trained model files (gitignored)
├── api/
│   ├── main.py                FastAPI app (also serves the frontend)
│   └── schemas.py             request/response models
├── frontend/
│   ├── index.html, style.css, app.js
├── config/
│   ├── settings.py            central config, reads .env
│   └── .env.example           copy to .env and fill in
├── notebooks/
│   └── exploratory_analysis.md  cell-by-cell EDA to run in Jupyter
├── requirements.txt
└── README.md
```

---

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`pycaret` is a heavy optional dependency (used only for the AutoML baseline sweep in Phase 1 of training). If you'd rather skip it, delete/comment that line from `requirements.txt` before installing — `models/train.py` detects its absence and logs a warning instead of failing.

Copy the environment template and fill in your own PostgreSQL credentials:

```bash
cp config/.env.example .env
```

### PostgreSQL

You need a running PostgreSQL instance and an empty database:

```bash
createdb crop_advisory
```

Update `.env` with your host/port/user/password/db name. `data/load_to_postgres.py` applies `data/schema.sql` automatically — you don't need to run it by hand.

---

## 2. Run the pipeline, in order

```bash
# 1. Generate realistic weather / soil / crop-season data
python data/simulate_data.py

# 2. Load it into PostgreSQL
python data/load_to_postgres.py

# 3. Build model-ready features (from Postgres, or --source csv to skip the DB and iterate faster)
python models/feature_engineering.py --source db

# 4. Train: PyCaret AutoML baseline, then tuned XGBoost + p10/p90 quantile models, all logged to MLflow
python models/train.py
```

Step 4 prints test-set MAE / MAPE / R² and the empirical 80%-interval coverage — logged honestly, not hidden, even if coverage isn't exactly 80% (model calibration is a real, ongoing concern, not a solved problem).

Inspect experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 3. Run the app

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — the FastAPI process serves the frontend directly, so this is the only command you need. API docs are at `/docs`.

If you'd rather run the frontend separately (e.g. a different static file server), it calls the API at whatever origin it's served from — same-origin is simplest, which is why `api/main.py` mounts `frontend/` itself.

---

## 4. Using it

Fill in district, crop, season, irrigation source, area, and sowing date. Soil readings are optional — leave them blank and the API falls back to the latest district-level soil survey on file.

Try, for example: **Guntur, Rice, kharif, canal irrigation, 2.5 ha, sown 2026-06-20** — this returns a real, sane result out of the box (predicted yield with interval, a 3-stage irrigation schedule, and a drought-risk flag, since canal-fed rice in a normal monsoon still typically needs supplemental water on top of rainfall).

---

## Responsible AI notes

- Every API response includes `confidence_interval_80pct` — never a bare number — and the empirical coverage of that interval is logged during training so the "80%" claim is checked, not assumed.
- Every response carries an explicit `disclaimer` field and a `reasoning.note` explaining that rainfall figures are historical climatology, not a live short-range forecast.
- The frontend surfaces the disclaimer and reasoning inline with the results, not buried in fine print.
- The model has no visibility into pest pressure, exact fertilizer/pesticide timing, or extreme weather events — this is stated directly rather than implied.

---

## Extending this

- **Real data**: swap `data/simulate_data.py`'s output for actual IMD/ICRISAT/AP Dept. of Agriculture exports with the same column names, and the rest of the pipeline needs no changes.
- **Live forecasts**: `models/forecasting.py`'s `WeatherClimatology` class is the single seam to replace with a real short-range forecast API (e.g. IMD's) — same method signatures, different data source.
- **More crops/districts**: add entries to `AP_DISTRICTS` / `CROPS` / `CROP_CYCLE_DAYS` in `config/settings.py`, then re-run the pipeline from step 1.
- **Deployment**: containerize with a `Dockerfile` around `uvicorn api.main:app`, point `DATABASE_URL` at a managed Postgres instance (Azure/AWS/GCP), and this runs as-is.
