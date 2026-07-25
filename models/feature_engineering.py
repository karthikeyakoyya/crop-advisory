"""
Builds the model-ready feature table from raw weather / soil / crop-season
data. Can read from PostgreSQL (production path) or directly from the
simulated CSVs in data/raw/ (fast local iteration, no DB required).

Run:
    python models/feature_engineering.py --source csv
    python models/feature_engineering.py --source db
Writes data/processed/features.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import CROP_CYCLE_DAYS, PROCESSED_DATA_DIR, RAW_DATA_DIR, DATABASE_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_from_csv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weather = pd.read_csv(RAW_DATA_DIR / "weather_daily.csv", parse_dates=["obs_date"])
    soil = pd.read_csv(RAW_DATA_DIR / "soil_health.csv")
    seasons = pd.read_csv(RAW_DATA_DIR / "crop_seasons.csv", parse_dates=["sowing_date", "harvest_date"])
    return weather, soil, seasons


def _load_from_db() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import create_engine
    engine = create_engine(DATABASE_URL)
    weather = pd.read_sql(
        "SELECT w.*, d.district_name FROM weather_daily w JOIN districts d ON d.district_id = w.district_id",
        engine, parse_dates=["obs_date"],
    )
    soil = pd.read_sql(
        "SELECT s.*, d.district_name FROM soil_health s JOIN districts d ON d.district_id = s.district_id",
        engine,
    )
    seasons = pd.read_sql(
        "SELECT c.*, d.district_name FROM crop_seasons c JOIN districts d ON d.district_id = c.district_id",
        engine, parse_dates=["sowing_date", "harvest_date"],
    )
    return weather, soil, seasons


def _rainfall_lag_features(weather: pd.DataFrame, district: str, sowing_date: pd.Timestamp) -> dict:
    """Rainfall totals in the 30/60/90 days *before* sowing — captures soil
    moisture carry-over and monsoon onset timing, both strong yield drivers."""
    dw = weather[weather.district_name == district]
    out = {}
    for window in (30, 60, 90):
        start = sowing_date - pd.Timedelta(days=window)
        mask = (dw.obs_date >= start) & (dw.obs_date < sowing_date)
        out[f"pre_sowing_rain_{window}d"] = float(dw.loc[mask, "rainfall_mm"].sum())
    return out


def _in_season_features(weather: pd.DataFrame, district: str, sowing: pd.Timestamp, harvest: pd.Timestamp) -> dict:
    dw = weather[weather.district_name == district]
    window = dw[(dw.obs_date >= sowing) & (dw.obs_date <= harvest)]
    if window.empty:
        return {
            "season_total_rainfall_mm": np.nan, "season_rain_days": 0,
            "season_avg_tmax_c": np.nan, "season_avg_tmin_c": np.nan,
            "season_avg_humidity_pct": np.nan, "temp_trend_c_per_month": np.nan,
        }
    # temperature trend: simple linear slope over the season, in C/month
    days = (window.obs_date - sowing).dt.days.values
    if len(days) > 1 and np.ptp(days) > 0:
        slope_per_day = np.polyfit(days, window.temp_max_c.values, 1)[0]
    else:
        slope_per_day = 0.0
    return {
        "season_total_rainfall_mm": float(window.rainfall_mm.sum()),
        "season_rain_days": int((window.rainfall_mm > 2).sum()),
        "season_avg_tmax_c": float(window.temp_max_c.mean()),
        "season_avg_tmin_c": float(window.temp_min_c.mean()),
        "season_avg_humidity_pct": float(window.humidity_pct.mean()),
        "temp_trend_c_per_month": float(slope_per_day * 30),
    }


def build_features(weather: pd.DataFrame, soil: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building features for %d crop-season records", len(seasons))
    records = []
    for _, row in seasons.iterrows():
        district = row["district_name"]
        sowing = row["sowing_date"]
        harvest = row["harvest_date"]
        cycle_days = (harvest - sowing).days
        cycle_ref = CROP_CYCLE_DAYS.get(row["crop_name"], cycle_days)

        feats = {
            "district_name": district,
            "crop_name": row["crop_name"],
            "season_name": row["season_name"],
            "irrigation_source": row["irrigation_source"],
            "area_hectares": row["area_hectares"],
            "sowing_month": sowing.month,
            "cycle_days": cycle_days,
            # growth-stage indicator: fraction of crop cycle elapsed at a
            # representative mid-season checkpoint — a real deployment would
            # compute this relative to "today" for an in-progress season.
            "growth_stage_fraction": min(cycle_days / max(cycle_ref, 1), 1.5),
        }
        feats.update(_rainfall_lag_features(weather, district, sowing))
        feats.update(_in_season_features(weather, district, sowing, harvest))

        soil_row = soil[soil.district_name == district]
        if not soil_row.empty:
            nearest = soil_row.iloc[(soil_row["survey_year"] - sowing.year).abs().argsort()[:1]]
            feats["soil_ph"] = float(nearest["soil_ph"].values[0])
            feats["organic_carbon_pct"] = float(nearest["organic_carbon_pct"].values[0])
            feats["nitrogen_kg_ha"] = float(nearest["nitrogen_kg_ha"].values[0])
            feats["phosphorus_kg_ha"] = float(nearest["phosphorus_kg_ha"].values[0])
            feats["potassium_kg_ha"] = float(nearest["potassium_kg_ha"].values[0])
            feats["soil_moisture_pct"] = float(nearest["soil_moisture_pct"].values[0])
        feats["target_yield_kg_per_ha"] = row["yield_kg_per_ha"]
        records.append(feats)

    df = pd.DataFrame(records)
    n_before = len(df)
    df = df.dropna(subset=["target_yield_kg_per_ha"])
    logger.info("Feature rows: %d (dropped %d with missing target)", len(df), n_before - len(df))
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["csv", "db"], default="csv")
    args = parser.parse_args()

    if args.source == "csv":
        weather, soil, seasons = _load_from_csv()
    else:
        weather, soil, seasons = _load_from_db()

    features = build_features(weather, soil, seasons)
    out_path = PROCESSED_DATA_DIR / "features.csv"
    features.to_csv(out_path, index=False)
    logger.info("Wrote %d feature rows to %s", len(features), out_path)


if __name__ == "__main__":
    main()
