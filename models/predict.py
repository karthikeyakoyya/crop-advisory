"""
Inference layer sitting between the FastAPI endpoints and the trained
models. Responsible for:
    - assembling a feature row from farmer input + weather forecast/soil data
    - producing a yield point estimate + 80% confidence interval
    - deriving an irrigation schedule and drought/excess-rain risk flags
    - attaching the responsible-AI disclaimer + reasoning trace

This module never overwrites a farmer's own judgment — every number it
returns is explicitly framed as advisory, and the reasoning behind each
recommendation is returned alongside it (see `reasoning` in the response)
so the output can be checked, not just trusted.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import CROP_CYCLE_DAYS, MODELS_DIR, RAW_DATA_DIR
from models.forecasting import WeatherClimatology

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Approximate total crop water requirement (mm) over the full growth cycle,
# from FAO crop-water-need guidelines. Used only to size the irrigation
# recommendation, not the yield model itself.
CROP_WATER_REQUIREMENT_MM = {
    "Rice": 1250,
    "Groundnut": 550,
    "Cotton": 700,
    "Chilli": 650,
    "Maize": 550,
    "Sugarcane": 1800,
}

# Fraction of total water need that falls in each third of the crop cycle —
# used to spread the irrigation recommendation across growth stages rather
# than dumping it all at once.
STAGE_WATER_SPLIT = {"establishment": 0.25, "vegetative_flowering": 0.45, "maturation": 0.30}


class YieldAdvisor:
    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or MODELS_DIR
        self._load_artifacts()
        self._climatology: WeatherClimatology | None = None

    def _load_artifacts(self) -> None:
        try:
            self.model = joblib.load(self.models_dir / "xgboost_yield_model.joblib")
            self.lower_model = joblib.load(self.models_dir / "quantile_lower_model.joblib")
            self.upper_model = joblib.load(self.models_dir / "quantile_upper_model.joblib")
            with open(self.models_dir / "model_metadata.json") as f:
                self.metadata = json.load(f)
            logger.info("Loaded model artifacts from %s", self.models_dir)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Model artifacts not found. Run models/feature_engineering.py "
                "then models/train.py before starting the API."
            ) from exc

    @property
    def climatology(self) -> WeatherClimatology:
        if self._climatology is None:
            self._climatology = WeatherClimatology(RAW_DATA_DIR / "weather_daily.csv")
        return self._climatology

    def _soil_defaults(self, district: str) -> dict:
        soil = pd.read_csv(RAW_DATA_DIR / "soil_health.csv")
        row = soil[soil.district_name == district].sort_values("survey_year", ascending=False)
        if row.empty:
            raise ValueError(f"No soil data on file for district '{district}'")
        r = row.iloc[0]
        return {
            "soil_ph": float(r.soil_ph),
            "organic_carbon_pct": float(r.organic_carbon_pct),
            "nitrogen_kg_ha": float(r.nitrogen_kg_ha),
            "phosphorus_kg_ha": float(r.phosphorus_kg_ha),
            "potassium_kg_ha": float(r.potassium_kg_ha),
            "soil_moisture_pct": float(r.soil_moisture_pct),
        }

    def _build_feature_row(
        self, district: str, crop: str, season_name: str, irrigation_source: str,
        area_hectares: float, sowing_date: date, soil_overrides: dict | None,
    ) -> pd.DataFrame:
        cycle_days = CROP_CYCLE_DAYS.get(crop, 120)
        harvest_date = sowing_date + timedelta(days=cycle_days)
        sowing_ts, harvest_ts = pd.Timestamp(sowing_date), pd.Timestamp(harvest_date)

        pre_sowing = {}
        for window in (30, 60, 90):
            start = sowing_ts - pd.Timedelta(days=window)
            totals = self.climatology.season_totals(district, start, sowing_ts - pd.Timedelta(days=1))
            pre_sowing[f"pre_sowing_rain_{window}d"] = totals["season_total_rainfall_mm"]

        season_stats = self.climatology.season_totals(district, sowing_ts, harvest_ts)
        # temp trend: compare first-half vs second-half average tmax as a
        # simple monthly-slope proxy, consistent with training-time feature
        mid = sowing_ts + (harvest_ts - sowing_ts) / 2
        first_half = self.climatology.season_totals(district, sowing_ts, mid)
        second_half = self.climatology.season_totals(district, mid, harvest_ts)
        months_apart = max((harvest_ts - sowing_ts).days / 30 / 2, 0.1)
        temp_trend = (second_half["season_avg_tmax_c"] - first_half["season_avg_tmax_c"]) / months_apart

        soil = self._soil_defaults(district)
        if soil_overrides:
            soil.update({k: v for k, v in soil_overrides.items() if v is not None})

        row = {
            "district_name": district,
            "crop_name": crop,
            "season_name": season_name,
            "irrigation_source": irrigation_source,
            "area_hectares": area_hectares,
            "sowing_month": sowing_date.month,
            "cycle_days": cycle_days,
            "growth_stage_fraction": 1.0,
            **pre_sowing,
            "season_total_rainfall_mm": season_stats["season_total_rainfall_mm"],
            "season_rain_days": season_stats["season_rain_days"],
            "season_avg_tmax_c": season_stats["season_avg_tmax_c"],
            "season_avg_tmin_c": season_stats["season_avg_tmin_c"],
            "season_avg_humidity_pct": season_stats["season_avg_humidity_pct"],
            "temp_trend_c_per_month": temp_trend,
            **soil,
        }
        return pd.DataFrame([row]), season_stats, harvest_date

    def _irrigation_schedule(self, crop: str, sowing_date: date, harvest_date: date, expected_rainfall_mm: float) -> list[dict]:
        total_need = CROP_WATER_REQUIREMENT_MM.get(crop, 600)
        deficit_mm = max(total_need - expected_rainfall_mm, 0)
        cycle_days = (harvest_date - sowing_date).days
        schedule = []
        stage_starts = {
            "establishment": sowing_date,
            "vegetative_flowering": sowing_date + timedelta(days=int(cycle_days * 0.25)),
            "maturation": sowing_date + timedelta(days=int(cycle_days * 0.70)),
        }
        for stage, fraction in STAGE_WATER_SPLIT.items():
            stage_deficit = round(deficit_mm * fraction, 1)
            schedule.append({
                "stage": stage.replace("_", " "),
                "recommended_start_date": stage_starts[stage].isoformat(),
                "irrigation_needed_mm": stage_deficit,
                "note": (
                    "Rainfall is expected to cover most of this stage's water need; "
                    "irrigate only if visible wilting or crusting appears."
                    if stage_deficit < total_need * fraction * 0.25 else
                    "Supplemental irrigation recommended around this stage to avoid yield loss."
                ),
            })
        return schedule

    def _risk_flags(self, crop: str, expected_rainfall_mm: float) -> list[dict]:
        total_need = CROP_WATER_REQUIREMENT_MM.get(crop, 600)
        ratio = expected_rainfall_mm / total_need if total_need else 1.0
        flags = []
        if ratio < 0.6:
            flags.append({
                "type": "drought_risk", "severity": "high" if ratio < 0.4 else "moderate",
                "message": (
                    f"Expected rainfall covers only ~{ratio*100:.0f}% of {crop}'s typical water need. "
                    "Plan for supplemental irrigation and consider drought-tolerant variety options."
                ),
            })
        elif ratio > 1.6:
            flags.append({
                "type": "excess_rainfall_risk", "severity": "moderate",
                "message": (
                    f"Expected rainfall is ~{ratio*100:.0f}% of {crop}'s typical need — "
                    "watch for waterlogging and ensure field drainage is clear."
                ),
            })
        return flags

    def predict(
        self, district: str, crop: str, season_name: str, irrigation_source: str,
        area_hectares: float, sowing_date: date, soil_overrides: dict | None = None,
    ) -> dict:
        if crop not in CROP_CYCLE_DAYS:
            raise ValueError(f"Unknown crop '{crop}'. Supported crops: {sorted(CROP_CYCLE_DAYS)}")

        feature_row, season_stats, harvest_date = self._build_feature_row(
            district, crop, season_name, irrigation_source, area_hectares, sowing_date, soil_overrides,
        )

        median_pred = float(self.model.predict(feature_row)[0])
        p10 = float(self.lower_model.predict(feature_row)[0])
        p90 = float(self.upper_model.predict(feature_row)[0])
        # guard against quantile crossing (rare, but possible with two
        # independently trained models) — enforce p10 <= median <= p90
        p10, p90 = min(p10, median_pred), max(p90, median_pred)

        irrigation_schedule = self._irrigation_schedule(
            crop, sowing_date, harvest_date, season_stats["season_total_rainfall_mm"],
        )
        risk_flags = self._risk_flags(crop, season_stats["season_total_rainfall_mm"])

        return {
            "district": district,
            "crop": crop,
            "sowing_date": sowing_date.isoformat(),
            "harvest_date": harvest_date.isoformat(),
            "predicted_yield_kg_per_ha": round(median_pred, 1),
            "confidence_interval_80pct": {
                "low": round(p10, 1),
                "high": round(p90, 1),
            },
            "expected_total_rainfall_mm": round(season_stats["season_total_rainfall_mm"], 1),
            "irrigation_schedule": irrigation_schedule,
            "risk_flags": risk_flags,
            "reasoning": {
                "model": "XGBoost regressor (production), GradientBoosting p10/p90 for interval",
                "key_drivers_considered": [
                    "pre-sowing 30/60/90-day rainfall", "in-season rainfall & rain days",
                    "temperature trend across the growth cycle", "soil pH, organic carbon, N-P-K, moisture",
                    "irrigation source", "district-specific rainfall climatology",
                ],
                "note": (
                    "Weather inputs are derived from historical district climatology, not a live "
                    "short-range forecast — treat the rainfall figures as a seasonal expectation, "
                    "not a day-by-day forecast."
                ),
            },
            "disclaimer": (
                "This is a decision-support estimate based on historical patterns, not a guarantee. "
                "Actual yield depends on factors this model cannot see (pest pressure, exact input "
                "timing, extreme weather). Use alongside local agricultural extension advice."
            ),
        }
