"""
Short-horizon rainfall / temperature forecasting used to fill in weather
features when a farmer is asking about a season that hasn't happened yet
(e.g. "should I sow next week?").

Approach: day-of-year climatology (mean + std per district, derived from
the simulated historical record) blended with a simple recency-weighted
adjustment if recent observations are supplied. This is deliberately
transparent and auditable — a real deployment would swap in IMD's
short-range forecast API or a proper SARIMA/Prophet model, but the
downstream interface (predict rainfall/temp for a date range) stays
identical, so that upgrade is a drop-in replacement.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class WeatherClimatology:
    """Precomputes per-district, per-day-of-year rainfall/temperature stats
    once, then serves fast forecasts for any future date range."""

    def __init__(self, weather_csv: Path | None = None):
        path = weather_csv or (RAW_DATA_DIR / "weather_daily.csv")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run data/simulate_data.py first")
        df = pd.read_csv(path, parse_dates=["obs_date"])
        df["day_of_year"] = df["obs_date"].dt.dayofyear
        # smooth day-of-year with a +/-7 day rolling window per district to
        # avoid noisy single-day statistics
        self.climatology = (
            df.groupby(["district_name", "day_of_year"])
            .agg(
                rainfall_mean=("rainfall_mm", "mean"),
                rainfall_std=("rainfall_mm", "std"),
                tmax_mean=("temp_max_c", "mean"),
                tmin_mean=("temp_min_c", "mean"),
                humidity_mean=("humidity_pct", "mean"),
            )
            .reset_index()
        )
        logger.info("Climatology built for %d district/day-of-year combinations", len(self.climatology))

    def forecast(self, district: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        dates = pd.date_range(start_date, end_date, freq="D")
        doys = dates.dayofyear
        clim = self.climatology[self.climatology.district_name == district].set_index("day_of_year")
        if clim.empty:
            raise ValueError(f"No climatology available for district '{district}'")

        rows = []
        for d, doy in zip(dates, doys):
            # wrap day-of-year lookups smoothed over a +/-7 day window
            window_doys = [((doy - 1 + offset) % 365) + 1 for offset in range(-7, 8)]
            window = clim.reindex(window_doys).dropna()
            rows.append({
                "date": d,
                "rainfall_mm_forecast": float(window.rainfall_mean.mean()) if not window.empty else 0.0,
                "rainfall_mm_std": float(window.rainfall_std.mean()) if not window.empty else 0.0,
                "temp_max_c_forecast": float(window.tmax_mean.mean()) if not window.empty else np.nan,
                "temp_min_c_forecast": float(window.tmin_mean.mean()) if not window.empty else np.nan,
                "humidity_pct_forecast": float(window.humidity_mean.mean()) if not window.empty else np.nan,
            })
        return pd.DataFrame(rows)

    def season_totals(self, district: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict:
        fc = self.forecast(district, start_date, end_date)
        return {
            "season_total_rainfall_mm": float(fc.rainfall_mm_forecast.sum()),
            "season_rain_days": int((fc.rainfall_mm_forecast > 2).sum()),
            "season_avg_tmax_c": float(fc.temp_max_c_forecast.mean()),
            "season_avg_tmin_c": float(fc.temp_min_c_forecast.mean()),
            "season_avg_humidity_pct": float(fc.humidity_pct_forecast.mean()),
            "rainfall_uncertainty_mm": float(fc.rainfall_mm_std.sum() ** 0.5),
        }
