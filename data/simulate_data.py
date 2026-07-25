"""
Simulates realistic Andhra Pradesh agricultural data:
    - Daily district-level weather (rainfall, temperature, humidity)
    - Annual soil health survey records
    - Season-level crop records (sowing/harvest dates, area, yield)

Why simulated data: IMD (India Meteorological Department) and ICRISAT
publish district-level rainfall and yield data, but their portals require
manual/registered downloads and are not reachable from an automated,
network-restricted environment. This generator produces data with the same
shape, ranges, and seasonal structure as the real sources (coastal vs.
Rayalaseema rainfall gradients, kharif/rabi seasonality, crop-specific yield
bands from published ICRISAT/AP Dept. of Agriculture bulletins) so the rest
of the pipeline — schema, features, modeling, API — is a straightforward
swap-in once real credentials/data access are available.

Run:
    python data/simulate_data.py
Outputs CSVs under data/raw/.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import AP_DISTRICTS, CROPS, CROP_CYCLE_DAYS, RANDOM_SEED, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

rng = np.random.default_rng(RANDOM_SEED)

YEARS = list(range(2015, 2025))

# Coastal districts see far heavier, more reliable monsoon rainfall than
# Rayalaseema (interior, rain-shadow) districts. This drives realistic
# yield variance across the state.
COASTAL_DISTRICTS = {
    "Visakhapatnam", "Vizianagaram", "Srikakulam",
    "East Godavari", "West Godavari", "Krishna", "Guntur", "Nellore", "Prakasam",
}
RAYALASEEMA_DISTRICTS = {"Kurnool", "Anantapur", "Chittoor", "YSR Kadapa"}

# Baseline yield (kg/ha) under "average" conditions, roughly aligned with
# published ICRISAT / state agriculture department averages.
BASELINE_YIELD_KG_HA = {
    "Rice": 3200,
    "Groundnut": 1400,
    "Cotton": 550,   # lint yield
    "Chilli": 1900,
    "Maize": 3600,
    "Sugarcane": 70000,
}

IRRIGATION_SOURCES = ["canal", "borewell", "tank", "rainfed"]
SEASON_BY_MONTH = {
    "kharif": (6, 10),   # Jun-Oct, monsoon-fed
    "rabi": (11, 3),     # Nov-Mar
}


def _daily_rainfall_mm(day_of_year: int, is_coastal: bool) -> float:
    """Seasonal rainfall curve peaking in the SW monsoon (Jun-Sep)."""
    monsoon_peak = 200  # ~ mid July
    spread = 55
    seasonal = np.exp(-((day_of_year - monsoon_peak) ** 2) / (2 * spread ** 2))
    # Northeast monsoon bump (Oct-Nov), stronger on the coast
    ne_peak = 300
    ne_spread = 25
    ne_bump = (0.5 if is_coastal else 0.2) * np.exp(-((day_of_year - ne_peak) ** 2) / (2 * ne_spread ** 2))
    base_intensity = 11 if is_coastal else 6
    intensity = base_intensity * (seasonal + ne_bump)
    # Rain is bursty: most days are dry, some days spike heavily.
    # Calibrated so a coastal district's Jun-Oct total lands around
    # 700-950mm and Rayalaseema around 350-500mm, in line with published
    # AP kharif rainfall bulletins.
    if rng.random() < (0.42 * (seasonal + ne_bump) + 0.02):
        return float(max(0.0, rng.gamma(shape=1.6, scale=max(intensity, 0.5))))
    return 0.0


def _temperature_c(day_of_year: int, is_coastal: bool) -> tuple[float, float]:
    """Return (max, min) temperature for the day."""
    annual = -np.cos(2 * np.pi * (day_of_year - 15) / 365)  # peak ~ late May
    base_max = 33 if is_coastal else 34
    base_min = 23 if is_coastal else 19  # Rayalaseema has wider diurnal swing
    tmax = base_max + 5 * annual + rng.normal(0, 1.2)
    tmin = base_min + 4 * annual + rng.normal(0, 1.0)
    return round(float(tmax), 1), round(float(min(tmin, tmax - 3)), 1)


def simulate_weather() -> pd.DataFrame:
    logger.info("Simulating daily weather for %d districts x %d years", len(AP_DISTRICTS), len(YEARS))
    rows = []
    for district in AP_DISTRICTS:
        is_coastal = district in COASTAL_DISTRICTS
        for year in YEARS:
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            for d in dates:
                doy = d.dayofyear
                rain = _daily_rainfall_mm(doy, is_coastal)
                tmax, tmin = _temperature_c(doy, is_coastal)
                humidity = float(np.clip(55 + 30 * (rain > 0) + rng.normal(0, 6), 25, 98))
                rows.append({
                    "district_name": district,
                    "obs_date": d.date().isoformat(),
                    "rainfall_mm": round(rain, 2),
                    "temp_max_c": tmax,
                    "temp_min_c": tmin,
                    "humidity_pct": round(humidity, 1),
                })
    df = pd.DataFrame(rows)
    logger.info("Weather rows generated: %d", len(df))
    return df


def simulate_soil_health() -> pd.DataFrame:
    logger.info("Simulating soil health survey records")
    rows = []
    survey_years = [2016, 2019, 2022]
    for district in AP_DISTRICTS:
        is_coastal = district in COASTAL_DISTRICTS
        for year in survey_years:
            rows.append({
                "district_name": district,
                "survey_year": year,
                "soil_ph": round(float(np.clip(rng.normal(6.8 if is_coastal else 7.6, 0.4), 4.5, 9.0)), 2),
                "organic_carbon_pct": round(float(np.clip(rng.normal(0.55 if is_coastal else 0.35, 0.12), 0.1, 1.5)), 3),
                "nitrogen_kg_ha": round(float(np.clip(rng.normal(280, 40), 100, 450)), 1),
                "phosphorus_kg_ha": round(float(np.clip(rng.normal(22, 6), 5, 60)), 1),
                "potassium_kg_ha": round(float(np.clip(rng.normal(240, 50), 80, 500)), 1),
                "soil_moisture_pct": round(float(np.clip(rng.normal(28 if is_coastal else 16, 5), 4, 45)), 1),
            })
    df = pd.DataFrame(rows)
    logger.info("Soil health rows generated: %d", len(df))
    return df


def _season_dates(year: int, season_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if season_name == "kharif":
        return pd.Timestamp(year=year, month=6, day=15), pd.Timestamp(year=year, month=10, day=15)
    return pd.Timestamp(year=year, month=11, day=15), pd.Timestamp(year=year + 1, month=3, day=15)


def simulate_crop_seasons(weather_df: pd.DataFrame, soil_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Simulating crop season / yield records")
    weather_df = weather_df.copy()
    weather_df["obs_date"] = pd.to_datetime(weather_df["obs_date"])
    rows = []

    for district in AP_DISTRICTS:
        is_coastal = district in COASTAL_DISTRICTS
        dw = weather_df[weather_df.district_name == district]
        crops_grown = CROPS if is_coastal else [c for c in CROPS if c != "Sugarcane"]

        for year in YEARS[:-1]:  # need full season window inside simulated years
            for crop in crops_grown:
                season_name = "kharif" if crop != "Chilli" else rng.choice(["kharif", "rabi"])
                sowing, harvest = _season_dates(year, season_name)
                cycle_days = CROP_CYCLE_DAYS[crop]
                harvest = sowing + pd.Timedelta(days=cycle_days)

                window = dw[(dw.obs_date >= sowing) & (dw.obs_date <= harvest)]
                total_rain = window.rainfall_mm.sum()
                avg_tmax = window.temp_max_c.mean()
                rain_days = int((window.rainfall_mm > 2).sum())

                # crude, transparent agronomic model: yield responds positively
                # to rainfall up to an optimum, then plateaus/declines (waterlogging),
                # and negatively to heat stress; soil quality scales the whole thing.
                optimum_rain = 900 if crop == "Rice" else 550
                rain_ratio = total_rain / optimum_rain
                rain_effect = 1 - 0.9 * (rain_ratio - 1) ** 2 if rain_ratio < 1.6 else 0.55
                rain_effect = float(np.clip(rain_effect, 0.25, 1.15))

                heat_stress = float(np.clip(1 - max(0, avg_tmax - 34) * 0.05, 0.6, 1.0))

                soil_row = soil_df[soil_df.district_name == district].iloc[
                    int(rng.integers(0, len(soil_df[soil_df.district_name == district])))
                ]
                soil_quality = float(np.clip(
                    0.5 + soil_row.organic_carbon_pct * 0.5 + (soil_row.soil_moisture_pct / 100), 0.5, 1.3
                ))

                irrigation = rng.choice(
                    IRRIGATION_SOURCES,
                    p=[0.30, 0.35, 0.10, 0.25] if is_coastal else [0.10, 0.45, 0.15, 0.30],
                )
                irrigation_boost = {"canal": 1.10, "borewell": 1.05, "tank": 1.0, "rainfed": 0.88}[irrigation]

                noise = float(rng.normal(1.0, 0.08))
                yield_kg_ha = (
                    BASELINE_YIELD_KG_HA[crop]
                    * rain_effect * heat_stress * soil_quality * irrigation_boost * noise
                )
                area_ha = float(np.clip(rng.gamma(shape=2.0, scale=120), 20, 3000))

                rows.append({
                    "district_name": district,
                    "crop_name": crop,
                    "season_name": season_name,
                    "sowing_date": sowing.date().isoformat(),
                    "harvest_date": harvest.date().isoformat(),
                    "area_hectares": round(area_ha, 1),
                    "yield_kg_per_ha": round(max(yield_kg_ha, 0), 1),
                    "irrigation_source": irrigation,
                    "total_rainfall_mm": round(float(total_rain), 1),
                    "rain_days": rain_days,
                    "avg_tmax_c": round(float(avg_tmax), 1),
                })

    df = pd.DataFrame(rows)
    logger.info("Crop season rows generated: %d", len(df))
    return df


def main() -> None:
    weather_df = simulate_weather()
    soil_df = simulate_soil_health()
    seasons_df = simulate_crop_seasons(weather_df, soil_df)

    weather_df.to_csv(RAW_DATA_DIR / "weather_daily.csv", index=False)
    soil_df.to_csv(RAW_DATA_DIR / "soil_health.csv", index=False)
    seasons_df.to_csv(RAW_DATA_DIR / "crop_seasons.csv", index=False)

    logger.info("Wrote CSVs to %s", RAW_DATA_DIR)
    logger.info(
        "Rows -> weather: %d, soil: %d, crop_seasons: %d",
        len(weather_df), len(soil_df), len(seasons_df),
    )


if __name__ == "__main__":
    main()
