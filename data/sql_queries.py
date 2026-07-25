"""
Parameterized SQL used by the feature engineering pipeline. Kept separate
from ad-hoc queries so every consumer (training, API) reads features the
same way.
"""

# District-level seasonal summary joined with the nearest soil survey.
Q_SEASON_SUMMARY = """
SELECT * FROM v_district_season_summary
ORDER BY district_id, crop_name, sowing_date;
"""

# Daily weather for a district within a date range — used to compute
# rainfall lag / rolling features for a specific sowing window.
Q_WEATHER_WINDOW = """
SELECT obs_date, rainfall_mm, temp_max_c, temp_min_c, humidity_pct
FROM weather_daily
WHERE district_id = :district_id
  AND obs_date BETWEEN :start_date AND :end_date
ORDER BY obs_date;
"""

# District-level average rainfall by calendar month, used as a prior when
# no live weather window is available yet (e.g. forecasting).
Q_MONTHLY_RAINFALL_CLIMATOLOGY = """
SELECT
    district_id,
    EXTRACT(MONTH FROM obs_date)::INT AS month,
    AVG(rainfall_mm) AS avg_rainfall_mm,
    STDDEV(rainfall_mm) AS std_rainfall_mm
FROM weather_daily
GROUP BY district_id, EXTRACT(MONTH FROM obs_date)
ORDER BY district_id, month;
"""

# Latest soil health record on file for a district.
Q_LATEST_SOIL = """
SELECT * FROM soil_health
WHERE district_id = :district_id
ORDER BY survey_year DESC
LIMIT 1;
"""

# Historical yield distribution for a district + crop, used to sanity-check
# model predictions and to build the "advisory" confidence narrative.
Q_HISTORICAL_YIELD_STATS = """
SELECT
    crop_name,
    COUNT(*) AS n_seasons,
    AVG(yield_kg_per_ha) AS avg_yield,
    STDDEV(yield_kg_per_ha) AS std_yield,
    MIN(yield_kg_per_ha) AS min_yield,
    MAX(yield_kg_per_ha) AS max_yield
FROM crop_seasons cs
JOIN districts d ON d.district_id = cs.district_id
WHERE d.district_name = :district_name AND cs.crop_name = :crop_name
GROUP BY crop_name;
"""
