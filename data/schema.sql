-- ---------------------------------------------------------------------------
-- Crop Yield & Irrigation Advisory System — PostgreSQL schema
-- Run with: psql -U <user> -d crop_advisory -f data/schema.sql
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS districts (
    district_id     SERIAL PRIMARY KEY,
    district_name   TEXT UNIQUE NOT NULL,
    latitude        NUMERIC(8, 4) NOT NULL,
    longitude       NUMERIC(8, 4) NOT NULL
);

CREATE TABLE IF NOT EXISTS weather_daily (
    weather_id      BIGSERIAL PRIMARY KEY,
    district_id     INTEGER NOT NULL REFERENCES districts(district_id),
    obs_date        DATE NOT NULL,
    rainfall_mm     NUMERIC(6, 2) NOT NULL DEFAULT 0,
    temp_max_c      NUMERIC(5, 2) NOT NULL,
    temp_min_c      NUMERIC(5, 2) NOT NULL,
    humidity_pct    NUMERIC(5, 2) NOT NULL,
    UNIQUE (district_id, obs_date)
);
CREATE INDEX IF NOT EXISTS idx_weather_district_date
    ON weather_daily (district_id, obs_date);

CREATE TABLE IF NOT EXISTS soil_health (
    soil_id             SERIAL PRIMARY KEY,
    district_id         INTEGER NOT NULL REFERENCES districts(district_id),
    survey_year         INTEGER NOT NULL,
    soil_ph             NUMERIC(4, 2) NOT NULL,
    organic_carbon_pct  NUMERIC(5, 3) NOT NULL,
    nitrogen_kg_ha      NUMERIC(7, 2) NOT NULL,
    phosphorus_kg_ha    NUMERIC(7, 2) NOT NULL,
    potassium_kg_ha     NUMERIC(7, 2) NOT NULL,
    soil_moisture_pct   NUMERIC(5, 2) NOT NULL,
    UNIQUE (district_id, survey_year)
);

CREATE TABLE IF NOT EXISTS crop_seasons (
    season_id       BIGSERIAL PRIMARY KEY,
    district_id     INTEGER NOT NULL REFERENCES districts(district_id),
    crop_name       TEXT NOT NULL,
    sowing_date     DATE NOT NULL,
    harvest_date    DATE NOT NULL,
    area_hectares   NUMERIC(10, 2) NOT NULL,
    yield_kg_per_ha NUMERIC(10, 2) NOT NULL,
    irrigation_source TEXT NOT NULL, -- 'canal', 'borewell', 'rainfed', 'tank'
    season_name     TEXT NOT NULL     -- 'kharif', 'rabi', 'zaid'
);
CREATE INDEX IF NOT EXISTS idx_crop_seasons_lookup
    ON crop_seasons (district_id, crop_name, sowing_date);

-- View: district-level seasonal summary used as a modeling feature source
CREATE OR REPLACE VIEW v_district_season_summary AS
SELECT
    cs.district_id,
    d.district_name,
    cs.crop_name,
    cs.season_name,
    EXTRACT(YEAR FROM cs.sowing_date)::INT AS sowing_year,
    cs.sowing_date,
    cs.harvest_date,
    cs.area_hectares,
    cs.yield_kg_per_ha,
    cs.irrigation_source,
    sh.soil_ph,
    sh.organic_carbon_pct,
    sh.nitrogen_kg_ha,
    sh.phosphorus_kg_ha,
    sh.potassium_kg_ha,
    sh.soil_moisture_pct
FROM crop_seasons cs
JOIN districts d ON d.district_id = cs.district_id
LEFT JOIN LATERAL (
    SELECT * FROM soil_health sh
    WHERE sh.district_id = cs.district_id
    ORDER BY ABS(sh.survey_year - EXTRACT(YEAR FROM cs.sowing_date))
    LIMIT 1
) sh ON TRUE;
