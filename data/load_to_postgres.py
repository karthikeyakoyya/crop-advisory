"""
Creates the PostgreSQL schema (if needed) and loads the simulated CSVs from
data/raw/ into it.

Run:
    python data/load_to_postgres.py
Requires a running PostgreSQL instance reachable via the credentials in .env.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import AP_DISTRICTS, DATABASE_URL, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_engine() -> Engine:
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connected to PostgreSQL successfully")
        return engine
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
        logger.error(
            "Could not connect to PostgreSQL at %s. Is the server running and "
            "are the credentials in .env correct? Original error: %s",
            DATABASE_URL, exc,
        )
        raise


def apply_schema(engine: Engine) -> None:
    logger.info("Applying schema from %s", SCHEMA_PATH)
    sql_text = SCHEMA_PATH.read_text()
    with engine.begin() as conn:
        for statement in sql_text.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
    logger.info("Schema applied")


def load_districts(engine: Engine) -> dict[str, int]:
    with engine.begin() as conn:
        for name, (lat, lon) in AP_DISTRICTS.items():
            conn.execute(
                text(
                    """
                    INSERT INTO districts (district_name, latitude, longitude)
                    VALUES (:name, :lat, :lon)
                    ON CONFLICT (district_name) DO NOTHING
                    """
                ),
                {"name": name, "lat": lat, "lon": lon},
            )
        rows = conn.execute(text("SELECT district_id, district_name FROM districts")).fetchall()
    mapping = {name: did for did, name in rows}
    logger.info("Districts loaded: %d", len(mapping))
    return mapping


def load_weather(engine: Engine, district_map: dict[str, int]) -> None:
    path = RAW_DATA_DIR / "weather_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run data/simulate_data.py first")
    df = pd.read_csv(path)
    df["district_id"] = df["district_name"].map(district_map)
    df = df.drop(columns=["district_name"])
    logger.info("Loading %d weather rows", len(df))
    df.to_sql("weather_daily", engine, if_exists="append", index=False, method="multi", chunksize=5000)


def load_soil(engine: Engine, district_map: dict[str, int]) -> None:
    path = RAW_DATA_DIR / "soil_health.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run data/simulate_data.py first")
    df = pd.read_csv(path)
    df["district_id"] = df["district_name"].map(district_map)
    df = df.drop(columns=["district_name"])
    logger.info("Loading %d soil health rows", len(df))
    df.to_sql("soil_health", engine, if_exists="append", index=False, method="multi", chunksize=1000)


def load_crop_seasons(engine: Engine, district_map: dict[str, int]) -> None:
    path = RAW_DATA_DIR / "crop_seasons.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run data/simulate_data.py first")
    df = pd.read_csv(path)
    df["district_id"] = df["district_name"].map(district_map)
    df = df[[
        "district_id", "crop_name", "sowing_date", "harvest_date",
        "area_hectares", "yield_kg_per_ha", "irrigation_source", "season_name",
    ]]
    logger.info("Loading %d crop season rows", len(df))
    df.to_sql("crop_seasons", engine, if_exists="append", index=False, method="multi", chunksize=2000)


def main() -> None:
    engine = get_engine()
    apply_schema(engine)
    district_map = load_districts(engine)
    load_weather(engine, district_map)
    load_soil(engine, district_map)
    load_crop_seasons(engine, district_map)
    logger.info("All data loaded successfully")


if __name__ == "__main__":
    main()
