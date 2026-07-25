"""
Central configuration for the Crop Yield & Irrigation Advisory System.

All scripts (data pipeline, modeling, API) import from here so that
connection strings, paths, and constants live in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if present. Safe no-op if the file is missing.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set. "
            f"Copy config/.env.example to .env and fill it in."
        )
    return value


# --- Database -------------------------------------------------------------
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "crop_advisory")
POSTGRES_USER = os.getenv("POSTGRES_USER", "crop_advisory_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "change_me")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

# --- MLflow -----------------------------------------------------------------
# SQLite backend (recommended by MLflow over the legacy filesystem store)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "crop_yield_advisory")

# --- API ---------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Reproducibility -----------------------------------------------------
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# --- Domain constants -----------------------------------------------------
# Districts of undivided Andhra Pradesh region covered by the simulated
# rainfall / yield dataset. Coordinates are approximate district centroids.
AP_DISTRICTS = {
    "Visakhapatnam": (17.6868, 83.2185),
    "Vizianagaram": (18.1067, 83.3956),
    "Srikakulam": (18.2949, 83.8938),
    "East Godavari": (17.0005, 82.2416),
    "West Godavari": (16.9107, 81.3399),
    "Krishna": (16.5062, 80.6480),
    "Guntur": (16.3067, 80.4365),
    "Prakasam": (15.5088, 79.9218),
    "Nellore": (14.4426, 79.9865),
    "Kurnool": (15.8281, 78.0373),
    "Anantapur": (14.6819, 77.6006),
    "Chittoor": (13.2172, 79.1003),
    "YSR Kadapa": (14.4674, 78.8241),
}

CROPS = ["Rice", "Groundnut", "Cotton", "Chilli", "Maize", "Sugarcane"]

# Approximate crop growth-cycle length in days, used to derive growth stage
CROP_CYCLE_DAYS = {
    "Rice": 120,
    "Groundnut": 110,
    "Cotton": 160,
    "Chilli": 150,
    "Maize": 100,
    "Sugarcane": 330,
}

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
