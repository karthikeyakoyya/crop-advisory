"""
FastAPI service for the Crop Yield & Irrigation Advisory System.

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
(from the project root, with the venv activated)

Docs available at /docs once running.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import AP_DISTRICTS, CROPS, LOG_LEVEL
from models.predict import YieldAdvisor
from api.schemas import (
    AdvisoryRequest, AdvisoryResponse, CropsResponse, DistrictsResponse, HealthResponse,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crop Yield & Irrigation Advisory API",
    description=(
        "Decision-support API for smallholder farmers and agricultural extension "
        "officers in Andhra Pradesh. Predictions are advisory only — see the "
        "`disclaimer` field on every response."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

_advisor: YieldAdvisor | None = None


def get_advisor() -> YieldAdvisor:
    global _advisor
    if _advisor is None:
        logger.info("Loading model artifacts (first request)")
        _advisor = YieldAdvisor()
    return _advisor


@app.on_event("startup")
def _startup() -> None:
    # Fail fast and loudly if model artifacts are missing, rather than
    # letting the first user request surface a confusing 500.
    try:
        get_advisor()
        logger.info("Model artifacts loaded successfully at startup")
    except RuntimeError as exc:
        logger.error("Startup check failed: %s", exc)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_advisor()
        return HealthResponse(status="ok", model_loaded=True)
    except RuntimeError:
        return HealthResponse(status="degraded", model_loaded=False)


@app.get("/api/districts", response_model=DistrictsResponse)
def list_districts() -> DistrictsResponse:
    return DistrictsResponse(districts=sorted(AP_DISTRICTS.keys()))


@app.get("/api/crops", response_model=CropsResponse)
def list_crops() -> CropsResponse:
    return CropsResponse(crops=CROPS)


@app.post("/api/advisory", response_model=AdvisoryResponse)
def get_advisory(request: AdvisoryRequest) -> AdvisoryResponse:
    if request.district not in AP_DISTRICTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown district '{request.district}'. See /api/districts for valid options.",
        )
    try:
        advisor = get_advisor()
        soil_overrides = request.soil.model_dump() if request.soil else None
        result = advisor.predict(
            district=request.district,
            crop=request.crop,
            season_name=request.season_name,
            irrigation_source=request.irrigation_source,
            area_hectares=request.area_hectares,
            sowing_date=request.sowing_date,
            soil_overrides=soil_overrides,
        )
        return AdvisoryResponse(**result)
    except ValueError as exc:
        logger.warning("Bad request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Model not ready: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error handling advisory request")
        raise HTTPException(status_code=500, detail="Internal error generating advisory") from exc


# Serve the frontend as static files at the root, so the whole app runs off
# a single `uvicorn` process during local development.
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
