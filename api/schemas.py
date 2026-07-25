"""Pydantic request/response models for the advisory API."""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SoilOverrides(BaseModel):
    """Optional farmer-supplied soil readings. Any field left out falls back
    to the most recent district-level soil health survey on file."""
    soil_ph: Optional[float] = Field(None, ge=3.5, le=10.0)
    organic_carbon_pct: Optional[float] = Field(None, ge=0, le=3)
    nitrogen_kg_ha: Optional[float] = Field(None, ge=0, le=1000)
    phosphorus_kg_ha: Optional[float] = Field(None, ge=0, le=200)
    potassium_kg_ha: Optional[float] = Field(None, ge=0, le=1000)
    soil_moisture_pct: Optional[float] = Field(None, ge=0, le=100)


class AdvisoryRequest(BaseModel):
    district: str = Field(..., description="Andhra Pradesh district name, e.g. 'Guntur'")
    crop: str = Field(..., description="Crop name, e.g. 'Rice'")
    season_name: str = Field("kharif", description="'kharif' or 'rabi'")
    irrigation_source: str = Field("borewell", description="'canal', 'borewell', 'tank', or 'rainfed'")
    area_hectares: float = Field(..., gt=0, le=5000, description="Land area under this crop, in hectares")
    sowing_date: date = Field(..., description="Planned or actual sowing date")
    soil: Optional[SoilOverrides] = None

    @field_validator("season_name")
    @classmethod
    def validate_season(cls, v: str) -> str:
        allowed = {"kharif", "rabi"}
        if v.lower() not in allowed:
            raise ValueError(f"season_name must be one of {allowed}")
        return v.lower()

    @field_validator("irrigation_source")
    @classmethod
    def validate_irrigation(cls, v: str) -> str:
        allowed = {"canal", "borewell", "tank", "rainfed"}
        if v.lower() not in allowed:
            raise ValueError(f"irrigation_source must be one of {allowed}")
        return v.lower()


class ConfidenceInterval(BaseModel):
    low: float
    high: float


class IrrigationStage(BaseModel):
    stage: str
    recommended_start_date: str
    irrigation_needed_mm: float
    note: str


class RiskFlag(BaseModel):
    type: str
    severity: str
    message: str


class Reasoning(BaseModel):
    model: str
    key_drivers_considered: list[str]
    note: str


class AdvisoryResponse(BaseModel):
    district: str
    crop: str
    sowing_date: str
    harvest_date: str
    predicted_yield_kg_per_ha: float
    confidence_interval_80pct: ConfidenceInterval
    expected_total_rainfall_mm: float
    irrigation_schedule: list[IrrigationStage]
    risk_flags: list[RiskFlag]
    reasoning: Reasoning
    disclaimer: str


class DistrictsResponse(BaseModel):
    districts: list[str]


class CropsResponse(BaseModel):
    crops: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
