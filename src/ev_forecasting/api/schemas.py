"""Pydantic schemas for FastAPI endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    environment: str
    model_loaded: bool
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    model_version: str
    stage: str
    mlflow_run_id: Optional[str]
    dataset_version: str
    dataset_hash: str
    feature_version: str
    metrics: Dict[str, Any]


class StationInfo(BaseModel):
    station_id: str
    site_id: str
    capacity_kwh: float


class ForecastRequest(BaseModel):
    station_id: str = Field(..., description="Target EV charging station identifier")
    horizon_hours: int = Field(
        default=168, ge=1, le=720, description="Forecast horizon in hours (1-720)"
    )


class ForecastItem(BaseModel):
    timestamp: str
    forecast_kwh: float
    capacity_kwh: float
    utilization_pct: float
    risk_level: str
    recommendation: str


class ScenarioItem(BaseModel):
    scenario_name: str
    capacity_multiplier: float
    peak_utilization_pct: float
    avg_utilization_pct: float
    critical_hours_count: int
    high_hours_count: int
    medium_hours_count: int
    low_hours_count: int


class ForecastResponse(BaseModel):
    prediction_id: str
    station_id: str
    horizon_hours: int
    model_name: str
    model_version: str
    dataset_version: str
    dataset_hash: str
    generated_at: str
    peak_forecast_kwh: float
    peak_utilization_pct: float
    critical_risk_hours: int
    forecast: List[ForecastItem]
    expansion_scenarios: List[ScenarioItem]


class BatchForecastRequest(BaseModel):
    station_ids: Optional[List[str]] = Field(
        default=None, description="Optional subset of stations to forecast"
    )
    horizon_hours: int = Field(default=168, ge=1, le=720)


class BatchForecastResponse(BaseModel):
    status: str = "SUCCESS"
    forecast_count: int
    output_parquet_path: str


class CapacityAnalyzeRequest(BaseModel):
    station_id: str
    custom_capacity_kwh: Optional[float] = Field(default=50.0, gt=0)
    forecast_kwh_values: List[float] = Field(..., min_length=1)


class CapacityAnalyzeResponse(BaseModel):
    station_id: str
    capacity_kwh: float
    peak_demand_kwh: float
    peak_utilization_pct: float
    critical_hours: int
    risk_level: str
    recommendation: str
