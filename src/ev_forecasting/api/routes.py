"""FastAPI router and endpoint implementations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from ev_forecasting.api.dependencies import (
    get_app_config,
    get_audit_logger,
    get_forecast_service,
)
from ev_forecasting.api.schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    CapacityAnalyzeRequest,
    CapacityAnalyzeResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfoResponse,
    StationInfo,
)
from ev_forecasting.audit.prediction_log import AuditLogger
from ev_forecasting.capacity.risk import CapacityRiskEngine
from ev_forecasting.config.settings import AppConfig
from ev_forecasting.forecasting.batch_forecast import BatchForecaster
from ev_forecasting.forecasting.forecast import ForecastService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(
    config: AppConfig = Depends(get_app_config),
    forecast_service: ForecastService = Depends(get_forecast_service),
) -> HealthResponse:
    """Check service health and model readiness."""
    return HealthResponse(
        status="healthy",
        environment=config.environment,
        model_loaded=forecast_service.model is not None,
        version="1.0.0",
    )


@router.get("/ready", tags=["Health"])
def readiness_check(
    forecast_service: ForecastService = Depends(get_forecast_service),
) -> Dict[str, str]:
    """Readiness probe for container orchestration."""
    if forecast_service.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or ready.",
        )
    return {"status": "ready"}


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info(
    forecast_service: ForecastService = Depends(get_forecast_service),
) -> ModelInfoResponse:
    """Retrieve production model provenance and metadata."""
    meta = forecast_service.metadata
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No production model registered yet.",
        )
    return ModelInfoResponse(
        model_name=meta.get("model_name", "unknown"),
        model_type=meta.get("model_type", "unknown"),
        model_version=meta.get("model_version", "v1.0"),
        stage=meta.get("stage", "Production"),
        mlflow_run_id=meta.get("mlflow_run_id"),
        dataset_version=meta.get("dataset_version", "1.0.0"),
        dataset_hash=meta.get("dataset_hash", "unknown"),
        feature_version=meta.get("feature_version", "v1.0"),
        metrics=meta.get("metrics", {}),
    )


@router.get("/stations", response_model=List[StationInfo], tags=["Stations"])
def get_available_stations(config: AppConfig = Depends(get_app_config)) -> List[StationInfo]:
    """List all available stations in the processed dataset."""
    hourly_path = Path(config.paths.processed_data_dir) / "hourly_demand.parquet"
    if not hourly_path.exists():
        # Fallback to config site definitions if data not yet processed
        return [
            StationInfo(
                station_id=f"{s.site_id}_ST_001",
                site_id=s.site_id,
                capacity_kwh=s.default_station_capacity_kwh,
            )
            for s in config.dataset.sites
        ]

    df = pd.read_parquet(hourly_path)
    stations = []
    for st_id, grp in df.groupby("station_id"):
        site_id = grp["site_id"].iloc[0]
        stations.append(
            StationInfo(
                station_id=st_id,
                site_id=site_id,
                capacity_kwh=config.capacity.default_station_capacity_kwh,
            )
        )
    return stations


@router.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def generate_forecast(
    req: ForecastRequest,
    forecast_service: ForecastService = Depends(get_forecast_service),
) -> ForecastResponse:
    """Generate multi-step hourly demand forecast and capacity analysis for a station."""
    try:
        res = forecast_service.predict_station_demand(
            station_id=req.station_id,
            horizon_hours=req.horizon_hours,
        )
        return ForecastResponse(**res)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error("Forecast generation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecasting error: {str(e)}",
        )


@router.post("/forecast/batch", response_model=BatchForecastResponse, tags=["Forecasting"])
def run_batch_forecast(
    req: BatchForecastRequest,
    config: AppConfig = Depends(get_app_config),
) -> BatchForecastResponse:
    """Trigger batch forecasting run across stations."""
    try:
        batcher = BatchForecaster(config)
        out_path = batcher.run_batch(
            station_ids=req.station_ids,
            horizon_hours=req.horizon_hours,
        )
        return BatchForecastResponse(
            status="SUCCESS",
            forecast_count=len(req.station_ids) if req.station_ids else 10,
            output_parquet_path=str(out_path),
        )
    except Exception as e:
        logger.error("Batch forecast error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/capacity/analyze", response_model=CapacityAnalyzeResponse, tags=["Capacity"])
def analyze_capacity(
    req: CapacityAnalyzeRequest,
    config: AppConfig = Depends(get_app_config),
) -> CapacityAnalyzeResponse:
    """Directly analyze capacity utilization and risk levels for provided forecast values."""
    capacity_engine = CapacityRiskEngine(config)
    cap = req.custom_capacity_kwh or config.capacity.default_station_capacity_kwh
    vals = np.array(req.forecast_kwh_values)
    peak_demand = float(np.max(vals))
    peak_util = float((peak_demand / cap) * 100.0)

    utils = (vals / cap) * 100.0
    crit_hours = int((utils >= config.capacity.thresholds.high).sum())
    risk_level = capacity_engine.classify_risk(peak_util)
    recommendation = capacity_engine.get_recommendation(risk_level)

    return CapacityAnalyzeResponse(
        station_id=req.station_id,
        capacity_kwh=cap,
        peak_demand_kwh=round(peak_demand, 3),
        peak_utilization_pct=round(peak_util, 2),
        critical_hours=crit_hours,
        risk_level=risk_level,
        recommendation=recommendation,
    )


@router.get("/audit/logs", tags=["Audit"])
def get_audit_logs(
    limit: int = Query(default=50, ge=1, le=500),
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> List[Dict[str, Any]]:
    """Query recent prediction audit logs from SQL database."""
    df_logs = audit_logger.get_recent_audit_logs(limit=limit)
    if df_logs.empty:
        return []
    return df_logs.to_dict(orient="records")
