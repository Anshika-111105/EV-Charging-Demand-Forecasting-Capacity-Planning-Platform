"""Data quality validation gates and schema definitions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from ev_forecasting.config.settings import AppConfig

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data quality validation gates fail."""

    pass


class SessionQualityReport(BaseModel):
    total_sessions: int
    valid_sessions: int
    invalid_sessions: int
    null_counts: Dict[str, int]
    negative_energy_count: int
    negative_duration_count: int
    duplicate_session_ids: int
    outliers_count: int
    status: str = "PASS"
    critical_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class HourlyQualityReport(BaseModel):
    total_records: int
    unique_stations: int
    unique_sites: int
    time_range_start: Optional[str]
    time_range_end: Optional[str]
    null_counts: Dict[str, int]
    duplicate_station_timestamps: int
    negative_energy_records: int
    unusually_high_spikes: int
    missing_hours_count: int
    status: str = "PASS"
    critical_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DataQualityValidator:
    """Validates raw sessions and aggregated hourly time-series against quality gates."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.reports_dir = Path(config.paths.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def validate_sessions_dataframe(
        self, df: pd.DataFrame, raise_on_failure: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate session-level dataframe against schema and business logic."""
        if df.empty:
            raise DataValidationError("Session dataframe is empty.")

        required_cols = [
            "session_id",
            "station_id",
            "site_id",
            "start_time",
            "end_time",
            "duration_hours",
            "kwh_delivered",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns in session data: {missing_cols}")

        total_sessions = len(df)
        null_counts = {col: int(df[col].isna().sum()) for col in required_cols}

        # Quality metrics
        negative_energy = int((df["kwh_delivered"] < 0).sum())
        negative_duration = int((df["duration_hours"] <= 0).sum())
        duplicates = int(df["session_id"].duplicated().sum())
        # An EV session exceeding 200 kWh on AC Level 2 charger is anomalous
        outliers = int((df["kwh_delivered"] > 200.0).sum())

        critical_errors: List[str] = []
        warnings: List[str] = []

        if null_counts["session_id"] > 0:
            critical_errors.append(
                f"Found {null_counts['session_id']} records with null session_id"
            )
        if null_counts["station_id"] > 0:
            critical_errors.append(
                f"Found {null_counts['station_id']} records with null station_id"
            )
        if negative_duration > 0:
            critical_errors.append(f"Found {negative_duration} records with non-positive duration")
        if negative_energy > 0:
            critical_errors.append(f"Found {negative_energy} records with negative energy")
        if duplicates > 0:
            critical_errors.append(f"Found {duplicates} duplicate session_id records")

        if outliers > 0:
            warnings.append(f"Found {outliers} sessions with unusually high energy (>200 kWh)")

        status = "FAIL" if critical_errors else "PASS"

        report = SessionQualityReport(
            total_sessions=total_sessions,
            valid_sessions=total_sessions - len(critical_errors),
            invalid_sessions=len(critical_errors),
            null_counts=null_counts,
            negative_energy_count=negative_energy,
            negative_duration_count=negative_duration,
            duplicate_session_ids=duplicates,
            outliers_count=outliers,
            status=status,
            critical_errors=critical_errors,
            warnings=warnings,
        )

        report_dict = report.model_dump()
        if raise_on_failure and status == "FAIL":
            raise DataValidationError(f"Session data quality validation failed: {critical_errors}")

        return (status == "PASS", report_dict)

    def validate_hourly_dataframe(
        self, df: pd.DataFrame, raise_on_failure: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate aggregated hourly station-level time series dataframe."""
        if df.empty:
            raise DataValidationError("Hourly time-series dataframe is empty.")

        required_cols = ["station_id", "site_id", "timestamp", "energy_demand_kwh"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns in hourly dataset: {missing_cols}")

        total_records = len(df)
        unique_stations = int(df["station_id"].nunique())
        unique_sites = int(df["site_id"].nunique())
        null_counts = {col: int(df[col].isna().sum()) for col in required_cols}

        # Check duplicate (station_id, timestamp)
        dup_station_ts = int(df.duplicated(subset=["station_id", "timestamp"]).sum())
        negative_energy = int((df["energy_demand_kwh"] < 0).sum())
        # Single station hourly demand > 150 kWh on level 2 is extreme
        high_spikes = int((df["energy_demand_kwh"] > 150.0).sum())

        critical_errors: List[str] = []
        warnings: List[str] = []

        if dup_station_ts > 0:
            critical_errors.append(
                f"Found {dup_station_ts} duplicate (station_id, timestamp) records"
            )
        if null_counts["timestamp"] > 0 or null_counts["energy_demand_kwh"] > 0:
            critical_errors.append(f"Found nulls in critical hourly fields: {null_counts}")
        if negative_energy > 0:
            critical_errors.append(f"Found {negative_energy} records with negative energy demand")

        if high_spikes > 0:
            warnings.append(f"Found {high_spikes} hourly demand readings exceeding 150 kWh")

        start_ts = str(df["timestamp"].min()) if not df["timestamp"].isna().all() else None
        end_ts = str(df["timestamp"].max()) if not df["timestamp"].isna().all() else None

        status = "FAIL" if critical_errors else "PASS"

        report = HourlyQualityReport(
            total_records=total_records,
            unique_stations=unique_stations,
            unique_sites=unique_sites,
            time_range_start=start_ts,
            time_range_end=end_ts,
            null_counts=null_counts,
            duplicate_station_timestamps=dup_station_ts,
            negative_energy_records=negative_energy,
            unusually_high_spikes=high_spikes,
            missing_hours_count=0,
            status=status,
            critical_errors=critical_errors,
            warnings=warnings,
        )

        report_dict = report.model_dump()
        report_path = self.reports_dir / "data_quality_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "validated_at": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "hourly_report": report_dict,
                },
                f,
                indent=2,
            )

        if raise_on_failure and status == "FAIL":
            raise DataValidationError(f"Hourly data quality validation failed: {critical_errors}")

        return (status == "PASS", report_dict)
