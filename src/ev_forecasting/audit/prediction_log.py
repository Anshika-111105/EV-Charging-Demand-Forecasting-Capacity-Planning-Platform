"""SQL Prediction Audit Trail and Lineage Logger."""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ev_forecasting.config.settings import AppConfig

logger = logging.getLogger(__name__)

Base = declarative_base()


class PredictionAuditRecord(Base):
    """SQLAlchemy model for recording every production forecast request and output."""

    __tablename__ = "prediction_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(64), index=True, nullable=False)
    station_id = Column(String(64), index=True, nullable=False)
    request_timestamp = Column(DateTime(timezone=True), nullable=False)
    forecast_timestamp = Column(DateTime(timezone=True), nullable=False)
    forecast_kwh = Column(Float, nullable=False)
    utilization_pct = Column(Float, nullable=True)
    risk_level = Column(String(32), nullable=True)
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(64), nullable=False)
    mlflow_run_id = Column(String(64), nullable=True)
    dataset_version = Column(String(32), nullable=False)
    dataset_hash = Column(String(64), nullable=False)
    feature_version = Column(String(32), nullable=False)
    input_hash = Column(String(64), nullable=False)
    status = Column(String(32), default="SUCCESS")


class AuditLogger:
    """Logs predictions and metadata to SQL database (PostgreSQL with SQLite fallback)."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.db_url = config.database_url
        try:
            from sqlalchemy.pool import StaticPool

            if ":memory:" in self.db_url:
                self.engine = create_engine(
                    self.db_url,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                    echo=False,
                )
            else:
                self.engine = create_engine(self.db_url, echo=False)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info("Initialized audit database at %s", self.db_url.split("@")[-1])
        except Exception as e:
            logger.warning(
                "Could not connect to database %s: %s. Falling back to SQLite.", self.db_url, e
            )
            self.engine = create_engine("sqlite:///./audit.db", echo=False)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)

    def log_forecast_batch(
        self,
        forecast_df: pd.DataFrame,
        model_metadata: Dict[str, Any],
        dataset_hash: str,
        feature_version: str,
    ) -> str:
        """Log a batch of forecast predictions with complete audit trail."""
        prediction_batch_id = str(uuid.uuid4())
        req_time = datetime.now(timezone.utc)

        # Compute deterministic hash of inputs
        input_hash = hashlib.sha256(
            f"{forecast_df['station_id'].tolist()}_{len(forecast_df)}".encode("utf-8")
        ).hexdigest()

        records = []
        for _, row in forecast_df.iterrows():
            ts = pd.to_datetime(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            record = PredictionAuditRecord(
                prediction_id=prediction_batch_id,
                station_id=str(row["station_id"]),
                request_timestamp=req_time,
                forecast_timestamp=ts,
                forecast_kwh=float(row["forecast_kwh"]),
                utilization_pct=float(row.get("utilization_pct", 0.0)),
                risk_level=str(row.get("risk_level", "UNKNOWN")),
                model_name=str(model_metadata.get("model_name", "production_model")),
                model_version=str(model_metadata.get("model_version", "v1.0")),
                mlflow_run_id=str(model_metadata.get("mlflow_run_id", "prod")),
                dataset_version=self.config.dataset.version,
                dataset_hash=dataset_hash,
                feature_version=feature_version,
                input_hash=input_hash,
                status="SUCCESS",
            )
            records.append(record)

        session = self.SessionLocal()
        try:
            session.add_all(records)
            session.commit()
            logger.info(
                "Logged %d predictions under Batch ID %s", len(records), prediction_batch_id
            )
        except Exception as e:
            session.rollback()
            logger.error("Failed to log predictions to audit database: %s", e)
        finally:
            session.close()

        return prediction_batch_id

    def get_recent_audit_logs(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve recent prediction audit records as a dataframe."""
        session = self.SessionLocal()
        try:
            records = (
                session.query(PredictionAuditRecord)
                .order_by(PredictionAuditRecord.id.desc())
                .limit(limit)
                .all()
            )
            if not records:
                return pd.DataFrame()
            data = [
                {
                    "id": r.id,
                    "prediction_id": r.prediction_id,
                    "station_id": r.station_id,
                    "request_timestamp": r.request_timestamp,
                    "forecast_timestamp": r.forecast_timestamp,
                    "forecast_kwh": r.forecast_kwh,
                    "utilization_pct": r.utilization_pct,
                    "risk_level": r.risk_level,
                    "model_name": r.model_name,
                    "model_version": r.model_version,
                    "dataset_hash": r.dataset_hash[:8] + "...",
                    "status": r.status,
                }
                for r in records
            ]
            return pd.DataFrame(data)
        finally:
            session.close()
