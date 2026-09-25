"""Online Forecasting Service with audit trail and capacity risk integration."""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ev_forecasting.audit.lineage import LineageTracker
from ev_forecasting.audit.prediction_log import AuditLogger
from ev_forecasting.capacity.risk import CapacityRiskEngine
from ev_forecasting.config.settings import AppConfig
from ev_forecasting.models.base import BaseForecaster
from ev_forecasting.models.xgboost import XGBoostForecaster
from ev_forecasting.registry.model_registry import ModelRegistryManager

logger = logging.getLogger(__name__)


class ForecastService:
    """Production forecasting service orchestrating models, capacity analysis, and audit logging."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.registry_mgr = ModelRegistryManager(config)
        self.capacity_engine = CapacityRiskEngine(config)
        self.audit_logger = AuditLogger(config)
        self.lineage_tracker = LineageTracker(config)
        self.model: Optional[BaseForecaster] = None
        self.metadata: Dict[str, Any] = {}
        self._load_production_model()

    def _load_production_model(self) -> None:
        """Load production model artifact from checkpoints directory."""
        prod_meta = self.registry_mgr.get_production_model_metadata()
        if prod_meta is None:
            logger.warning("No registered production model found in registry.")
            return

        self.metadata = prod_meta
        art_path = Path(prod_meta["model_artifact_path"])
        if not art_path.exists():
            logger.warning("Production artifact does not exist at %s", art_path)
            return

        # Load forecaster
        model_name = prod_meta["model_name"]
        if "xgboost" in model_name:
            self.model = XGBoostForecaster.load(art_path)
        else:
            with open(art_path, "rb") as f:
                self.model = pickle.load(f)

        logger.info("Loaded production model '%s' (%s)", model_name, prod_meta["model_version"])

    def predict_station_demand(
        self,
        station_id: str,
        horizon_hours: int = 168,
        df_history: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Generate forecast and capacity risk for a single station."""
        if self.model is None:
            self._load_production_model()
            if self.model is None:
                raise RuntimeError("No production model loaded in ForecastService.")

        if df_history is None:
            # Load from processed hourly dataset
            hourly_path = Path(self.config.paths.processed_data_dir) / "hourly_demand.parquet"
            if not hourly_path.exists():
                raise FileNotFoundError(f"Processed dataset not found at {hourly_path}")
            df_history = pd.read_parquet(hourly_path)

        st_history = df_history[df_history["station_id"] == station_id].copy()
        if st_history.empty:
            raise ValueError(f"Station ID '{station_id}' not found in historical data.")

        # Generate forecast
        raw_forecast_df = self.model.predict(
            horizon_hours=horizon_hours,
            df_history=df_history,
        )

        # Filter to requested station
        st_forecast = raw_forecast_df[raw_forecast_df["station_id"] == station_id].copy()
        if st_forecast.empty:
            raise ValueError(f"No forecast generated for station '{station_id}'.")

        # Capacity risk analysis
        analyzed_df = self.capacity_engine.analyze_forecast(st_forecast)

        # Scenarios evaluation
        scenarios = self.capacity_engine.evaluate_expansion_scenarios(st_forecast)

        # Log to SQL audit trail
        dataset_hash = self.metadata.get("dataset_hash", "unknown")
        feature_version = self.metadata.get("feature_version", "v1.0")
        batch_id = self.audit_logger.log_forecast_batch(
            forecast_df=analyzed_df,
            model_metadata=self.metadata,
            dataset_hash=dataset_hash,
            feature_version=feature_version,
        )

        # Prepare formatted items
        forecast_items = []
        for _, row in analyzed_df.iterrows():
            forecast_items.append(
                {
                    "timestamp": str(row["timestamp"]),
                    "forecast_kwh": round(float(row["forecast_kwh"]), 3),
                    "capacity_kwh": round(float(row["capacity_kwh"]), 2),
                    "utilization_pct": round(float(row["utilization_pct"]), 2),
                    "risk_level": str(row["risk_level"]),
                    "recommendation": str(row["recommendation"]),
                }
            )

        return {
            "prediction_id": batch_id,
            "station_id": station_id,
            "horizon_hours": horizon_hours,
            "model_name": self.metadata.get("model_name", self.model.name),
            "model_version": self.metadata.get("model_version", "v1.0"),
            "dataset_version": self.config.dataset.version,
            "dataset_hash": dataset_hash,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "peak_forecast_kwh": round(float(analyzed_df["forecast_kwh"].max()), 3),
            "peak_utilization_pct": round(float(analyzed_df["utilization_pct"].max()), 2),
            "critical_risk_hours": int((analyzed_df["risk_level"] == "CRITICAL").sum()),
            "forecast": forecast_items,
            "expansion_scenarios": scenarios,
        }
