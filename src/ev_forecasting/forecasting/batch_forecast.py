"""Batch forecasting job for multi-station demand and capacity generation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ev_forecasting.config.settings import AppConfig
from ev_forecasting.forecasting.forecast import ForecastService

logger = logging.getLogger(__name__)


class BatchForecaster:
    """Executes scheduled batch forecasting across all stations."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.forecast_service = ForecastService(config)
        self.output_dir = Path(config.paths.processed_data_dir) / "forecasts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_batch(
        self,
        station_ids: Optional[List[str]] = None,
        horizon_hours: int = 168,
    ) -> Path:
        """Generate and save forecasts for all eligible stations."""
        hourly_path = Path(self.config.paths.processed_data_dir) / "hourly_demand.parquet"
        if not hourly_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {hourly_path}")

        df_history = pd.read_parquet(hourly_path)
        all_stations = df_history["station_id"].unique().tolist()
        target_stations = station_ids or all_stations

        logger.info(
            "Starting batch forecasting for %d stations (horizon=%dh)",
            len(target_stations),
            horizon_hours,
        )

        batch_results = []
        for st_id in target_stations:
            try:
                res = self.forecast_service.predict_station_demand(
                    station_id=st_id,
                    horizon_hours=horizon_hours,
                    df_history=df_history,
                )
                for item in res["forecast"]:
                    batch_results.append(
                        {
                            "prediction_id": res["prediction_id"],
                            "station_id": st_id,
                            "timestamp": item["timestamp"],
                            "forecast_kwh": item["forecast_kwh"],
                            "capacity_kwh": item["capacity_kwh"],
                            "utilization_pct": item["utilization_pct"],
                            "risk_level": item["risk_level"],
                            "model_name": res["model_name"],
                            "model_version": res["model_version"],
                            "generated_at": res["generated_at"],
                        }
                    )
            except Exception as e:
                logger.error("Failed batch forecast for station %s: %s", st_id, e)

        batch_df = pd.DataFrame(batch_results)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_file = self.output_dir / f"batch_forecast_{timestamp_str}.parquet"
        batch_df.to_parquet(out_file, index=False)
        logger.info(
            "Batch forecasting complete. Saved %d predictions to %s", len(batch_df), out_file
        )
        return out_file
