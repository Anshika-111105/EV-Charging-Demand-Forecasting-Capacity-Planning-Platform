"""Facebook Prophet time-series forecaster."""

from __future__ import annotations

import logging
import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

from ev_forecasting.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet model for multi-station EV demand forecasting."""

    def __init__(
        self,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        name: str = "prophet",
    ):
        super().__init__(name=name, model_type="prophet")
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.station_models: Dict[str, Prophet] = {}
        self.last_timestamps: Dict[str, pd.Timestamp] = {}

    def fit(self, df_train: pd.DataFrame, **kwargs: Any) -> ProphetForecaster:
        """Fit Prophet models per station."""
        start_time = time.perf_counter()
        self.station_models = {}
        self.last_timestamps = {}

        df_sorted = df_train.sort_values(self.timestamp_col)
        for station_id, group in df_sorted.groupby("station_id"):
            prophet_df = pd.DataFrame(
                {
                    "ds": group[self.timestamp_col].dt.tz_localize(None),
                    "y": group[self.target_col].values,
                }
            ).dropna()

            self.last_timestamps[station_id] = group[self.timestamp_col].iloc[-1]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    m = Prophet(
                        daily_seasonality=self.daily_seasonality,
                        weekly_seasonality=self.weekly_seasonality,
                        yearly_seasonality=self.yearly_seasonality,
                    )
                    m.fit(prophet_df)
                    self.station_models[station_id] = m
                except Exception as e:
                    logger.warning("Prophet fitting failed for station %s: %s", station_id, e)
                    self.station_models[station_id] = None

        self.is_fitted = True
        self.fitted_at = datetime.now(timezone.utc).isoformat()
        self.training_duration_seconds = time.perf_counter() - start_time
        return self

    def predict(
        self,
        horizon_hours: int,
        df_history: Optional[pd.DataFrame] = None,
        df_future: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate Prophet predictions over future horizon."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")

        results = []
        for station_id, m in self.station_models.items():
            last_ts = self.last_timestamps[station_id]
            future_ts_utc = pd.date_range(
                start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="1h", tz="UTC"
            )

            if m is not None:
                future_df = pd.DataFrame({"ds": future_ts_utc.tz_localize(None)})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast = m.predict(future_df)
                yhat = np.clip(forecast["yhat"].values, 0.0, None)
            else:
                yhat = np.zeros(horizon_hours)

            st_df = pd.DataFrame(
                {
                    "station_id": station_id,
                    "timestamp": future_ts_utc,
                    "forecast_kwh": yhat,
                    "model_name": self.name,
                }
            )
            results.append(st_df)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def save(self, filepath: Path) -> None:
        """Serialize Prophet model state."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> ProphetForecaster:
        """Load serialized Prophet forecaster."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
