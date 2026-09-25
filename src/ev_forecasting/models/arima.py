"""ARIMA statistical time-series forecaster."""

from __future__ import annotations

import logging
import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from ev_forecasting.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class ARIMAForecaster(BaseForecaster):
    """Statistical Autoregressive Integrated Moving Average (ARIMA) Forecaster."""

    def __init__(self, order: Tuple[int, int, int] = (2, 1, 1), name: str = "arima"):
        super().__init__(name=name, model_type="statistical")
        self.order = tuple(order)
        self.station_models: Dict[str, Any] = {}
        self.last_timestamps: Dict[str, pd.Timestamp] = {}
        self.station_means: Dict[str, float] = {}

    def fit(self, df_train: pd.DataFrame, **kwargs: Any) -> ARIMAForecaster:
        """Fit ARIMA model for each station."""
        start_time = time.perf_counter()
        self.station_models = {}
        self.last_timestamps = {}
        self.station_means = {}

        df_sorted = df_train.sort_values(self.timestamp_col)
        for station_id, group in df_sorted.groupby("station_id"):
            series = group.set_index(self.timestamp_col)[self.target_col].asfreq("1h").fillna(0.0)
            self.last_timestamps[station_id] = series.index[-1]
            self.station_means[station_id] = float(series.mean())

            # Fit ARIMA with warnings suppressed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # Keep training window manageable for performance (last 720 hours)
                    train_series = series.iloc[-720:] if len(series) > 720 else series
                    model = ARIMA(train_series, order=self.order)
                    res = model.fit()
                    self.station_models[station_id] = res
                except Exception as e:
                    logger.warning(
                        "ARIMA fit failed for station %s (%s). Using fallback mean.", station_id, e
                    )
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
        """Forecast future demand for each station over the horizon."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")

        results = []
        for station_id, res in self.station_models.items():
            last_ts = self.last_timestamps[station_id]
            future_ts = pd.date_range(
                start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="1h"
            )

            if res is not None:
                try:
                    forecast_res = res.forecast(steps=horizon_hours)
                    vals = np.clip(forecast_res.values, 0.0, None)
                except Exception as e:
                    logger.warning("Forecast error for %s: %s", station_id, e)
                    vals = np.full(horizon_hours, self.station_means.get(station_id, 0.0))
            else:
                vals = np.full(horizon_hours, self.station_means.get(station_id, 0.0))

            st_df = pd.DataFrame(
                {
                    "station_id": station_id,
                    "timestamp": future_ts,
                    "forecast_kwh": vals,
                    "model_name": self.name,
                }
            )
            results.append(st_df)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def save(self, filepath: Path) -> None:
        """Serialize ARIMA forecaster state."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Store metadata and results
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> ARIMAForecaster:
        """Load serialized ARIMA forecaster."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
