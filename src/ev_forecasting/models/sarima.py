"""Seasonal SARIMA statistical forecaster."""

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
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ev_forecasting.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class SARIMAForecaster(BaseForecaster):
    """Seasonal Autoregressive Integrated Moving Average with Exogenous regressors (SARIMA)."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24),
        name: str = "sarima",
    ):
        super().__init__(name=name, model_type="statistical")
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.station_models: Dict[str, Any] = {}
        self.last_timestamps: Dict[str, pd.Timestamp] = {}
        self.station_means: Dict[str, float] = {}

    def fit(self, df_train: pd.DataFrame, **kwargs: Any) -> SARIMAForecaster:
        """Fit SARIMA models with seasonal components per station."""
        start_time = time.perf_counter()
        self.station_models = {}
        self.last_timestamps = {}
        self.station_means = {}

        df_sorted = df_train.sort_values(self.timestamp_col)
        for station_id, group in df_sorted.groupby("station_id"):
            series = group.set_index(self.timestamp_col)[self.target_col].asfreq("1h").fillna(0.0)
            self.last_timestamps[station_id] = series.index[-1]
            self.station_means[station_id] = float(series.mean())

            # Use last 336 hours (14 days) to keep SARIMA fit fast and stable
            train_series = series.iloc[-336:] if len(series) > 336 else series

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = SARIMAX(
                        train_series,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = model.fit(disp=False, maxiter=50)
                    self.station_models[station_id] = res
                except Exception as e:
                    logger.warning("SARIMA fit failed for station %s: %s", station_id, e)
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
        """Forecast SARIMA demand over specified horizon."""
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
                    logger.warning("SARIMA forecast error for %s: %s", station_id, e)
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
        """Serialize SARIMA forecaster."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> SARIMAForecaster:
        """Load serialized SARIMA forecaster."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
