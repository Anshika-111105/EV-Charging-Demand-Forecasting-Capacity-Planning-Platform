"""Seasonal Naive baseline forecaster for multi-station demand."""

from __future__ import annotations

import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ev_forecasting.models.base import BaseForecaster


class SeasonalNaiveForecaster(BaseForecaster):
    """Predicts future demand using the historical value from exactly s periods ago."""

    def __init__(self, seasonality: int = 168, name: Optional[str] = None):
        model_name = name or f"seasonal_naive_{seasonality}"
        super().__init__(name=model_name, model_type="baseline")
        self.seasonality = seasonality
        self.station_history: Dict[str, pd.Series] = {}
        self.last_timestamps: Dict[str, pd.Timestamp] = {}

    def fit(self, df_train: pd.DataFrame, **kwargs: Any) -> SeasonalNaiveForecaster:
        """Fit seasonal naive by recording the latest seasonal pattern per station."""
        start_time = time.perf_counter()
        self.station_history = {}
        self.last_timestamps = {}

        df_sorted = df_train.sort_values([self.timestamp_col])
        for station_id, group in df_sorted.groupby("station_id"):
            series = group.set_index(self.timestamp_col)[self.target_col]
            # Store at least the last seasonality steps
            self.station_history[station_id] = series.iloc[-self.seasonality :]
            self.last_timestamps[station_id] = series.index[-1]

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
        """Generate seasonal naive forecasts for all stations across the horizon."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")

        history_map = self.station_history
        last_ts_map = self.last_timestamps

        if df_history is not None and not df_history.empty:
            history_map = {}
            last_ts_map = {}
            for station_id, group in df_history.groupby("station_id"):
                series = group.set_index(self.timestamp_col)[self.target_col]
                history_map[station_id] = series.iloc[-self.seasonality :]
                last_ts_map[station_id] = series.index[-1]

        results = []
        for station_id, hist_series in history_map.items():
            last_ts = last_ts_map[station_id]
            future_ts = pd.date_range(
                start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="1h"
            )

            hist_vals = hist_series.values
            n_hist = len(hist_vals)
            if n_hist == 0:
                forecast_vals = np.zeros(horizon_hours)
            else:
                # Tile the seasonal pattern
                repeats = (horizon_hours // n_hist) + 2
                tiled = np.tile(hist_vals, repeats)
                forecast_vals = tiled[:horizon_hours]

            st_df = pd.DataFrame(
                {
                    "station_id": station_id,
                    "timestamp": future_ts,
                    "forecast_kwh": np.clip(forecast_vals, 0.0, None),
                    "model_name": self.name,
                }
            )
            results.append(st_df)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def save(self, filepath: Path) -> None:
        """Serialize baseline forecaster to pickle file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> SeasonalNaiveForecaster:
        """Load baseline forecaster from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
