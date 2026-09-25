"""Global Multi-Station XGBoost Forecaster."""

from __future__ import annotations

import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ev_forecasting.config.settings import XGBoostModelConfig
from ev_forecasting.features.pipeline import FeaturePipeline
from ev_forecasting.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """Global feature-based gradient boosted trees forecaster for multi-station EV demand."""

    def __init__(
        self,
        config: Optional[XGBoostModelConfig] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
        name: str = "xgboost_global",
    ):
        super().__init__(name=name, model_type="gradient_boosting")
        self.config = config or XGBoostModelConfig()
        self.feature_pipeline = feature_pipeline
        self.model: Optional[XGBRegressor] = None
        self.feature_columns: List[str] = []
        self.train_history: Optional[pd.DataFrame] = None

    def fit(
        self,
        df_train: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> XGBoostForecaster:
        """Fit global XGBoost model across all stations."""
        start_time = time.perf_counter()

        # Extract features if not already in dataframe
        if self.feature_pipeline is not None and (
            feature_columns is None or "lag_1h" not in df_train.columns
        ):
            df_feat, feat_cols = self.feature_pipeline.fit_transform(df_train)
            self.feature_columns = feat_cols
        else:
            df_feat = df_train
            self.feature_columns = feature_columns or [
                c
                for c in df_feat.columns
                if c
                not in [
                    "timestamp",
                    "station_id",
                    "site_id",
                    self.target_col,
                    "session_count",
                    "active_sessions",
                ]
            ]

        # Drop rows with NaN in target, and fill initial unobserved lag/rolling NaNs with 0.0
        clean_df = df_feat.dropna(subset=[self.target_col]).copy()
        if clean_df.empty:
            raise ValueError("No valid rows remaining after feature construction.")

        X = clean_df[self.feature_columns].fillna(0.0)
        y = clean_df[self.target_col]

        self.model = XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.model.fit(X, y)

        self.train_history = df_train.sort_values([self.timestamp_col]).copy()
        self.is_fitted = True
        self.fitted_at = datetime.now(timezone.utc).isoformat()
        self.training_duration_seconds = time.perf_counter() - start_time
        logger.info(
            "Fitted XGBoost model on %d samples (%d features) in %.2fs",
            len(clean_df),
            len(self.feature_columns),
            self.training_duration_seconds,
        )
        return self

    def predict(
        self,
        horizon_hours: int,
        df_history: Optional[pd.DataFrame] = None,
        df_future: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate iterative recursive predictions forward for all stations."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted.")

        history_df = df_history if df_history is not None else self.train_history
        if history_df is None or history_df.empty:
            raise ValueError("No historical data provided for recursive feature generation.")

        # If full feature dataframe for future is already provided (e.g. In backtesting test slice)
        if df_future is not None and all(c in df_future.columns for c in self.feature_columns):
            preds = self.model.predict(df_future[self.feature_columns])
            out = df_future[["station_id", "timestamp"]].copy()
            out["forecast_kwh"] = np.clip(preds, 0.0, None)
            out["model_name"] = self.name
            return out

        # Iterative recursive forecasting step-by-step
        current_history = history_df.copy()
        all_forecasts = []
        stations = current_history["station_id"].unique().tolist()
        last_global_ts = current_history[self.timestamp_col].max()

        for step in range(1, horizon_hours + 1):
            next_ts = last_global_ts + pd.Timedelta(hours=step)

            # Create placeholder rows for next timestamp for each station
            step_rows = []
            for st_id in stations:
                site_id = current_history[current_history["station_id"] == st_id]["site_id"].iloc[0]
                step_rows.append(
                    {
                        "station_id": st_id,
                        "site_id": site_id,
                        "timestamp": next_ts,
                        self.target_col: np.nan,
                        "session_count": 0,
                        "active_sessions": 0,
                    }
                )
            step_df = pd.DataFrame(step_rows)
            combined = pd.concat([current_history, step_df], ignore_index=True)

            if self.feature_pipeline is not None:
                feat_df = self.feature_pipeline.transform(combined)
            else:
                feat_df = combined

            target_slice = feat_df[feat_df[self.timestamp_col] == next_ts]
            # Fill missing feature columns if any
            for c in self.feature_columns:
                if c not in target_slice.columns:
                    target_slice[c] = 0.0

            X_step = target_slice[self.feature_columns].fillna(0.0)
            step_preds = np.clip(self.model.predict(X_step), 0.0, None)

            for idx, st_id in enumerate(stations):
                pred_val = float(step_preds[idx])
                all_forecasts.append(
                    {
                        "station_id": st_id,
                        "timestamp": next_ts,
                        "forecast_kwh": pred_val,
                        "model_name": self.name,
                    }
                )
                # Update current_history so subsequent lags/rolling use previous predictions
                step_df.loc[step_df["station_id"] == st_id, self.target_col] = pred_val

            current_history = pd.concat([current_history, step_df], ignore_index=True)

        return pd.DataFrame(all_forecasts)

    def save(self, filepath: Path) -> None:
        """Serialize XGBoost model and pipeline."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "config": self.config,
                    "feature_pipeline": self.feature_pipeline,
                    "feature_columns": self.feature_columns,
                    "metadata": self.get_metadata(),
                },
                f,
            )

    @classmethod
    def load(cls, filepath: Path) -> XGBoostForecaster:
        """Load serialized XGBoost forecaster."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        obj = cls(config=data.get("config"), feature_pipeline=data.get("feature_pipeline"))
        obj.model = data.get("model")
        obj.feature_columns = data.get("feature_columns", [])
        obj.is_fitted = True
        return obj
