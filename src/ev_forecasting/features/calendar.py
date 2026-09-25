"""Calendar and cyclical temporal feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ev_forecasting.config.settings import CalendarFeaturesConfig


class CalendarFeatureExtractor:
    """Extracts calendar and cyclical trigonometric features from timestamps."""

    def __init__(self, config: CalendarFeaturesConfig):
        self.config = config

    def transform(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """Add calendar feature columns to dataframe."""
        df_out = df.copy()
        ts = pd.to_datetime(df_out[timestamp_col], utc=True)

        if self.config.include_hour:
            df_out["hour"] = ts.dt.hour
            if self.config.include_cyclical:
                df_out["hour_sin"] = np.sin(2 * np.pi * df_out["hour"] / 24.0)
                df_out["hour_cos"] = np.cos(2 * np.pi * df_out["hour"] / 24.0)

        if self.config.include_day_of_week:
            df_out["day_of_week"] = ts.dt.dayofweek
            if self.config.include_cyclical:
                df_out["day_of_week_sin"] = np.sin(2 * np.pi * df_out["day_of_week"] / 7.0)
                df_out["day_of_week_cos"] = np.cos(2 * np.pi * df_out["day_of_week"] / 7.0)

        if self.config.include_day_of_month:
            df_out["day_of_month"] = ts.dt.day

        if self.config.include_month:
            df_out["month"] = ts.dt.month
            if self.config.include_cyclical:
                df_out["month_sin"] = np.sin(2 * np.pi * (df_out["month"] - 1) / 12.0)
                df_out["month_cos"] = np.cos(2 * np.pi * (df_out["month"] - 1) / 12.0)

        if self.config.include_is_weekend:
            df_out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        return df_out
