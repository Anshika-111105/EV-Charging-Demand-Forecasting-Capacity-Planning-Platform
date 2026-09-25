"""Causal lag feature generation for time-series forecasting."""

from __future__ import annotations

import pandas as pd

from ev_forecasting.config.settings import LagFeaturesConfig


class LagFeatureExtractor:
    """Extracts causal historical lag features grouped by entity/station."""

    def __init__(self, config: LagFeaturesConfig):
        self.config = config

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str = "energy_demand_kwh",
        group_col: str = "station_id",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Add lag features per station. Sorted strictly chronologically."""
        df_out = df.copy()
        df_out = df_out.sort_values([group_col, timestamp_col]).reset_index(drop=True)

        for lag in self.config.hours:
            col_name = f"lag_{lag}h"
            df_out[col_name] = df_out.groupby(group_col)[target_col].shift(lag)

        return df_out
