"""Causal rolling window feature engineering."""

from __future__ import annotations

import pandas as pd

from ev_forecasting.config.settings import RollingFeaturesConfig


class RollingFeatureExtractor:
    """Extracts strictly causal rolling window statistics shifted by 1 hour to prevent lookahead."""

    def __init__(self, config: RollingFeaturesConfig):
        self.config = config

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str = "energy_demand_kwh",
        group_col: str = "station_id",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Add causal rolling statistics per station."""
        df_out = df.copy()
        df_out = df_out.sort_values([group_col, timestamp_col]).reset_index(drop=True)

        # Shift target by 1 hour so the rolling window at row t only sees observations up to t-1
        shifted_target = df_out.groupby(group_col)[target_col].shift(1)

        for window in self.config.windows:
            rolling_obj = shifted_target.groupby(df_out[group_col]).rolling(
                window=window, min_periods=min(window // 4, 6)
            )

            for stat in self.config.statistics:
                col_name = f"rolling_{stat}_{window}h"
                if stat == "mean":
                    df_out[col_name] = rolling_obj.mean().reset_index(drop=True)
                elif stat == "std":
                    df_out[col_name] = rolling_obj.std().reset_index(drop=True).fillna(0.0)
                elif stat == "min":
                    df_out[col_name] = rolling_obj.min().reset_index(drop=True)
                elif stat == "max":
                    df_out[col_name] = rolling_obj.max().reset_index(drop=True)

        return df_out
