"""Capacity utilization calculation module."""

from __future__ import annotations

import pandas as pd

from ev_forecasting.capacity.capacity import StationCapacityManager


class UtilizationCalculator:
    """Calculates hourly utilization percentages from demand forecasts and station capacities."""

    def __init__(self, capacity_manager: StationCapacityManager):
        self.capacity_mgr = capacity_manager

    def calculate_utilization(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Compute utilization percentage for each forecasted hour."""
        df_out = forecast_df.copy()

        # Map station capacity
        df_out["capacity_kwh"] = df_out["station_id"].apply(self.capacity_mgr.get_station_capacity)
        df_out["utilization_pct"] = (
            (df_out["forecast_kwh"] / df_out["capacity_kwh"]) * 100.0
        ).clip(lower=0.0)

        return df_out
