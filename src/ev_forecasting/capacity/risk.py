"""Capacity risk classification and what-if expansion scenarios."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from ev_forecasting.capacity.capacity import StationCapacityManager
from ev_forecasting.capacity.utilization import UtilizationCalculator
from ev_forecasting.config.settings import AppConfig

logger = logging.getLogger(__name__)


class CapacityRiskEngine:
    """Classifies capacity bottlenecks into risk levels and evaluates what-if upgrade scenarios."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.capacity_mgr = StationCapacityManager(config)
        self.util_calc = UtilizationCalculator(self.capacity_mgr)
        self.thresholds = config.capacity.thresholds

    def classify_risk(self, utilization_pct: float) -> str:
        """Deterministic risk tier mapping."""
        if utilization_pct >= self.thresholds.high:
            return "CRITICAL"
        elif utilization_pct >= self.thresholds.medium:
            return "HIGH"
        elif utilization_pct >= self.thresholds.low:
            return "MEDIUM"
        return "LOW"

    def get_recommendation(self, risk_level: str) -> str:
        """Generate actionable engineering recommendation for risk tier."""
        if risk_level == "CRITICAL":
            return "Immediate capacity expansion required or dynamic grid power curtailment."
        elif risk_level == "HIGH":
            return "High queue risk: Schedule smart charging load redistribution."
        elif risk_level == "MEDIUM":
            return "Normal peak: Monitor charging demand and reserve margins."
        return "Optimal headroom: No operational action required."

    def analyze_forecast(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Attach utilization percentages, risk classifications, and recommendations to forecast."""
        df_util = self.util_calc.calculate_utilization(forecast_df)
        df_util["risk_level"] = df_util["utilization_pct"].apply(self.classify_risk)
        df_util["recommendation"] = df_util["risk_level"].apply(self.get_recommendation)
        return df_util

    def evaluate_expansion_scenarios(self, forecast_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Evaluate baseline and what-if capacity upgrade scenarios (+10%, +20%, +50%)."""
        scenarios_results = []
        scenarios = self.config.capacity.scenarios or [
            {"name": "Current Capacity", "multiplier": 1.0},
            {"name": "+10% Capacity", "multiplier": 1.10},
            {"name": "+20% Capacity", "multiplier": 1.20},
            {"name": "+50% Capacity", "multiplier": 1.50},
        ]

        for sc in scenarios:
            name = sc.name if hasattr(sc, "name") else sc["name"]
            mult = sc.multiplier if hasattr(sc, "multiplier") else sc["multiplier"]

            temp_df = forecast_df.copy()
            base_capacity = temp_df["station_id"].apply(self.capacity_mgr.get_station_capacity)
            expanded_capacity = base_capacity * mult

            util_pct = ((temp_df["forecast_kwh"] / expanded_capacity) * 100.0).clip(lower=0.0)
            risk_levels = util_pct.apply(self.classify_risk)

            scenarios_results.append(
                {
                    "scenario_name": name,
                    "capacity_multiplier": mult,
                    "peak_utilization_pct": float(util_pct.max()),
                    "avg_utilization_pct": float(util_pct.mean()),
                    "critical_hours_count": int((risk_levels == "CRITICAL").sum()),
                    "high_hours_count": int((risk_levels == "HIGH").sum()),
                    "medium_hours_count": int((risk_levels == "MEDIUM").sum()),
                    "low_hours_count": int((risk_levels == "LOW").sum()),
                }
            )

        return scenarios_results
