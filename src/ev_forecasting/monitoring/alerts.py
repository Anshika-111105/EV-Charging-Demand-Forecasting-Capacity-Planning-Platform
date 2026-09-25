"""Monitoring alerts and retraining policy evaluation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from ev_forecasting.config.settings import AppConfig
from ev_forecasting.monitoring.drift import DriftDetector

logger = logging.getLogger(__name__)


class MonitoringManager:
    """Evaluates data freshness, drift triggers, and automated retraining policies."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.drift_detector = DriftDetector(
            significance_level=config.monitoring.drift_significance_level,
            psi_threshold=config.monitoring.psi_threshold_significant,
        )

    def evaluate_monitoring_state(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Perform comprehensive monitoring check."""
        drift_results = self.drift_detector.evaluate_drift(baseline_df, current_df, feature_names)

        high_drift_features = [d["feature_name"] for d in drift_results if d["severity"] == "HIGH"]
        retrain_recommended = len(high_drift_features) > 0

        alert_message = (
            f"Significant drift detected in {len(high_drift_features)} features ({high_drift_features}). Retraining recommended."
            if retrain_recommended
            else "System is healthy. No significant distribution drift detected."
        )

        return {
            "status": "ALERT" if retrain_recommended else "HEALTHY",
            "retraining_recommended": retrain_recommended,
            "alert_message": alert_message,
            "drift_features_evaluated": len(drift_results),
            "high_drift_count": len(high_drift_features),
            "drift_report": drift_results,
        }
