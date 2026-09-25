"""Statistical distribution drift detection (KS-Test and PSI)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between baseline and current distributions."""
    b_clean = baseline[~np.isnan(baseline)]
    c_clean = current[~np.isnan(current)]

    if len(b_clean) < 10 or len(c_clean) < 10:
        return 0.0

    # Determine quantile bins from baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(b_clean, quantiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    bins[0] = -np.inf
    bins[-1] = np.inf

    b_counts, _ = np.histogram(b_clean, bins=bins)
    c_counts, _ = np.histogram(c_clean, bins=bins)

    # Convert to proportions with smoothing epsilon
    eps = 1e-4
    b_prop = (b_counts / len(b_clean)) + eps
    c_prop = (c_counts / len(c_clean)) + eps

    psi_val = np.sum((c_prop - b_prop) * np.log(c_prop / b_prop))
    return float(psi_val)


def calculate_ks_drift(baseline: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """Calculate two-sample Kolmogorov-Smirnov test statistic and p-value."""
    b_clean = baseline[~np.isnan(baseline)]
    c_clean = current[~np.isnan(current)]

    if len(b_clean) < 5 or len(c_clean) < 5:
        return {"ks_statistic": 0.0, "p_value": 1.0}

    res = stats.ks_2samp(b_clean, c_clean)
    return {"ks_statistic": float(res.statistic), "p_value": float(res.pvalue)}


class DriftDetector:
    """Monitors feature and target demand distribution drift."""

    def __init__(self, significance_level: float = 0.05, psi_threshold: float = 0.25):
        self.significance_level = significance_level
        self.psi_threshold = psi_threshold

    def evaluate_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features_to_check: List[str],
    ) -> List[Dict[str, Any]]:
        """Evaluate KS test and PSI across checked features."""
        drift_report = []

        for feat in features_to_check:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                continue

            b_vals = baseline_df[feat].dropna().values
            c_vals = current_df[feat].dropna().values

            if len(b_vals) < 5 or len(c_vals) < 5:
                continue

            psi_score = calculate_psi(b_vals, c_vals)
            ks_res = calculate_ks_drift(b_vals, c_vals)

            drift_detected = (ks_res["p_value"] < self.significance_level) or (
                psi_score >= self.psi_threshold
            )

            drift_report.append(
                {
                    "feature_name": feat,
                    "psi_score": round(psi_score, 4),
                    "ks_statistic": round(ks_res["ks_statistic"], 4),
                    "p_value": round(ks_res["p_value"], 6),
                    "drift_detected": drift_detected,
                    "severity": "HIGH"
                    if psi_score >= self.psi_threshold
                    else ("MODERATE" if psi_score >= 0.1 else "LOW"),
                }
            )

        return drift_report
