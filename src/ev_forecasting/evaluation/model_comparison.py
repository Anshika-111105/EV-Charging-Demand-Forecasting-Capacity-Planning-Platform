"""Model comparison, ranking, and automated candidate selection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ev_forecasting.config.settings import AppConfig

logger = logging.getLogger(__name__)


class ModelComparator:
    """Aggregates backtesting results, ranks models, and enforces selection quality gates."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.metrics_dir = Path(config.paths.metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison_table(self, all_eval_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate fold-level metrics per model into a clean comparison leaderboard."""
        if all_eval_df.empty:
            return pd.DataFrame()

        summary = (
            all_eval_df.groupby(["model_name", "model_type", "horizon_hours"])
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                mape_mean=("mape", "mean"),
                smape_mean=("smape", "mean"),
                wape_mean=("wape", "mean"),
                r2_mean=("r2", "mean"),
                avg_training_seconds=("training_seconds", "mean"),
                total_folds=("fold_idx", "count"),
            )
            .reset_index()
        )

        # Sort by primary metric (RMSE mean ascending, then MAE mean ascending)
        summary = summary.sort_values(["horizon_hours", "rmse_mean", "mae_mean"]).reset_index(
            drop=True
        )

        # Save to disk
        summary_csv = self.metrics_dir / "model_comparison_leaderboard.csv"
        summary.to_csv(summary_csv, index=False)

        summary_json = self.metrics_dir / "model_comparison_leaderboard.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(orient="records"), f, indent=2)

        logger.info("Saved model comparison leaderboard to %s", summary_csv)
        return summary

    def select_production_candidate(
        self, comparison_df: pd.DataFrame, target_horizon: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Select production candidate using objective multi-criteria quality gates."""
        if comparison_df.empty:
            raise ValueError("Comparison dataframe is empty.")

        horizon = target_horizon or self.config.backtesting.primary_horizon
        h_df = comparison_df[comparison_df["horizon_hours"] == horizon].copy()
        if h_df.empty:
            h_df = comparison_df.copy()

        criteria = self.config.registry.promotion_criteria
        baseline_name = criteria.baseline_model

        baseline_row = h_df[h_df["model_name"] == baseline_name]
        baseline_rmse = (
            float(baseline_row["rmse_mean"].iloc[0]) if not baseline_row.empty else float("inf")
        )
        (
            float(baseline_row["mae_mean"].iloc[0]) if not baseline_row.empty else float("inf")
        )

        eligible_models = []
        for _, row in h_df.iterrows():
            m_name = row["model_name"]
            rmse = float(row["rmse_mean"])
            mae = float(row["mae_mean"])
            mape = float(row["mape_mean"])

            beats_baseline = (rmse <= baseline_rmse) or (m_name == baseline_name)

            score = rmse * 0.6 + mae * 0.4
            eligible_models.append(
                {
                    "model_name": m_name,
                    "model_type": row["model_type"],
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "beats_baseline": beats_baseline,
                    "composite_score": score,
                }
            )

        # Rank by composite score
        eligible_df = pd.DataFrame(eligible_models).sort_values("composite_score")
        best_model_name = eligible_df.iloc[0]["model_name"]
        best_details = eligible_df.iloc[0].to_dict()

        logger.info(
            "Selected production model candidate: %s (RMSE=%.2f, MAE=%.2f, beats baseline=%s)",
            best_model_name,
            best_details["rmse"],
            best_details["mae"],
            best_details["beats_baseline"],
        )

        return best_model_name, best_details
