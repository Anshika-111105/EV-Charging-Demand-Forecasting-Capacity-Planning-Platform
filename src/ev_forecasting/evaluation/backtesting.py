"""Expanding-window walk-forward backtesting execution engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ev_forecasting.config.settings import AppConfig
from ev_forecasting.evaluation.metrics import evaluate_forecast
from ev_forecasting.models.base import BaseForecaster
from ev_forecasting.splitting.backtest_split import ExpandingWindowSplitter

logger = logging.getLogger(__name__)


class BacktestingEngine:
    """Executes walk-forward expanding window backtesting across all candidate models."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.splitter = ExpandingWindowSplitter(config.backtesting)
        self.backtests_dir = Path(config.paths.backtests_dir)
        self.backtests_dir.mkdir(parents=True, exist_ok=True)

    def run_backtest_for_model(
        self,
        model: BaseForecaster,
        df_full: pd.DataFrame,
        horizons: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Run expanding window backtesting for a model across all horizons and folds."""
        target_horizons = horizons or self.config.backtesting.horizons
        all_fold_evals = []

        for h in target_horizons:
            folds = self.splitter.generate_folds(df_full, horizon_hours=h)

            for fold in folds:
                logger.info(
                    "Evaluating %s on Fold %d (Horizon=%dh, Train: %s to %s, Test: %s to %s)",
                    model.name,
                    fold.fold_idx,
                    h,
                    fold.train_start.strftime("%Y-%m-%d"),
                    fold.train_end.strftime("%Y-%m-%d"),
                    fold.test_start.strftime("%Y-%m-%d"),
                    fold.test_end.strftime("%Y-%m-%d"),
                )

                # Fit on expanding historical train fold
                model.fit(fold.train_df)

                # Predict test period
                preds_df = model.predict(
                    horizon_hours=h,
                    df_history=fold.train_df,
                    df_future=fold.test_df,
                )

                # Merge predictions with ground truth
                merged = pd.merge(
                    fold.test_df[["station_id", "site_id", "timestamp", "energy_demand_kwh"]],
                    preds_df[["station_id", "timestamp", "forecast_kwh", "model_name"]],
                    on=["station_id", "timestamp"],
                    how="inner",
                )

                # Save fold-level predictions to disk
                fold_pred_file = (
                    self.backtests_dir / f"{model.name}_h{h}_fold{fold.fold_idx:03d}.parquet"
                )
                merged.to_parquet(fold_pred_file, index=False)

                # Calculate overall fold metrics
                metrics = evaluate_forecast(
                    merged["energy_demand_kwh"].values, merged["forecast_kwh"].values
                )
                metrics.update(
                    {
                        "model_name": model.name,
                        "model_type": model.model_type,
                        "horizon_hours": h,
                        "fold_idx": fold.fold_idx,
                        "train_end": str(fold.train_end),
                        "test_start": str(fold.test_start),
                        "test_end": str(fold.test_end),
                        "n_samples": len(merged),
                        "training_seconds": model.training_duration_seconds,
                    }
                )
                all_fold_evals.append(metrics)

        return pd.DataFrame(all_fold_evals)
