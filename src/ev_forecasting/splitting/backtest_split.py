"""Expanding-window walk-forward backtesting splitter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import pandas as pd

from ev_forecasting.config.settings import BacktestingConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestFold:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    horizon_hours: int


class ExpandingWindowSplitter:
    """Generates walk-forward expanding window folds for temporal backtesting."""

    def __init__(self, config: BacktestingConfig):
        self.config = config

    def generate_folds(
        self,
        df: pd.DataFrame,
        horizon_hours: int = 168,
        timestamp_col: str = "timestamp",
    ) -> List[BacktestFold]:
        """Generate a list of BacktestFold instances."""
        if df.empty:
            return []

        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        unique_timestamps = (
            df_sorted[timestamp_col].drop_duplicates().sort_values().reset_index(drop=True)
        )
        total_hours = len(unique_timestamps)

        min_train = min(self.config.min_train_hours, int(total_hours * 0.5))
        step = self.config.step_hours
        n_folds = self.config.n_folds

        # Ensure we have enough points for backtesting
        min_train + (n_folds * horizon_hours)
        if total_hours < min_train + horizon_hours:
            logger.warning(
                "Insufficient time series length (%d hours) for min_train=%d and horizon=%d",
                total_hours,
                min_train,
                horizon_hours,
            )
            # Adjust min_train dynamically if needed
            min_train = max(int(total_hours * 0.5), 24)

        folds: List[BacktestFold] = []
        current_train_end_idx = min_train

        for fold_i in range(1, n_folds + 1):
            if current_train_end_idx + horizon_hours > total_hours:
                break

            train_cutoff = unique_timestamps.iloc[current_train_end_idx - 1]
            test_start = unique_timestamps.iloc[current_train_end_idx]
            test_end_idx = min(current_train_end_idx + horizon_hours - 1, total_hours - 1)
            test_cutoff = unique_timestamps.iloc[test_end_idx]

            train_data = df_sorted[df_sorted[timestamp_col] <= train_cutoff].copy()
            test_data = df_sorted[
                (df_sorted[timestamp_col] >= test_start) & (df_sorted[timestamp_col] <= test_cutoff)
            ].copy()

            if not train_data.empty and not test_data.empty:
                folds.append(
                    BacktestFold(
                        fold_idx=fold_i,
                        train_start=unique_timestamps.iloc[0],
                        train_end=train_cutoff,
                        test_start=test_start,
                        test_end=test_cutoff,
                        train_df=train_data,
                        test_df=test_data,
                        horizon_hours=horizon_hours,
                    )
                )

            current_train_end_idx += step

        logger.info("Generated %d walk-forward folds for horizon=%dh", len(folds), horizon_hours)
        return folds
