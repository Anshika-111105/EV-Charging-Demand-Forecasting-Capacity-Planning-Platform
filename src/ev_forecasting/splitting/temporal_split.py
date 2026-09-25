"""Chronological temporal dataset splitting module."""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

from ev_forecasting.config.settings import SplittingConfig

logger = logging.getLogger(__name__)


class TemporalSplitter:
    """Performs strict chronological train, validation, and test splitting without lookahead."""

    def __init__(self, config: SplittingConfig):
        self.config = config

    def split(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train, validation, and test sets chronologically."""
        if df.empty:
            raise ValueError("Cannot split empty dataframe")

        # Sort strictly by timestamp
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        unique_timestamps = (
            df_sorted[timestamp_col].drop_duplicates().sort_values().reset_index(drop=True)
        )
        n_times = len(unique_timestamps)

        train_idx_end = int(n_times * self.config.train_ratio)
        val_idx_end = int(n_times * (self.config.train_ratio + self.config.validation_ratio))

        train_cutoff = unique_timestamps.iloc[train_idx_end - 1]
        val_cutoff = unique_timestamps.iloc[val_idx_end - 1]

        train_df = df_sorted[df_sorted[timestamp_col] <= train_cutoff].copy()
        val_df = df_sorted[
            (df_sorted[timestamp_col] > train_cutoff) & (df_sorted[timestamp_col] <= val_cutoff)
        ].copy()
        test_df = df_sorted[df_sorted[timestamp_col] > val_cutoff].copy()

        # Integrity checks
        assert train_df[timestamp_col].max() < val_df[timestamp_col].min(), (
            "Train/Val temporal overlap detected!"
        )
        assert val_df[timestamp_col].max() < test_df[timestamp_col].min(), (
            "Val/Test temporal overlap detected!"
        )

        logger.info(
            "Temporal split completed: Train (%d rows, %s to %s), Val (%d rows, %s to %s), Test (%d rows, %s to %s)",
            len(train_df),
            train_df[timestamp_col].min(),
            train_df[timestamp_col].max(),
            len(val_df),
            val_df[timestamp_col].min(),
            val_df[timestamp_col].max(),
            len(test_df),
            test_df[timestamp_col].min(),
            test_df[timestamp_col].max(),
        )

        return train_df, val_df, test_df
