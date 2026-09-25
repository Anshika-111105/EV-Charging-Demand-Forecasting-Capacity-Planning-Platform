"""Data cleaning and normalization for EV charging sessions."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ev_forecasting.config.settings import AppConfig

logger = logging.getLogger(__name__)


class SessionDataCleaner:
    """Cleans and standardizes raw parsed EV charging sessions."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.interim_dir = Path(config.paths.interim_data_dir)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def clean_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean sessions dataframe by removing duplicates, fixing timestamps, and capping anomalies."""
        if df.empty:
            return df

        initial_count = len(df)
        df_clean = df.copy()

        # 1. Ensure UTC datetimes
        df_clean["start_time"] = pd.to_datetime(df_clean["start_time"], utc=True)
        df_clean["end_time"] = pd.to_datetime(df_clean["end_time"], utc=True)

        # 2. Filter out invalid temporal order or zero duration
        df_clean = df_clean[df_clean["end_time"] >= df_clean["start_time"]]
        df_clean["duration_hours"] = (
            df_clean["end_time"] - df_clean["start_time"]
        ).dt.total_seconds() / 3600.0
        # Avoid zero duration
        df_clean["duration_hours"] = df_clean["duration_hours"].clip(lower=1.0 / 3600.0)

        # 3. Deduplicate by session_id (keep first)
        df_clean = df_clean.drop_duplicates(subset=["session_id"])

        # 4. Cap unrealistic energy spikes (e.g. max 150 kWh on typical L2 chargers)
        df_clean["kwh_delivered"] = df_clean["kwh_delivered"].clip(lower=0.0, upper=150.0)

        # 5. Drop records with missing station_id or site_id
        df_clean = df_clean.dropna(subset=["station_id", "site_id", "start_time"])

        logger.info(
            "Cleaned sessions: retained %d of %d original records (%.1f%%)",
            len(df_clean),
            initial_count,
            (len(df_clean) / max(initial_count, 1)) * 100,
        )

        output_path = self.interim_dir / "sessions_cleaned.parquet"
        df_clean.to_parquet(output_path, index=False)
        logger.info("Saved cleaned sessions to %s", output_path)

        return df_clean
