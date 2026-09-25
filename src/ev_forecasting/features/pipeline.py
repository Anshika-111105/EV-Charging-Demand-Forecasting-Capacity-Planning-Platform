"""Feature engineering pipeline orchestrator and manifest logger."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ev_forecasting.config.settings import AppConfig
from ev_forecasting.features.calendar import CalendarFeatureExtractor
from ev_forecasting.features.lags import LagFeatureExtractor
from ev_forecasting.features.rolling import RollingFeatureExtractor

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrates all feature extraction stages and maintains feature metadata."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.calendar_extractor = CalendarFeatureExtractor(config.features.calendar)
        self.lag_extractor = LagFeatureExtractor(config.features.lags)
        self.rolling_extractor = RollingFeatureExtractor(config.features.rolling)
        self.manifest_dir = Path(config.paths.manifest_dir)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.station_encoder: Optional[OneHotEncoder] = None
        self.feature_columns: List[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "energy_demand_kwh",
        group_col: str = "station_id",
        timestamp_col: str = "timestamp",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Fit encoders and transform raw hourly data into model-ready features."""
        df_feat = df.copy()

        # 1. Extract calendar features
        df_feat = self.calendar_extractor.transform(df_feat, timestamp_col=timestamp_col)

        # 2. Extract lag features
        df_feat = self.lag_extractor.transform(
            df_feat, target_col=target_col, group_col=group_col, timestamp_col=timestamp_col
        )

        # 3. Extract rolling features
        df_feat = self.rolling_extractor.transform(
            df_feat, target_col=target_col, group_col=group_col, timestamp_col=timestamp_col
        )

        # 4. Encode station and site categorical identifiers
        categorical_cols = [c for c in ["station_id", "site_id"] if c in df_feat.columns]
        if categorical_cols:
            self.station_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_cats = self.station_encoder.fit_transform(df_feat[categorical_cols])
            encoded_col_names = self.station_encoder.get_feature_names_out(
                categorical_cols
            ).tolist()
            # Drop any prior encoded columns if present to avoid duplicate names
            df_feat = df_feat.drop(
                columns=[c for c in encoded_col_names if c in df_feat.columns], errors="ignore"
            )
            encoded_df = pd.DataFrame(encoded_cats, columns=encoded_col_names, index=df_feat.index)
            df_feat = pd.concat([df_feat, encoded_df], axis=1)

        # Ensure no duplicate columns
        df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()].copy()

        # Drop non-feature metadata columns from final feature list
        exclude_cols = {
            "timestamp",
            "station_id",
            "site_id",
            "session_count",
            "active_sessions",
            target_col,
        }
        self.feature_columns = [c for c in df_feat.columns if c not in exclude_cols]

        self.save_feature_manifest(self.feature_columns, len(df_feat))
        logger.info("Engineered %d features for %d rows", len(self.feature_columns), len(df_feat))
        return df_feat, self.feature_columns

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str = "energy_demand_kwh",
        group_col: str = "station_id",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Apply fitted transforms to new / test dataset."""
        df_feat = df.copy()
        df_feat = self.calendar_extractor.transform(df_feat, timestamp_col=timestamp_col)
        df_feat = self.lag_extractor.transform(
            df_feat, target_col=target_col, group_col=group_col, timestamp_col=timestamp_col
        )
        df_feat = self.rolling_extractor.transform(
            df_feat, target_col=target_col, group_col=group_col, timestamp_col=timestamp_col
        )

        categorical_cols = [c for c in ["station_id", "site_id"] if c in df_feat.columns]
        if categorical_cols and self.station_encoder is not None:
            encoded_cats = self.station_encoder.transform(df_feat[categorical_cols])
            encoded_col_names = self.station_encoder.get_feature_names_out(
                categorical_cols
            ).tolist()
            df_feat = df_feat.drop(
                columns=[c for c in encoded_col_names if c in df_feat.columns], errors="ignore"
            )
            encoded_df = pd.DataFrame(encoded_cats, columns=encoded_col_names, index=df_feat.index)
            df_feat = pd.concat([df_feat, encoded_df], axis=1)

        df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()].copy()
        return df_feat

    def save_feature_manifest(self, feature_names: List[str], sample_count: int) -> Dict[str, Any]:
        """Save feature engineering manifest."""
        manifest = {
            "feature_version": self.config.features.version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_features": len(feature_names),
            "feature_names": feature_names,
            "lag_hours": self.config.features.lags.hours,
            "rolling_windows": self.config.features.rolling.windows,
            "sample_count": sample_count,
        }

        manifest_path = self.manifest_dir / "feature_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
