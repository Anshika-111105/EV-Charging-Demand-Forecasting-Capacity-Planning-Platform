"""Explicit tests to detect any temporal leakage in feature generation and splitting."""

import numpy as np
import pandas as pd
import pytest

from ev_forecasting.config.settings import load_config
from ev_forecasting.features.pipeline import FeaturePipeline
from ev_forecasting.splitting.temporal_split import TemporalSplitter


def test_strict_temporal_split_order(synthetic_hourly):
    config = load_config("configs/test.yaml")
    splitter = TemporalSplitter(config.splitting)

    train_df, val_df, test_df = splitter.split(synthetic_hourly)

    train_max_ts = train_df["timestamp"].max()
    val_min_ts = val_df["timestamp"].min()
    val_max_ts = val_df["timestamp"].max()
    test_min_ts = test_df["timestamp"].min()

    # Assert strict chronological boundary order
    assert train_max_ts < val_min_ts, "Temporal leakage: Train timestamps overlap with Validation!"
    assert val_max_ts < test_min_ts, "Temporal leakage: Validation timestamps overlap with Test!"


def test_future_value_perturbation_leakage_detector(synthetic_hourly):
    """Perturb future target values at time T_future.
    Verify that feature matrix for all times t < T_future remains 100% identical.
    """
    config = load_config("configs/test.yaml")
    pipeline = FeaturePipeline(config)

    # 1. Base transform
    df_feat_base = pipeline.transform(synthetic_hourly)

    # 2. Mutate future rows (e.g. Set last 24 hours to 9999.0 kWh)
    perturbed_df = synthetic_hourly.copy()
    cutoff_time = perturbed_df["timestamp"].max() - pd.Timedelta(hours=24)
    perturbed_df.loc[perturbed_df["timestamp"] >= cutoff_time, "energy_demand_kwh"] = 9999.0

    # 3. Transform perturbed dataframe
    df_feat_perturbed = pipeline.transform(perturbed_df)

    # 4. Check feature values before cutoff_time
    historical_mask = df_feat_base["timestamp"] < cutoff_time
    base_past = df_feat_base[historical_mask].dropna()
    perturbed_past = df_feat_perturbed[historical_mask].dropna()

    feat_cols = [c for c in df_feat_base.columns if "lag_" in c or "rolling_" in c]
    for col in feat_cols:
        diff = np.abs(base_past[col].values - perturbed_past[col].values)
        max_diff = np.max(diff) if len(diff) > 0 else 0.0
        assert max_diff == pytest.approx(0.0, abs=1e-7), (
            f"LEAKAGE DETECTED in column '{col}': modifying future data changed past feature values!"
        )
