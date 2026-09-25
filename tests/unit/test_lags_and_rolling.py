"""Unit tests for lag and causal rolling statistics."""

import pandas as pd

from ev_forecasting.config.settings import LagFeaturesConfig, RollingFeaturesConfig
from ev_forecasting.features.lags import LagFeatureExtractor
from ev_forecasting.features.rolling import RollingFeatureExtractor


def test_lag_features():
    cfg = LagFeaturesConfig(hours=[1, 2, 24])
    extractor = LagFeatureExtractor(cfg)

    dates = pd.date_range("2020-01-01", periods=30, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "station_id": ["ST_01"] * 30,
            "energy_demand_kwh": list(range(30)),
        }
    )

    res = extractor.transform(df)

    assert "lag_1h" in res.columns
    assert "lag_2h" in res.columns
    assert "lag_24h" in res.columns

    # Check lag 1 is shifted by 1
    assert pd.isna(res["lag_1h"].iloc[0])
    assert res["lag_1h"].iloc[1] == 0
    assert res["lag_1h"].iloc[5] == 4


def test_rolling_causality():
    cfg = RollingFeaturesConfig(windows=[6], statistics=["mean", "max"])
    extractor = RollingFeatureExtractor(cfg)

    dates = pd.date_range("2020-01-01", periods=10, freq="1h", tz="UTC")
    # Values: 10, 20, 30, 40, ...
    vals = [10.0 * (i + 1) for i in range(10)]
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "station_id": ["ST_01"] * 10,
            "energy_demand_kwh": vals,
        }
    )

    res = extractor.transform(df)

    assert "rolling_mean_6h" in res.columns
    assert "rolling_max_6h" in res.columns

    # At row index 2 (target=30), shifted target sees [10, 20]. Rolling mean should be 15.0, NOT including 30.0!
    assert res["rolling_mean_6h"].iloc[2] == 15.0
    assert res["rolling_max_6h"].iloc[2] == 20.0
