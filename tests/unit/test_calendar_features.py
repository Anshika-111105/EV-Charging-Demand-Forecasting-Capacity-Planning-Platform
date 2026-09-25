"""Unit tests for calendar and cyclical features."""

import pandas as pd

from ev_forecasting.config.settings import CalendarFeaturesConfig
from ev_forecasting.features.calendar import CalendarFeatureExtractor


def test_calendar_feature_extraction():
    cfg = CalendarFeaturesConfig(
        include_hour=True,
        include_day_of_week=True,
        include_day_of_month=True,
        include_month=True,
        include_is_weekend=True,
        include_cyclical=True,
    )
    extractor = CalendarFeatureExtractor(cfg)

    df = pd.DataFrame(
        {"timestamp": pd.date_range("2020-01-01 00:00:00", periods=48, freq="1h", tz="UTC")}
    )

    res = extractor.transform(df)

    assert "hour" in res.columns
    assert "hour_sin" in res.columns
    assert "hour_cos" in res.columns
    assert "day_of_week" in res.columns
    assert "is_weekend" in res.columns

    # Verify sine/cosine range [-1, 1]
    assert res["hour_sin"].min() >= -1.0
    assert res["hour_sin"].max() <= 1.0
    assert res["hour_cos"].min() >= -1.0
    assert res["hour_cos"].max() <= 1.0

    # Verify hour values [0, 23]
    assert res["hour"].min() == 0
    assert res["hour"].max() == 23
