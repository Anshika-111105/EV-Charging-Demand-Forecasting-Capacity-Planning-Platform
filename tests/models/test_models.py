"""Tests for forecasting models (Baselines, ARIMA, SARIMA, Prophet, XGBoost)."""

import tempfile
from pathlib import Path

import numpy as np

from ev_forecasting.config.settings import XGBoostModelConfig
from ev_forecasting.features.pipeline import FeaturePipeline
from ev_forecasting.models.arima import ARIMAForecaster
from ev_forecasting.models.prophet import ProphetForecaster
from ev_forecasting.models.sarima import SARIMAForecaster
from ev_forecasting.models.seasonal_naive import SeasonalNaiveForecaster
from ev_forecasting.models.xgboost import XGBoostForecaster


def test_seasonal_naive_forecaster(synthetic_hourly):
    model = SeasonalNaiveForecaster(seasonality=24)
    model.fit(synthetic_hourly)

    assert model.is_fitted is True
    assert model.training_duration_seconds >= 0

    preds = model.predict(horizon_hours=24)
    assert not preds.empty
    assert len(preds) == 24 * synthetic_hourly["station_id"].nunique()
    assert preds["forecast_kwh"].isna().sum() == 0
    assert (preds["forecast_kwh"] < 0).sum() == 0

    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        model.save(tmp_path)
        loaded = SeasonalNaiveForecaster.load(tmp_path)
        loaded_preds = loaded.predict(horizon_hours=24)
        np.testing.assert_array_almost_equal(
            preds["forecast_kwh"].values, loaded_preds["forecast_kwh"].values
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def test_arima_forecaster(synthetic_hourly):
    model = ARIMAForecaster(order=(1, 0, 0))
    model.fit(synthetic_hourly)

    preds = model.predict(horizon_hours=12)
    assert not preds.empty
    assert preds["forecast_kwh"].isna().sum() == 0
    assert (preds["forecast_kwh"] < 0).sum() == 0


def test_sarima_forecaster(synthetic_hourly):
    model = SARIMAForecaster(order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    model.fit(synthetic_hourly)

    preds = model.predict(horizon_hours=12)
    assert not preds.empty
    assert preds["forecast_kwh"].isna().sum() == 0


def test_prophet_forecaster(synthetic_hourly):
    model = ProphetForecaster(daily_seasonality=True, weekly_seasonality=False)
    model.fit(synthetic_hourly)

    preds = model.predict(horizon_hours=12)
    assert not preds.empty
    assert preds["forecast_kwh"].isna().sum() == 0


def test_xgboost_forecaster(synthetic_hourly, test_config):
    pipe = FeaturePipeline(test_config)
    feat_df, feat_cols = pipe.fit_transform(synthetic_hourly)

    model = XGBoostForecaster(
        config=XGBoostModelConfig(n_estimators=10, max_depth=3),
        feature_pipeline=pipe,
    )
    model.fit(feat_df, feature_columns=feat_cols)

    preds = model.predict(horizon_hours=12, df_history=synthetic_hourly)
    assert not preds.empty
    assert preds["forecast_kwh"].isna().sum() == 0
    assert (preds["forecast_kwh"] < 0).sum() == 0
