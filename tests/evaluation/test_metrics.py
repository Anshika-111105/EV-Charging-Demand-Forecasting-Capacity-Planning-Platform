"""Tests for forecasting metrics and zero-demand edge cases."""

import numpy as np
import pytest

from ev_forecasting.evaluation.metrics import (
    calculate_mae,
    calculate_peak_error,
    calculate_rmse,
    calculate_wape,
    evaluate_forecast,
)


def test_standard_metrics_calculation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 40.0])

    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    wape = calculate_wape(y_true, y_pred)
    peak_err = calculate_peak_error(y_true, y_pred)

    assert mae == pytest.approx(1.75, abs=1e-4)  # (|2| + |2| + |3| + |0|) / 4 = 1.75
    assert rmse == pytest.approx(np.sqrt((4 + 4 + 9 + 0) / 4), abs=1e-4)
    assert wape == pytest.approx((7.0 / 100.0) * 100.0, abs=1e-4)  # 7.0%
    assert peak_err == 0.0


def test_zero_actual_demand_safety():
    """Verify that zero actual values do not trigger divide-by-zero or inf in MAPE/sMAPE."""
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 0.0, 0.5])

    res = evaluate_forecast(y_true, y_pred)

    assert not np.isinf(res["mape"])
    assert not np.isnan(res["mape"])
    assert not np.isinf(res["smape"])
    assert not np.isnan(res["smape"])
    assert res["mae"] > 0
