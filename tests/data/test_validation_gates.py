"""Tests for data validation gates and anomaly rejection."""

import pandas as pd
import pytest

from ev_forecasting.config.settings import load_config
from ev_forecasting.data.validation import DataQualityValidator, DataValidationError


def test_session_validation_success(synthetic_sessions):
    config = load_config("configs/test.yaml")
    validator = DataQualityValidator(config)

    passed, report = validator.validate_sessions_dataframe(
        synthetic_sessions, raise_on_failure=True
    )
    assert passed is True
    assert report["status"] == "PASS"
    assert report["total_sessions"] > 0


def test_session_validation_negative_energy_failure():
    config = load_config("configs/test.yaml")
    validator = DataQualityValidator(config)

    invalid_df = pd.DataFrame(
        {
            "session_id": ["s1", "s2"],
            "station_id": ["st1", "st2"],
            "site_id": ["caltech", "caltech"],
            "start_time": pd.to_datetime(["2020-01-01 08:00:00", "2020-01-01 09:00:00"], utc=True),
            "end_time": pd.to_datetime(["2020-01-01 10:00:00", "2020-01-01 11:00:00"], utc=True),
            "duration_hours": [2.0, 2.0],
            "kwh_delivered": [10.0, -5.0],  # Negative energy!
        }
    )

    with pytest.raises(DataValidationError):
        validator.validate_sessions_dataframe(invalid_df, raise_on_failure=True)


def test_hourly_validation_duplicate_timestamps():
    config = load_config("configs/test.yaml")
    validator = DataQualityValidator(config)

    dup_df = pd.DataFrame(
        {
            "station_id": ["st1", "st1"],
            "site_id": ["caltech", "caltech"],
            "timestamp": pd.to_datetime(["2020-01-01 08:00:00", "2020-01-01 08:00:00"], utc=True),
            "energy_demand_kwh": [10.0, 15.0],
        }
    )

    with pytest.raises(DataValidationError):
        validator.validate_hourly_dataframe(dup_df, raise_on_failure=True)
