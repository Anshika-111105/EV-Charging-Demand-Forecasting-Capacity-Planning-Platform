"""Pytest configuration and shared test fixtures."""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ev_forecasting.config.settings import AppConfig, load_config
from tests.fixtures.synthetic_data import generate_synthetic_sessions


@pytest.fixture(scope="session")
def test_config() -> AppConfig:
    """Load test configuration with temporary test paths."""
    config = load_config("configs/test.yaml")
    return config


@pytest.fixture
def temp_workspace(test_config):
    """Provide clean temporary directories for test runs."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="ev_test_"))
    orig_raw = test_config.paths.raw_data_dir
    orig_interim = test_config.paths.interim_data_dir
    orig_processed = test_config.paths.processed_data_dir

    test_config.paths.raw_data_dir = str(tmp_dir / "raw")
    test_config.paths.interim_data_dir = str(tmp_dir / "interim")
    test_config.paths.processed_data_dir = str(tmp_dir / "processed")
    test_config.paths.manifest_dir = str(tmp_dir / "manifests")
    test_config.paths.reports_dir = str(tmp_dir / "reports")
    test_config.paths.metrics_dir = str(tmp_dir / "reports/metrics")
    test_config.paths.figures_dir = str(tmp_dir / "reports/figures")
    test_config.paths.backtests_dir = str(tmp_dir / "reports/backtests")
    test_config.paths.model_artifacts_dir = str(tmp_dir / "models")

    yield test_config

    shutil.rmtree(tmp_dir, ignore_errors=True)
    test_config.paths.raw_data_dir = orig_raw
    test_config.paths.interim_data_dir = orig_interim
    test_config.paths.processed_data_dir = orig_processed


@pytest.fixture
def synthetic_sessions() -> pd.DataFrame:
    """Generate 14 days of synthetic session data."""
    return generate_synthetic_sessions(n_days=14, stations_per_site=2, random_seed=42)


@pytest.fixture
def synthetic_hourly(synthetic_sessions, test_config) -> pd.DataFrame:
    """Generate clean aggregated hourly dataset from synthetic sessions."""
    from ev_forecasting.data.aggregation import TimeSeriesAggregator
    from ev_forecasting.data.cleaning import SessionDataCleaner

    cleaner = SessionDataCleaner(test_config)
    cleaned = cleaner.clean_sessions(synthetic_sessions)
    aggregator = TimeSeriesAggregator(test_config)
    raw_hourly = aggregator.allocate_session_energy_hourly(cleaned)
    return aggregator.build_regular_grid(raw_hourly)
