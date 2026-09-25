"""Integration tests for data ingestion, preprocessing, and model pipelines."""

from pathlib import Path

from ev_forecasting.data.aggregation import TimeSeriesAggregator
from ev_forecasting.data.cleaning import SessionDataCleaner
from ev_forecasting.data.ingestion import ACNDataIngestor
from ev_forecasting.data.parsing import ACNDataParser
from ev_forecasting.data.validation import DataQualityValidator
from ev_forecasting.features.pipeline import FeaturePipeline
from tests.fixtures.synthetic_data import create_synthetic_raw_files


def test_data_pipeline_integration(temp_workspace):
    config = temp_workspace
    raw_dir = Path(config.paths.raw_data_dir)

    # 1. Create physical raw sample files
    create_synthetic_raw_files(raw_dir, n_sessions_per_site=3)

    # 2. Ingestion manifest generation
    ingestor = ACNDataIngestor(config)
    manifest = ingestor.generate_manifest()
    assert manifest["total_files"] >= 9
    assert len(manifest["sites"]) == 3

    # 3. Parse raw sessions
    parser = ACNDataParser(config)
    sessions_df, _ = parser.parse_all_sessions(raw_dir)
    assert not sessions_df.empty

    # 4. Validate sessions
    validator = DataQualityValidator(config)
    passed, _ = validator.validate_sessions_dataframe(sessions_df)
    assert passed is True

    # 5. Clean sessions
    cleaner = SessionDataCleaner(config)
    cleaned = cleaner.clean_sessions(sessions_df)
    assert len(cleaned) == len(sessions_df)

    # 6. Aggregate to regular hourly
    aggregator = TimeSeriesAggregator(config)
    raw_hourly = aggregator.allocate_session_energy_hourly(cleaned)
    hourly_grid = aggregator.build_regular_grid(raw_hourly, min_station_records=1)
    assert not hourly_grid.empty
    assert (hourly_grid["energy_demand_kwh"] < 0).sum() == 0

    # 7. Feature pipeline
    feat_pipe = FeaturePipeline(config)
    feat_df, feat_cols = feat_pipe.fit_transform(hourly_grid)
    assert len(feat_cols) > 0
    assert "lag_1h" in feat_df.columns
    assert "rolling_mean_24h" in feat_df.columns
