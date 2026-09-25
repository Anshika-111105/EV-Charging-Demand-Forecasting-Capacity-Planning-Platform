"""End-to-end system test verifying full pipeline execution and auditability."""

from pathlib import Path

from ev_forecasting.audit.lineage import LineageTracker
from ev_forecasting.audit.prediction_log import AuditLogger
from ev_forecasting.data.aggregation import TimeSeriesAggregator
from ev_forecasting.data.cleaning import SessionDataCleaner
from ev_forecasting.data.ingestion import ACNDataIngestor
from ev_forecasting.data.parsing import ACNDataParser
from ev_forecasting.data.validation import DataQualityValidator
from ev_forecasting.forecasting.forecast import ForecastService
from ev_forecasting.training.train_all_models import ModelTrainingPipeline
from tests.fixtures.synthetic_data import create_synthetic_raw_files


def test_full_system_end_to_end(temp_workspace):
    config = temp_workspace
    raw_dir = Path(config.paths.raw_data_dir)

    # 1. Ingestion of raw session files
    create_synthetic_raw_files(raw_dir, n_sessions_per_site=5)
    ingestor = ACNDataIngestor(config)
    manifest = ingestor.generate_manifest()
    assert manifest["total_files"] >= 15
    assert len(manifest["sha256"]) == 64

    # 2. Parsing and Validation Gate
    parser = ACNDataParser(config)
    sessions_df, _ = parser.parse_all_sessions(raw_dir)
    validator = DataQualityValidator(config)
    passed_s, _ = validator.validate_sessions_dataframe(sessions_df)
    assert passed_s is True

    # 3. Cleaning & Aggregation
    cleaner = SessionDataCleaner(config)
    cleaned = cleaner.clean_sessions(sessions_df)
    aggregator = TimeSeriesAggregator(config)
    raw_hourly = aggregator.allocate_session_energy_hourly(cleaned)
    hourly_df = aggregator.build_regular_grid(raw_hourly, min_station_records=1)
    passed_h, _ = validator.validate_hourly_dataframe(hourly_df)
    assert passed_h is True

    # 4. Training, Backtesting, and Model Registry Promotion
    trainer = ModelTrainingPipeline(config)
    train_results = trainer.run(hourly_df)
    assert "best_model_name" in train_results
    assert "registry_info" in train_results

    # 5. Online Forecast Service & Capacity Risk
    service = ForecastService(config)
    test_station = hourly_df["station_id"].iloc[0]
    forecast_res = service.predict_station_demand(
        station_id=test_station,
        horizon_hours=24,
        df_history=hourly_df,
    )
    assert forecast_res["station_id"] == test_station
    assert len(forecast_res["forecast"]) == 24
    assert "expansion_scenarios" in forecast_res
    assert len(forecast_res["expansion_scenarios"]) > 0

    # 6. Audit Trail & Lineage Verification
    audit_logger = AuditLogger(config)
    logs_df = audit_logger.get_recent_audit_logs(limit=10)
    assert not logs_df.empty

    lineage = LineageTracker(config)
    lineage_graph = lineage.get_complete_lineage(forecast_res["prediction_id"])
    assert lineage_graph["dataset_provenance"]["sha256"] == manifest["sha256"]
    assert lineage_graph["model_provenance"]["model_name"] == train_results["best_model_name"]
