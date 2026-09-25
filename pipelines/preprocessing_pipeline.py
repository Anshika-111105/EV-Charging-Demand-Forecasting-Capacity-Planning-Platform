"""CLI Pipeline for raw session parsing, validation, cleaning, and hourly aggregation."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ev_forecasting.config.settings import load_config
from ev_forecasting.data.aggregation import TimeSeriesAggregator
from ev_forecasting.data.cleaning import SessionDataCleaner
from ev_forecasting.data.parsing import ACNDataParser
from ev_forecasting.data.validation import DataQualityValidator
from ev_forecasting.features.pipeline import FeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocessing_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Preprocess EV charging sessions to hourly demand")
    parser.add_argument(
        "--config", type=str, default="configs/development.yaml", help="Config file path"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Starting preprocessing pipeline (%s)...", args.config)

    # 1. Parse raw sessions
    parser_module = ACNDataParser(config)
    sessions_df, _ = parser_module.parse_all_sessions()
    if sessions_df.empty:
        logger.error(
            "No sessions found to process in %s. Run ingest_pipeline first.",
            config.paths.raw_data_dir,
        )
        sys.exit(1)

    # 2. Validate session quality gates
    validator = DataQualityValidator(config)
    passed_sessions, sess_report = validator.validate_sessions_dataframe(
        sessions_df, raise_on_failure=False
    )
    logger.info(
        "Session validation status: %s (Valid: %d/%d)",
        sess_report["status"],
        sess_report["valid_sessions"],
        sess_report["total_sessions"],
    )

    # 3. Clean sessions
    cleaner = SessionDataCleaner(config)
    cleaned_sessions = cleaner.clean_sessions(sessions_df)

    # 4. Hourly aggregation with proper hour-boundary energy allocation
    aggregator = TimeSeriesAggregator(config)
    hourly_raw = aggregator.allocate_session_energy_hourly(cleaned_sessions)
    hourly_complete = aggregator.build_regular_grid(hourly_raw)

    # 5. Validate hourly time series
    passed_hourly, hourly_report = validator.validate_hourly_dataframe(
        hourly_complete, raise_on_failure=True
    )
    logger.info(
        "Hourly dataset validation status: %s (%d records, %d stations)",
        hourly_report["status"],
        hourly_report["total_records"],
        hourly_report["unique_stations"],
    )

    # 6. Fit feature pipeline and save feature manifest
    feat_pipeline = FeaturePipeline(config)
    _, feat_cols = feat_pipeline.fit_transform(hourly_complete)
    logger.info("Engineered %d features and updated feature manifest.", len(feat_cols))


if __name__ == "__main__":
    main()
