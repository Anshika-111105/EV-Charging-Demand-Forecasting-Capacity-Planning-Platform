"""CLI Pipeline for data quality and statistical drift monitoring."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ev_forecasting.config.settings import load_config
from ev_forecasting.features.pipeline import FeaturePipeline
from ev_forecasting.monitoring.alerts import MonitoringManager
from ev_forecasting.splitting.temporal_split import TemporalSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("monitoring_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Evaluate drift and monitoring status")
    parser.add_argument(
        "--config", type=str, default="configs/development.yaml", help="Config file path"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Starting monitoring pipeline (%s)...", args.config)

    hourly_path = Path(config.paths.processed_data_dir) / "hourly_demand.parquet"
    if not hourly_path.exists():
        logger.error("Processed dataset not found at %s", hourly_path)
        sys.exit(1)

    df_hourly = pd.read_parquet(hourly_path)
    splitter = TemporalSplitter(config.splitting)
    train_df, _, test_df = splitter.split(df_hourly)

    feat_pipeline = FeaturePipeline(config)
    train_feat, feat_names = feat_pipeline.fit_transform(train_df)
    test_feat = feat_pipeline.transform(test_df)

    # Check drift across target and key lag/rolling features
    key_features = ["energy_demand_kwh", "lag_1h", "lag_24h", "rolling_mean_24h"]
    valid_keys = [k for k in key_features if k in train_feat.columns and k in test_feat.columns]

    mgr = MonitoringManager(config)
    report = mgr.evaluate_monitoring_state(train_feat, test_feat, valid_keys)

    # Save monitoring report
    report_file = Path(config.paths.reports_dir) / "monitoring_drift_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Monitoring check status: %s (Retrain recommended: %s)",
        report["status"],
        report["retraining_recommended"],
    )
    logger.info("Saved monitoring report to %s", report_file)


if __name__ == "__main__":
    main()
