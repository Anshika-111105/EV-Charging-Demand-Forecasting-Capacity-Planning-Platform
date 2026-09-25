"""CLI Pipeline for batch forecasting generation."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ev_forecasting.config.settings import load_config
from ev_forecasting.forecasting.batch_forecast import BatchForecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("forecasting_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Generate batch forecasts across all stations")
    parser.add_argument(
        "--config", type=str, default="configs/development.yaml", help="Config file path"
    )
    parser.add_argument("--horizon", type=int, default=168, help="Forecast horizon in hours")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(
        "Starting batch forecasting pipeline (%s, horizon=%dh)...", args.config, args.horizon
    )

    batcher = BatchForecaster(config)
    out_file = batcher.run_batch(horizon_hours=args.horizon)
    logger.info("Batch forecasting finished. Output file: %s", out_file)


if __name__ == "__main__":
    main()
