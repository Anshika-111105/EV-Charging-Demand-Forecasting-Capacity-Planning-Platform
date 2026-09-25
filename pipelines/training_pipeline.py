"""CLI Pipeline for multi-model training, expanding backtesting, and model registry promotion."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ev_forecasting.config.settings import load_config
from ev_forecasting.training.train_all_models import ModelTrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("training_pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="Train, backtest, and promote EV forecasting models"
    )
    parser.add_argument(
        "--config", type=str, default="configs/development.yaml", help="Config file path"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Starting model training pipeline (%s)...", args.config)

    trainer = ModelTrainingPipeline(config)
    results = trainer.run()

    logger.info("Training pipeline completed successfully!")
    logger.info("Winning production model: %s", results["best_model_name"])
    logger.info("Registry promotion info: %s", results["registry_info"])


if __name__ == "__main__":
    main()
