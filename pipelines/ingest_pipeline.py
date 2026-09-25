"""CLI Pipeline for ACN-Data ingestion and manifest generation."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ev_forecasting.config.settings import load_config
from ev_forecasting.data.ingestion import ACNDataIngestor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Ingest ACN-Data static snapshot")
    parser.add_argument(
        "--config", type=str, default="configs/development.yaml", help="Config file path"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Starting ingestion pipeline with config: %s", args.config)

    ingestor = ACNDataIngestor(config)
    stats = ingestor.ingest_remote_dataset()
    logger.info("Ingestion complete. Stats: %s", stats)

    manifest = ingestor.generate_manifest()
    logger.info(
        "Generated dataset manifest: %d files, %d stations, sha256=%s...",
        manifest["total_files"],
        manifest["stations_count"],
        manifest["sha256"][:12],
    )


if __name__ == "__main__":
    main()
