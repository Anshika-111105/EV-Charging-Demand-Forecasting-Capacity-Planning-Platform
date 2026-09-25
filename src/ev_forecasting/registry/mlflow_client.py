"""MLflow experiment tracking client and artifact manager."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

from ev_forecasting.config.settings import AppConfig

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Manages experiment tracking, hyperparameter logging, and model artifact storage."""

    def __init__(self, config: AppConfig):
        self.config = config
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        self.tracking_uri = config.tracking_uri
        if (
            not self.tracking_uri.startswith("http")
            and not self.tracking_uri.startswith("sqlite")
            and not self.tracking_uri.startswith("postgresql")
        ):
            self.tracking_uri = f"sqlite:///{Path(self.tracking_uri).resolve()}/../mlflow.db"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment_name = config.registry.experiment_name
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self._setup_experiment()

    def _setup_experiment(self) -> str:
        """Create or retrieve active MLflow experiment."""
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            artifact_loc = Path(self.config.paths.model_artifacts_dir).resolve().as_uri()
            exp_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=artifact_loc,
            )
        else:
            exp_id = exp.experiment_id
        mlflow.set_experiment(self.experiment_name)
        return exp_id

    def log_training_run(
        self,
        model_name: str,
        model_type: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        dataset_hash: str,
        feature_version: str,
        artifacts: Optional[Dict[str, Path]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Log a complete training & evaluation run to MLflow."""
        run_name = f"{model_name}_{dataset_hash[:8]}"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Log tags
            mlflow.set_tags(
                {
                    "model_name": model_name,
                    "model_type": model_type,
                    "dataset_hash": dataset_hash,
                    "feature_version": feature_version,
                    "environment": self.config.environment,
                }
            )
            if tags:
                mlflow.set_tags(tags)

            # Log parameters
            clean_params = {
                k: str(v) if isinstance(v, (list, dict)) else v for k, v in params.items()
            }
            mlflow.log_params(clean_params)

            # Log metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            # Log artifacts (e.g. plot figures, model card, config)
            if artifacts:
                for art_name, art_path in artifacts.items():
                    if art_path and Path(art_path).exists():
                        try:
                            mlflow.log_artifact(
                                str(Path(art_path).resolve()), artifact_path="evaluation_artifacts"
                            )
                        except Exception as e:
                            logger.warning("Could not log MLflow artifact %s: %s", art_name, e)

            logger.info("Logged MLflow run: %s (Run ID: %s)", run_name, run_id)
            return run_id
