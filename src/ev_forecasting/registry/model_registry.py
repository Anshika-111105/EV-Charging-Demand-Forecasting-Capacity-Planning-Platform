"""Model registry, versioning, and promotion management."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ev_forecasting.config.settings import AppConfig
from ev_forecasting.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class ModelRegistryManager:
    """Manages model artifact persistence, registry metadata, and promotion quality gates."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.artifacts_dir = Path(config.paths.model_artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.artifacts_dir / "registry_metadata.json"

    def register_and_promote(
        self,
        model: BaseForecaster,
        metrics: Dict[str, Any],
        dataset_hash: str,
        feature_version: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save model as current production model and record complete provenance metadata."""
        version_str = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
        model_file = self.artifacts_dir / f"{model.name}_{version_str}.pkl"
        prod_model_file = self.artifacts_dir / "production_model.pkl"

        # Save specific versioned artifact
        model.save(model_file)
        # Save canonical production artifact
        model.save(prod_model_file)

        registry_record = {
            "model_name": model.name,
            "model_type": model.model_type,
            "model_version": version_str,
            "stage": "Production",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": run_id or "local_run",
            "dataset_hash": dataset_hash,
            "feature_version": feature_version,
            "target_variable": self.config.dataset.target_variable,
            "metrics": metrics,
            "model_artifact_path": str(prod_model_file),
            "versioned_artifact_path": str(model_file),
            "feature_columns": model.feature_columns,
        }

        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry_record, f, indent=2)

        logger.info(
            "Registered and promoted model '%s' (%s) to Production stage at %s",
            model.name,
            version_str,
            prod_model_file,
        )
        return registry_record

    def get_production_model_metadata(self) -> Optional[Dict[str, Any]]:
        """Retrieve current production model registry metadata."""
        if not self.registry_file.exists():
            return None
        with open(self.registry_file, "r", encoding="utf-8") as f:
            return json.load(f)
