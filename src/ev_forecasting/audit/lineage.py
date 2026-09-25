"""End-to-end data and model lineage tracker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ev_forecasting.config.settings import AppConfig


class LineageTracker:
    """Provides complete traceability from Raw Data -> Features -> Model -> Prediction."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.manifest_dir = Path(config.paths.manifest_dir)

    def get_complete_lineage(self, prediction_id: Optional[str] = None) -> Dict[str, Any]:
        """Construct full lineage graph dictionary."""
        data_manifest_file = self.manifest_dir / "dataset_manifest.json"
        feature_manifest_file = self.manifest_dir / "feature_manifest.json"
        registry_file = Path(self.config.paths.model_artifacts_dir) / "registry_metadata.json"

        data_manifest = {}
        if data_manifest_file.exists():
            with open(data_manifest_file, "r", encoding="utf-8") as f:
                data_manifest = json.load(f)

        feature_manifest = {}
        if feature_manifest_file.exists():
            with open(feature_manifest_file, "r", encoding="utf-8") as f:
                feature_manifest = json.load(f)

        registry_metadata = {}
        if registry_file.exists():
            with open(registry_file, "r", encoding="utf-8") as f:
                registry_metadata = json.load(f)

        return {
            "prediction_id": prediction_id or "latest_inference",
            "dataset_provenance": {
                "name": data_manifest.get("dataset_name", self.config.dataset.name),
                "source": data_manifest.get("source_url", self.config.dataset.source_url),
                "sha256": data_manifest.get("sha256", "unverified"),
                "total_files": data_manifest.get("total_files", 0),
                "ingested_at": data_manifest.get("ingested_at", "unknown"),
            },
            "feature_provenance": {
                "version": feature_manifest.get("feature_version", self.config.features.version),
                "total_features": feature_manifest.get("total_features", 0),
                "feature_names": feature_manifest.get("feature_names", []),
            },
            "model_provenance": {
                "model_name": registry_metadata.get("model_name", "unregistered"),
                "model_version": registry_metadata.get("model_version", "v0"),
                "stage": registry_metadata.get("stage", "None"),
                "mlflow_run_id": registry_metadata.get("mlflow_run_id", "local"),
                "promoted_at": registry_metadata.get("promoted_at", "unknown"),
                "metrics": registry_metadata.get("metrics", {}),
            },
        }
