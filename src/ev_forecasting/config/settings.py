"""Configuration loader and schemas for EV Demand Forecasting Platform."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class SiteConfig(BaseModel):
    site_id: str
    name: str
    location: str
    nominal_voltage_v: float = 208.0
    default_station_capacity_kwh: float = 50.0


class DatasetConfig(BaseModel):
    name: str = "ACN-Data"
    source_url: str = "https://ev.caltech.edu/dataset"
    static_snapshot_url: str = "https://raw.githubusercontent.com/tongxin-li/ACN-Data-Static/main"
    version: str = "1.0.0"
    target_variable: str = "energy_demand_kwh"
    frequency: str = "1h"
    timezone: str = "UTC"
    sites: List[SiteConfig] = Field(default_factory=list)


class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    interim_data_dir: str = "data/interim"
    processed_data_dir: str = "data/processed"
    manifest_dir: str = "data/manifests"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"
    metrics_dir: str = "reports/metrics"
    backtests_dir: str = "reports/backtests"
    model_cards_dir: str = "reports/model_cards"
    model_artifacts_dir: str = "models/checkpoints"


class CalendarFeaturesConfig(BaseModel):
    include_hour: bool = True
    include_day_of_week: bool = True
    include_day_of_month: bool = True
    include_month: bool = True
    include_is_weekend: bool = True
    include_cyclical: bool = True


class LagFeaturesConfig(BaseModel):
    hours: List[int] = Field(default_factory=lambda: [1, 2, 3, 24, 48, 72, 168])


class RollingFeaturesConfig(BaseModel):
    windows: List[int] = Field(default_factory=lambda: [24, 168])
    statistics: List[str] = Field(default_factory=lambda: ["mean", "std", "min", "max"])


class FeaturesConfig(BaseModel):
    version: str = "v1.0"
    calendar: CalendarFeaturesConfig = Field(default_factory=CalendarFeaturesConfig)
    lags: LagFeaturesConfig = Field(default_factory=LagFeaturesConfig)
    rolling: RollingFeaturesConfig = Field(default_factory=RollingFeaturesConfig)


class SplittingConfig(BaseModel):
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


class BacktestingConfig(BaseModel):
    horizons: List[int] = Field(default_factory=lambda: [24, 48, 168])
    primary_horizon: int = 168
    min_train_hours: int = 720
    step_hours: int = 168
    n_folds: int = 3


class BaselineModelConfig(BaseModel):
    seasonal_periods: List[int] = Field(default_factory=lambda: [24, 168])


class ARIMAModelConfig(BaseModel):
    order: List[int] = Field(default_factory=lambda: [2, 1, 1])


class SARIMAModelConfig(BaseModel):
    order: List[int] = Field(default_factory=lambda: [1, 1, 1])
    seasonal_order: List[int] = Field(default_factory=lambda: [1, 0, 1, 24])


class ProphetModelConfig(BaseModel):
    growth: str = "linear"
    daily_seasonality: bool = True
    weekly_seasonality: bool = True
    yearly_seasonality: bool = False


class XGBoostModelConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


class ModelsConfig(BaseModel):
    baseline: BaselineModelConfig = Field(default_factory=BaselineModelConfig)
    arima: ARIMAModelConfig = Field(default_factory=ARIMAModelConfig)
    sarima: SARIMAModelConfig = Field(default_factory=SARIMAModelConfig)
    prophet: ProphetModelConfig = Field(default_factory=ProphetModelConfig)
    xgboost: XGBoostModelConfig = Field(default_factory=XGBoostModelConfig)


class PromotionCriteriaConfig(BaseModel):
    max_mape_threshold: float = 30.0
    must_beat_baseline: bool = True
    baseline_model: str = "seasonal_naive_168"
    metric_priority: List[str] = Field(default_factory=lambda: ["rmse", "mae", "mape"])


class RegistryConfig(BaseModel):
    experiment_name: str = "ev_demand_forecasting"
    registered_model_name: str = "ev_charging_demand_forecaster"
    promotion_criteria: PromotionCriteriaConfig = Field(default_factory=PromotionCriteriaConfig)


class ScenarioConfig(BaseModel):
    name: str
    multiplier: float


class CapacityThresholdsConfig(BaseModel):
    low: float = 70.0
    medium: float = 85.0
    high: float = 95.0


class CapacityConfig(BaseModel):
    default_station_capacity_kwh: float = 50.0
    thresholds: CapacityThresholdsConfig = Field(default_factory=CapacityThresholdsConfig)
    scenarios: List[ScenarioConfig] = Field(default_factory=list)


class RetrainingPolicyConfig(BaseModel):
    min_new_samples: int = 168
    performance_degradation_pct: float = 15.0


class MonitoringConfig(BaseModel):
    drift_significance_level: float = 0.05
    psi_threshold_moderate: float = 0.10
    psi_threshold_significant: float = 0.25
    retraining_policy: RetrainingPolicyConfig = Field(default_factory=RetrainingPolicyConfig)


class IngestionSettings(BaseModel):
    max_files_per_site: int = 100
    fetch_remote_if_missing: bool = True


class AppConfig(BaseModel):
    environment: str = "development"
    tracking_uri: str = "./mlruns"
    database_url: str = "sqlite:///./audit.db"
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = dict(dict1)
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from base.yaml and optional environment overrides."""
    base_path = Path("configs/base.yaml")
    config_dict: Dict[str, Any] = {}
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

    if config_path:
        override_path = Path(config_path)
        if override_path.exists():
            with open(override_path, "r", encoding="utf-8") as f:
                override_dict = yaml.safe_load(f) or {}
                config_dict = deep_merge(config_dict, override_dict)

    # Environment variable overrides
    if os.getenv("ENVIRONMENT"):
        config_dict["environment"] = os.getenv("ENVIRONMENT")
    if os.getenv("MLFLOW_TRACKING_URI"):
        config_dict["tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI")
    if os.getenv("DATABASE_URL"):
        config_dict["database_url"] = os.getenv("DATABASE_URL")

    return AppConfig(**config_dict)
