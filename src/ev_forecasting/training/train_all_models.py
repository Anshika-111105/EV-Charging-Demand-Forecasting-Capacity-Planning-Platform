"""Training, backtesting, and model promotion pipeline execution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ev_forecasting.config.settings import AppConfig
from ev_forecasting.evaluation.backtesting import BacktestingEngine
from ev_forecasting.evaluation.error_analysis import ErrorAnalyzer
from ev_forecasting.evaluation.model_comparison import ModelComparator
from ev_forecasting.features.pipeline import FeaturePipeline
from ev_forecasting.models.arima import ARIMAForecaster
from ev_forecasting.models.base import BaseForecaster
from ev_forecasting.models.prophet import ProphetForecaster
from ev_forecasting.models.sarima import SARIMAForecaster
from ev_forecasting.models.seasonal_naive import SeasonalNaiveForecaster
from ev_forecasting.models.xgboost import XGBoostForecaster
from ev_forecasting.registry.mlflow_client import MLflowTracker
from ev_forecasting.registry.model_registry import ModelRegistryManager
from ev_forecasting.splitting.temporal_split import TemporalSplitter

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """Orchestrates end-to-end multi-model benchmarking, backtesting, and registry promotion."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.mlflow_tracker = MLflowTracker(config)
        self.registry_mgr = ModelRegistryManager(config)
        self.comparator = ModelComparator(config)
        self.error_analyzer = ErrorAnalyzer(config)
        self.backtest_engine = BacktestingEngine(config)

    def run(self, df_hourly: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Execute the full training, backtesting, comparison, and promotion workflow."""
        if df_hourly is None:
            hourly_path = Path(self.config.paths.processed_data_dir) / "hourly_demand.parquet"
            if not hourly_path.exists():
                raise FileNotFoundError(f"Processed dataset not found at {hourly_path}")
            df_hourly = pd.read_parquet(hourly_path)

        # 1. Read dataset manifest to obtain dataset hash
        manifest_path = Path(self.config.paths.manifest_dir) / "dataset_manifest.json"
        dataset_hash = "unknown_hash"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                dataset_hash = manifest_data.get("sha256", "unknown_hash")

        # 2. Chronological temporal split
        splitter = TemporalSplitter(self.config.splitting)
        train_df, val_df, test_df = splitter.split(df_hourly)

        # 3. Fit feature pipeline on train split
        feat_pipeline = FeaturePipeline(self.config)
        train_feat_df, feature_cols = feat_pipeline.fit_transform(train_df)
        feat_pipeline.transform(df_hourly)

        # 4. Instantiate candidate models
        models_to_benchmark: List[BaseForecaster] = [
            SeasonalNaiveForecaster(seasonality=24, name="seasonal_naive_24"),
            SeasonalNaiveForecaster(seasonality=168, name="seasonal_naive_168"),
            ARIMAForecaster(order=tuple(self.config.models.arima.order), name="arima"),
            SARIMAForecaster(
                order=tuple(self.config.models.sarima.order),
                seasonal_order=tuple(self.config.models.sarima.seasonal_order),
                name="sarima",
            ),
            ProphetForecaster(
                daily_seasonality=self.config.models.prophet.daily_seasonality,
                weekly_seasonality=self.config.models.prophet.weekly_seasonality,
                name="prophet",
            ),
            XGBoostForecaster(
                config=self.config.models.xgboost,
                feature_pipeline=feat_pipeline,
                name="xgboost_global",
            ),
        ]

        # 5. Run expanding-window backtesting for each model
        all_evals = []
        fitted_models: Dict[str, BaseForecaster] = {}

        for model in models_to_benchmark:
            logger.info(">>> Running Walk-Forward Backtesting for %s <<<", model.name)
            model_eval_df = self.backtest_engine.run_backtest_for_model(model, df_hourly)
            all_evals.append(model_eval_df)
            fitted_models[model.name] = model

        combined_eval_df = pd.concat(all_evals, ignore_index=True)

        # 6. Generate comparison table & rank models
        comparison_df = self.comparator.generate_comparison_table(combined_eval_df)

        # 7. Generate comparison visualizations
        bar_chart_path = self.error_analyzer.plot_model_comparison_bar(comparison_df)

        # 8. Log runs to MLflow
        run_ids = {}
        for m_name, group in combined_eval_df.groupby("model_name"):
            metrics_dict = {
                "rmse_mean": float(group["rmse"].mean()),
                "mae_mean": float(group["mae"].mean()),
                "mape_mean": float(group["mape"].mean()),
                "smape_mean": float(group["smape"].mean()),
                "wape_mean": float(group["wape"].mean()),
            }
            run_id = self.mlflow_tracker.log_training_run(
                model_name=m_name,
                model_type=group["model_type"].iloc[0],
                params={"config_version": self.config.dataset.version},
                metrics=metrics_dict,
                dataset_hash=dataset_hash,
                feature_version=self.config.features.version,
                artifacts={"comparison_plot": bar_chart_path},
            )
            run_ids[m_name] = run_id

        # 9. Select production candidate using quality gates
        best_model_name, best_details = self.comparator.select_production_candidate(comparison_df)
        winning_model = fitted_models[best_model_name]

        # Train winning model on entire historical dataset up to test split
        train_val_combined = pd.concat([train_df, val_df], ignore_index=True)
        if isinstance(winning_model, XGBoostForecaster):
            winning_feat_df, _ = feat_pipeline.fit_transform(train_val_combined)
            winning_model.fit(winning_feat_df, feature_columns=feature_cols)
        else:
            winning_model.fit(train_val_combined)

        # 10. Register and promote winning model
        registry_info = self.registry_mgr.register_and_promote(
            model=winning_model,
            metrics=best_details,
            dataset_hash=dataset_hash,
            feature_version=self.config.features.version,
            run_id=run_ids.get(best_model_name, "prod_run"),
        )

        # 11. Plot actual vs predicted on test set for winning model
        test_preds = winning_model.predict(
            horizon_hours=self.config.backtesting.primary_horizon,
            df_history=train_val_combined,
            df_future=feat_pipeline.transform(test_df)
            if isinstance(winning_model, XGBoostForecaster)
            else test_df,
        )
        test_merged = pd.merge(
            test_df[["station_id", "timestamp", "energy_demand_kwh"]],
            test_preds[["station_id", "timestamp", "forecast_kwh"]],
            on=["station_id", "timestamp"],
            how="inner",
        )
        if not test_merged.empty:
            st_example = test_merged["station_id"].iloc[0]
            self.error_analyzer.plot_actual_vs_predicted(
                test_merged,
                station_id=st_example,
                title=f"Production Forecast ({best_model_name}) - {st_example}",
                filename="production_actual_vs_predicted.png",
            )
            self.error_analyzer.plot_residual_distribution(test_merged)

        return {
            "best_model_name": best_model_name,
            "registry_info": registry_info,
            "comparison_summary": comparison_df.to_dict(orient="records"),
        }
