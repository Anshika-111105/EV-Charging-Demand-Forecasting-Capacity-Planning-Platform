"""Error analysis, residual diagnostics, and automated figure generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ev_forecasting.config.settings import AppConfig

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Computes residual diagnostics and generates publication-quality figures."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.figures_dir = Path(config.paths.figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def plot_actual_vs_predicted(
        self,
        df_pred: pd.DataFrame,
        station_id: Optional[str] = None,
        title: str = "Actual vs Predicted EV Demand",
        filename: str = "actual_vs_predicted.png",
    ) -> Path:
        """Plot actual vs predicted time series."""
        fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

        plot_df = df_pred.copy()
        if station_id and "station_id" in plot_df.columns:
            plot_df = plot_df[plot_df["station_id"] == station_id]

        # Plot first 168 hours if too long
        if len(plot_df) > 336:
            plot_df = plot_df.iloc[:336]

        ax.plot(
            plot_df["timestamp"],
            plot_df["energy_demand_kwh"],
            label="Actual Demand (kWh)",
            color="#2563EB",
            linewidth=2.0,
            marker="o",
            markersize=3,
        )
        ax.plot(
            plot_df["timestamp"],
            plot_df["forecast_kwh"],
            label="Forecast Demand (kWh)",
            color="#DC2626",
            linestyle="--",
            linewidth=2.0,
            marker="x",
            markersize=3,
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Timestamp (UTC)", fontsize=11)
        ax.set_ylabel("Energy Demand (kWh)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        out_path = self.figures_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info("Saved actual vs predicted figure to %s", out_path)
        return out_path

    def plot_residual_distribution(
        self, df_pred: pd.DataFrame, filename: str = "residual_distribution.png"
    ) -> Path:
        """Plot histogram and KDE of forecasting residuals."""
        residuals = df_pred["energy_demand_kwh"] - df_pred["forecast_kwh"]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        sns.histplot(residuals, kde=True, ax=ax, color="#4F46E5", bins=40)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")

        ax.set_title(
            "Forecast Residual Distribution ($y - \\hat{y}$)", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Residual (kWh)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()

        out_path = self.figures_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info("Saved residual distribution figure to %s", out_path)
        return out_path

    def plot_model_comparison_bar(
        self, comparison_df: pd.DataFrame, filename: str = "model_comparison_bar.png"
    ) -> Path:
        """Plot side-by-side comparison of RMSE and MAE across models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Plot for primary horizon
        p_horizon = self.config.backtesting.primary_horizon
        h_df = comparison_df[comparison_df["horizon_hours"] == p_horizon]
        if h_df.empty:
            h_df = comparison_df

        models = h_df["model_name"].tolist()
        rmse_vals = h_df["rmse_mean"].tolist()
        mae_vals = h_df["mae_mean"].tolist()

        colors = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EF4444"][: len(models)]

        ax1.bar(models, rmse_vals, color=colors, edgecolor="black", alpha=0.85)
        ax1.set_title(f"RMSE by Model (Horizon: {p_horizon}h)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("RMSE (kWh)", fontsize=11)
        ax1.tick_params(axis="x", rotation=30)
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        ax2.bar(models, mae_vals, color=colors, edgecolor="black", alpha=0.85)
        ax2.set_title(f"MAE by Model (Horizon: {p_horizon}h)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("MAE (kWh)", fontsize=11)
        ax2.tick_params(axis="x", rotation=30)
        ax2.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        out_path = self.figures_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        logger.info("Saved model comparison bar chart to %s", out_path)
        return out_path
