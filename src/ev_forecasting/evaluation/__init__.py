"""Evaluation and benchmarking package."""

from ev_forecasting.evaluation.backtesting import BacktestingEngine
from ev_forecasting.evaluation.error_analysis import ErrorAnalyzer
from ev_forecasting.evaluation.metrics import (
    calculate_mae,
    calculate_mape,
    calculate_peak_error,
    calculate_rmse,
    calculate_smape,
    calculate_wape,
    evaluate_forecast,
)
from ev_forecasting.evaluation.model_comparison import ModelComparator

__all__ = [
    "calculate_mae",
    "calculate_rmse",
    "calculate_mape",
    "calculate_smape",
    "calculate_wape",
    "calculate_peak_error",
    "evaluate_forecast",
    "BacktestingEngine",
    "ModelComparator",
    "ErrorAnalyzer",
]
