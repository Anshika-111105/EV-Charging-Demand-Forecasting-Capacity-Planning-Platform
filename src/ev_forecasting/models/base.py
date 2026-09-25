"""Base Forecaster Abstract Class and Interfaces."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class BaseForecaster(abc.ABC):
    """Abstract base class for all time-series forecasting models."""

    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
        self.is_fitted = False
        self.fitted_at: Optional[str] = None
        self.training_duration_seconds: float = 0.0
        self.target_col: str = "energy_demand_kwh"
        self.timestamp_col: str = "timestamp"
        self.feature_columns: List[str] = []

    @abc.abstractmethod
    def fit(self, df_train: pd.DataFrame, **kwargs: Any) -> BaseForecaster:
        """Fit model on historical training dataframe."""
        pass

    @abc.abstractmethod
    def predict(
        self,
        horizon_hours: int,
        df_history: Optional[pd.DataFrame] = None,
        df_future: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate forecasts for the specified horizon."""
        pass

    @abc.abstractmethod
    def save(self, filepath: Path) -> None:
        """Serialize model to disk."""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, filepath: Path) -> BaseForecaster:
        """Deserialize model from disk."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata dictionary."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
            "fitted_at": self.fitted_at,
            "training_duration_seconds": self.training_duration_seconds,
            "feature_columns": self.feature_columns,
        }
