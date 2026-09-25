"""Unit tests for capacity planning and risk tier calculations."""

import pandas as pd

from ev_forecasting.capacity.risk import CapacityRiskEngine
from ev_forecasting.config.settings import load_config


def test_capacity_and_risk():
    config = load_config("configs/test.yaml")
    engine = CapacityRiskEngine(config)

    # Test risk tier classification
    assert engine.classify_risk(50.0) == "LOW"
    assert engine.classify_risk(75.0) == "MEDIUM"
    assert engine.classify_risk(90.0) == "HIGH"
    assert engine.classify_risk(98.0) == "CRITICAL"

    # Test utilization calculation on dataframe
    df_fc = pd.DataFrame(
        {
            "station_id": ["caltech_ST_001", "caltech_ST_001", "caltech_ST_001"],
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="1h", tz="UTC"),
            "forecast_kwh": [25.0, 45.0, 50.0],  # Capacities are 50 kWh
        }
    )

    analyzed = engine.analyze_forecast(df_fc)

    assert "utilization_pct" in analyzed.columns
    assert "risk_level" in analyzed.columns
    assert analyzed["utilization_pct"].iloc[0] == 50.0
    assert analyzed["risk_level"].iloc[0] == "LOW"
    assert analyzed["utilization_pct"].iloc[1] == 90.0
    assert analyzed["risk_level"].iloc[1] == "HIGH"
    assert analyzed["utilization_pct"].iloc[2] == 100.0
    assert analyzed["risk_level"].iloc[2] == "CRITICAL"
