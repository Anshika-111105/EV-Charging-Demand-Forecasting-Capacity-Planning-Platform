"""Tests for FastAPI HTTP endpoints."""

import pytest
from fastapi.testclient import TestClient

from ev_forecasting.api.main import app
from ev_forecasting.models.seasonal_naive import SeasonalNaiveForecaster


@pytest.fixture
def client(synthetic_hourly, test_config):
    """Set up TestClient with a pre-warmed production model."""
    from ev_forecasting.api.dependencies import get_forecast_service

    svc = get_forecast_service()
    model = SeasonalNaiveForecaster(seasonality=24)
    model.fit(synthetic_hourly)
    svc.model = model
    svc.metadata = {
        "model_name": "seasonal_naive_24",
        "model_type": "baseline",
        "model_version": "v_test",
        "dataset_version": "1.0.0",
        "dataset_hash": "test_hash",
        "feature_version": "v1.0",
        "metrics": {"rmse": 1.5, "mae": 1.0},
    }

    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "seasonal_naive_24"
    assert "metrics" in data


def test_stations_endpoint(client):
    response = client.get("/stations")
    assert response.status_code == 200
    stations = response.json()
    assert len(stations) > 0
    assert "station_id" in stations[0]


def test_capacity_analyze_endpoint(client):
    payload = {
        "station_id": "test_st_01",
        "custom_capacity_kwh": 50.0,
        "forecast_kwh_values": [10.0, 25.0, 48.0, 52.0],
    }
    response = client.post("/capacity/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["station_id"] == "test_st_01"
    assert data["risk_level"] == "CRITICAL"  # 52 / 50 = 104% > 95%
    assert data["critical_hours"] >= 1


def test_invalid_forecast_request(client):
    # Horizon 0 is invalid (must be >= 1)
    payload = {"station_id": "caltech_ST_001", "horizon_hours": 0}
    response = client.post("/forecast", json=payload)
    assert response.status_code == 422  # Pydantic validation error
