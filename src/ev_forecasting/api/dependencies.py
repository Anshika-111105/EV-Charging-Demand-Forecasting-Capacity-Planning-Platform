"""API dependency injection."""

from __future__ import annotations

from typing import Optional

from ev_forecasting.audit.prediction_log import AuditLogger
from ev_forecasting.config.settings import AppConfig, load_config
from ev_forecasting.forecasting.forecast import ForecastService
from ev_forecasting.monitoring.alerts import MonitoringManager

_config: Optional[AppConfig] = None
_forecast_service: Optional[ForecastService] = None
_audit_logger: Optional[AuditLogger] = None
_monitoring_mgr: Optional[MonitoringManager] = None


def get_app_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_forecast_service() -> ForecastService:
    global _forecast_service
    if _forecast_service is None:
        _forecast_service = ForecastService(get_app_config())
    return _forecast_service


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(get_app_config())
    return _audit_logger


def get_monitoring_manager() -> MonitoringManager:
    global _monitoring_mgr
    if _monitoring_mgr is None:
        _monitoring_mgr = MonitoringManager(get_app_config())
    return _monitoring_mgr
