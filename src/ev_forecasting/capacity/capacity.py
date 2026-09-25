"""Station capacity configuration and management."""

from __future__ import annotations

from typing import Dict

from ev_forecasting.config.settings import AppConfig


class StationCapacityManager:
    """Manages physical power and energy capacities per EV charging station."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.default_capacity = config.capacity.default_station_capacity_kwh
        self.station_capacities: Dict[str, float] = {}
        self._init_site_capacities()

    def _init_site_capacities(self) -> None:
        """Initialize capacities from site configurations."""
        for site in self.config.dataset.sites:
            pass  # Default capacities can be customized per site if needed

    def get_station_capacity(self, station_id: str) -> float:
        """Get capacity in kWh/hour for a station."""
        return self.station_capacities.get(station_id, self.default_capacity)

    def set_station_capacity(self, station_id: str, capacity_kwh: float) -> None:
        """Set explicit capacity for a specific station."""
        self.station_capacities[station_id] = float(capacity_kwh)
