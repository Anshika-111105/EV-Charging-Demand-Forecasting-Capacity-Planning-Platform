"""Hourly time-series aggregation for multi-station EV charging demand."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ev_forecasting.config.settings import AppConfig

logger = logging.getLogger(__name__)


class TimeSeriesAggregator:
    """Aggregates irregular charging sessions into regular hourly station-level demand series."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.processed_dir = Path(config.paths.processed_data_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def allocate_session_energy_hourly(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Distribute each session's energy across the exact hourly bins it spans."""
        hourly_records = []

        for _, row in sessions_df.iterrows():
            st = row["start_time"]
            et = row["end_time"]
            total_kwh = float(row["kwh_delivered"])
            station_id = str(row["station_id"])
            site_id = str(row["site_id"])
            session_id = str(row["session_id"])

            if pd.isna(st) or pd.isna(et) or total_kwh < 0:
                continue

            # Floor start time and ceil end time to hourly boundaries
            st_hour = st.floor("1h")
            et_hour = et.floor("1h")

            if st_hour == et_hour:
                # Entire session fits in single hour
                hourly_records.append(
                    {
                        "station_id": station_id,
                        "site_id": site_id,
                        "timestamp": st_hour,
                        "energy_kwh": total_kwh,
                        "session_count": 1,
                        "session_id": session_id,
                    }
                )
            else:
                # Multi-hour session: calculate exact overlap duration per hour
                total_duration_sec = max((et - st).total_seconds(), 1.0)
                current_h = st_hour
                while current_h <= et_hour:
                    next_h = current_h + pd.Timedelta(hours=1)
                    overlap_start = max(st, current_h)
                    overlap_end = min(et, next_h)
                    overlap_sec = max((overlap_end - overlap_start).total_seconds(), 0.0)

                    if overlap_sec > 0 and total_duration_sec > 0:
                        fraction = overlap_sec / total_duration_sec
                        hourly_records.append(
                            {
                                "station_id": station_id,
                                "site_id": site_id,
                                "timestamp": current_h,
                                "energy_kwh": total_kwh * fraction,
                                "session_count": 1,
                                "session_id": session_id,
                            }
                        )
                    current_h = next_h

        if not hourly_records:
            return pd.DataFrame(
                columns=["station_id", "site_id", "timestamp", "energy_demand_kwh", "session_count"]
            )

        raw_hourly = pd.DataFrame(hourly_records)
        # Aggregate across simultaneous sessions in the same hour
        aggregated = (
            raw_hourly.groupby(["station_id", "site_id", "timestamp"])
            .agg(
                energy_demand_kwh=("energy_kwh", "sum"),
                session_count=("session_count", "sum"),
                active_sessions=("session_id", "nunique"),
            )
            .reset_index()
        )

        return aggregated

    def build_regular_grid(
        self, aggregated_df: pd.DataFrame, min_station_records: int = 10
    ) -> pd.DataFrame:
        """Create a complete continuous hourly temporal grid for all stations."""
        if aggregated_df.empty:
            return aggregated_df

        # Filter out stations with too few total records to ensure stability
        station_counts = aggregated_df["station_id"].value_counts()
        valid_stations = station_counts[station_counts >= min_station_records].index.tolist()
        if not valid_stations:
            valid_stations = aggregated_df["station_id"].unique().tolist()

        filtered_agg = aggregated_df[aggregated_df["station_id"].isin(valid_stations)].copy()

        # Build full date range from global min to global max
        global_min = filtered_agg["timestamp"].min()
        global_max = filtered_agg["timestamp"].max()
        all_hours = pd.date_range(start=global_min, end=global_max, freq="1h", tz="UTC")

        full_station_frames = []
        for station_id, group in filtered_agg.groupby("station_id"):
            site_id = group["site_id"].iloc[0]
            # Create full hourly reindexed grid for this station
            grid_df = pd.DataFrame({"timestamp": all_hours})
            grid_df["station_id"] = station_id
            grid_df["site_id"] = site_id

            merged = pd.merge(
                grid_df,
                group[["timestamp", "energy_demand_kwh", "session_count", "active_sessions"]],
                on="timestamp",
                how="left",
            )
            # In EV charging at an active operational station, missing session entries indicate zero demand
            merged["energy_demand_kwh"] = merged["energy_demand_kwh"].fillna(0.0).clip(lower=0.0)
            merged["session_count"] = merged["session_count"].fillna(0).astype(int)
            merged["active_sessions"] = merged["active_sessions"].fillna(0).astype(int)

            full_station_frames.append(merged)

        complete_df = pd.concat(full_station_frames, ignore_index=True)
        complete_df = complete_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

        logger.info(
            "Built complete regular hourly dataset: %d total rows across %d stations (%s to %s)",
            len(complete_df),
            complete_df["station_id"].nunique(),
            global_min,
            global_max,
        )

        output_path = self.processed_dir / "hourly_demand.parquet"
        complete_df.to_parquet(output_path, index=False)
        logger.info("Saved processed hourly dataset to %s", output_path)

        return complete_df
