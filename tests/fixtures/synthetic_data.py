"""Synthetic deterministic data fixture generators for testing and CI."""

from __future__ import annotations

import gzip
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def generate_synthetic_sessions(
    n_days: int = 30,
    sites: List[str] = ["caltech", "jpl", "office_01"],
    stations_per_site: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic deterministic EV charging sessions."""
    np.random.seed(random_seed)
    start_date = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    records = []
    session_counter = 1000

    for day in range(n_days):
        current_day = start_date + timedelta(days=day)
        is_weekend = current_day.weekday() >= 5

        for site_id in sites:
            # Different arrival probabilities based on site and weekend
            if site_id == "jpl" and is_weekend:
                base_sessions_per_station = 1  # low weekend workplace
            elif site_id == "caltech" and is_weekend:
                base_sessions_per_station = 3  # campus weekend
            else:
                base_sessions_per_station = 5  # typical weekday

            for st_num in range(1, stations_per_site + 1):
                station_id = f"{site_id}_ST_{st_num:03d}"
                n_sessions = np.random.poisson(base_sessions_per_station)

                for _ in range(n_sessions):
                    session_counter += 1
                    # Realistic peak arrival times (8-10 AM or 12-14 PM)
                    arrival_hour = int(np.clip(np.random.normal(9, 2.5), 6, 21))
                    arrival_minute = int(np.random.randint(0, 60))
                    session_start = current_day.replace(hour=arrival_hour, minute=arrival_minute)

                    duration_h = float(np.clip(np.random.exponential(3.0), 0.5, 8.0))
                    session_end = session_start + timedelta(hours=duration_h)

                    # 6.6 kW L2 charging rate average
                    energy_kwh = float(np.clip(duration_h * np.random.uniform(4.0, 6.6), 1.0, 45.0))

                    records.append(
                        {
                            "session_id": f"sess_{session_counter}",
                            "site_id": site_id,
                            "station_id": station_id,
                            "raw_station_code": f"ST_{st_num:03d}",
                            "start_time": session_start,
                            "end_time": session_end,
                            "duration_hours": duration_h,
                            "kwh_delivered": energy_kwh,
                            "data_points": int(duration_h * 120),
                            "file_path": f"synthetic/{site_id}/{station_id}_{session_counter}.csv.gz",
                        }
                    )

    df = pd.DataFrame(records)
    return df.sort_values(["station_id", "start_time"]).reset_index(drop=True)


def create_synthetic_raw_files(
    target_dir: Path, n_sessions_per_site: int = 5, n_days: int = 14
) -> List[Path]:
    """Create physical sample `.csv.gz` files for testing the parser and ingestion pipelines."""
    target_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    sites = ["caltech", "jpl", "office_01"]

    for site in sites:
        site_dir = target_dir / site / "Default_Garage"
        site_dir.mkdir(parents=True, exist_ok=True)

        for d in range(n_days):
            for i in range(1, n_sessions_per_site + 1):
                sess_time = datetime(2020, 1, 1, 8, 0, 0, tzinfo=timezone.utc) + timedelta(
                    days=d, hours=i
                )
                time_str = sess_time.strftime("%Y-%m-%dT%H-%M-%S-000000")
                fname = f"1-1-{i:03d}-{1000 * d + i}-{time_str}.csv.gz"
                fpath = site_dir / fname

                # Generate sample 1-minute time series
                n_steps = 60
                ts_list = [(sess_time + timedelta(minutes=m)).isoformat() for m in range(n_steps)]
                current_a = np.random.uniform(15.0, 30.0, size=n_steps)
                voltage_v = np.full(n_steps, 208.0)
                energy_kwh = np.cumsum((current_a * voltage_v / 1000.0) * (1.0 / 60.0))

                df = pd.DataFrame(
                    {
                        "Unnamed: 0": ts_list,
                        "Charging Current (A)": current_a,
                        "Actual Pilot (A)": np.full(n_steps, 32.0),
                        "Voltage (V)": voltage_v,
                        "Charging State": ["Charging"] * n_steps,
                        "Energy Delivered (kWh)": energy_kwh,
                        "Power (kW)": (current_a * voltage_v) / 1000.0,
                    }
                )

                with gzip.open(fpath, "wt", encoding="utf-8") as gz:
                    df.to_csv(gz, index=False)

                created_files.append(fpath)

    return created_files
