"""Parser for ACN EV charging session time-series files."""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ev_forecasting.config.settings import AppConfig

logger = logging.getLogger(__name__)


class ACNDataParser:
    """Parses raw ACN-Data `.csv.gz` files into structured session and measurement data."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.site_voltages = {site.site_id: site.nominal_voltage_v for site in config.dataset.sites}

    def parse_filename(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata encoded in ACN filenames.
        Format: {site_prefix}-{space_id}-{station_id}-{session_id}-{ISO_timestamp}.csv.gz
        """
        # Determine site from parent directory structure if available
        # e.g., data/raw/caltech/California_Garage_01/filename.csv.gz
        parts = file_path.name.replace(".csv.gz", "").replace(".csv", "").split("-")

        parent_parts = file_path.parts
        site_id = "unknown"
        for candidate in ["caltech", "jpl", "office_01"]:
            if candidate in parent_parts:
                site_id = candidate
                break

        station_id = parts[2] if len(parts) >= 3 else "unknown_station"
        session_stem = file_path.name.replace(".csv.gz", "").replace(".csv", "")
        session_id = f"{site_id}_{session_stem}"

        return {
            "site_id": site_id,
            "station_id": f"{site_id}_{station_id}",
            "raw_station_code": station_id,
            "session_id": session_id,
            "file_path": str(file_path),
        }

    def parse_session_file(
        self, file_path: Path
    ) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
        """Parse a single session file and return session summary and measurement dataframe."""
        meta = self.parse_filename(file_path)
        site_id = meta["site_id"]
        nominal_v = self.site_voltages.get(site_id, 208.0)

        try:
            if str(file_path).endswith(".gz"):
                with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(file_path)

            if df.empty or len(df.columns) == 0:
                logger.warning("Empty or unreadable file: %s", file_path)
                return None, None

            # First column is timestamp
            ts_col = df.columns[0]
            df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

            if len(df) == 0:
                return None, None

            # Clean column names
            current_col = None
            energy_col = None
            voltage_col = None

            for col in df.columns:
                c_lower = col.lower()
                if "current" in c_lower and "a" in c_lower:
                    current_col = col
                elif "energy" in c_lower and "kwh" in c_lower:
                    energy_col = col
                elif "voltage" in c_lower or "v" in c_lower and len(col) <= 12:
                    voltage_col = col
                elif "power" in c_lower and "kw" in c_lower:
                    pass

            start_time = df["timestamp"].iloc[0]
            end_time = df["timestamp"].iloc[-1]
            duration_hours = max((end_time - start_time).total_seconds() / 3600.0, 1.0 / 3600.0)

            # Energy calculation / reconstruction
            kwh_delivered = 0.0
            if energy_col and df[energy_col].notna().sum() > 0:
                max_e = df[energy_col].max()
                min_e = df[energy_col].min()
                if not np.isnan(max_e) and max_e > 0:
                    kwh_delivered = float(max_e - (min_e if min_e >= 0 else 0))

            # Fallback: integrate current * voltage over time intervals
            if kwh_delivered <= 0.0 and current_col and df[current_col].notna().sum() > 0:
                currents = df[current_col].fillna(0.0).values
                voltages = (
                    df[voltage_col].fillna(nominal_v).values
                    if voltage_col
                    else np.full_like(currents, nominal_v)
                )

                # dt in hours
                ts_seconds = df["timestamp"].astype("int64") // 10**9
                dt_hours = np.diff(ts_seconds, prepend=ts_seconds.iloc[0]) / 3600.0
                # limit single interval dt to max 0.5 hours to avoid huge gaps inflating energy
                dt_hours = np.clip(dt_hours, 0.0, 0.5)
                power_kw = (voltages * currents) / 1000.0
                kwh_delivered = float(np.sum(power_kw * dt_hours))

            # Clean and prepare interval dataframe for aggregation
            measurements = pd.DataFrame(
                {
                    "timestamp": df["timestamp"],
                    "station_id": meta["station_id"],
                    "site_id": meta["site_id"],
                    "session_id": meta["session_id"],
                    "current_a": df[current_col].fillna(0.0) if current_col else 0.0,
                    "energy_kwh_cumulative": df[energy_col].ffill().fillna(0.0)
                    if energy_col
                    else 0.0,
                }
            )

            session_summary = {
                "session_id": meta["session_id"],
                "site_id": meta["site_id"],
                "station_id": meta["station_id"],
                "raw_station_code": meta["raw_station_code"],
                "start_time": start_time,
                "end_time": end_time,
                "duration_hours": float(duration_hours),
                "kwh_delivered": max(float(kwh_delivered), 0.0),
                "data_points": int(len(df)),
                "file_path": meta["file_path"],
            }

            return session_summary, measurements

        except Exception as e:
            logger.error("Error parsing %s: %s", file_path, e)
            return None, None

    def parse_all_sessions(
        self, raw_dir: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse all session files in raw directory and return session & interval dataframes."""
        source_dir = raw_dir or Path(self.config.paths.raw_data_dir)
        file_list = sorted(list(source_dir.glob("**/*.csv.gz")) + list(source_dir.glob("**/*.csv")))

        session_records: List[Dict[str, Any]] = []
        all_measurements: List[pd.DataFrame] = []

        logger.info("Parsing %d session files from %s", len(file_list), source_dir)
        for fpath in file_list:
            if fpath.name.startswith("."):
                continue
            summary, measurements = self.parse_session_file(fpath)
            if summary is not None and summary["kwh_delivered"] >= 0:
                session_records.append(summary)
                if measurements is not None and not measurements.empty:
                    all_measurements.append(measurements)

        sessions_df = pd.DataFrame(session_records)
        measurements_df = (
            pd.concat(all_measurements, ignore_index=True) if all_measurements else pd.DataFrame()
        )

        logger.info(
            "Successfully parsed %d sessions (%d measurement rows)",
            len(sessions_df),
            len(measurements_df),
        )
        return sessions_df, measurements_df
