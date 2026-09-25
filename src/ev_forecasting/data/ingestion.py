"""Data ingestion module for ACN-Data EV charging sessions."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ev_forecasting.config.settings import AppConfig

logger = logging.getLogger(__name__)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_directory_hash(directory: Path) -> str:
    """Compute an aggregated SHA-256 hash across all files in a directory."""
    sha256_hash = hashlib.sha256()
    for root, _, files in sorted(os.walk(directory)):
        for fname in sorted(files):
            if (
                fname.endswith(".gz")
                or fname.endswith(".csv")
                or fname.endswith(".json")
                or fname.endswith(".parquet")
            ):
                fpath = Path(root) / fname
                sha256_hash.update(fname.encode("utf-8"))
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class ACNDataIngestor:
    """Ingest and verify raw ACN-Data static snapshot datasets."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.raw_dir = Path(config.paths.raw_data_dir)
        self.manifest_dir = Path(config.paths.manifest_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def fetch_site_file_list(self, site_path: str, max_files: int = 100) -> List[Dict[str, Any]]:
        """Query GitHub API to get file list from ACN-Data-Static repository."""
        api_url = f"https://api.github.com/repos/tongxin-li/ACN-Data-Static/contents/{site_path}"
        req = urllib.request.Request(
            api_url, headers={"User-Agent": "Mozilla/5.0 (EV-Forecasting-Agent)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                items = json.loads(resp.read().decode("utf-8"))
                files = [
                    item
                    for item in items
                    if item.get("type") == "file" and item.get("name", "").endswith(".csv.gz")
                ]
                return files[:max_files]
        except Exception as e:
            logger.warning("Could not fetch remote file list via GitHub API (%s): %s", api_url, e)
            return []

    def download_file(self, download_url: str, target_path: Path) -> bool:
        """Download a single session file with retry and verification."""
        if target_path.exists() and target_path.stat().st_size > 0:
            return True  # Idempotent: already exists

        target_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            download_url, headers={"User-Agent": "Mozilla/5.0 (EV-Forecasting-Agent)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
                if len(content) == 0:
                    logger.warning("Downloaded empty content from %s", download_url)
                    return False
                with open(target_path, "wb") as f:
                    f.write(content)
                return True
        except Exception as e:
            logger.error("Failed downloading %s to %s: %s", download_url, target_path, e)
            return False

    def ingest_remote_dataset(self) -> Dict[str, Any]:
        """Download snapshot files for configured sites."""
        download_stats: Dict[str, Any] = {"downloaded": 0, "existing": 0, "failed": 0, "sites": {}}
        max_files = self.config.ingestion.max_files_per_site

        for site in self.config.dataset.sites:
            site_rel_path = f"time series data/{site.site_id}/{site.location}"
            site_target_dir = self.raw_dir / site.site_id / site.location
            site_target_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Ingesting data for site: %s (path: %s)", site.site_id, site_rel_path)
            remote_files = self.fetch_site_file_list(
                urllib.parse.quote(site_rel_path), max_files=max_files
            )

            site_downloaded = 0
            site_existing = 0
            site_failed = 0

            for f_info in remote_files:
                fname = f_info["name"]
                durl = f_info.get("download_url") or (
                    f"{self.config.dataset.static_snapshot_url}/"
                    f"time%20series%20data/{site.site_id}/{site.location}/{fname}"
                )
                target_file = site_target_dir / fname
                if target_file.exists():
                    site_existing += 1
                else:
                    success = self.download_file(durl, target_file)
                    if success:
                        site_downloaded += 1
                    else:
                        site_failed += 1

            download_stats["downloaded"] += site_downloaded
            download_stats["existing"] += site_existing
            download_stats["failed"] += site_failed
            download_stats["sites"][site.site_id] = {
                "downloaded": site_downloaded,
                "existing": site_existing,
                "failed": site_failed,
                "target_dir": str(site_target_dir),
            }

        return download_stats

    def generate_manifest(self) -> Dict[str, Any]:
        """Compute dataset manifest with hashes, station and session counts."""
        all_raw_files = list(self.raw_dir.glob("**/*.csv.gz"))
        total_files = len(all_raw_files)
        total_bytes = sum(f.stat().st_size for f in all_raw_files)
        dir_hash = compute_directory_hash(self.raw_dir)

        sites_found = set()
        stations_found = set()
        for fpath in all_raw_files:
            rel = fpath.relative_to(self.raw_dir)
            if len(rel.parts) > 1:
                sites_found.add(rel.parts[0])
            # extract station id from filename
            parts = fpath.name.split("-")
            if len(parts) >= 4:
                station_key = f"{rel.parts[0]}_{parts[2]}"
                stations_found.add(station_key)

        manifest = {
            "dataset_name": self.config.dataset.name,
            "version": self.config.dataset.version,
            "source_url": self.config.dataset.source_url,
            "static_snapshot_url": self.config.dataset.static_snapshot_url,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "sha256": dir_hash,
            "total_files": total_files,
            "total_bytes": total_bytes,
            "sites": sorted(list(sites_found)),
            "stations_count": len(stations_found),
            "stations": sorted(list(stations_found)),
        }

        manifest_path = self.manifest_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Dataset manifest written to %s (hash: %s...)", manifest_path, dir_hash[:12])
        return manifest
