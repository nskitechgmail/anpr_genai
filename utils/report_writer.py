"""
utils/report_writer.py — Thread-safe CSV + JSON violation report writer.

Creates time-stamped files in outputs/reports/ per session.
Each write() call appends one violation record.
"""
from __future__ import annotations
import csv
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

log = logging.getLogger("ReportWriter")

_FIELDS = [
    "timestamp", "camera_id", "frame",
    "plate", "plate_raw", "plate_conf", "plate_valid", "plate_enhanced",
    "vehicle_class", "violation",
    "helmet", "helmet_conf",
    "seatbelt", "seatbelt_conf",
    "image_saved",
]


class ReportWriter:
    """
    Thread-safe CSV + JSON report writer.

    Parameters
    ----------
    settings : Settings — provides output_dir and camera_id
    """

    def __init__(self, settings):
        self.cfg      = settings
        self._lock    = threading.Lock()
        self._records: list[dict] = []
        self._frame   = 0

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(settings.output_dir) / "reports"
        base.mkdir(parents=True, exist_ok=True)

        csv_path        = base / f"violations_{ts}.csv"
        self._json_path = base / f"violations_{ts}.json"

        self._csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_FIELDS,
                                          extrasaction="ignore")
        self._csv_writer.writeheader()
        self._csv_file.flush()
        log.info(f"ReportWriter → {csv_path}")

    def write(self, record: dict):
        """Append one violation record (dict keyed by _FIELDS)."""
        with self._lock:
            self._frame += 1
            record.setdefault("frame",     self._frame)
            record.setdefault("camera_id", self.cfg.camera_id)
            self._csv_writer.writerow(record)
            self._csv_file.flush()
            self._records.append(record.copy())
            # Rewrite JSON on every write (small sessions only)
            self._json_path.write_text(
                json.dumps(self._records, indent=2, default=str),
                encoding="utf-8")

    def get_all(self) -> list[dict]:
        """Return all records written this session (for API)."""
        with self._lock:
            return list(self._records)

    def close(self):
        with self._lock:
            if not self._csv_file.closed:
                self._csv_file.close()
        log.info("ReportWriter closed.")
