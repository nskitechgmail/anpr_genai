"""
utils/heatmap.py — Traffic density heatmap overlay (Sprint 3).

Accumulates vehicle positions across frames using a Gaussian kernel,
renders a colour-coded OpenCV heatmap overlay, and saves snapshots.
"""
from __future__ import annotations
import cv2
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("Heatmap")


class TrafficHeatmap:
    """
    Incremental traffic density heatmap.

    Parameters
    ----------
    width, height : int   — frame dimensions for the accumulator grid
    decay         : float — per-frame decay (0–1). Lower = faster fade
    output_dir    : str   — directory for saved snapshots
    """

    def __init__(self, width: int = 1280, height: int = 720,
                 decay: float = 0.995, output_dir: str = "outputs/heatmaps"):
        self.width      = width
        self.height     = height
        self.decay      = decay
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._density      = np.zeros((height, width), dtype=np.float32)
        self._frame_count  = 0
        self._hourly_counts: dict[int, int] = defaultdict(int)
        self._kernel       = self._make_gaussian_kernel(radius=30)

    # ── Public API ────────────────────────────────────────────────

    def update(self, detections: list):
        """Add vehicle centres from current frame into the accumulator."""
        self._frame_count += 1
        hour = datetime.now().hour
        self._density *= self.decay

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            self._add_gaussian(cx, cy)
            self._hourly_counts[hour] += 1

    def render(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """Overlay the heatmap on *frame* and return the blended result."""
        h, w = frame.shape[:2]
        density = self._density
        if (h, w) != (self.height, self.width):
            density = cv2.resize(density, (w, h))

        max_val = density.max()
        if max_val > 0:
            d_norm = (density / max_val * 255).astype(np.uint8)
        else:
            d_norm = density.astype(np.uint8)

        coloured = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1.0 - alpha, coloured, alpha, 0)

    def save_snapshot(self, tag: str = "") -> Path:
        """Save the current density grid as a standalone PNG."""
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"heatmap_{ts}{'_'+tag if tag else ''}.png"
        path  = self.output_dir / fname

        max_val = self._density.max()
        if max_val > 0:
            d_norm = (self._density / max_val * 255).astype(np.uint8)
        else:
            d_norm = self._density.astype(np.uint8)

        cv2.imwrite(str(path), cv2.applyColorMap(d_norm, cv2.COLORMAP_JET))
        log.info(f"Heatmap snapshot → {path}")
        return path

    def get_stats(self) -> dict:
        """Return serialisable stats dict for REST API."""
        return {
            "frame_count"  : self._frame_count,
            "peak_density" : float(self._density.max()),
            "hourly_counts": dict(self._hourly_counts),
        }

    def reset(self):
        self._density[:]  = 0
        self._hourly_counts.clear()
        self._frame_count = 0

    # ── Internal ──────────────────────────────────────────────────

    def _add_gaussian(self, cx: int, cy: int):
        """Splat a Gaussian centred at (cx, cy) into the density grid."""
        kr = self._kernel.shape[0] // 2
        x1 = cx - kr;  y1 = cy - kr
        x2 = cx + kr + 1;  y2 = cy + kr + 1

        gx1 = max(0, -x1);  gy1 = max(0, -y1)
        gx2 = self._kernel.shape[1] - max(0, x2 - self.width)
        gy2 = self._kernel.shape[0] - max(0, y2 - self.height)

        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(self.width, x2);  y2 = min(self.height, y2)

        if x2 > x1 and y2 > y1 and gx2 > gx1 and gy2 > gy1:
            self._density[y1:y2, x1:x2] += self._kernel[gy1:gy2, gx1:gx2]

    @staticmethod
    def _make_gaussian_kernel(radius: int = 30) -> np.ndarray:
        size  = 2 * radius + 1
        sigma = radius / 2.5
        k     = np.zeros((size, size), dtype=np.float32)
        cx = cy = radius
        for y in range(size):
            for x in range(size):
                k[y, x] = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
        return k / k.sum()
