"""
core/pipeline.py — ANPR pipeline orchestrator.

  • Video capture loop (webcam / file / RTSP)
  • Frame-by-frame processing with FPS targeting
  • Temporal violation confirmation (N-frame smoother)
  • Heatmap accumulation (Sprint 3)
  • Repeat-violator alert triggering (Sprint 3)
  • Thread-safe state sharing with GUI

NOTE: FrameStats is defined HERE and imported by annotator.py,
      dashboard.py and test_suite.py.  It is NOT in plate_recogniser.py.
"""
from __future__ import annotations
import cv2, time, logging, threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

log = logging.getLogger("Pipeline")


# ── FrameStats — single canonical definition ──────────────────────────────
@dataclass
class FrameStats:
    frame_num:   int   = 0
    fps:         float = 0.0
    vehicles:    int   = 0
    violations:  int   = 0
    plates_read: int   = 0


# ── ViolationTracker ──────────────────────────────────────────────────────
class ViolationTracker:
    """Confirm a violation only after N consecutive frames."""

    def __init__(self, n_frames: int = 3):
        self.n = n_frames
        self._counts:   dict[str, deque] = defaultdict(
            lambda: deque(maxlen=n_frames))
        self._reported: set[str] = set()

    def update(self, plate_text: str, violation: str) -> bool:
        key = f"{plate_text}:{violation}"
        self._counts[key].append(1)
        if (len(self._counts[key]) == self.n
                and sum(self._counts[key]) == self.n
                and key not in self._reported):
            self._reported.add(key)
            return True
        return False

    def reset(self, plate_text: str):
        for k in [k for k in self._counts if k.startswith(plate_text)]:
            self._counts[k].clear()
            self._reported.discard(k)


# ── ANPRPipeline ──────────────────────────────────────────────────────────
class ANPRPipeline:
    """
    Top-level pipeline.

    GUI usage:
        pipeline = ANPRPipeline(settings)
        pipeline.start()
        frame, stats, detections = pipeline.get_latest()
        pipeline.stop()

    Headless usage:
        pipeline = ANPRPipeline(settings)
        pipeline.run_headless()
    """

    def __init__(self, settings):
        self.cfg      = settings

        # Lazy imports avoid circular dependency at module load time
        from models.model_manager  import ModelManager
        from core.plate_recogniser import PlateRecogniser
        from utils.annotator       import FrameAnnotator
        from utils.report_writer   import ReportWriter
        from utils.anonymiser      import FaceAnonymiser

        self.models     = ModelManager(settings)
        self.recogniser: Optional[PlateRecogniser] = None
        self.annotator  = FrameAnnotator(settings)
        self.reporter   = ReportWriter(settings)
        self.anonymiser = FaceAnonymiser()
        self.tracker    = ViolationTracker(settings.violation_frames)

        # Sprint 3 features
        self.heatmap = None
        self.alerts  = None
        if settings.enable_heatmap:
            from utils.heatmap import TrafficHeatmap
            self.heatmap = TrafficHeatmap(
                output_dir=str(Path(settings.output_dir) / "heatmaps"))
        if settings.enable_alerts:
            from utils.alerts import AlertSystem
            self.alerts = AlertSystem(
                threshold=settings.alert_repeat_threshold,
                settings=settings)

        # Shared state (GUI ↔ pipeline thread)
        self._running      = False
        self._latest_frame = None
        self._latest_stats = FrameStats()
        self._latest_dets  = []
        self._lock         = threading.Lock()
        self._thread       = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self):
        from core.plate_recogniser import PlateRecogniser
        self.models.load_all()
        self.recogniser = PlateRecogniser(self.models, self.cfg)
        self._running   = True
        self._thread    = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Pipeline thread started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self.reporter.close()
        if self.heatmap:
            self.heatmap.save_snapshot(tag="final")
        log.info("Pipeline stopped.")

    def get_latest(self) -> tuple:
        """Return (annotated_frame, FrameStats, detections) — thread-safe."""
        with self._lock:
            return self._latest_frame, self._latest_stats, list(self._latest_dets)

    def process_single_image(self, image_path: str):
        """Process one image file. Returns (annotated_frame, detections)."""
        from core.plate_recogniser import PlateRecogniser
        self.models.load_all()
        self.recogniser = PlateRecogniser(self.models, self.cfg)
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        # Resize very large images to keep processing fast
        h, w = frame.shape[:2]
        if max(h, w) > 1920:
            scale = 1920 / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        dets      = self.recogniser.process_frame(frame)
        annotated = self.annotator.draw(frame, dets,
                                        FrameStats(vehicles=len(dets)))
        return annotated, dets

    def run_headless(self):
        from core.plate_recogniser import PlateRecogniser
        self.models.load_all()
        self.recogniser = PlateRecogniser(self.models, self.cfg)
        cap = self._open_capture()
        if not cap.isOpened():
            log.error("Could not open video source.")
            return
        log.info("Headless pipeline started. Ctrl+C to stop.")
        self._running = True
        stats = FrameStats()
        try:
            self._capture_loop(cap, stats, headless=True)
        except KeyboardInterrupt:
            log.info("Interrupted.")
        finally:
            cap.release()
            self.reporter.close()
            if self.heatmap:
                self.heatmap.save_snapshot(tag="final")
            self._print_summary(stats)

    # ── Internal ──────────────────────────────────────────────────────────

    def _loop(self):
        cap   = self._open_capture()
        stats = FrameStats()
        self._capture_loop(cap, stats, headless=False)
        cap.release()

    def _capture_loop(self, cap, stats: FrameStats, headless: bool):
        fps_timer  = time.time()
        fps_frames = 0
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30

        while self._running:
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.cfg.source, str) and \
                        self.cfg.source.lower().endswith(
                            (".mp4", ".avi", ".mov", ".mkv")):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            stats.frame_num += 1
            skip = max(1, int(source_fps / max(1, self.cfg.fps_target)))
            if stats.frame_num % skip != 0:
                continue

            # ── Detect ────────────────────────────────────────────────────
            dets = self.recogniser.process_frame(frame)

            # ── Anonymise ─────────────────────────────────────────────────
            if self.cfg.anonymise_faces:
                frame = self.anonymiser.blur_faces(frame)

            # ── Heatmap ───────────────────────────────────────────────────
            if self.heatmap:
                self.heatmap.update(dets)
                frame = self.heatmap.render(frame)

            # ── Cumulative counts ─────────────────────────────────────────
            stats.vehicles    += len(dets)
            stats.plates_read += sum(1 for d in dets if d.plate)

            # ── Violation confirmation + reporting ────────────────────────
            for det in dets:
                if det.has_violation() and det.plate:
                    if self.tracker.update(det.plate.text, det.violation):
                        stats.violations += 1
                        self._save_violation(frame, det, stats)
                        if self.alerts:
                            self.alerts.record(det.plate.text, det.violation)

            # ── Annotate ──────────────────────────────────────────────────
            annotated = self.annotator.draw(frame, dets, stats)

            # ── FPS ───────────────────────────────────────────────────────
            fps_frames += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                stats.fps  = fps_frames / (now - fps_timer)
                fps_timer  = now
                fps_frames = 0

            # ── Share with GUI ────────────────────────────────────────────
            with self._lock:
                self._latest_frame = annotated
                self._latest_stats = FrameStats(
                    frame_num  = stats.frame_num,
                    fps        = stats.fps,
                    vehicles   = len(dets),
                    violations = stats.violations,
                    plates_read= stats.plates_read,
                )
                self._latest_dets = list(dets)

            if headless:
                self._print_frame_info(stats, dets)

    def _save_violation(self, frame: np.ndarray, det, stats: FrameStats):
        from core.plate_recogniser import VehicleDetection
        ts      = datetime.now()
        plate   = det.plate.normalised() if det.plate else "UNKNOWN"
        img_path = ""
        if self.cfg.save_violations:
            out_dir = Path(self.cfg.output_dir) / "violations"
            fname   = f"{plate.replace(' ','')}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
            path    = out_dir / fname
            cv2.imwrite(str(path), frame)
            img_path = str(path)
        self.reporter.write({
            "timestamp"     : ts.isoformat(),
            "camera_id"     : self.cfg.camera_id,
            "frame"         : stats.frame_num,
            "plate"         : plate,
            "plate_raw"     : det.plate.raw_text      if det.plate else "",
            "plate_conf"    : round(det.plate.confidence * 100, 1)
                              if det.plate else 0.0,
            "plate_valid"   : det.plate.valid_format  if det.plate else False,
            "plate_enhanced": det.plate.enhanced      if det.plate else False,
            "vehicle_class" : det.vehicle_class,
            "violation"     : det.violation,
            "helmet"        : det.helmet,
            "helmet_conf"   : round(det.helmet_conf * 100, 1),
            "seatbelt"      : det.seatbelt,
            "seatbelt_conf" : round(det.seatbelt_conf * 100, 1),
            "image_saved"   : img_path,
        })

    def _open_capture(self) -> cv2.VideoCapture:
        src = self.cfg.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error(f"Cannot open: {src}")
        return cap

    def _print_frame_info(self, stats: FrameStats, dets: list):
        plates = [d.plate.normalised() for d in dets if d.plate]
        viols  = sum(1 for d in dets if d.has_violation())
        print(f"\r[{stats.frame_num:06d}] FPS:{stats.fps:5.1f} | "
              f"Vehicles:{len(dets)} | Plates:{plates} | Violations:{viols}",
              end="", flush=True)

    def _print_summary(self, stats: FrameStats):
        print(f"\n{'='*55}")
        print(f"  Frames : {stats.frame_num}")
        print(f"  Vehicles: {stats.vehicles}  Plates: {stats.plates_read}")
        print(f"  Violations: {stats.violations}")
        print(f"  Reports → {self.cfg.output_dir}/reports/")
        print(f"{'='*55}")
