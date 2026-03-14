"""
api/server.py — FastAPI REST API for Smart City ANPR System.

Endpoints:
    GET  /               Health check
    GET  /detections     Latest N vehicle detections
    GET  /violations     All confirmed violations (paginated)
    GET  /stats          Live system statistics
    GET  /heatmap/stats  Traffic density heatmap statistics
    POST /config         Update live configuration
    GET  /docs           Auto-generated Swagger UI (FastAPI built-in)

Run standalone:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Or via main.py:
    python main.py --headless --api --source 0
    python main.py --api-only
"""
from __future__ import annotations
import time
import logging
from datetime import datetime
from typing import List, Optional

log = logging.getLogger("API")

# ── FastAPI import with graceful degradation ──────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    log.warning("fastapi not installed — REST API disabled. "
                "Install with: pip install fastapi uvicorn")


# ══════════════════════════════════════════════════════════════════════════
#  Pydantic response / request models
# ══════════════════════════════════════════════════════════════════════════

if _FASTAPI_AVAILABLE:

    class ConfigUpdate(BaseModel):
        conf_thresh: Optional[float] = Field(
            None, ge=0.1, le=1.0,
            description="YOLO detection confidence threshold")
        use_genai:   Optional[bool]  = Field(
            None, description="Enable/disable Real-ESRGAN enhancement")
        fps_target:  Optional[int]   = Field(
            None, ge=1, le=120,
            description="Target processing FPS")
        anonymise:   Optional[bool]  = Field(
            None, description="Enable/disable face anonymisation")

    class DetectionResponse(BaseModel):
        id:            int
        timestamp:     str
        camera_id:     str
        vehicle_class: str
        confidence:    float
        plate_text:    Optional[str]
        plate_conf:    Optional[float]
        plate_enhanced: bool
        plate_valid:   bool
        helmet:        Optional[bool]
        seatbelt:      Optional[bool]
        violation:     str

    class ViolationResponse(BaseModel):
        id:            int
        timestamp:     str
        camera_id:     str
        plate:         str
        vehicle_class: str
        violation:     str
        plate_conf:    float
        plate_enhanced: bool
        image_saved:   str

    class StatsResponse(BaseModel):
        uptime_seconds:   float
        frame_count:      int
        fps:              float
        total_vehicles:   int
        total_plates:     int
        total_violations: int
        genai_enabled:    bool
        camera_id:        str
        source:           str
        alert_stats:      Optional[dict] = None


# ══════════════════════════════════════════════════════════════════════════
#  App factory
# ══════════════════════════════════════════════════════════════════════════

def create_app(pipeline=None, settings=None) -> "FastAPI":
    """
    Create and return the FastAPI application.

    Parameters
    ----------
    pipeline : ANPRPipeline | None
        Live pipeline instance. None = stub/test mode (API still responds).
    settings : Settings | None
        Runtime settings. Required for /config endpoint.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError("fastapi is not installed. "
                          "Run: pip install fastapi uvicorn")

    app = FastAPI(
        title       = "Smart City ANPR System API",
        description = (
            "REST API for the Multi-Modal Vehicle Detection and "
            "License Plate Recognition system.\n\n"
            "**Tech stack:** YOLOv9 · Real-ESRGAN · EasyOCR · MobileNetV3 · "
            "MediaPipe\n\n"
            "**SRM Institute of Science and Technology — "
            "Dept. of Computational Intelligence, 2024–25**"
        ),
        version  = "1.0.0",
        docs_url = "/docs",
        redoc_url= "/redoc",
    )

    # ── In-memory detection / violation stores ────────────────────────────
    # Populated on every call to /detections and /violations by pulling
    # the live pipeline state — no background thread required.
    _detections:  list[dict] = []
    _violations:  list[dict] = []
    _det_counter              = {"n": 0}
    _viol_counter             = {"n": 0}
    _start_time               = time.time()

    # ── Pipeline state sync ───────────────────────────────────────────────

    def _sync_from_pipeline():
        """Pull latest detections from the running pipeline into local stores."""
        if pipeline is None:
            return
        try:
            _frame, _stats, dets = pipeline.get_latest()
        except Exception:
            return

        for det in dets:
            _det_counter["n"] += 1
            _detections.append({
                "id"            : _det_counter["n"],
                "timestamp"     : datetime.now().isoformat(),
                "camera_id"     : settings.camera_id if settings else "CCTV-001",
                "vehicle_class" : det.vehicle_class,
                "confidence"    : round(det.confidence, 3),
                "plate_text"    : det.plate.text            if det.plate else None,
                "plate_conf"    : round(det.plate.confidence, 3) if det.plate else None,
                "plate_enhanced": det.plate.enhanced        if det.plate else False,
                "plate_valid"   : det.plate.valid_format    if det.plate else False,
                "helmet"        : det.helmet,
                "seatbelt"      : det.seatbelt,
                "violation"     : det.violation,
            })
            if det.has_violation() and det.plate:
                _viol_counter["n"] += 1
                _violations.append({
                    "id"            : _viol_counter["n"],
                    "timestamp"     : datetime.now().isoformat(),
                    "camera_id"     : settings.camera_id if settings else "CCTV-001",
                    "plate"         : det.plate.normalised(),
                    "vehicle_class" : det.vehicle_class,
                    "violation"     : det.violation,
                    "plate_conf"    : round(det.plate.confidence * 100, 1),
                    "plate_enhanced": det.plate.enhanced,
                    "image_saved"   : "",
                })

        # Cap store size to avoid unbounded memory growth
        if len(_detections) > 1000:
            del _detections[:500]
        if len(_violations) > 500:
            del _violations[:250]

    # ── Routes ────────────────────────────────────────────────────────────

    @app.get("/", tags=["Health"])
    async def root():
        """Health check — confirms the API is online."""
        return {
            "status" : "online",
            "system" : "Smart City ANPR",
            "version": "1.0.0",
            "uptime" : round(time.time() - _start_time, 1),
            "docs"   : "/docs",
        }

    @app.get(
        "/detections",
        response_model=List[DetectionResponse],
        tags=["Data"],
        summary="Latest vehicle detections",
    )
    async def get_detections(
        limit: int = Query(20, ge=1, le=200,
                           description="Max number of detections to return"),
    ):
        """
        Returns the most recent vehicle detections from the live pipeline.
        Each entry contains vehicle class, plate text, safety compliance,
        and violation status.
        """
        _sync_from_pipeline()
        page = _detections[-limit:]
        return [DetectionResponse(**d) for d in page]

    @app.get(
        "/violations",
        response_model=List[ViolationResponse],
        tags=["Data"],
        summary="Confirmed violations (paginated)",
    )
    async def get_violations(
        skip:  int = Query(0,  ge=0,   description="Number of records to skip"),
        limit: int = Query(50, ge=1, le=500,
                           description="Max records to return"),
    ):
        """
        Returns confirmed traffic violations logged this session.
        A violation is confirmed after N consecutive frame detections
        (temporal smoother). Supports pagination via skip/limit.
        """
        # Also pull from reporter if pipeline is live
        if pipeline is not None:
            try:
                records = pipeline.reporter.get_all()
                for i, r in enumerate(records):
                    if not any(v.get("id") == i + 1 for v in _violations):
                        _viol_counter["n"] += 1
                        _violations.append({
                            "id"            : _viol_counter["n"],
                            "timestamp"     : r.get("timestamp", ""),
                            "camera_id"     : r.get("camera_id", ""),
                            "plate"         : r.get("plate", ""),
                            "vehicle_class" : r.get("vehicle_class", ""),
                            "violation"     : r.get("violation", ""),
                            "plate_conf"    : r.get("plate_conf", 0.0),
                            "plate_enhanced": r.get("plate_enhanced", False),
                            "image_saved"   : r.get("image_saved", ""),
                        })
            except Exception:
                pass

        page = _violations[skip: skip + limit]
        return [ViolationResponse(**v) for v in page]

    @app.get(
        "/stats",
        response_model=StatsResponse,
        tags=["Data"],
        summary="Live system statistics",
    )
    async def get_stats():
        """
        Returns real-time pipeline statistics including FPS, vehicle counts,
        violation totals, GenAI status, and optional alert stats.
        """
        uptime = round(time.time() - _start_time, 1)

        if pipeline is None:
            return StatsResponse(
                uptime_seconds   = uptime,
                frame_count      = 0,
                fps              = 0.0,
                total_vehicles   = 0,
                total_plates     = 0,
                total_violations = 0,
                genai_enabled    = settings.use_genai if settings else False,
                camera_id        = settings.camera_id if settings else "N/A",
                source           = str(settings.source) if settings else "N/A",
            )

        try:
            _frame, stats, _dets = pipeline.get_latest()
            alert_stats = (
                pipeline.alerts.get_stats()
                if hasattr(pipeline, "alerts") and pipeline.alerts else None
            )
            return StatsResponse(
                uptime_seconds   = uptime,
                frame_count      = stats.frame_num,
                fps              = round(stats.fps, 1),
                total_vehicles   = stats.vehicles,
                total_plates     = stats.plates_read,
                total_violations = stats.violations,
                genai_enabled    = settings.use_genai if settings else False,
                camera_id        = settings.camera_id if settings else "N/A",
                source           = str(settings.source) if settings else "N/A",
                alert_stats      = alert_stats,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/heatmap/stats", tags=["Data"], summary="Traffic density heatmap stats")
    async def get_heatmap_stats():
        """
        Returns traffic density heatmap statistics (hourly vehicle counts,
        peak density, frame count). Requires pipeline started with --heatmap.
        """
        if pipeline and hasattr(pipeline, "heatmap") and pipeline.heatmap:
            return pipeline.heatmap.get_stats()
        return {
            "message": (
                "Heatmap not active. "
                "Start the pipeline with --heatmap flag to enable."
            )
        }

    @app.post("/config", tags=["Control"], summary="Update live configuration")
    async def update_config(update: ConfigUpdate):
        """
        Update system configuration at runtime. Changes take effect on
        the next processed frame — no restart required.

        Adjustable fields: conf_thresh, use_genai, fps_target, anonymise.
        """
        if settings is None:
            raise HTTPException(
                status_code=503,
                detail="No live pipeline settings available. "
                       "Start the pipeline before updating config.",
            )
        changed = {}
        if update.conf_thresh is not None:
            settings.conf_thresh = update.conf_thresh
            changed["conf_thresh"] = update.conf_thresh
        if update.use_genai is not None:
            settings.use_genai = update.use_genai
            changed["use_genai"] = update.use_genai
        if update.fps_target is not None:
            settings.fps_target = update.fps_target
            changed["fps_target"] = update.fps_target
        if update.anonymise is not None:
            settings.anonymise_faces = update.anonymise
            changed["anonymise_faces"] = update.anonymise

        log.info(f"Config updated via API: {changed}")
        return {"status": "ok", "updated": changed}

    return app


# ── Standalone app instance (used by: uvicorn api.server:app) ─────────────
if _FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None
