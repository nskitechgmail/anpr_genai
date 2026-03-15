"""
config/settings.py — Single source of truth for all runtime configuration.
All pipeline modules read exclusively from this Settings dataclass.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    # ── Input ─────────────────────────────────────────────────────────────
    source:    str = "0"          # "0"=webcam | video path | rtsp://...
    camera_id: str = "CCTV-001"

    # ── Model / inference ──────────────────────────────────────────────────
    detector_model: str   = "yolov8n"   # yolov8n (fast CPU) | yolov9c (accurate)
    conf_thresh:    float = 0.25        # YOLO confidence threshold
    iou_thresh:     float = 0.45        # YOLO NMS IoU
    device:         str   = "auto"      # auto | cpu | cuda | mps
    use_genai:      bool  = False       # Real-ESRGAN — set True if GPU available
    esrgan_scale:   int   = 4
    esrgan_tile:    int   = 0           # 0 = auto; use 256 for low VRAM

    # ── OCR ────────────────────────────────────────────────────────────────
    ocr_languages: list = field(default_factory=lambda: ["en"])

    # ── Safety compliance ──────────────────────────────────────────────────
    helmet_conf:      float = 0.55
    seatbelt_conf:    float = 0.55
    violation_frames: int   = 3    # consecutive frames before confirming

    # ── Plate ──────────────────────────────────────────────────────────────
    plate_min_area: int  = 150     # min pixel area of plate crop

    # ── Processing ─────────────────────────────────────────────────────────
    fps_target:     int  = 30
    show_plate_crop: bool = True   # show inset thumbnail in annotator

    # ── Privacy ────────────────────────────────────────────────────────────
    anonymise_faces: bool = True

    # ── Heatmap ────────────────────────────────────────────────────────────
    enable_heatmap: bool = False

    # ── Alerts ─────────────────────────────────────────────────────────────
    enable_alerts:          bool = False
    alert_repeat_threshold: int  = 3
    alert_smtp_host:  str = ""
    alert_smtp_port:  int = 587
    alert_smtp_user:  str = ""
    alert_smtp_pass:  str = ""
    alert_email_from: str = ""
    alert_email_to:   str = ""
    alert_twilio_sid:   str = ""
    alert_twilio_token: str = ""
    alert_sms_to:   str = ""
    alert_sms_from: str = ""

    # ── REST API ───────────────────────────────────────────────────────────
    api_port: int = 8000

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir:      str  = "outputs"
    save_violations: bool = True

    # ─────────────────────────────────────────────────────────────────────

    def __post_init__(self):
        # Cast numeric string source to int (webcam index)
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)

        # Load alert credentials from environment variables if not provided
        env_map = {
            "alert_smtp_host":   "ANPR_ALERT_SMTP_HOST",
            "alert_smtp_user":   "ANPR_ALERT_SMTP_USER",
            "alert_smtp_pass":   "ANPR_ALERT_SMTP_PASS",
            "alert_email_from":  "ANPR_ALERT_EMAIL_FROM",
            "alert_email_to":    "ANPR_ALERT_EMAIL_TO",
            "alert_twilio_sid":  "ANPR_TWILIO_SID",
            "alert_twilio_token":"ANPR_TWILIO_TOKEN",
            "alert_sms_to":      "ANPR_ALERT_SMS_TO",
            "alert_sms_from":    "ANPR_ALERT_SMS_FROM",
        }
        for attr, env_key in env_map.items():
            if not getattr(self, attr):
                val = os.environ.get(env_key, "")
                if val:
                    setattr(self, attr, val)

        port_env = os.environ.get("ANPR_ALERT_SMTP_PORT", "")
        if port_env.isdigit():
            self.alert_smtp_port = int(port_env)

        # Auto-resolve device
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

        # Create output directories
        for sub in ("reports", "violations", "heatmaps"):
            Path(self.output_dir, sub).mkdir(parents=True, exist_ok=True)
