"""
models/model_manager.py — Lazy-loaded model registry.

Models:
  .detector         — YOLOv8/v9 vehicle detector
  .ocr              — EasyOCR reader
  .enhancer         — Real-ESRGAN upscaler
  .safety_classifier — SafetyClassifier (helmet + seatbelt)
"""
from __future__ import annotations
import logging
import urllib.request
from pathlib import Path

log = logging.getLogger("ModelManager")

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

ESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)


# ── Simulation stubs ───────────────────────────────────────────────────────

class _SimulatedDetector:
    def __call__(self, frame, **kw): return []

class _SimulatedOCR:
    def readtext(self, img, **kw):
        import random
        states = ["MH","KA","TN","DL","UP","AP","TS"]
        series = ["AB","CD","EF","XY"]
        p = (f"{random.choice(states)}{random.randint(1,99):02d}"
             f"{random.choice(series)}{random.randint(1000,9999)}")
        return [([[0,0],[100,0],[100,30],[0,30]], p, 0.82)]

class _PILEnhancer:
    def enhance(self, img, outscale: int = 4, **kw):
        from PIL import Image, ImageEnhance
        import numpy as np
        pil = Image.fromarray(img[..., ::-1])
        pil = pil.resize((pil.width*outscale, pil.height*outscale),
                         Image.LANCZOS)
        pil = ImageEnhance.Sharpness(pil).enhance(2.5)
        pil = ImageEnhance.Contrast(pil).enhance(1.4)
        return np.array(pil)[..., ::-1], None


# ── ModelManager ───────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self, settings):
        self.cfg            = settings
        self._detector      = None
        self._ocr           = None
        self._enhancer      = None
        self._safety_clf    = None

    # ── Properties ────────────────────────────────────────────────

    @property
    def detector(self):
        if self._detector is None: self._load_detector()
        return self._detector

    @property
    def ocr(self):
        if self._ocr is None: self._load_ocr()
        return self._ocr

    @property
    def enhancer(self):
        if self._enhancer is None and getattr(self.cfg,"use_genai",False):
            self._load_enhancer()
        return self._enhancer

    @property
    def safety_classifier(self):
        """SafetyClassifier instance — helmet + seatbelt detection."""
        if self._safety_clf is None:
            self._load_safety()
        return self._safety_clf

    # ── Bulk loader ────────────────────────────────────────────────

    def load_all(self):
        log.info("Loading models …")
        self._load_detector()
        self._load_ocr()
        if getattr(self.cfg, "use_genai", False):
            self._load_enhancer()
        self._load_safety()
        log.info("All models ready.")

    # ── Individual loaders ─────────────────────────────────────────

    def _load_detector(self):
        try:
            from ultralytics import YOLO
            model_name = getattr(self.cfg, "detector_model", "yolov8n")
            log.info(f"Loading YOLO: {model_name}")
            try:
                self._detector = YOLO(f"{model_name}.pt")
            except Exception:
                log.warning(f"{model_name}.pt not found — trying yolov8n.pt")
                self._detector = YOLO("yolov8n.pt")
            self._detector.to(self.cfg.device)
            log.info(f"  ✓ Detector ready on {self.cfg.device}")
        except ImportError:
            log.warning("ultralytics not installed — simulation mode")
            self._detector = _SimulatedDetector()
        except Exception as e:
            log.warning(f"Detector load error ({e}) — simulation mode")
            self._detector = _SimulatedDetector()

    def _load_ocr(self):
        try:
            import easyocr
            log.info("Loading EasyOCR …")
            gpu = (self.cfg.device != "cpu")
            self._ocr = easyocr.Reader(
                getattr(self.cfg, "ocr_languages", ["en"]),
                gpu=gpu,
                model_storage_directory=str(WEIGHTS_DIR / "easyocr"),
                verbose=False,
            )
            log.info("  ✓ EasyOCR ready")
        except ImportError:
            log.warning("easyocr not installed — simulation mode")
            self._ocr = _SimulatedOCR()
        except Exception as e:
            log.warning(f"EasyOCR error ({e}) — simulation mode")
            self._ocr = _SimulatedOCR()

    def _load_enhancer(self):
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model_path = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"
            if not model_path.exists():
                log.info(f"Downloading Real-ESRGAN weights (~64 MB) …")
                urllib.request.urlretrieve(ESRGAN_URL, str(model_path))
                log.info("  ✓ Download complete")
            log.info("Loading Real-ESRGAN …")
            net = RRDBNet(num_in_ch=3, num_out_ch=3,
                          num_feat=64, num_block=23, num_grow_ch=32,
                          scale=self.cfg.esrgan_scale)
            self._enhancer = RealESRGANer(
                scale      = self.cfg.esrgan_scale,
                model_path = str(model_path),
                model      = net,
                tile       = self.cfg.esrgan_tile,
                tile_pad   = 10, pre_pad=0,
                half       = (self.cfg.device not in ("cpu","mps")),
                device     = self.cfg.device,
            )
            log.info("  ✓ Real-ESRGAN ready")
        except ImportError:
            log.warning("realesrgan/basicsr not installed — PIL fallback")
            self._enhancer = _PILEnhancer()
        except Exception as e:
            log.warning(f"Real-ESRGAN error ({e}) — PIL fallback")
            self._enhancer = _PILEnhancer()

    def _load_safety(self):
        """
        Load SafetyClassifier — the dedicated helmet + seatbelt module.
        Tries YOLO helmet model → MobileNetV3 → improved HSV heuristic.
        """
        try:
            from models.safety_classifier import SafetyClassifier
            self._safety_clf = SafetyClassifier(
                device=getattr(self.cfg, "device", "cpu"))
            log.info("  ✓ SafetyClassifier ready")
        except Exception as e:
            log.warning(f"SafetyClassifier load error ({e}) — using null")
            self._safety_clf = _NullSafety()


class _NullSafety:
    """Last-resort stub if safety_classifier module fails."""
    def classify(self, roi, vehicle_class="Car"):
        return {"helmet": None, "helmet_conf": 0.0,
                "seatbelt": None, "seatbelt_conf": 0.0}
