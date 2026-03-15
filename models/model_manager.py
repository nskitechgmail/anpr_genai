"""
models/model_manager.py — Lazy-loaded model registry.

Loads all ML models once; provides graceful simulation fallbacks so the
system runs on any laptop — even without GPU or heavy ML dependencies.

Property names:
  .detector         — YOLOv8/v9 Ultralytics model
  .ocr              — EasyOCR reader
  .enhancer         — Real-ESRGAN upscaler
  .safety_classifier — MobileNetV3 helmet/seatbelt classifier
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


# ══════════════════════════════════════════════════════════════════
#  Simulation stubs
# ══════════════════════════════════════════════════════════════════

class _SimulatedDetector:
    def __call__(self, frame, **kw):
        return []


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


class _NullSafetyClassifier:
    def classify(self, roi):
        return {"helmet": None, "helmet_conf": 0.0,
                "seatbelt": None, "seatbelt_conf": 0.0}


# ══════════════════════════════════════════════════════════════════
#  ModelManager
# ══════════════════════════════════════════════════════════════════

class ModelManager:
    def __init__(self, settings):
        self.cfg              = settings
        self._detector        = None
        self._ocr             = None
        self._enhancer        = None
        self._safety_clf      = None   # internal; exposed as .safety_classifier

    # ── Properties ────────────────────────────────────────────────

    @property
    def detector(self):
        if self._detector is None:
            self._load_detector()
        return self._detector

    @property
    def ocr(self):
        if self._ocr is None:
            self._load_ocr()
        return self._ocr

    @property
    def enhancer(self):
        if self._enhancer is None and getattr(self.cfg, "use_genai", False):
            self._load_enhancer()
        return self._enhancer

    @property
    def safety_classifier(self):
        """Exposed as safety_classifier (used by plate_recogniser.py)."""
        if self._safety_clf is None:
            self._load_safety()
        return self._safety_clf

    # ── Bulk loader ───────────────────────────────────────────────

    def load_all(self):
        log.info("Loading models …")
        self._load_detector()
        self._load_ocr()
        if getattr(self.cfg, "use_genai", False):
            self._load_enhancer()
        self._load_safety()
        log.info("All models ready.")

    # ── Individual loaders ────────────────────────────────────────

    def _load_detector(self):
        try:
            from ultralytics import YOLO
            model_name = getattr(self.cfg, "detector_model", "yolov8n")
            log.info(f"Loading YOLO: {model_name}")
            # Try the configured model; fall back to yolov8n if not found
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
                self._download_esrgan(model_path)
            log.info("Loading Real-ESRGAN …")
            net = RRDBNet(num_in_ch=3, num_out_ch=3,
                          num_feat=64, num_block=23, num_grow_ch=32,
                          scale=self.cfg.esrgan_scale)
            self._enhancer = RealESRGANer(
                scale      = self.cfg.esrgan_scale,
                model_path = str(model_path),
                model      = net,
                tile       = self.cfg.esrgan_tile,
                tile_pad   = 10,
                pre_pad    = 0,
                half       = (self.cfg.device not in ("cpu", "mps")),
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
        try:
            import torch
            import torchvision.models as tv
            import torchvision.transforms as T
            model_path = WEIGHTS_DIR / "safety_classifier.pth"
            if not model_path.exists():
                log.warning("Safety classifier weights not found — heuristics mode")
                self._safety_clf = _NullSafetyClassifier()
                return
            log.info("Loading safety classifier …")
            model = tv.mobilenet_v3_small(weights=None)
            model.classifier[-1] = torch.nn.Linear(
                model.classifier[-1].in_features, 4)
            model.load_state_dict(
                torch.load(str(model_path), map_location=self.cfg.device))
            model.eval().to(self.cfg.device)
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
            self._safety_clf = _TorchSafetyClassifier(
                model, transform, self.cfg.device)
            log.info("  ✓ Safety classifier ready")
        except ImportError:
            log.warning("torch/torchvision not available — heuristics mode")
            self._safety_clf = _NullSafetyClassifier()
        except Exception as e:
            log.warning(f"Safety classifier error ({e}) — heuristics mode")
            self._safety_clf = _NullSafetyClassifier()

    def _download_esrgan(self, dest: Path):
        log.info(f"Downloading Real-ESRGAN weights (~64 MB) → {dest}")
        urllib.request.urlretrieve(ESRGAN_URL, str(dest))
        log.info("  ✓ Download complete")


class _TorchSafetyClassifier:
    def __init__(self, model, transform, device):
        self.model     = model
        self.transform = transform
        self.device    = device

    def classify(self, roi):
        import torch, cv2
        try:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            inp = self.transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = torch.softmax(self.model(inp), dim=1)[0].cpu().tolist()
            # Indices: 0=helmet_ok, 1=no_helmet, 2=belt_ok, 3=no_belt
            return {
                "helmet":        out[0] > out[1],
                "helmet_conf":   max(out[0], out[1]),
                "seatbelt":      out[2] > out[3],
                "seatbelt_conf": max(out[2], out[3]),
            }
        except Exception as e:
            log.debug(f"Safety classify error: {e}")
            return {"helmet": None, "helmet_conf": 0.0,
                    "seatbelt": None, "seatbelt_conf": 0.0}
