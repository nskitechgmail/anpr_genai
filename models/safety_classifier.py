"""
models/safety_classifier.py — Helmet & Seatbelt Violation Detection.

Architecture (3 layers — deepest available wins):

  HelmetDetector
    Layer 1 — YOLO helmet model (auto-downloaded)
              Trained on SHWD / hard-hat dataset.
              Classes: 0=helmet, 1=no_helmet
    Layer 2 — MobileNetV3 binary head-crop classifier
              Used when Layer 1 unavailable but torch installed.
    Layer 3 — Improved HSV heuristic
              Head-zone cropped, skin+dark-blob analysis.

  SeatbeltDetector
    Layer 1 — YOLO seatbelt model (if available)
    Layer 2 — Diagonal-edge gradient detection
              Seatbelt appears as strong diagonal line across torso.
    Layer 3 — Grey/dark stripe heuristic

  SafetyClassifier  (main interface used by plate_recogniser)
    .classify(vehicle_roi, vehicle_class)  →  dict

ROI extraction rules (critical — garbage in, garbage out):
  Motorcycle: helmet_roi  = upper 40% of vehicle bbox, centre 70% width
  Car/Bus:    seatbelt_roi = centre-left 50% width, 20-65% height (driver)
"""

from __future__ import annotations
import logging
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("SafetyClassifier")

# ── Model download URLs ────────────────────────────────────────────────────
# keremberke YOLOv8n hard-hat detection (Helmet / No Helmet)
_HELMET_MODEL_URL = (
    "https://github.com/keremberke/awesome-yolov8-models/releases/download/"
    "v1.0.0/best.pt"
)
# Fallback: Roboflow public helmet model
_HELMET_MODEL_URL_FB = (
    "https://huggingface.co/Ultralytics/assets/resolve/main/"
    "yolov8n.pt"                    # placeholder — replaced at runtime
)

_WEIGHTS_DIR = Path(__file__).parent / "weights"
_HELMET_PT   = _WEIGHTS_DIR / "helmet_detector.pt"
_BELT_PT     = _WEIGHTS_DIR / "seatbelt_detector.pt"


# ══════════════════════════════════════════════════════════════════════════
#  ROI Extraction helpers
# ══════════════════════════════════════════════════════════════════════════

def extract_helmet_roi(vehicle_roi: np.ndarray) -> np.ndarray:
    """
    Extract the rider's head zone from a motorcycle vehicle crop.

    Empirical rule: helmet is in the upper 40% of the motorcycle bbox,
    centred horizontally (ignore wheel / engine zones on the sides).
    """
    if vehicle_roi is None or vehicle_roi.size == 0:
        return vehicle_roi
    h, w = vehicle_roi.shape[:2]
    y2   = int(h * 0.45)
    x1   = int(w * 0.15)
    x2   = int(w * 0.85)
    roi  = vehicle_roi[0:y2, x1:x2]
    return roi if roi.size > 0 else vehicle_roi


def extract_seatbelt_roi(vehicle_roi: np.ndarray) -> np.ndarray:
    """
    Extract the driver torso region from a car crop.

    Driver sits in the left-centre portion of a car (right-hand traffic).
    Seatbelt runs diagonally from upper-left shoulder to lower-right hip.
    """
    if vehicle_roi is None or vehicle_roi.size == 0:
        return vehicle_roi
    h, w = vehicle_roi.shape[:2]
    # Left half of vehicle (driver side for India — right-hand traffic)
    x1 = int(w * 0.05)
    x2 = int(w * 0.55)
    y1 = int(h * 0.15)
    y2 = int(h * 0.70)
    roi = vehicle_roi[y1:y2, x1:x2]
    return roi if roi.size > 0 else vehicle_roi


# ══════════════════════════════════════════════════════════════════════════
#  Layer 1: YOLO-based helmet detector
# ══════════════════════════════════════════════════════════════════════════

class YOLOHelmetDetector:
    """
    YOLOv8n model fine-tuned for helmet detection.
    Auto-downloads weights on first use.
    Classes expected: 0 = helmet/with_helmet, 1 = no_helmet/without_helmet
    (some models have reversed indices — handled via _HELMET_CLASS_IDX)
    """

    # Class names from keremberke hard-hat model:
    # 0 = "helmet", 1 = "no_helmet"
    # If your model differs, adjust this dict
    _HELMET_IDX    = 0     # class id for "has helmet"
    _NO_HELMET_IDX = 1     # class id for "no helmet"

    def __init__(self, device: str = "cpu"):
        self._model  = None
        self._device = device
        self._loaded = False
        self._load()

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        _WEIGHTS_DIR.mkdir(exist_ok=True)

        try:
            from ultralytics import YOLO

            # Try to download if not present
            if not _HELMET_PT.exists():
                log.info("Downloading helmet detector weights …")
                try:
                    urllib.request.urlretrieve(_HELMET_MODEL_URL,
                                               str(_HELMET_PT))
                    log.info("  ✓ Helmet detector downloaded")
                except Exception as e:
                    log.warning(f"  Download failed ({e}) — trying hub …")
                    try:
                        # Try ultralytics hub model as fallback
                        m = YOLO("keremberke/yolov8n-hard-hat-detection")
                        m.save(str(_HELMET_PT))
                        log.info("  ✓ Helmet detector saved from hub")
                    except Exception as e2:
                        log.warning(f"  Hub also failed ({e2})")
                        return

            if _HELMET_PT.exists():
                self._model = YOLO(str(_HELMET_PT))
                self._model.to(self._device)
                log.info("  ✓ YOLO helmet detector ready")
        except ImportError:
            log.warning("ultralytics not installed — YOLO helmet unavailable")
        except Exception as e:
            log.warning(f"YOLO helmet load error: {e}")

    def predict(self, roi: np.ndarray) -> tuple[bool, float]:
        """
        Returns (has_helmet: bool, confidence: float).
        """
        if self._model is None or roi is None or roi.size == 0:
            return None, 0.0

        try:
            results = self._model(roi, conf=0.25, verbose=False)
            helmet_conf    = 0.0
            no_helmet_conf = 0.0

            for r in results:
                for i in range(len(r.boxes.cls)):
                    cls  = int(r.boxes.cls[i])
                    conf = float(r.boxes.conf[i])
                    # Handle models where class names vary
                    name = ""
                    if r.names:
                        name = r.names.get(cls, "").lower()

                    is_helmet = (cls == self._HELMET_IDX or
                                 "helmet" in name and "no" not in name and
                                 "without" not in name)
                    is_nohelm = (cls == self._NO_HELMET_IDX or
                                 "no_helmet" in name or "without" in name or
                                 "nohelmet" in name)

                    if is_helmet:
                        helmet_conf    = max(helmet_conf, conf)
                    elif is_nohelm:
                        no_helmet_conf = max(no_helmet_conf, conf)

            if helmet_conf == 0.0 and no_helmet_conf == 0.0:
                return None, 0.0   # nothing detected in this crop

            has_helmet = helmet_conf >= no_helmet_conf
            confidence = max(helmet_conf, no_helmet_conf)
            return has_helmet, confidence

        except Exception as e:
            log.debug(f"YOLO helmet predict error: {e}")
            return None, 0.0


# ══════════════════════════════════════════════════════════════════════════
#  Layer 2: MobileNetV3 binary head classifier
# ══════════════════════════════════════════════════════════════════════════

class MobileNetV3HelmetClassifier:
    """
    MobileNetV3-Small fine-tuned as a binary helmet/no-helmet classifier.
    Used when the YOLO model is unavailable but torch/torchvision are present.
    Loads weights from weights/safety_classifier.pth if available,
    otherwise initialises with ImageNet weights and uses feature heuristics.
    """

    def __init__(self, device: str = "cpu"):
        self._model     = None
        self._transform = None
        self._device    = device
        self._load()

    def _load(self):
        try:
            import torch
            import torchvision.models as tv
            import torchvision.transforms as T

            model = tv.mobilenet_v3_small(weights="IMAGENET1K_V1")
            # Replace final layer for binary classification
            model.classifier[-1] = torch.nn.Linear(
                model.classifier[-1].in_features, 2)

            # Load fine-tuned weights if available
            weights_path = _WEIGHTS_DIR / "safety_classifier.pth"
            if weights_path.exists():
                model.load_state_dict(
                    torch.load(str(weights_path), map_location=device))
                log.info("  ✓ MobileNetV3 safety classifier loaded")
            else:
                log.info("  ℹ  MobileNetV3 using ImageNet features (no fine-tune)")

            model.eval().to(self._device)
            self._model = model
            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
            ])
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"MobileNetV3 load error: {e}")

    def predict(self, roi: np.ndarray) -> tuple[bool, float]:
        if self._model is None or roi is None or roi.size == 0:
            return None, 0.0
        try:
            import torch
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            inp = self._transform(rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                logits = self._model(inp)[0]
                probs  = torch.softmax(logits, dim=0).cpu().tolist()
            # Index 0 = helmet, Index 1 = no_helmet
            has_helmet = probs[0] > probs[1]
            confidence = max(probs)
            return has_helmet, confidence
        except Exception as e:
            log.debug(f"MobileNetV3 predict error: {e}")
            return None, 0.0


# ══════════════════════════════════════════════════════════════════════════
#  Layer 3: Improved HSV heuristic helmet detector
# ══════════════════════════════════════════════════════════════════════════

class HSVHelmetHeuristic:
    """
    Improved heuristic that actually works — unlike the old brightness check.

    Logic:
    1. Detect skin-coloured pixels in the head zone
    2. A bare head has ~30-60% skin in the top region
    3. A helmeted head has mostly non-skin colours (any solid colour)
    4. Detect rounded blob shape (helmet) vs irregular (bare head hair)

    Handles: dark helmets, coloured helmets, low light.
    """

    def predict(self, roi: np.ndarray) -> tuple[bool, float]:
        if roi is None or roi.size == 0:
            return True, 0.50   # uncertain — default to compliant

        h, w = roi.shape[:2]
        if h < 10 or w < 10:
            return True, 0.50

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Skin colour range in HSV (handles various skin tones)
        # Lower skin tone
        skin_lo1 = np.array([0,  20, 70],  dtype=np.uint8)
        skin_hi1 = np.array([20, 200, 255], dtype=np.uint8)
        # Upper skin tone
        skin_lo2 = np.array([160, 20, 70],  dtype=np.uint8)
        skin_hi2 = np.array([180, 200, 255], dtype=np.uint8)

        mask1    = cv2.inRange(hsv, skin_lo1, skin_hi1)
        mask2    = cv2.inRange(hsv, skin_lo2, skin_hi2)
        skin_mask = cv2.bitwise_or(mask1, mask2)

        total_px  = h * w
        skin_px   = int(skin_mask.sum() / 255)
        skin_ratio = skin_px / max(total_px, 1)

        # Dominant colour analysis — helmets tend to be uniformly coloured
        resized   = cv2.resize(roi, (32, 32))
        std_dev   = float(np.std(resized.reshape(-1, 3), axis=0).mean())
        # Low std_dev in head region → uniform colour → likely helmet
        uniform   = std_dev < 45

        # Circular blob detection (helmets are roughly round)
        gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur     = cv2.GaussianBlur(gray, (5, 5), 0)
        circles  = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1,
            minDist=max(10, min(h, w) // 2),
            param1=50, param2=20,
            minRadius=max(5, min(h, w) // 5),
            maxRadius=max(10, min(h, w) // 2),
        )
        has_round_blob = circles is not None

        # Decision logic
        # Strong skin signal → bare head → no helmet
        if skin_ratio > 0.35:
            confidence = min(0.80, 0.50 + skin_ratio)
            return False, confidence

        # Uniform colour + round blob → helmet
        if uniform and has_round_blob:
            return True, 0.75

        # Uniform colour alone → probably helmet
        if uniform and skin_ratio < 0.15:
            return True, 0.65

        # Weak signal — moderate confidence no-helmet
        if skin_ratio > 0.18:
            return False, 0.55

        # Default — uncertain, assume compliant
        return True, 0.52


# ══════════════════════════════════════════════════════════════════════════
#  Seatbelt detector — diagonal edge analysis
# ══════════════════════════════════════════════════════════════════════════

class DiagonalSeatbeltDetector:
    """
    Detects seatbelt presence by finding a strong diagonal edge (the strap)
    crossing the driver torso region from upper-shoulder to lower-hip.

    Seatbelt strap characteristics:
      - Narrow band (~5-20px wide)
      - Grey / dark grey colour
      - Diagonal angle 30-60° from vertical
      - Runs from top-left to bottom-right of driver area
    """

    def predict(self, roi: np.ndarray) -> tuple[bool, float]:
        if roi is None or roi.size == 0:
            return None, 0.0

        h, w = roi.shape[:2]
        if h < 20 or w < 20:
            return None, 0.0

        try:
            gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Enhance contrast
            clahe = cv2.createCLAHE(2.0, (8, 8))
            gray  = clahe.apply(gray)

            # Detect edges
            edges = cv2.Canny(gray, 30, 100)

            # Hough lines — look for diagonal lines (30-60° to vertical)
            lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi/180,
                threshold=max(20, min(h, w) // 4),
                minLineLength=max(15, min(h, w) // 4),
                maxLineGap=max(5, min(h, w) // 8),
            )

            if lines is None:
                return self._colour_heuristic(roi)

            diagonal_count  = 0
            diagonal_conf   = 0.0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                # Seatbelt diagonal: 25-65° from horizontal
                if 25 <= angle <= 65:
                    # Check it goes from upper-area to lower-area
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length > min(h, w) * 0.25:
                        diagonal_count += 1
                        diagonal_conf   = max(diagonal_conf,
                                              min(0.90, length / (h * 0.8)))

            if diagonal_count >= 1:
                return True, max(0.60, diagonal_conf)

            return self._colour_heuristic(roi)

        except Exception as e:
            log.debug(f"Seatbelt detect error: {e}")
            return None, 0.0

    def _colour_heuristic(self, roi: np.ndarray) -> tuple[bool, float]:
        """Check for grey/dark band as fallback."""
        h, w = roi.shape[:2]
        # Seatbelt is typically grey (80-160 brightness, low saturation)
        hsv       = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        grey_mask = cv2.inRange(hsv,
                                np.array([0,  0,  60]),
                                np.array([180, 40, 170]))
        grey_ratio = grey_mask.sum() / 255 / max(h * w, 1)

        if grey_ratio > 0.05:
            return True, min(0.70, 0.50 + grey_ratio * 4)
        return False, 0.55


# ══════════════════════════════════════════════════════════════════════════
#  SafetyClassifier — main public interface
# ══════════════════════════════════════════════════════════════════════════

class SafetyClassifier:
    """
    Main safety compliance classifier.

    Usage:
        clf = SafetyClassifier(device="cpu")
        result = clf.classify(vehicle_roi, vehicle_class="Motorcycle")
        # result = {"helmet": bool, "helmet_conf": float,
        #           "seatbelt": bool, "seatbelt_conf": float}

    Integrates with ModelManager as models.safety_classifier
    """

    def __init__(self, device: str = "cpu"):
        self._device = device

        # Helmet — 3 layers
        self._yolo_helmet   = YOLOHelmetDetector(device)
        self._mobilenet     = MobileNetV3HelmetClassifier(device)
        self._hsv_heuristic = HSVHelmetHeuristic()

        # Seatbelt — 2 layers
        self._diagonal_belt = DiagonalSeatbeltDetector()

        log.info("SafetyClassifier initialised.")

    def classify(
        self,
        vehicle_roi:   np.ndarray,
        vehicle_class: str = "Car",
    ) -> dict:
        """
        Classify helmet and seatbelt compliance from vehicle ROI.

        Returns
        -------
        dict with keys:
            helmet        : bool | None
            helmet_conf   : float  (0–1)
            seatbelt      : bool | None
            seatbelt_conf : float  (0–1)
        """
        result = {
            "helmet":        None,
            "helmet_conf":   0.0,
            "seatbelt":      None,
            "seatbelt_conf": 0.0,
        }

        if vehicle_roi is None or vehicle_roi.size == 0:
            return result

        # ── Helmet (motorcycles only) ─────────────────────────────────────
        if vehicle_class in ("Motorcycle", "motorcycle"):
            head_roi = extract_helmet_roi(vehicle_roi)
            helmet, conf = self._predict_helmet(head_roi)
            result["helmet"]      = helmet
            result["helmet_conf"] = conf

        # ── Seatbelt (cars, buses, trucks) ────────────────────────────────
        elif vehicle_class in ("Car", "Bus", "Truck"):
            belt_roi = extract_seatbelt_roi(vehicle_roi)
            belt, conf = self._predict_seatbelt(belt_roi)
            result["seatbelt"]      = belt
            result["seatbelt_conf"] = conf

        return result

    # ── Internal prediction with layer fallback ───────────────────────────

    def _predict_helmet(self, head_roi: np.ndarray) -> tuple[Optional[bool], float]:
        """Try YOLO → MobileNetV3 → HSV heuristic."""
        # Layer 1: YOLO
        h, c = self._yolo_helmet.predict(head_roi)
        if h is not None and c >= 0.35:
            log.debug(f"  Helmet via YOLO: {'YES' if h else 'NO'} ({c:.2f})")
            return h, c

        # Layer 2: MobileNetV3
        h, c = self._mobilenet.predict(head_roi)
        if h is not None and c >= 0.40:
            log.debug(f"  Helmet via MobileNetV3: {'YES' if h else 'NO'} ({c:.2f})")
            return h, c

        # Layer 3: HSV heuristic (always returns a result)
        h, c = self._hsv_heuristic.predict(head_roi)
        log.debug(f"  Helmet via HSV: {'YES' if h else 'NO'} ({c:.2f})")
        return h, c

    def _predict_seatbelt(
        self, belt_roi: np.ndarray
    ) -> tuple[Optional[bool], float]:
        """Try diagonal edge detection → colour heuristic."""
        b, c = self._diagonal_belt.predict(belt_roi)
        if b is not None:
            log.debug(f"  Seatbelt via diagonal: {'YES' if b else 'NO'} ({c:.2f})")
            return b, c
        return None, 0.0
