"""
core/plate_recogniser.py — End-to-end licence plate recognition pipeline.

Stage 1 : YOLO vehicle detection
Stage 2 : Plate localisation (contour + aspect-ratio filter)
Stage 3 : GenAI enhancement via Real-ESRGAN (or PIL fallback)
Stage 4 : CLAHE + Otsu pre-processing → EasyOCR
Stage 5 : Indian plate format validation + post-processing correction
"""

from __future__ import annotations
import cv2, re, time, logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("PlateRecogniser")

# ── Indian licence plate regex ─────────────────────────────────────────────
# Covers: MH12AB1234 / MH 12 AB 1234 / MH-12-AB-1234 / MH12A1234 etc.
_PLATE_RE = re.compile(
    r"^[A-Z]{2}[\s\-]?\d{1,2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{1,4}$",
    re.IGNORECASE,
)

# COCO class IDs that correspond to vehicles
_VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Common OCR correction map (visually similar characters on plates)
_OCR_FIXES = {
    "O": "0", "I": "1", "Z": "2", "S": "5", "B": "8",
}


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class PlateResult:
    text:          str
    confidence:    float
    bbox:          tuple          # (x1,y1,x2,y2) in original frame coords
    plate_crop:    Optional[np.ndarray] = None   # enhanced plate image (BGR)
    enhanced:      bool = False
    valid_format:  bool = False
    raw_text:      str  = ""      # OCR output before post-processing

    def __post_init__(self):
        self.valid_format = bool(_PLATE_RE.match(self.text))

    def normalised(self) -> str:
        """Return plate text in canonical upper-case spaced form."""
        t = re.sub(r"[\s\-]+", " ", self.text.upper().strip())
        return t


@dataclass
class VehicleDetection:
    vehicle_class: str
    bbox:          tuple          # (x1,y1,x2,y2)
    confidence:    float
    plate:         Optional[PlateResult] = None
    helmet:        Optional[bool] = None
    helmet_conf:   float = 0.0
    seatbelt:      Optional[bool] = None
    seatbelt_conf: float = 0.0
    violation:     str = "Compliant"
    timestamp:     float = field(default_factory=time.time)

    def has_violation(self) -> bool:
        return self.violation != "Compliant"


@dataclass
class FrameStats:
    """Per-frame pipeline statistics passed to the annotator and dashboard."""
    fps:                 float = 0.0
    vehicle_count:       int   = 0
    plates_read:         int   = 0
    violations:          int   = 0
    helmet_violations:   int   = 0
    seatbelt_violations: int   = 0
    genai_enabled:       bool  = False
    camera_id:           str   = "CCTV-001"
    frame_number:        int   = 0
    processing_ms:       float = 0.0


# ── Plate Recogniser ───────────────────────────────────────────────────────

class PlateRecogniser:
    """
    Orchestrates the full plate detection + recognition pipeline.

    Parameters
    ----------
    models   : ModelManager  — pre-loaded model registry
    settings : Settings      — runtime configuration
    """

    def __init__(self, models, settings):
        self.models = models
        self.cfg    = settings

    # ── Public API ────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> list[VehicleDetection]:
        """Process one frame end-to-end. Returns VehicleDetection list."""
        detections: list[VehicleDetection] = []

        vehicles = self._detect_vehicles(frame)

        for vdet in vehicles:
            x1, y1, x2, y2 = vdet["bbox"]
            vcls            = vdet["class"]
            vconf           = vdet["conf"]

            roi = self._safe_crop(frame, x1, y1, x2, y2, pad=10)
            if roi is None:
                continue

            plate_result = self._detect_and_read_plate(frame, roi, x1, y1)

            helmet_ok, helmet_conf, belt_ok, belt_conf = \
                self._check_safety(roi, vcls)

            violation = self._classify_violation(
                vcls, helmet_ok, helmet_conf, belt_ok, belt_conf
            )

            detections.append(VehicleDetection(
                vehicle_class = vcls,
                bbox          = (x1, y1, x2, y2),
                confidence    = vconf,
                plate         = plate_result,
                helmet        = helmet_ok,
                helmet_conf   = helmet_conf,
                seatbelt      = belt_ok,
                seatbelt_conf = belt_conf,
                violation     = violation,
            ))

        return detections

    # ── Stage 1: YOLO detection ───────────────────────────────────────────

    def _detect_vehicles(self, frame: np.ndarray) -> list[dict]:
        results = []
        try:
            yolo_results = self.models.detector(
                frame,
                conf    = self.cfg.conf_thresh,
                iou     = self.cfg.iou_thresh,
                classes = list(_VEHICLE_CLASSES.keys()),
                verbose = False,
            )
            for r in yolo_results:
                boxes = r.boxes
                for i in range(len(boxes.xyxy)):
                    cls_id = int(boxes.cls[i])
                    if cls_id not in _VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])
                    results.append({
                        "bbox" : (x1, y1, x2, y2),
                        "class": _VEHICLE_CLASSES[cls_id],
                        "conf" : float(boxes.conf[i]),
                    })
        except Exception as e:
            log.debug(f"YOLO inference error: {e}")
            # CPU fallback simulation (for demo without GPU)
            h, w = frame.shape[:2]
            results.append({
                "bbox" : (w // 6, h // 3, 5 * w // 6, 9 * h // 10),
                "class": "Car",
                "conf" : 0.75,
            })
        return results

    # ── Stage 2: Smart plate localisation ────────────────────────────────

    def _locate_plate_in_roi(self, roi: np.ndarray,
                              vehicle_class: str = "Car") -> Optional[tuple]:
        """
        Smart plate localisation:
        - Searches lower 50-60% of ROI only (plates never at the top)
        - Rejects crops wider than 80% of ROI (advertisement boards)
        - Scores candidates by area + centre position + bottom position
        Returns (x1,y1,x2,y2) in ROI coordinates, or None.
        """
        h, w = roi.shape[:2]

        # Search only lower portion — plates are never on the roof
        if vehicle_class == "Motorcycle":
            search_top = int(h * 0.60)   # bikes: plate at very bottom
        else:
            search_top = int(h * 0.50)   # cars/buses/trucks: lower half

        search_roi = roi[search_top:, :]
        rh, rw = search_roi.shape[:2]

        if rh < 10 or rw < 10:
            return None

        # Pre-process
        gray   = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        gray   = cv2.bilateralFilter(gray, 9, 75, 75)
        edges  = cv2.Canny(gray, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)

            # Size filters
            if cw < 60 or ch < 15:         # too small
                continue
            if cw > rw * 0.80:             # too wide = advertisement board
                continue
            if ch > rh * 0.45:             # too tall = not a plate
                continue

            # Aspect ratio: Indian plates are 2.5x to 6.5x wide
            aspect = cw / max(ch, 1)
            if not (2.5 <= aspect <= 6.5):
                continue

            # Score: prefer bottom-centre candidates
            area         = cw * ch
            centre_x     = cx + cw / 2
            centre_score = 1.0 - abs(centre_x - rw / 2) / max(rw / 2, 1)
            bottom_score = cy / max(rh, 1)
            score        = area * (1 + centre_score * 0.5 + bottom_score * 0.3)

            candidates.append((score, cx, cy, cw, ch))

        if candidates:
            candidates.sort(reverse=True)
            _, cx, cy, cw, ch = candidates[0]
            pad = 4
            # Translate back to full ROI coordinates
            y1 = max(0, search_top + cy - pad)
            y2 = min(h, search_top + cy + ch + pad)
            x1 = max(0, cx - pad)
            x2 = min(w, cx + cw + pad)
            return (x1, y1, x2, y2)

        # Fallback: bottom-centre strip
        x1 = w // 4
        y1 = int(h * 0.65)
        x2 = 3 * w // 4
        y2 = int(h * 0.88)
        area = (x2 - x1) * (y2 - y1)
        if area >= getattr(self.cfg, 'plate_min_area', 200):
            return (x1, y1, x2, y2)

        return None

    # ── Stage 3 + 4: Enhancement + OCR ───────────────────────────────────

    def _detect_and_read_plate(
        self,
        full_frame:  np.ndarray,
        vehicle_roi: np.ndarray,
        roi_x:       int,
        roi_y:       int,
    ) -> Optional[PlateResult]:

        vcls = "Car"   # default for localisation hinting
        plate_bbox_local = self._locate_plate_in_roi(vehicle_roi, vcls)
        if plate_bbox_local is None:
            return None

        lx1, ly1, lx2, ly2 = plate_bbox_local
        plate_crop = self._safe_crop(vehicle_roi, lx1, ly1, lx2, ly2)
        if plate_crop is None or plate_crop.size == 0:
            return None

        # Absolute bbox in full frame
        abs_bbox = (
            roi_x + lx1, roi_y + ly1,
            roi_x + lx2, roi_y + ly2,
        )

        enhanced = False

        # Stage 3: GenAI enhancement (Real-ESRGAN) or PIL fallback
        if getattr(self.cfg, 'use_genai', False):
            plate_crop, enhanced = self._enhance_plate(plate_crop)
        else:
            # CPU fallback: bicubic upscale x4
            h, w = plate_crop.shape[:2]
            plate_crop = cv2.resize(
                plate_crop, (w * 4, h * 4),
                interpolation=cv2.INTER_CUBIC
            )

        # Stage 4: OCR
        text, conf = self._run_ocr(plate_crop)

        if not text:
            return None

        raw_text = text
        text     = self._post_correct(text)

        return PlateResult(
            text       = text,
            confidence = conf,
            bbox       = abs_bbox,
            plate_crop = plate_crop,
            enhanced   = enhanced,
            raw_text   = raw_text,
        )

    def _enhance_plate(self, plate_img: np.ndarray):
        """Apply Real-ESRGAN or fall back to PIL bicubic upscale."""
        try:
            enhanced_img, _ = self.models.enhancer.enhance(
                plate_img,
                outscale=getattr(self.cfg, 'esrgan_scale', 4),
            )
            return enhanced_img, True
        except Exception as e:
            log.debug(f"ESRGAN enhancement failed: {e} — using PIL fallback")
            h, w = plate_img.shape[:2]
            img = cv2.resize(plate_img, (w * 4, h * 4),
                             interpolation=cv2.INTER_CUBIC)
            return img, False

    # ── Stage 4: Multi-version OCR preprocessing ─────────────────────────

    def _preprocess_plate(self, plate_img: np.ndarray) -> list:
        """
        Enhanced preprocessing for unclear/blurry plates.
        Returns 4 candidate versions — Otsu, Adaptive, Inverted, Grayscale.
        Works entirely on CPU without Real-ESRGAN.
        """
        if plate_img is None or plate_img.size == 0:
            return [plate_img]

        # Step 1: Upscale to minimum readable size
        h, w = plate_img.shape[:2]
        target_w = max(w * 4, 280)
        target_h = max(h * 4, 70)
        img = cv2.resize(plate_img, (target_w, target_h),
                         interpolation=cv2.INTER_CUBIC)

        # Step 2: Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Step 3: Sharpen
        kernel_sharp = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        img = cv2.filter2D(img, -1, kernel_sharp)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 4: CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # Step 5: Four candidate binarisations
        _, v_otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        v_adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8)

        _, v_inv = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        v_gray = gray

        return [
            cv2.cvtColor(v_otsu,  cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(v_adapt, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(v_inv,   cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(v_gray,  cv2.COLOR_GRAY2BGR),
        ]

    def _run_ocr(self, plate_img: np.ndarray) -> tuple[str, float]:
        """
        Run OCR across all 4 preprocessed versions.
        Returns (best_text, best_confidence).
        """
        if plate_img is None or plate_img.size == 0:
            return "", 0.0

        versions  = self._preprocess_plate(plate_img)
        best_text = ""
        best_conf = 0.0
        allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

        for img in versions:
            try:
                results = self.models.ocr.readtext(
                    img,
                    allowlist       = allowlist,
                    detail          = 1,
                    paragraph       = False,
                    min_size        = 10,
                    text_threshold  = 0.4,
                    low_text        = 0.3,
                    link_threshold  = 0.3,
                )
                for (_, text, conf) in results:
                    text = text.strip().upper()
                    if len(text) >= 4 and conf > best_conf:
                        best_text = text
                        best_conf = conf
            except Exception as e:
                log.debug(f"OCR error on version: {e}")
                continue

        return best_text, best_conf

    # ── Stage 5: Post-correction ──────────────────────────────────────────

    def _post_correct(self, text: str) -> str:
        """
        Position-aware OCR correction for Indian plate format:
        SS NN LL NNNN  (state, district, series, number)
        """
        text     = re.sub(r"[\s\-]+", "", text.upper())
        text     = re.sub(r"[^A-Z0-9]", "", text)
        corrected = list(text)

        letter_positions = {0, 1}   # state code — must be letters
        digit_positions  = {2, 3}   # district  — must be digits

        for i, ch in enumerate(corrected):
            if i in letter_positions and ch.isdigit():
                rev = {v: k for k, v in _OCR_FIXES.items()}
                corrected[i] = rev.get(ch, ch)
            elif i in digit_positions and ch.isalpha():
                corrected[i] = _OCR_FIXES.get(ch, ch)

        return "".join(corrected)

    # ── Safety compliance ─────────────────────────────────────────────────

    def _check_safety(
        self, roi: np.ndarray, vehicle_class: str
    ) -> tuple[Optional[bool], float, Optional[bool], float]:
        """
        Returns (helmet_ok, helmet_conf, seatbelt_ok, seatbelt_conf).
        Uses MobileNetV3 classifier if available, otherwise heuristics.
        """
        helmet_ok   = None
        helmet_conf = 0.0
        belt_ok     = None
        belt_conf   = 0.0

        try:
            clf = self.models.safety_classifier
            if clf is None:
                raise AttributeError("No classifier")

            result      = clf.classify(roi)
            helmet_ok   = result.get("helmet")
            helmet_conf = result.get("helmet_conf", 0.0)
            belt_ok     = result.get("seatbelt")
            belt_conf   = result.get("seatbelt_conf", 0.0)
        except Exception:
            # Heuristic fallback using brightness as crude proxy
            gray        = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_bright = float(np.mean(gray))
            if vehicle_class == "Motorcycle":
                h     = roi.shape[0]
                upper = cv2.cvtColor(roi[:h // 3], cv2.COLOR_BGR2GRAY)
                helmet_ok   = float(np.mean(upper)) > 60
                helmet_conf = 0.65
            elif vehicle_class in ("Car", "Bus", "Truck"):
                belt_ok   = mean_bright > 50
                belt_conf = 0.60

        return helmet_ok, helmet_conf, belt_ok, belt_conf

    def _classify_violation(
        self,
        vehicle_class: str,
        helmet_ok:  Optional[bool], helmet_conf: float,
        belt_ok:    Optional[bool], belt_conf:   float,
    ) -> str:
        if vehicle_class == "Motorcycle":
            if helmet_ok is False and helmet_conf >= getattr(self.cfg, 'helmet_conf', 0.55):
                return "No Helmet"
        elif vehicle_class in ("Car", "Bus", "Truck"):
            if belt_ok is False and belt_conf >= getattr(self.cfg, 'seatbelt_conf', 0.55):
                return "No Seat Belt"
        return "Compliant"

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _safe_crop(
        img: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        pad: int = 0,
    ) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2].copy()
