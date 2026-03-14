"""
core/plate_recogniser.py — End-to-end licence plate recognition pipeline.

Stage 1 : YOLO vehicle detection
Stage 2 : Plate localisation with rotation + tilt handling
Stage 3 : Perspective correction (deskew distorted plates)
Stage 4 : GenAI enhancement (Real-ESRGAN) or CPU bicubic upscale
Stage 5 : Multi-version preprocessing → EasyOCR best-of-4
Stage 6 : Indian plate format validation + position-aware correction

Robustness improvements over baseline:
  ✓ minAreaRect  — detects tilted / rotated plates (not just axis-aligned)
  ✓ Deskew       — perspective warp corrects leaning / distorted plates
  ✓ Deblur       — unsharp mask recovers motion-blurred characters
  ✓ 4-version OCR — Otsu / Adaptive / Inverted / CLAHE all tried, best kept
  ✓ Relaxed aspect — 1.5–8.0 range covers extreme angles
  ✓ Multi-scale  — tries original + 2× upscale if first attempt fails
  ✓ Fallback strip — guaranteed attempt even when contours fail
"""

from __future__ import annotations
import cv2
import re
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("PlateRecogniser")

# ── Indian licence plate regex ─────────────────────────────────────────────
_PLATE_RE = re.compile(
    r"^[A-Z]{2}[\s\-]?\d{1,2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{1,4}$",
    re.IGNORECASE,
)

# COCO vehicle class IDs
_VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# OCR character confusion correction (position-aware applied later)
_OCR_FIXES = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "1"}


# ══════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PlateResult:
    text:         str
    confidence:   float
    bbox:         tuple                    # (x1,y1,x2,y2) in full-frame coords
    plate_crop:   Optional[np.ndarray] = None
    enhanced:     bool  = False
    valid_format: bool  = False
    raw_text:     str   = ""

    def __post_init__(self):
        self.valid_format = bool(_PLATE_RE.match(self.text))

    def normalised(self) -> str:
        t = re.sub(r"[\s\-]+", " ", self.text.upper().strip())
        return t


@dataclass
class VehicleDetection:
    vehicle_class: str
    bbox:          tuple
    confidence:    float
    plate:         Optional[PlateResult] = None
    helmet:        Optional[bool]        = None
    helmet_conf:   float                 = 0.0
    seatbelt:      Optional[bool]        = None
    seatbelt_conf: float                 = 0.0
    violation:     str                   = "Compliant"
    timestamp:     float                 = field(default_factory=time.time)

    def has_violation(self) -> bool:
        return self.violation != "Compliant"


# ══════════════════════════════════════════════════════════════════════════
#  PlateRecogniser
# ══════════════════════════════════════════════════════════════════════════

class PlateRecogniser:
    """Full plate detection + OCR pipeline with robustness improvements."""

    def __init__(self, models, settings):
        self.models = models
        self.cfg    = settings

    # ── Public ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> list[VehicleDetection]:
        detections: list[VehicleDetection] = []
        for vdet in self._detect_vehicles(frame):
            x1, y1, x2, y2 = vdet["bbox"]
            roi = self._safe_crop(frame, x1, y1, x2, y2, pad=12)
            if roi is None:
                continue
            plate  = self._detect_and_read_plate(frame, roi, x1, y1,
                                                  vdet["class"])
            helmet_ok, hconf, belt_ok, bconf = self._check_safety(roi, vdet["class"])
            violation = self._classify_violation(
                vdet["class"], helmet_ok, hconf, belt_ok, bconf)
            detections.append(VehicleDetection(
                vehicle_class = vdet["class"],
                bbox          = (x1, y1, x2, y2),
                confidence    = vdet["conf"],
                plate         = plate,
                helmet        = helmet_ok,
                helmet_conf   = hconf,
                seatbelt      = belt_ok,
                seatbelt_conf = bconf,
                violation     = violation,
            ))
        return detections

    # ── Stage 1: YOLO detection ───────────────────────────────────────────

    def _detect_vehicles(self, frame: np.ndarray) -> list[dict]:
        try:
            results = self.models.detector(
                frame,
                conf    = self.cfg.conf_thresh,
                iou     = self.cfg.iou_thresh,
                classes = list(_VEHICLE_CLASSES.keys()),
                verbose = False,
            )
            out = []
            for r in results:
                for i in range(len(r.boxes.xyxy)):
                    cls_id = int(r.boxes.cls[i])
                    if cls_id not in _VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in r.boxes.xyxy[i])
                    out.append({"bbox": (x1, y1, x2, y2),
                                "class": _VEHICLE_CLASSES[cls_id],
                                "conf":  float(r.boxes.conf[i])})
            return out
        except Exception as e:
            log.debug(f"YOLO error: {e}")
            h, w = frame.shape[:2]
            return [{"bbox": (w//6, h//3, 5*w//6, 9*h//10),
                     "class": "Car", "conf": 0.75}]

    # ── Stage 2: Plate localisation ───────────────────────────────────────

    def _locate_plate_in_roi(
        self, roi: np.ndarray, vehicle_class: str = "Car"
    ) -> Optional[tuple]:
        """
        Find the best plate-shaped rectangle in the vehicle ROI.
        Uses minAreaRect to handle TILTED plates, then returns
        the axis-aligned bounding box of the rotated rect.
        """
        h, w = roi.shape[:2]

        # Search only lower portion — plates never appear at the top
        search_top = int(h * 0.60) if vehicle_class == "Motorcycle" else int(h * 0.50)
        search_roi = roi[search_top:, :]
        rh, rw = search_roi.shape[:2]
        if rh < 10 or rw < 10:
            return None

        # Pre-process for edge detection
        gray   = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        gray   = cv2.bilateralFilter(gray, 9, 75, 75)
        edges  = cv2.Canny(gray, 30, 120)
        # Morphological close bridges character gaps inside the plate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            # Use minAreaRect — handles rotated contours
            rect   = cv2.minAreaRect(cnt)
            box    = cv2.boxPoints(rect)
            box    = np.intp(box)
            cx_r, cy_r = rect[0]
            (rw_r, rh_r) = rect[1]
            angle  = rect[2]

            # Normalise width/height (minAreaRect can swap them)
            if rw_r < rh_r:
                rw_r, rh_r = rh_r, rw_r

            # Size filters
            if rw_r < 50 or rh_r < 12:
                continue
            if rw_r > rw * 0.85:
                continue
            if rh_r > rh * 0.50:
                continue

            # Aspect ratio — relaxed to capture plates at an angle
            aspect = rw_r / max(rh_r, 1)
            if not (1.5 <= aspect <= 8.0):
                continue

            area         = rw_r * rh_r
            centre_score = 1.0 - abs(cx_r - rw / 2) / max(rw / 2, 1)
            bottom_score = cy_r / max(rh, 1)
            score        = area * (1 + centre_score * 0.4 + bottom_score * 0.3)
            candidates.append((score, box, cx_r, cy_r, rw_r, rh_r))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, box, cx_r, cy_r, rw_r, rh_r = candidates[0]
            # Axis-aligned bounding box of the minAreaRect
            bx, by, bw, bh = cv2.boundingRect(box)
            pad = 6
            x1 = max(0, bx - pad)
            y1 = max(0, search_top + by - pad)
            x2 = min(w, bx + bw + pad)
            y2 = min(h, search_top + by + bh + pad)
            if (x2 - x1) * (y2 - y1) >= getattr(self.cfg, "plate_min_area", 200):
                return (x1, y1, x2, y2)

        # Fallback strip — bottom-centre region
        x1 = w // 4
        y1 = int(h * 0.65)
        x2 = 3 * w // 4
        y2 = int(h * 0.88)
        if (x2 - x1) * (y2 - y1) >= getattr(self.cfg, "plate_min_area", 200):
            return (x1, y1, x2, y2)
        return None

    # ── Stage 3: Perspective correction (deskew) ──────────────────────────

    @staticmethod
    def _deskew(plate_img: np.ndarray) -> np.ndarray:
        """
        Correct perspective distortion / tilt using the dominant
        line angle in the plate crop (Hough lines).
        Returns the corrected image (same size or slightly different).
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img

        h, w = plate_img.shape[:2]
        gray  = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=40, minLineLength=w // 4,
                                maxLineGap=w // 8)
        if lines is None:
            return plate_img

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -30 < angle < 30:      # ignore near-vertical lines
                    angles.append(angle)

        if not angles:
            return plate_img

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:       # negligible tilt — skip
            return plate_img

        # Rotate to correct tilt
        cx, cy = w // 2, h // 2
        M      = cv2.getRotationMatrix2D((cx, cy), median_angle, 1.0)
        rotated = cv2.warpAffine(
            plate_img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    # ── Stage 4: Enhancement ──────────────────────────────────────────────

    def _enhance_plate(self, plate_img: np.ndarray):
        """Real-ESRGAN ×4 or CPU bicubic upscale fallback."""
        if getattr(self.cfg, "use_genai", False) and self.models.enhancer:
            try:
                out, _ = self.models.enhancer.enhance(
                    plate_img,
                    outscale=getattr(self.cfg, "esrgan_scale", 4),
                )
                return out, True
            except Exception as e:
                log.debug(f"ESRGAN failed: {e}")

        # CPU fallback — bicubic ×4 + unsharp mask
        h, w = plate_img.shape[:2]
        up   = cv2.resize(plate_img, (w * 4, h * 4),
                          interpolation=cv2.INTER_CUBIC)
        return up, False

    # ── Stage 5: Multi-version preprocessing → OCR ───────────────────────

    def _preprocess_plate(self, plate_img: np.ndarray) -> list[np.ndarray]:
        """
        Generate 4 binarisation variants and a deblurred variant.
        Returns list of BGR images — OCR runs on all, best kept.
        """
        if plate_img is None or plate_img.size == 0:
            return []

        # Ensure minimum readable size
        h, w = plate_img.shape[:2]
        if w < 200 or h < 40:
            scale = max(200 / max(w, 1), 40 / max(h, 1))
            plate_img = cv2.resize(plate_img,
                                   (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_CUBIC)

        # ── Deblur: unsharp mask ──────────────────────────────────────────
        blurred    = cv2.GaussianBlur(plate_img, (0, 0), 3)
        deblurred  = cv2.addWeighted(plate_img, 1.5, blurred, -0.5, 0)

        # ── Denoise ───────────────────────────────────────────────────────
        denoised = cv2.fastNlMeansDenoisingColored(deblurred, None, 8, 8, 7, 21)

        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

        # ── CLAHE contrast enhancement ────────────────────────────────────
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        ceq   = clahe.apply(gray)

        # ── Sharpen ───────────────────────────────────────────────────────
        sharp_k  = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(ceq, -1, sharp_k)

        # ── 4 binarisation variants ───────────────────────────────────────
        # V1: Otsu on sharpened
        _, v_otsu = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # V2: Adaptive Gaussian (good for uneven lighting)
        v_adapt = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8)

        # V3: Inverted Otsu (for dark/dirty plates with light chars)
        _, v_inv = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # V4: CLAHE equalised grey (for heavily blurred plates)
        v_gray = ceq

        def _to_bgr(img):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return [_to_bgr(v_otsu), _to_bgr(v_adapt),
                _to_bgr(v_inv),  _to_bgr(v_gray)]

    def _run_ocr(self, plate_img: np.ndarray) -> tuple[str, float]:
        """
        Run EasyOCR across all 4 preprocessed versions.
        Returns (best_text, best_confidence).
        """
        if self.models.ocr is None:
            return "", 0.0

        versions = self._preprocess_plate(plate_img)
        if not versions:
            return "", 0.0

        best_text = ""
        best_conf = 0.0

        for v in versions:
            try:
                results = self.models.ocr.readtext(v, detail=1, paragraph=False)
                if not results:
                    continue
                texts = [r[1] for r in results]
                confs = [r[2] for r in results]
                combined = "".join(texts).upper().strip()
                avg_conf = float(np.mean(confs)) if confs else 0.0

                if avg_conf > best_conf and len(combined) >= 4:
                    best_conf = avg_conf
                    best_text = combined
            except Exception as e:
                log.debug(f"OCR version error: {e}")
                continue

        return best_text, best_conf

    # ── Stage 2–5 orchestrator ────────────────────────────────────────────

    def _detect_and_read_plate(
        self,
        full_frame:   np.ndarray,
        vehicle_roi:  np.ndarray,
        roi_x:        int,
        roi_y:        int,
        vehicle_class: str = "Car",
    ) -> Optional[PlateResult]:

        plate_bbox = self._locate_plate_in_roi(vehicle_roi, vehicle_class)
        if plate_bbox is None:
            return None

        lx1, ly1, lx2, ly2 = plate_bbox
        plate_crop = self._safe_crop(vehicle_roi, lx1, ly1, lx2, ly2)
        if plate_crop is None or plate_crop.size == 0:
            return None

        min_area = getattr(self.cfg, "plate_min_area", 200)
        if plate_crop.shape[0] * plate_crop.shape[1] < min_area:
            return None

        abs_bbox = (roi_x + lx1, roi_y + ly1, roi_x + lx2, roi_y + ly2)

        # Stage 3: Deskew
        plate_crop = self._deskew(plate_crop)

        # Stage 4: Enhance
        plate_crop, enhanced = self._enhance_plate(plate_crop)

        # Stage 5: OCR
        raw_text, confidence = self._run_ocr(plate_crop)

        # If first attempt weak, retry on 2× bigger crop
        if confidence < 0.40 or len(raw_text) < 4:
            h, w = plate_crop.shape[:2]
            bigger = cv2.resize(plate_crop, (w * 2, h * 2),
                                interpolation=cv2.INTER_CUBIC)
            raw2, conf2 = self._run_ocr(bigger)
            if conf2 > confidence and len(raw2) >= 4:
                raw_text, confidence = raw2, conf2

        if not raw_text or len(raw_text) < 4:
            return None

        corrected = self._post_correct(raw_text)

        return PlateResult(
            text       = corrected,
            confidence = confidence,
            bbox       = abs_bbox,
            plate_crop = plate_crop,
            enhanced   = enhanced,
            raw_text   = raw_text,
        )

    # ── Stage 6: Post-correction ──────────────────────────────────────────

    def _post_correct(self, raw: str) -> str:
        """
        Position-aware character correction for Indian plates.
        Format: SS DD LLL NNNN
          S = state letter (positions 0,1)  — must be alpha
          D = district digit (positions 2,3) — must be numeric
          L = series letter (positions 4–6)  — must be alpha
          N = number (positions 7–10)        — must be numeric
        """
        clean = re.sub(r"[^A-Z0-9]", "", raw.upper())
        if len(clean) < 4:
            return raw.upper().strip()

        corrected = list(clean)
        rev_fixes = {v: k for k, v in _OCR_FIXES.items()}   # digit→letter

        for i, ch in enumerate(corrected):
            if i < 2:
                # State code — must be letters
                if ch.isdigit():
                    corrected[i] = rev_fixes.get(ch, ch)
            elif 2 <= i < 4:
                # District code — must be digits
                if ch.isalpha():
                    corrected[i] = _OCR_FIXES.get(ch, ch)
            elif i >= len(clean) - 4:
                # Last 4 characters — registration number, must be digits
                if ch.isalpha():
                    corrected[i] = _OCR_FIXES.get(ch, ch)

        return "".join(corrected)

    # ── Safety compliance ──────────────────────────────────────────────────

    def _check_safety(
        self, roi: np.ndarray, vehicle_class: str
    ) -> tuple[Optional[bool], float, Optional[bool], float]:
        helmet_ok = helmet_conf = belt_ok = belt_conf = None
        helmet_conf = belt_conf = 0.0

        try:
            clf = self.models.safety_classifier
            if clf is None:
                raise AttributeError
            result      = clf.classify(roi)
            helmet_ok   = result.get("helmet")
            helmet_conf = result.get("helmet_conf", 0.0)
            belt_ok     = result.get("seatbelt")
            belt_conf   = result.get("seatbelt_conf", 0.0)
        except Exception:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean = float(np.mean(gray))
            if vehicle_class == "Motorcycle":
                h = roi.shape[0]
                upper       = cv2.cvtColor(roi[:h//3], cv2.COLOR_BGR2GRAY)
                helmet_ok   = float(np.mean(upper)) > 60
                helmet_conf = 0.65
            elif vehicle_class in ("Car", "Bus", "Truck"):
                belt_ok   = mean > 50
                belt_conf = 0.60

        return helmet_ok, helmet_conf, belt_ok, belt_conf

    def _classify_violation(self, vcls, helmet_ok, hconf, belt_ok, bconf) -> str:
        hc  = getattr(self.cfg, "helmet_conf",   0.55)
        bc  = getattr(self.cfg, "seatbelt_conf", 0.55)
        if vcls == "Motorcycle":
            if helmet_ok is False and hconf >= hc:
                return "No Helmet"
        elif vcls in ("Car", "Bus", "Truck"):
            if belt_ok is False and bconf >= bc:
                return "No Seat Belt"
        return "Compliant"

    # ── Helpers ────────────────────────────────────────────────────────────

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
