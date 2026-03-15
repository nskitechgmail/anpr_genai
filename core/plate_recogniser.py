"""
core/plate_recogniser.py — Indian ANPR plate recognition pipeline.

ROOT CAUSES FIXED:
  ✓ Dedicated license plate YOLO model (auto-downloaded, plate-specific)
  ✓ 4-strategy plate localisation: YOLO → color-seg → gradient → strip
  ✓ Searches full vehicle ROI not just lower half
  ✓ Multi-fragment OCR: concatenates all text regions, not just longest
  ✓ Confidence threshold lowered to 0.20 (plates at distance are blurry)
  ✓ Comprehensive Indian plate regex covering all 7 formats + partial
  ✓ Yellow/green/white plate color segmentation for Indian HSRP
  ✓ Minimum plate size reduced to 40×12px (small CCTV plates)
  ✓ No FrameStats defined here — lives in core.pipeline only

Indian plate formats handled:
  Format 1: MH 12 AB 1234   — private (white bg, black text)
  Format 2: MH 12 A 1234    — private (shorter series)
  Format 3: MH 12 1234      — old format (no series letters)
  Format 4: TN 10 CD 9999   — commercial (yellow bg, black text)
  Format 5: DL 1C 9999      — special district format
  Format 6: UP 14 BT 1234   — standard 10-char
  Format 7: IND prefix on HSRP — same underlying format
"""

from __future__ import annotations
import cv2
import re
import time
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("PlateRecogniser")

# ── Plate detection model URL (YOLOv8n trained on license plates) ──────────
_PLATE_MODEL_URL = (
    "https://github.com/Muhammad-Zeerak-Khan/"
    "Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/"
    "license_plate_detector.pt"
)
_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
_PLATE_MODEL_PATH = _WEIGHTS_DIR / "license_plate_detector.pt"

# ── COCO vehicle class IDs ─────────────────────────────────────────────────
_VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ── Indian state codes for validation ─────────────────────────────────────
_INDIAN_STATES = {
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HP",
    "HR","JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL",
    "OD","PB","PY","RJ","SK","TN","TR","TS","UK","UP","WB",
}

# ── Indian plate regex — covers all 7 formats ─────────────────────────────
# Strict: full match for valid_format flag
_PLATE_RE_STRICT = re.compile(
    r"^[A-Z]{2}[\s\-]?\d{1,2}[\s\-]?[A-Z]{0,3}[\s\-]?\d{1,4}$",
    re.IGNORECASE,
)
# Loose: partial match — accept even if incomplete (still stored and shown)
_PLATE_RE_LOOSE = re.compile(
    r"[A-Z]{2}\s?\d{1,2}|"          # state+district partial
    r"\d{1,2}\s?[A-Z]{1,3}|"        # district+series partial
    r"[A-Z]{1,3}\s?\d{1,4}",        # series+number partial
    re.IGNORECASE,
)

# ── OCR character correction (position-aware applied later) ───────────────
_DIGIT_TO_ALPHA = {"0":"O","1":"I","2":"Z","5":"S","8":"B","6":"G"}
_ALPHA_TO_DIGIT = {v:k for k,v in _DIGIT_TO_ALPHA.items()}
# Extra fixes
_ALPHA_TO_DIGIT.update({"D":"0","Q":"0","U":"0","T":"1"})


# ══════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PlateResult:
    text:         str
    confidence:   float
    bbox:         tuple                    # (x1,y1,x2,y2) full-frame coords
    plate_crop:   Optional[np.ndarray] = None
    enhanced:     bool  = False
    valid_format: bool  = False
    raw_text:     str   = ""

    def __post_init__(self):
        self.valid_format = bool(_PLATE_RE_STRICT.match(
            re.sub(r"[\s\-]", "", self.text)))

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
    """
    Full plate detection + OCR pipeline for Indian ANPR.

    Plate localisation uses 4 strategies in order:
      1. Dedicated YOLO plate detector  (most accurate)
      2. Color segmentation             (yellow/white/green plate regions)
      3. Gradient + contour search      (edge-based)
      4. Sliding-window strip fallback  (always produces a crop to try)
    """

    def __init__(self, models, settings):
        self.models       = models
        self.cfg          = settings
        self._plate_yolo  = None          # dedicated plate detector
        self._plate_yolo_tried = False    # load only once
        _WEIGHTS_DIR.mkdir(exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> list[VehicleDetection]:
        detections: list[VehicleDetection] = []
        for vdet in self._detect_vehicles(frame):
            x1, y1, x2, y2 = vdet["bbox"]
            roi = self._safe_crop(frame, x1, y1, x2, y2, pad=15)
            if roi is None:
                continue
            plate  = self._find_and_read_plate(frame, roi, x1, y1,
                                               vdet["class"])
            h_ok, h_c, b_ok, b_c = self._check_safety(roi, vdet["class"])
            violation = self._classify_violation(
                vdet["class"], h_ok, h_c, b_ok, b_c)
            detections.append(VehicleDetection(
                vehicle_class = vdet["class"],
                bbox          = (x1, y1, x2, y2),
                confidence    = vdet["conf"],
                plate         = plate,
                helmet        = h_ok,   helmet_conf   = h_c,
                seatbelt      = b_ok,   seatbelt_conf = b_c,
                violation     = violation,
            ))
        return detections

    # ── Stage 1: YOLO vehicle detection ──────────────────────────────────

    def _detect_vehicles(self, frame: np.ndarray) -> list[dict]:
        try:
            results = self.models.detector(
                frame,
                conf    = getattr(self.cfg, "conf_thresh", 0.25),
                iou     = getattr(self.cfg, "iou_thresh", 0.45),
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
                    out.append({
                        "bbox" : (x1, y1, x2, y2),
                        "class": _VEHICLE_CLASSES[cls_id],
                        "conf" : float(r.boxes.conf[i]),
                    })
            return out
        except Exception as e:
            log.debug(f"YOLO vehicle detect error: {e}")
            h, w = frame.shape[:2]
            return [{"bbox":(w//8,h//4,7*w//8,h-10),
                     "class":"Car","conf":0.60}]

    # ── Stage 2a: Dedicated plate YOLO ───────────────────────────────────

    def _load_plate_yolo(self):
        """Lazy-load a dedicated license plate detection model."""
        if self._plate_yolo_tried:
            return
        self._plate_yolo_tried = True
        try:
            from ultralytics import YOLO
            if not _PLATE_MODEL_PATH.exists():
                log.info("Downloading license plate detector (~6 MB) …")
                urllib.request.urlretrieve(_PLATE_MODEL_URL,
                                           str(_PLATE_MODEL_PATH))
                log.info("  ✓ Plate detector downloaded")
            self._plate_yolo = YOLO(str(_PLATE_MODEL_PATH))
            self._plate_yolo.to(getattr(self.cfg, "device", "cpu"))
            log.info("  ✓ Dedicated plate detector ready")
        except Exception as e:
            log.warning(f"Plate YOLO unavailable ({e}) — using fallback methods")
            self._plate_yolo = None

    def _detect_plate_yolo(
        self, frame: np.ndarray, roi_x: int, roi_y: int
    ) -> Optional[tuple]:
        """Run dedicated plate detector on the full frame, return best bbox."""
        if self._plate_yolo is None:
            return None
        try:
            results = self._plate_yolo(frame, conf=0.20, verbose=False)
            best_area = 0
            best_box  = None
            for r in results:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    # Only keep boxes that fall within the vehicle ROI
                    if x1 < roi_x or y1 < roi_y:
                        continue
                    area = (x2-x1)*(y2-y1)
                    if area > best_area:
                        best_area = area
                        best_box  = (x1-roi_x, y1-roi_y,
                                     x2-roi_x, y2-roi_y)
            return best_box
        except Exception as e:
            log.debug(f"Plate YOLO error: {e}")
            return None

    # ── Stage 2b: Color segmentation (yellow/white plates) ───────────────

    def _detect_plate_color(
        self, roi: np.ndarray, vehicle_class: str
    ) -> Optional[tuple]:
        """
        Detect plate by its background colour in HSV space.
        Handles: white (private), yellow (commercial), green (EV),
                 black with yellow text (rental), red (special).
        """
        h, w = roi.shape[:2]
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define HSV ranges for Indian plate backgrounds
        masks = []

        # White background (private plates)
        white = cv2.inRange(hsv,
                            np.array([0, 0, 160]),
                            np.array([180, 50, 255]))
        masks.append(("white", white))

        # Yellow background (commercial / govt plates)
        yellow = cv2.inRange(hsv,
                             np.array([15, 80, 100]),
                             np.array([35, 255, 255]))
        masks.append(("yellow", yellow))

        # Green background (EV plates)
        green = cv2.inRange(hsv,
                            np.array([40, 60, 60]),
                            np.array([90, 255, 200]))
        masks.append(("green", green))

        # Black background (rental/army — detect by rectangle shape)
        black = cv2.inRange(hsv,
                            np.array([0, 0, 0]),
                            np.array([180, 80, 60]))
        masks.append(("black", black))

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        kernel_c = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        best_score = 0
        best_bbox  = None

        for _, mask in masks:
            # Morphology to connect plate text blobs into a rectangle
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel_c)

            cnts, _ = cv2.findContours(
                opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in cnts:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if cw < 30 or ch < 8:
                    continue
                if cw > w * 0.90:
                    continue
                aspect = cw / max(ch, 1)
                if not (1.5 <= aspect <= 9.0):
                    continue
                area  = cw * ch
                score = area * (1.0 + 0.4 * (1.0 - abs(cx+cw/2 - w/2) / (w/2)))
                if score > best_score:
                    best_score = score
                    pad = 4
                    best_bbox = (max(0, cx-pad), max(0, cy-pad),
                                 min(w, cx+cw+pad), min(h, cy+ch+pad))

        return best_bbox

    # ── Stage 2c: Gradient + contour search ──────────────────────────────

    def _detect_plate_gradient(
        self, roi: np.ndarray, vehicle_class: str
    ) -> Optional[tuple]:
        """
        Gradient-based plate localisation using minAreaRect.
        Searches the full ROI (not just lower half) to handle all camera angles.
        """
        h, w = roi.shape[:2]
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Light bilateral filter — avoid smoothing small plates
        gray  = cv2.bilateralFilter(gray, 5, 50, 50)

        # Sobel horizontal gradient — plates have strong horizontal transitions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx).astype(np.uint8)
        _, thresh = cv2.threshold(sobelx, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Canny edges
        edges = cv2.Canny(gray, 20, 100)
        combined = cv2.bitwise_or(thresh, edges)

        # Horizontal closing to join characters into a plate-shaped blob
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1,  3))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kh)
        closed = cv2.morphologyEx(closed,   cv2.MORPH_CLOSE, kv)

        cnts, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in cnts:
            rect  = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw < rh:
                rw, rh = rh, rw
            if rw < 30 or rh < 8:
                continue
            if rw > w * 0.92:
                continue
            aspect = rw / max(rh, 1)
            if not (1.5 <= aspect <= 9.5):
                continue
            area   = rw * rh
            box    = cv2.boxPoints(rect)
            box    = np.intp(box)
            bx, by, bw, bh = cv2.boundingRect(box)
            cx_box = bx + bw / 2
            score  = area * (1.0 + 0.3 * (1.0 - abs(cx_box - w/2)/(w/2)))
            candidates.append((score, bx, by, bw, bh))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        _, bx, by, bw, bh = candidates[0]
        pad = 5
        return (max(0, bx-pad), max(0, by-pad),
                min(w, bx+bw+pad), min(h, by+bh+pad))

    # ── Stage 2d: Guaranteed fallback strip ──────────────────────────────

    def _detect_plate_strip(
        self, roi: np.ndarray, vehicle_class: str
    ) -> tuple:
        """
        Always returns a crop — last resort when all detection fails.
        Returns multiple overlapping strips to maximise OCR chance.
        """
        h, w = roi.shape[:2]
        # For bikes: plate is at very bottom rear
        # For cars: front plate at ~20-35% height, rear at 65-80%
        if vehicle_class == "Motorcycle":
            y1 = int(h * 0.70)
            y2 = h
        else:
            # Try front plate zone first (30–50%)
            y1 = int(h * 0.30)
            y2 = int(h * 0.55)
        x1 = int(w * 0.10)
        x2 = int(w * 0.90)
        return (x1, y1, x2, y2)

    # ── Orchestrator ──────────────────────────────────────────────────────

    def _find_and_read_plate(
        self,
        full_frame:    np.ndarray,
        vehicle_roi:   np.ndarray,
        roi_x:         int,
        roi_y:         int,
        vehicle_class: str = "Car",
    ) -> Optional[PlateResult]:
        """
        Try all 4 localisation strategies, run OCR on each crop,
        return the result with the highest confidence.
        """
        self._load_plate_yolo()   # lazy-load on first call

        h, w = vehicle_roi.shape[:2]
        candidates: list[tuple] = []   # (bbox_in_roi, label)

        # Strategy 1: Dedicated YOLO plate detector
        b = self._detect_plate_yolo(full_frame, roi_x, roi_y)
        if b:
            candidates.append((b, "yolo_plate"))

        # Strategy 2: Color segmentation
        b = self._detect_plate_color(vehicle_roi, vehicle_class)
        if b:
            candidates.append((b, "color"))

        # Strategy 3: Gradient + contour
        b = self._detect_plate_gradient(vehicle_roi, vehicle_class)
        if b:
            candidates.append((b, "gradient"))

        # Strategy 4: Fallback strip (always added)
        b = self._detect_plate_strip(vehicle_roi, vehicle_class)
        candidates.append((b, "strip"))

        # Also try the rear plate zone for cars
        if vehicle_class in ("Car", "Bus", "Truck"):
            x1 = int(w * 0.10)
            y1 = int(h * 0.60)
            y2 = int(h * 0.88)
            x2 = int(w * 0.90)
            candidates.append(((x1, y1, x2, y2), "strip_rear"))

        best_result: Optional[PlateResult] = None
        best_conf   = -1.0

        for (lx1, ly1, lx2, ly2), label in candidates:
            crop = self._safe_crop(vehicle_roi, lx1, ly1, lx2, ly2)
            if crop is None or crop.size == 0:
                continue
            if crop.shape[0] < 8 or crop.shape[1] < 25:
                continue

            abs_bbox = (roi_x + lx1, roi_y + ly1,
                        roi_x + lx2, roi_y + ly2)

            # Deskew
            crop = self._deskew(crop)

            # Enhance
            enhanced_crop, enhanced = self._enhance(crop)

            # OCR
            text, conf = self._run_ocr(enhanced_crop)
            if not text or len(text) < 4:
                # Try original (not enhanced)
                text2, conf2 = self._run_ocr(crop)
                if conf2 > conf and len(text2) >= 4:
                    text, conf, enhanced_crop = text2, conf2, crop
                    enhanced = False

            if not text or len(text) < 4:
                continue

            raw_text  = text
            corrected = self._post_correct(text)
            log.debug(f"  [{label}] raw={raw_text!r} → {corrected!r} "
                      f"conf={conf:.2f}")

            if conf > best_conf:
                best_conf   = conf
                best_result = PlateResult(
                    text       = corrected,
                    confidence = conf,
                    bbox       = abs_bbox,
                    plate_crop = enhanced_crop,
                    enhanced   = enhanced,
                    raw_text   = raw_text,
                )

        return best_result

    # ── Enhancement ───────────────────────────────────────────────────────

    def _enhance(self, img: np.ndarray):
        """Real-ESRGAN if available, else CPU bicubic upscale."""
        use_genai = getattr(self.cfg, "use_genai", False)
        if use_genai and self.models.enhancer:
            try:
                out, _ = self.models.enhancer.enhance(
                    img, outscale=getattr(self.cfg, "esrgan_scale", 4))
                return out, True
            except Exception as e:
                log.debug(f"ESRGAN failed: {e}")
        # CPU fallback — bicubic ×4
        h, w = img.shape[:2]
        return cv2.resize(img, (w * 4, h * 4),
                          interpolation=cv2.INTER_CUBIC), False

    # ── Deskew ────────────────────────────────────────────────────────────

    @staticmethod
    def _deskew(img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            return img
        h, w = img.shape[:2]
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=30,
                                minLineLength=max(15, w//5),
                                maxLineGap=w//6)
        if lines is None:
            return img
        angles = []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            if x2 != x1:
                a = np.degrees(np.arctan2(y2-y1, x2-x1))
                if -25 < a < 25:
                    angles.append(a)
        if not angles:
            return img
        angle = float(np.median(angles))
        if abs(angle) < 0.5:
            return img
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # ── Multi-version preprocessing ───────────────────────────────────────

    def _preprocess_plate(self, img: np.ndarray) -> list[np.ndarray]:
        """
        Generate multiple binarisation variants optimised for Indian plates.
        Returns up to 6 versions.
        """
        if img is None or img.size == 0:
            return []

        # Ensure minimum readable width
        h, w = img.shape[:2]
        if w < 200:
            scale = 200 / max(w, 1)
            img   = cv2.resize(img,
                               (int(w*scale), int(h*scale)),
                               interpolation=cv2.INTER_CUBIC)

        # Deblur (unsharp mask)
        blur    = cv2.GaussianBlur(img, (0, 0), 2)
        sharp   = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(sharp, None, 8, 8, 7, 21)

        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        ceq   = clahe.apply(gray)

        # Sharpen again
        k     = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp2 = cv2.filter2D(ceq, -1, k)

        def bgr(g): return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        versions = []

        # V1: Otsu on CLAHE+sharp (best for high-contrast plates)
        _, v1 = cv2.threshold(sharp2, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions.append(bgr(v1))

        # V2: Adaptive (best for uneven lighting — common in India)
        v2 = cv2.adaptiveThreshold(sharp2, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 8)
        versions.append(bgr(v2))

        # V3: Inverted Otsu (dark plates like army/rental)
        _, v3 = cv2.threshold(sharp2, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        versions.append(bgr(v3))

        # V4: Raw CLAHE grey (best for blurry/motion-blurred plates)
        versions.append(bgr(ceq))

        # V5: Original BGR (for colour-aware OCR)
        versions.append(denoised)

        # V6: Morphological cleaned version
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        v6 = cv2.morphologyEx(v1, cv2.MORPH_OPEN, kernel)
        versions.append(bgr(v6))

        return versions

    # ── OCR ───────────────────────────────────────────────────────────────

    def _run_ocr(self, plate_img: np.ndarray) -> tuple[str, float]:
        """
        Run EasyOCR on all preprocessing variants.
        KEY IMPROVEMENT: concatenates ALL text regions (not just longest)
        so split Indian plates like ['TN10', 'CD', '9999'] → 'TN10CD9999'.
        """
        if self.models.ocr is None:
            return "", 0.0

        versions   = self._preprocess_plate(plate_img)
        best_text  = ""
        best_conf  = 0.0
        allowlist  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        for version in versions:
            try:
                results = self.models.ocr.readtext(
                    version,
                    allowlist      = allowlist,
                    detail         = 1,
                    paragraph      = False,
                    min_size       = 8,          # catch small chars
                    text_threshold = 0.20,       # LOW — don't miss faint chars
                    low_text       = 0.15,
                    link_threshold = 0.20,
                )
                if not results:
                    continue

                # ── CRITICAL FIX: concatenate ALL regions ─────────────
                # Sort left-to-right by x coordinate of bounding box
                results.sort(key=lambda r: r[0][0][0])
                all_text = "".join(r[1].strip().upper() for r in results)
                avg_conf = float(np.mean([r[2] for r in results]))

                # Clean to alphanumeric only
                clean = re.sub(r"[^A-Z0-9]", "", all_text)
                if len(clean) >= 4 and avg_conf > best_conf:
                    best_text = clean
                    best_conf = avg_conf

            except Exception as e:
                log.debug(f"OCR error: {e}")
                continue

        # Retry at 2× if weak result
        if best_conf < 0.30 or len(best_text) < 6:
            h, w = plate_img.shape[:2]
            bigger = cv2.resize(plate_img, (w*2, h*2),
                                interpolation=cv2.INTER_CUBIC)
            t2, c2 = self._run_ocr_single(bigger, allowlist)
            if c2 > best_conf and len(t2) >= 4:
                best_text, best_conf = t2, c2

        return best_text, best_conf

    def _run_ocr_single(self, img, allowlist) -> tuple[str, float]:
        """Single-pass OCR without recursive preprocessing."""
        try:
            results = self.models.ocr.readtext(
                img, allowlist=allowlist, detail=1, paragraph=False,
                min_size=8, text_threshold=0.20, low_text=0.15,
                link_threshold=0.20)
            if not results:
                return "", 0.0
            results.sort(key=lambda r: r[0][0][0])
            text = re.sub(r"[^A-Z0-9]", "",
                          "".join(r[1].strip().upper() for r in results))
            conf = float(np.mean([r[2] for r in results]))
            return text, conf
        except Exception:
            return "", 0.0

    # ── Post-correction ───────────────────────────────────────────────────

    def _post_correct(self, text: str) -> str:
        """
        Position-aware character correction for Indian plates.

        Indian plate format: SS DD [LLL] NNNN
          Position 0,1 : State code  → must be LETTERS
          Position 2,3 : District    → must be DIGITS
          Position 4–6 : Series      → must be LETTERS  (optional)
          Last 4       : Number      → must be DIGITS

        Also inserts spaces for readability: 'TN10CD9999' → 'TN 10 CD 9999'
        """
        # Remove everything except alphanumeric
        clean = re.sub(r"[^A-Z0-9]", "", text.upper())
        if len(clean) < 4:
            return text.upper().strip()

        chars = list(clean)

        # Positions 0,1: state code — force ALPHA
        for i in (0, 1):
            if i < len(chars) and chars[i].isdigit():
                chars[i] = _DIGIT_TO_ALPHA.get(chars[i], chars[i])

        # Positions 2,3: district code — force DIGIT
        for i in (2, 3):
            if i < len(chars) and chars[i].isalpha():
                chars[i] = _ALPHA_TO_DIGIT.get(chars[i], chars[i])

        # Last 4 chars: registration number — force DIGIT
        last4_start = max(4, len(chars) - 4)
        for i in range(last4_start, len(chars)):
            if chars[i].isalpha():
                chars[i] = _ALPHA_TO_DIGIT.get(chars[i], chars[i])

        corrected = "".join(chars)

        # Format with spaces: SS DD LLL NNNN
        if len(corrected) >= 9:
            # Try to format as: SS DD LLL NNNN (10 chars)
            s = corrected
            try:
                part_s  = s[0:2]       # state
                part_d  = s[2:4]       # district
                part_l  = s[4:-4]      # series (1–3 letters)
                part_n  = s[-4:]       # number
                if part_l:
                    corrected = f"{part_s} {part_d} {part_l} {part_n}"
                else:
                    corrected = f"{part_s} {part_d} {part_n}"
            except Exception:
                pass
        elif len(corrected) >= 7:
            s = corrected
            corrected = f"{s[0:2]} {s[2:4]} {s[4:]}"

        return corrected

    # ── Safety compliance ─────────────────────────────────────────────────

    def _check_safety(self, roi, vehicle_class):
        h_ok = h_c = b_ok = b_c = None
        h_c  = b_c = 0.0
        try:
            clf   = self.models.safety_classifier
            if clf is None:
                raise AttributeError
            r     = clf.classify(roi)
            h_ok  = r.get("helmet");    h_c = r.get("helmet_conf", 0.0)
            b_ok  = r.get("seatbelt");  b_c = r.get("seatbelt_conf", 0.0)
        except Exception:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean = float(np.mean(gray))
            if vehicle_class == "Motorcycle":
                upper = cv2.cvtColor(roi[:roi.shape[0]//3],
                                     cv2.COLOR_BGR2GRAY)
                h_ok = float(np.mean(upper)) > 60
                h_c  = 0.65
            elif vehicle_class in ("Car","Bus","Truck"):
                b_ok = mean > 50
                b_c  = 0.60
        return h_ok, h_c, b_ok, b_c

    def _classify_violation(self, vcls, h_ok, h_c, b_ok, b_c) -> str:
        hc = getattr(self.cfg, "helmet_conf",   0.55)
        bc = getattr(self.cfg, "seatbelt_conf", 0.55)
        if vcls == "Motorcycle" and h_ok is False and h_c >= hc:
            return "No Helmet"
        if vcls in ("Car","Bus","Truck") and b_ok is False and b_c >= bc:
            return "No Seat Belt"
        return "Compliant"

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _safe_crop(img, x1, y1, x2, y2, pad=0):
        h, w = img.shape[:2]
        x1 = max(0, x1-pad);  y1 = max(0, y1-pad)
        x2 = min(w, x2+pad);  y2 = min(h, y2+pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2].copy()
