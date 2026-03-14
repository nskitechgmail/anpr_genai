"""
utils/anonymiser.py — Face anonymisation for DPDP Act 2023 compliance.

Robustness improvements for tilted / side-profile / angled faces:

  ✓ MediaPipe model_selection=1  — full-range model, handles tilts up to ±45°
  ✓ Lower confidence threshold   — 0.35 (was 0.5) catches partially visible faces
  ✓ Multi-angle sweep            — frame rotated ±15° and ±30°, detections merged
  ✓ Profile Haar cascade         — haarcascade_profileface catches pure side views
  ✓ Larger blur padding          — 25% pad (was 15%) covers hair/chin missed by bbox
  ✓ Elliptical blur mask         — follows face shape, less rectangular artefact
  ✓ NMS deduplication            — overlapping boxes from multi-angle merged cleanly
"""

from __future__ import annotations
import cv2
import logging
import math
import numpy as np
from typing import List, Tuple

log = logging.getLogger("FaceAnonymiser")

# Padding around detected face bbox (fraction of face size)
_PAD_FACTOR  = 0.25   # was 0.15 — larger covers tilted faces better
_BLUR_KERNEL = 51     # must be odd; size of Gaussian kernel


class FaceAnonymiser:
    """
    Detects and blurs all faces in a video frame.

    Detection strategy (in order, results merged):
      1. MediaPipe full-range model  — handles frontal + tilted + partial
      2. Multi-angle sweep (±15°, ±30°) — catches extreme tilts
      3. Profile Haar cascade         — side-view faces
      4. Frontal Haar cascade         — frontal fallback

    All detections are NMS-deduplicated before blurring so no face is
    blurred twice (which would over-darken the region).
    """

    # Rotation angles to sweep (degrees). 0° is always included implicitly.
    _SWEEP_ANGLES = [-30, -15, 15, 30]

    def __init__(self):
        self._mp_face    = None
        self._haar_front = None
        self._haar_prof  = None
        self._initialized = False

    # ── Public API ────────────────────────────────────────────────────────

    def blur_faces(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect and blur all faces in *frame* (BGR).
        Returns a new frame with blurred regions; original is NOT modified.
        """
        if frame is None or frame.size == 0:
            return frame

        self._init_models()
        out = frame.copy()
        h, w = out.shape[:2]

        # Collect all face bboxes from all detectors + all angles
        all_boxes = self._detect_all_angles(frame)

        # Deduplicate with NMS (IoU threshold 0.3)
        if len(all_boxes) > 1:
            all_boxes = self._nms(all_boxes, iou_thresh=0.30)

        for (x1, y1, x2, y2) in all_boxes:
            fw = x2 - x1
            fh = y2 - y1
            # Padded region
            px = int(fw * _PAD_FACTOR)
            py = int(fh * _PAD_FACTOR)
            rx1 = max(0, x1 - px)
            ry1 = max(0, y1 - py)
            rx2 = min(w, x2 + px)
            ry2 = min(h, y2 + py)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            self._apply_blur(out, rx1, ry1, rx2, ry2)

        return out

    # ── Multi-angle detection ─────────────────────────────────────────────

    def _detect_all_angles(self, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        """Run detection at 0° and sweep angles, unify all results."""
        h, w = frame.shape[:2]
        all_boxes: List[Tuple[int,int,int,int]] = []

        # Always run at 0° first
        all_boxes.extend(self._detect_faces(frame))

        # Sweep rotations for tilted faces
        for angle in self._SWEEP_ANGLES:
            # Rotate frame
            M       = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(frame, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            boxes_rot = self._detect_faces(rotated)
            if not boxes_rot:
                continue

            # Un-rotate bboxes back to original frame coords
            M_inv = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
            for (bx1, by1, bx2, by2) in boxes_rot:
                corners = np.array([
                    [bx1, by1, 1], [bx2, by1, 1],
                    [bx2, by2, 1], [bx1, by2, 1],
                ], dtype=np.float32)
                transformed = (M_inv @ corners.T).T
                txs = transformed[:, 0]
                tys = transformed[:, 1]
                ox1 = max(0, int(txs.min()))
                oy1 = max(0, int(tys.min()))
                ox2 = min(w, int(txs.max()))
                oy2 = min(h, int(tys.max()))
                if ox2 > ox1 and oy2 > oy1:
                    all_boxes.append((ox1, oy1, ox2, oy2))

        return all_boxes

    # ── Single-angle detection ────────────────────────────────────────────

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        """Run all available detectors on *frame*, return merged bbox list."""
        h, w = frame.shape[:2]
        boxes: List[Tuple[int,int,int,int]] = []

        # ── MediaPipe (primary) ───────────────────────────────────────────
        if self._mp_face is not None:
            try:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._mp_face.process(rgb)
                if results.detections:
                    for det in results.detections:
                        # Only keep detections above threshold
                        score = det.score[0] if det.score else 0
                        if score < 0.35:
                            continue
                        bb  = det.location_data.relative_bounding_box
                        x1  = max(0, int(bb.xmin * w))
                        y1  = max(0, int(bb.ymin * h))
                        x2  = min(w, int((bb.xmin + bb.width) * w))
                        y2  = min(h, int((bb.ymin + bb.height) * h))
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2, y2))
            except Exception as e:
                log.debug(f"MediaPipe detect error: {e}")

        # ── Profile Haar (side faces) ─────────────────────────────────────
        if self._haar_prof is not None:
            try:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray  = cv2.equalizeHist(gray)
                faces = self._haar_prof.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4,
                    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                for (fx, fy, fw, fh) in faces:
                    boxes.append((fx, fy, fx + fw, fy + fh))
                    # Mirror — profile cascade is left-facing only, also check right
                    flipped = cv2.flip(frame, 1)
                    gray_f  = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
                    gray_f  = cv2.equalizeHist(gray_f)
                    faces_r = self._haar_prof.detectMultiScale(
                        gray_f, scaleFactor=1.1, minNeighbors=4,
                        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                    for (rx, ry, rw, rh) in faces_r:
                        # Un-flip x coordinate
                        ux = w - rx - rw
                        boxes.append((ux, ry, ux + rw, ry + rh))
            except Exception as e:
                log.debug(f"Profile Haar error: {e}")

        # ── Frontal Haar fallback ─────────────────────────────────────────
        if self._haar_front is not None and not boxes:
            # Only run frontal Haar if nothing else found (performance)
            try:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray  = cv2.equalizeHist(gray)
                faces = self._haar_front.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                for (fx, fy, fw, fh) in faces:
                    boxes.append((fx, fy, fx + fw, fy + fh))
            except Exception as e:
                log.debug(f"Frontal Haar error: {e}")

        return boxes

    # ── Blur application ──────────────────────────────────────────────────

    @staticmethod
    def _apply_blur(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ):
        """
        Apply strong Gaussian blur to region [y1:y2, x1:x2].
        Uses an elliptical mask so blur follows face shape naturally.
        """
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        k = _BLUR_KERNEL | 1   # ensure odd
        blurred = cv2.GaussianBlur(roi, (k, k), 0)

        # Elliptical mask — softer edges, less rectangular artefact
        rh, rw = roi.shape[:2]
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cx, cy = rw // 2, rh // 2
        cv2.ellipse(mask, (cx, cy), (rw // 2, rh // 2),
                    0, 0, 360, 255, -1)
        mask_3 = cv2.merge([mask, mask, mask])

        # Blend: blurred where mask=255, original elsewhere
        frame[y1:y2, x1:x2] = np.where(mask_3 == 255, blurred, roi)

    # ── NMS ───────────────────────────────────────────────────────────────

    @staticmethod
    def _nms(
        boxes: List[Tuple[int,int,int,int]],
        iou_thresh: float = 0.30,
    ) -> List[Tuple[int,int,int,int]]:
        """
        Non-maximum suppression — merge overlapping boxes from
        different detectors / angles. Keeps the largest box.
        """
        if not boxes:
            return []
        boxes_arr = np.array(boxes, dtype=np.float32)
        x1 = boxes_arr[:, 0]; y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]; y2 = boxes_arr[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]   # largest first

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            # IoU with remaining
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            iou   = inter / np.maximum(areas[order[1:]] + areas[i] - inter, 1)
            order = order[1:][iou < iou_thresh]

        return [boxes[k] for k in keep]

    # ── Model initialisation ──────────────────────────────────────────────

    def _init_models(self):
        if self._initialized:
            return
        self._initialized = True

        # ── MediaPipe ─────────────────────────────────────────────────────
        try:
            import mediapipe as mp
            # model_selection=1 → full-range model, handles tilts up to ±45°
            # model_selection=0 → short-range frontal only (was the old value)
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.35,   # was 0.5 — more sensitive
            )
            log.info("FaceAnonymiser: MediaPipe full-range model loaded (model_selection=1)")
        except Exception as e:
            log.warning(f"MediaPipe unavailable ({e}) — using Haar only")

        # ── Profile Haar cascade (side faces) ─────────────────────────────
        try:
            profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
            clf = cv2.CascadeClassifier(profile_path)
            if not clf.empty():
                self._haar_prof = clf
                log.info("FaceAnonymiser: Profile Haar cascade loaded")
            else:
                log.warning("Profile Haar cascade not found in OpenCV data")
        except Exception as e:
            log.debug(f"Profile Haar load error: {e}")

        # ── Frontal Haar cascade (fallback) ───────────────────────────────
        try:
            front_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            clf = cv2.CascadeClassifier(front_path)
            if not clf.empty():
                self._haar_front = clf
                log.info("FaceAnonymiser: Frontal Haar cascade loaded")
        except Exception as e:
            log.debug(f"Frontal Haar load error: {e}")

        if self._mp_face is None and self._haar_front is None:
            log.error("FaceAnonymiser: NO face detector available — "
                      "faces will not be blurred!")
