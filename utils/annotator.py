"""
utils/annotator.py — Draws detection overlays, violation badges, and HUD.

Imports FrameStats from core.pipeline (the single canonical definition).
"""
from __future__ import annotations
import cv2
import time
import numpy as np

# FrameStats lives in core.pipeline — NOT in core.plate_recogniser
from core.pipeline        import FrameStats
from core.plate_recogniser import VehicleDetection

COLOURS = {
    "Compliant"    : (50,  205,  50),
    "No Helmet"    : (0,     0, 220),
    "No Seat Belt" : (0,   140, 255),
    "Unknown"      : (180, 180, 180),
    "plate_box"    : (255, 200,   0),
    "hud_bg"       : (20,   20,  30),
    "hud_text"     : (200, 230, 255),
    "hud_accent"   : (0,   200, 255),
    "hud_warning"  : (0,    80, 255),
}
FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


class FrameAnnotator:
    def __init__(self, settings):
        self.cfg = settings

    # ── Main ──────────────────────────────────────────────────────

    def draw(self,
             frame: np.ndarray,
             detections: list,
             stats: FrameStats) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            self._draw_vehicle(out, det)
        self._draw_hud(out, detections, stats)
        return out

    # ── Vehicle box ───────────────────────────────────────────────

    def _draw_vehicle(self, frame: np.ndarray, det: VehicleDetection):
        x1, y1, x2, y2 = det.bbox
        col = COLOURS.get(det.violation, COLOURS["Unknown"])

        # Corner-style bounding box
        self._draw_corners(frame, x1, y1, x2, y2, col)

        # Class label bar
        self._label_bar(frame, x1, y1,
                        f"{det.vehicle_class}  {det.confidence:.0%}", col)

        # Violation / compliant badge
        if det.violation != "Compliant":
            self._viol_badge(frame, x1, y2, det.violation, col)
        else:
            self._small_badge(frame, x1, y2, "✓ Compliant", COLOURS["Compliant"])

        # Safety icons (H / B)
        self._draw_safety_icons(frame, x2, y1, det)

        # Plate overlay
        if det.plate:
            self._draw_plate_overlay(frame, det)

    def _draw_corners(self, frame, x1, y1, x2, y2, col, t=2, c=20):
        segs = [
            [(x1,y1),(x1+c,y1)], [(x1,y1),(x1,y1+c)],
            [(x2,y1),(x2-c,y1)], [(x2,y1),(x2,y1+c)],
            [(x1,y2),(x1+c,y2)], [(x1,y2),(x1,y2-c)],
            [(x2,y2),(x2-c,y2)], [(x2,y2),(x2,y2-c)],
        ]
        for p1, p2 in segs:
            cv2.line(frame, p1, p2, col, t+1)
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 1)

    def _label_bar(self, frame, x, y, text, col):
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
        by = max(th + 6, y - 6)
        cv2.rectangle(frame, (x, by-th-4), (x+tw+8, by+4), col, -1)
        cv2.putText(frame, text, (x+4, by), FONT, 0.55,
                    (255,255,255), 1, cv2.LINE_AA)

    def _viol_badge(self, frame, x, y, text, col):
        badge = f"! {text}"
        (tw, th), _ = cv2.getTextSize(badge, FONT, 0.60, 2)
        alpha = 0.80 + 0.20 * abs(np.sin(time.time() * 4))
        ov = frame.copy()
        cv2.rectangle(ov, (x, y+4), (x+tw+12, y+th+16), col, -1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
        cv2.putText(frame, badge, (x+6, y+th+8), FONT, 0.60,
                    (255,255,255), 2, cv2.LINE_AA)

    def _small_badge(self, frame, x, y, text, col):
        (tw, th), _ = cv2.getTextSize(text, FONT_SMALL, 0.50, 1)
        cv2.rectangle(frame, (x, y+2), (x+tw+8, y+th+10), col, -1)
        cv2.putText(frame, text, (x+4, y+th+4), FONT_SMALL, 0.50,
                    (0,0,0), 1, cv2.LINE_AA)

    def _draw_safety_icons(self, frame, x2, y1, det: VehicleDetection):
        ix, iy = x2+4, y1
        if det.helmet is not None:
            col  = COLOURS["Compliant"] if det.helmet else COLOURS["No Helmet"]
            icon = "H:OK" if det.helmet else "H:NO"
            cv2.rectangle(frame, (ix,iy), (ix+46,iy+20), col, -1)
            cv2.putText(frame, icon, (ix+2,iy+14), FONT_SMALL, 0.9,
                        (255,255,255), 1, cv2.LINE_AA)
            iy += 24
        if det.seatbelt is not None:
            col  = COLOURS["Compliant"] if det.seatbelt else COLOURS["No Seat Belt"]
            icon = "B:OK" if det.seatbelt else "B:NO"
            cv2.rectangle(frame, (ix,iy), (ix+46,iy+20), col, -1)
            cv2.putText(frame, icon, (ix+2,iy+14), FONT_SMALL, 0.9,
                        (255,255,255), 1, cv2.LINE_AA)

    def _draw_plate_overlay(self, frame: np.ndarray, det: VehicleDetection):
        plate = det.plate
        px1, py1, px2, py2 = plate.bbox
        cv2.rectangle(frame, (px1,py1), (px2,py2), COLOURS["plate_box"], 2)

        label = f"{plate.normalised()}  ({int(plate.confidence*100)}%)"
        if plate.enhanced:
            label += " [AI]"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 2)
        tx = max(0, px1)
        ty = py2 + th + 10
        if ty > frame.shape[0] - 5:
            ty = py1 - 6
        cv2.rectangle(frame, (tx-2, ty-th-4), (tx+tw+4, ty+4), (0,30,80), -1)
        cv2.putText(frame, label, (tx, ty), FONT, 0.65,
                    (0,255,255), 2, cv2.LINE_AA)

        # Plate crop inset
        if getattr(self.cfg, "show_plate_crop", True) and \
                plate.plate_crop is not None:
            try:
                crop = plate.plate_crop
                if len(crop.shape) == 2:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                h, w = frame.shape[:2]
                th2, tw2 = 40, 120
                ex = max(0, det.bbox[2] - tw2 - 4)
                ey = det.bbox[1] + 24
                if ey+th2 < h and ex+tw2 < w:
                    frame[ey:ey+th2, ex:ex+tw2] = cv2.resize(crop, (tw2, th2))
                    cv2.rectangle(frame, (ex,ey), (ex+tw2,ey+th2),
                                  COLOURS["plate_box"], 1)
            except Exception:
                pass

    # ── HUD ───────────────────────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray, detections: list,
                  stats: FrameStats):
        h, w = frame.shape[:2]
        viols = [d for d in detections if d.has_violation()]

        # Top bar
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,36), COLOURS["hud_bg"], -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame,
                    "SMART CITY ANPR  |  SRM Institute of Science & Technology",
                    (10,24), FONT_SMALL, 0.55, COLOURS["hud_accent"],
                    1, cv2.LINE_AA)
        ts = time.strftime("%d/%m/%Y  %H:%M:%S")
        cam_txt = f"{getattr(self.cfg,'camera_id','CCTV-001')}  |  {ts}"
        (cw,_),_ = cv2.getTextSize(cam_txt, FONT_SMALL, 0.50, 1)
        cv2.putText(frame, cam_txt, (w-cw-10, 24), FONT_SMALL, 0.50,
                    COLOURS["hud_text"], 1, cv2.LINE_AA)

        # Bottom stats bar
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0,h-32), (w,h), COLOURS["hud_bg"], -1)
        cv2.addWeighted(ov2, 0.80, frame, 0.20, 0, frame)
        items = [
            ("FPS",       f"{stats.fps:.1f}"),
            ("Vehicles",  str(len(detections))),
            ("Plates",    str(sum(1 for d in detections if d.plate))),
            ("Violations",str(len(viols))),
            ("Frame",     str(stats.frame_num)),
            ("GenAI",     "ON" if getattr(self.cfg,"use_genai",False) else "OFF"),
        ]
        sx = 10
        for label, val in items:
            col = COLOURS["hud_warning"] if label=="Violations" and viols \
                  else COLOURS["hud_accent"]
            txt = f"{label}: {val}"
            cv2.putText(frame, txt, (sx, h-10), FONT_SMALL, 0.50,
                        col, 1, cv2.LINE_AA)
            (tw,_),_ = cv2.getTextSize(txt, FONT_SMALL, 0.50, 1)
            sx += tw + 25

        # Pulsing violation alert banner
        if viols:
            banner = f"!  {len(viols)} VIOLATION(S) DETECTED"
            (bw,bh),_ = cv2.getTextSize(banner, FONT, 0.75, 2)
            bx = (w - bw) // 2
            alpha = 0.6 + 0.4*abs(np.sin(time.time()*3))
            ov3 = frame.copy()
            cv2.rectangle(ov3, (bx-10,38), (bx+bw+10,72), (0,0,180), -1)
            cv2.addWeighted(ov3, alpha, frame, 1-alpha, 0, frame)
            cv2.putText(frame, banner, (bx,65), FONT, 0.75,
                        (255,255,255), 2, cv2.LINE_AA)
