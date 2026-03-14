import re

path = 'core/plate_recogniser.py'
code = open(path, encoding='utf-8').read()

new_localise = '''
def _localise_plate(self, vehicle_roi, vehicle_class='Car'):
    """
    Smart plate localisation:
    - Searches LOWER 50% of ROI only (plates are never at the top)
    - Stricter aspect ratio: 2.5 to 6.0
    - Minimum width: 60px, minimum height: 15px
    - Rejects crops wider than 80% of ROI (advertisement boards)
    - Prefers crops in the bottom-centre region
    """
    import cv2, numpy as np

    h, w = vehicle_roi.shape[:2]

    # ── Search only the lower half of the vehicle ROI ──────────────────
    if vehicle_class in ('Motorcycle', 'motorcycle'):
        # Plates are at the very bottom on bikes
        search_top = int(h * 0.60)
    else:
        # Cars, buses, trucks — plate in lower 50%
        search_top = int(h * 0.50)

    roi = vehicle_roi[search_top:, :]
    rh, rw = roi.shape[:2]

    # ── Pre-process ────────────────────────────────────────────────────
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray  = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)

        # ── Size filters ────────────────────────────────────────────
        if cw < 60 or ch < 15:          # too small
            continue
        if cw > rw * 0.80:              # too wide = advertisement board
            continue
        if ch > rh * 0.40:              # too tall = not a plate
            continue

        # ── Aspect ratio filter ─────────────────────────────────────
        aspect = cw / ch
        if not (2.5 <= aspect <= 6.5):
            continue

        # ── Score: prefer bottom-centre ─────────────────────────────
        area         = cw * ch
        centre_x     = cx + cw / 2
        centre_score = 1.0 - abs(centre_x - rw / 2) / (rw / 2)
        bottom_score = cy / rh          # higher = closer to bottom
        score        = area * (1 + centre_score * 0.5 + bottom_score * 0.3)

        candidates.append((score, cx, cy, cw, ch))

    if candidates:
        candidates.sort(reverse=True)
        _, cx, cy, cw, ch = candidates[0]
        # Translate back to full ROI coordinates
        pad = 4
        y1 = max(0,  search_top + cy - pad)
        y2 = min(h,  search_top + cy + ch + pad)
        x1 = max(0,  cx - pad)
        x2 = min(w,  cx + cw + pad)
        return vehicle_roi[y1:y2, x1:x2]

    # ── Fallback: bottom-centre strip ──────────────────────────────────
    fy1 = int(h * 0.72)
    fy2 = int(h * 0.90)
    fx1 = int(w * 0.15)
    fx2 = int(w * 0.85)
    return vehicle_roi[fy1:fy2, fx1:fx2]
'''

# Replace existing _localise_plate method
pattern = r'def _localise_plate\(self.*?(?=\n    def |\nclass |\Z)'
if re.search(pattern, code, re.DOTALL):
    code = re.sub(pattern, new_localise.strip(), code, flags=re.DOTALL)
    print("Replaced existing _localise_plate method.")
else:
    # Append if not found
    code = code.rstrip() + '\n' + new_localise + '\n'
    print("Added new _localise_plate method.")

open(path, 'w', encoding='utf-8').write(code)
print("Done!")