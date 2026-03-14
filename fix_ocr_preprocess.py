import re

path = 'core/plate_recogniser.py'
code = open(path, encoding='utf-8').read()

new_preprocess = '''
def _preprocess_plate(self, plate_img):
    """
    Enhanced preprocessing pipeline for unclear/blurry plates.
    Works without GPU or Real-ESRGAN.
    Tries 4 different methods and returns the best result for OCR.
    """
    import cv2, numpy as np

    if plate_img is None or plate_img.size == 0:
        return plate_img

    # ── Step 1: Upscale to minimum readable size ───────────────────────
    h, w = plate_img.shape[:2]
    target_w = max(w * 4, 280)
    target_h = max(h * 4, 70)
    img = cv2.resize(plate_img, (target_w, target_h),
                     interpolation=cv2.INTER_CUBIC)

    # ── Step 2: Denoise ────────────────────────────────────────────────
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # ── Step 3: Sharpen ────────────────────────────────────────────────
    kernel_sharp = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    img = cv2.filter2D(img, -1, kernel_sharp)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 4: CLAHE contrast enhancement ────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # ── Step 5: Generate 4 candidate versions ─────────────────────────
    # Version A: Otsu threshold
    _, v_otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Version B: Adaptive threshold
    v_adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8)

    # Version C: Inverted Otsu (for dark backgrounds)
    _, v_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Version D: Raw enhanced grayscale
    v_gray = gray

    return [
        cv2.cvtColor(v_otsu,  cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(v_adapt, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(v_inv,   cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(v_gray,  cv2.COLOR_GRAY2BGR),
    ]
'''

new_ocr = '''
def _run_ocr(self, plate_img):
    """
    Run OCR on all 4 preprocessed versions and return
    the result with the highest confidence score.
    """
    import cv2

    if plate_img is None or plate_img.size == 0:
        return "", 0.0

    versions = self._preprocess_plate(plate_img)
    if not isinstance(versions, list):
        versions = [versions]

    best_text  = ""
    best_conf  = 0.0

    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

    for img in versions:
        try:
            results = self.models.ocr.readtext(
                img,
                allowlist=allowlist,
                detail=1,
                paragraph=False,
                min_size=10,
                text_threshold=0.5,
                low_text=0.3,
                link_threshold=0.3,
            )
            for (_, text, conf) in results:
                text = text.strip().upper()
                if len(text) >= 4 and conf > best_conf:
                    best_text = text
                    best_conf = conf
        except Exception:
            continue

    # ── Apply position-aware Indian plate correction ───────────────────
    best_text = self._correct_plate_text(best_text)
    return best_text, best_conf


def _correct_plate_text(self, text):
    """Position-aware correction for Indian plate format: SS NN LL NNNN"""
    import re
    text = text.replace(" ", "").upper()

    # Remove non-alphanumeric
    text = re.sub(r"[^A-Z0-9]", "", text)

    if len(text) < 4:
        return text

    corrected = list(text)
    for i, ch in enumerate(corrected):
        if i < 2:
            # State code — must be letters
            if ch == "0": corrected[i] = "O"
            if ch == "1": corrected[i] = "I"
            if ch == "5": corrected[i] = "S"
            if ch == "8": corrected[i] = "B"
        elif i < 4:
            # District code — must be digits
            if ch == "O": corrected[i] = "0"
            if ch == "I": corrected[i] = "1"
            if ch == "S": corrected[i] = "5"
            if ch == "B": corrected[i] = "8"
            if ch == "Z": corrected[i] = "2"

    return "".join(corrected)
'''

# Replace _preprocess_plate
p1 = r'def _preprocess_plate\(self.*?(?=\n    def |\nclass |\Z)'
if re.search(p1, code, re.DOTALL):
    code = re.sub(p1, new_preprocess.strip(), code, flags=re.DOTALL)
    print("Replaced _preprocess_plate")
else:
    code = code.rstrip() + '\n' + new_preprocess + '\n'
    print("Added _preprocess_plate")

# Replace _run_ocr
p2 = r'def _run_ocr\(self.*?(?=\n    def |\nclass |\Z)'
if re.search(p2, code, re.DOTALL):
    code = re.sub(p2, new_ocr.strip(), code, flags=re.DOTALL)
    print("Replaced _run_ocr")
else:
    code = code.rstrip() + '\n' + new_ocr + '\n'
    print("Added _run_ocr")

open(path, 'w', encoding='utf-8').write(code)
print("Done!")