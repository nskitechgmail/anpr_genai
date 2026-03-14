import cv2
import sys

# ── Load image ────────────────────────────────────────────────────────────────
img_path = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'
frame = cv2.imread(img_path)
if frame is None:
    print(f"ERROR: Cannot read image: {img_path}")
    sys.exit(1)

print(f"Image loaded: {frame.shape[1]}x{frame.shape[0]} px")

# ── Stage 1: YOLO vehicle detection ───────────────────────────────────────────
print("\n--- Stage 1: Vehicle Detection ---")
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model(frame, conf=0.20, iou=0.40, verbose=False)
boxes = []
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            boxes.append((x1,y1,x2,y2,cls,conf))
            print(f"  Vehicle detected: class={cls} conf={conf:.2f} box=({x1},{y1},{x2},{y2})")

if not boxes:
    print("  WARNING: No vehicles detected — try a clearer image with a car/bike")
    sys.exit(1)

# ── Stage 2: Plate localisation ───────────────────────────────────────────────
print("\n--- Stage 2: Plate Localisation ---")
for (x1,y1,x2,y2,cls,conf) in boxes:
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    print(f"  Vehicle ROI size: {w}x{h} px")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(blur, 30, 120)
    combined = cv2.bitwise_or(thresh, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Total contours found: {len(contours)}")

    candidates = []
    for cnt in contours:
        rx,ry,rw,rh = cv2.boundingRect(cnt)
        area = rw * rh
        aspect = rw / rh if rh > 0 else 0
        if 2.0 <= aspect <= 7.0 and area >= 200:
            candidates.append((area, rx, ry, rw, rh))

    print(f"  Plate candidates (aspect 2-7, area>=200): {len(candidates)}")
    if candidates:
        candidates.sort(reverse=True)
        _, rx,ry,rw,rh = candidates[0]
        plate_crop = roi[ry:ry+rh, rx:rx+rw]
        print(f"  Best plate crop: {rw}x{rh} px")
        cv2.imwrite('debug_plate_crop.jpg', plate_crop)
        print(f"  Saved: debug_plate_crop.jpg")
    else:
        # Fallback crop
        fallback = roi[int(h*0.6):int(h*0.85), int(w*0.15):int(w*0.85)]
        cv2.imwrite('debug_plate_crop.jpg', fallback)
        print(f"  No candidates — saved fallback crop: debug_plate_crop.jpg")

# ── Stage 3: OCR ──────────────────────────────────────────────────────────────
print("\n--- Stage 3: OCR ---")
import easyocr
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
plate_img = cv2.imread('debug_plate_crop.jpg')

if plate_img is not None:
    # Upscale for better OCR
    plate_img = cv2.resize(plate_img,
        (plate_img.shape[1]*4, plate_img.shape[0]*4),
        interpolation=cv2.INTER_CUBIC)

    results_ocr = reader.readtext(plate_img,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
        detail=1)

    if results_ocr:
        for (bbox, text, prob) in results_ocr:
            print(f"  OCR result: '{text}'  confidence: {prob:.2f}")
    else:
        print("  OCR found nothing — plate crop may be too blurry or small")
        print("  Check debug_plate_crop.jpg to see what was cropped")

print("\nDone! Check debug_plate_crop.jpg to see what the system cropped.")