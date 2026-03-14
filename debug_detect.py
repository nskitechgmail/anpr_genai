"""
debug_detect.py — Shows exactly what is being detected on your image.
Usage: python debug_detect.py your_image.jpg
"""
import sys, os, cv2
sys.path.insert(0, os.getcwd())

img_path = sys.argv[1] if len(sys.argv) > 1 else None
if not img_path:
    print("Usage: python debug_detect.py your_image.jpg")
    sys.exit(1)

frame = cv2.imread(img_path)
if frame is None:
    print(f"ERROR: Cannot read {img_path}")
    sys.exit(1)

print(f"Image: {frame.shape[1]}x{frame.shape[0]} px")

# Load pipeline
from config.settings import Settings
from models.model_manager import ModelManager
from core.plate_recogniser import PlateRecogniser

cfg = Settings()
cfg.use_genai   = False
cfg.device      = "cpu"
cfg.conf_thresh = 0.20
cfg.plate_min_area = 100

mm = ModelManager(cfg)
mm.load_all()

rec = PlateRecogniser(mm, cfg)
dets = rec.process_frame(frame)

print(f"\nDetections: {len(dets)}")
for i, d in enumerate(dets):
    plate = d.plate.text if d.plate else "NO PLATE"
    conf  = d.plate.confidence if d.plate else 0
    print(f"  [{i+1}] {d.vehicle_class:12s} | plate='{plate}' "
          f"conf={conf:.2f} | violation={d.violation}")

# Draw manually and save
for d in dets:
    x1,y1,x2,y2 = d.bbox
    color = (0,0,255) if d.has_violation() else (0,255,0)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    label = f"{d.vehicle_class}"
    if d.plate:
        label += f" | {d.plate.text}"
    if d.has_violation():
        label += f" | {d.violation}"
    cv2.putText(frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite("debug_output.jpg", frame)
print("\nSaved: debug_output.jpg")
print("Open it to see what was detected.")