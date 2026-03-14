import sys, os
sys.path.insert(0, os.getcwd())

print("=" * 50)
print("ANPR SYSTEM DIAGNOSTIC")
print("=" * 50)

# Test 1: Settings
print("\n[1] Testing Settings...")
try:
    from config.settings import Settings
    cfg = Settings()
    print(f"  OK — source={cfg.source}, device={cfg.device}")
    print(f"  conf_thresh={cfg.conf_thresh}, use_genai={cfg.use_genai}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: YOLO
print("\n[2] Testing YOLO...")
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("  OK — YOLO loaded")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 3: EasyOCR
print("\n[3] Testing EasyOCR...")
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("  OK — EasyOCR loaded")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: ModelManager
print("\n[4] Testing ModelManager...")
try:
    from config.settings import Settings
    from models.model_manager import ModelManager
    cfg = Settings()
    cfg.use_genai = False
    cfg.device = "cpu"
    mm = ModelManager(cfg)
    mm.load_all()
    print("  OK — all models loaded")
except Exception as e:
    print(f"  FAIL at: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Pipeline
print("\n[5] Testing Pipeline...")
try:
    from config.settings import Settings
    from core.pipeline import ANPRPipeline
    cfg = Settings()
    cfg.use_genai = False
    cfg.device = "cpu"
    cfg.source = "0"
    pipeline = ANPRPipeline(cfg)
    print("  OK — pipeline created")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Single image
print("\n[6] Testing single image processing...")
try:
    import cv2, numpy as np
    # Create a blank test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "MH 12 AB 1234", (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    cv2.imwrite("test_input.jpg", img)

    from config.settings import Settings
    from core.pipeline import ANPRPipeline
    cfg = Settings()
    cfg.use_genai = False
    cfg.device = "cpu"
    pipeline = ANPRPipeline(cfg)
    result, dets = pipeline.process_single_image("test_input.jpg")
    print(f"  OK — got {len(dets)} detections")
    cv2.imwrite("test_output.jpg", result)
    print("  Saved test_output.jpg")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("DIAGNOSTIC COMPLETE")
print("=" * 50)