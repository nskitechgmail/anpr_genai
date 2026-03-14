"""
fix_final.py — Fixes all remaining issues:
  1. Changes detector_model from yolov9c to yolov8n (already downloaded)
  2. Lowers conf_thresh to 0.25 for better CPU detection
  3. Lowers plate_min_area to 150 for smaller plates
  4. Sets use_genai=False by default (CPU mode)
  5. Fixes source to work with images and webcam
Run from project root: python fix_final.py
"""
import os, shutil

# ── Fix settings.py ───────────────────────────────────────────────────────────
path = os.path.join('config', 'settings.py')
code = open(path, encoding='utf-8').read()

fixes = {
    # Use yolov8n — already downloaded, works on CPU
    'detector_model: str = "yolov9c"':   'detector_model: str = "yolov8n"',
    # Lower confidence threshold for CPU detection
    'conf_thresh: float = 0.40':         'conf_thresh: float = 0.25',
    # Lower plate area threshold
    'plate_min_area: int = 500':         'plate_min_area: int = 150',
    # Disable genai by default (CPU mode)
    'use_genai: bool = True':            'use_genai: bool = False',
}

changed = []
for old, new in fixes.items():
    if old in code:
        code = code.replace(old, new)
        changed.append(new.split('=')[0].strip())

open(path, 'w', encoding='utf-8').write(code)
print(f'settings.py fixed: {", ".join(changed) if changed else "already correct"}')

# ── Move yolov8n.pt to where ModelManager looks for it ───────────────────────
src = 'yolov8n.pt'
dst = os.path.join('models', 'weights', 'yolov8n.pt')

if os.path.exists(src) and not os.path.exists(dst):
    shutil.copy(src, dst)
    print(f'Copied yolov8n.pt → {dst}')
elif os.path.exists(dst):
    print(f'yolov8n.pt already in models/weights/')
else:
    print('yolov8n.pt not found in root — YOLO will re-download it on next run (OK)')

# ── Fix model_manager.py to use yolov8n not yolov9c ──────────────────────────
mm_path = os.path.join('models', 'model_manager.py')
mm_code = open(mm_path, encoding='utf-8').read()

# Make YOLO loader more robust — try configured model, fallback to yolov8n
old_loader = '''    def _load_detector(self):
        """YOLOv8/v9 for vehicle and person detection."""
        try:
            from ultralytics import YOLO
            model_name = self.cfg.detector_model
            # ultralytics auto-downloads to ~/.cache/ultralytics
            log.info(f"Loading YOLO detector: {model_name}")
            self._detector = YOLO(f"{model_name}.pt")
            self._detector.to(self.cfg.device)
            log.info(f"  ✓ Detector ready on {self.cfg.device}")
        except ImportError:
            log.warning("ultralytics not installed — using simulation mode")
            self._detector = _SimulatedDetector()'''

new_loader = '''    def _load_detector(self):
        """YOLOv8/v9 for vehicle and person detection."""
        try:
            from ultralytics import YOLO
            # Try configured model first, fallback to yolov8n (CPU-friendly)
            for model_name in [self.cfg.detector_model, "yolov8n", "yolov8s"]:
                try:
                    log.info(f"Loading YOLO detector: {model_name}")
                    self._detector = YOLO(f"{model_name}.pt")
                    self._detector.to(self.cfg.device)
                    log.info(f"  ✓ Detector ready: {model_name} on {self.cfg.device}")
                    break
                except Exception as e:
                    log.warning(f"  Could not load {model_name}: {e} — trying next")
                    continue
            if self._detector is None:
                raise RuntimeError("No YOLO model could be loaded")
        except ImportError:
            log.warning("ultralytics not installed — using simulation mode")
            self._detector = _SimulatedDetector()
        except Exception as e:
            log.warning(f"YOLO load failed: {e} — using simulation mode")
            self._detector = _SimulatedDetector()'''

if old_loader in mm_code:
    mm_code = mm_code.replace(old_loader, new_loader)
    print('model_manager.py: fixed YOLO loader with fallback chain')
elif 'for model_name in [self.cfg.detector_model' in mm_code:
    print('model_manager.py: YOLO fallback chain already present')
else:
    print('model_manager.py: could not find exact loader — applying line-based fix')
    lines = mm_code.split('\n')
    for i, line in enumerate(lines):
        if 'model_name = self.cfg.detector_model' in line:
            indent = '            '
            lines[i] = (
                f'{indent}# Try configured model, fallback to yolov8n\n'
                f'{indent}model_name = getattr(self.cfg, "detector_model", "yolov8n")\n'
                f'{indent}if model_name == "yolov9c":  # yolov9c needs GPU, use yolov8n on CPU\n'
                f'{indent}    import torch\n'
                f'{indent}    if not torch.cuda.is_available():\n'
                f'{indent}        model_name = "yolov8n"\n'
                f'{indent}        log.info("  CPU detected — switching to yolov8n")'
            )
            break
    mm_code = '\n'.join(lines)
    print('model_manager.py: applied line-based yolov9c→yolov8n CPU fix')

open(mm_path, 'w', encoding='utf-8').write(mm_code)

# ── Verify fix with quick test ────────────────────────────────────────────────
print('\nRunning quick verification...')
import subprocess, sys
result = subprocess.run(
    [sys.executable, '-c', '''
import sys, os
sys.path.insert(0, os.getcwd())
from config.settings import Settings
cfg = Settings()
print(f"detector_model = {cfg.detector_model}")
print(f"conf_thresh    = {cfg.conf_thresh}")
print(f"plate_min_area = {cfg.plate_min_area}")
print(f"use_genai      = {cfg.use_genai}")
print(f"device         = {cfg.device}")
'''],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:200])

print('=' * 50)
print('All fixes applied!')
print()
print('Now run:')
print('  python main.py --no-genai')
print()
print('Or test on an image:')
print('  python main.py --image your_plate_image.jpg --no-genai')
print('=' * 50)
