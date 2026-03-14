"""
fix_mediapipe_manager.py — Complete fix for model_manager.py mediapipe issue.
Run from project root: python fix_mediapipe_manager.py
"""

path = 'models/model_manager.py'
code = open(path, encoding='utf-8').read()

# Add helper classes right after imports (top of file, outside any class)
haar_classes = '''
# ── MediaPipe-compatible face detector fallbacks ───────────────────────────

class _HaarFaceDetector:
    """OpenCV Haar cascade — used when MediaPipe is unavailable."""
    def __init__(self):
        import cv2
        self._clf = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def process(self, rgb_frame):
        import cv2
        ih, iw = rgb_frame.shape[:2]
        bgr  = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._clf.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

        class _BB:
            pass
        class _LD:
            pass
        class _Det:
            pass
        class _Result:
            pass

        dets = []
        for (x, y, w, h) in faces:
            bb        = _BB()
            bb.xmin   = x / iw
            bb.ymin   = y / ih
            bb.width  = w / iw
            bb.height = h / ih
            ld = _LD()
            ld.relative_bounding_box = bb
            det = _Det()
            det.location_data = ld
            dets.append(det)

        result = _Result()
        result.detections = dets
        return result

'''

# Only add if not already present
if '_HaarFaceDetector' not in code:
    # Insert after the last import line
    lines = code.split('\n')
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_at = i
    lines.insert(insert_at + 1, haar_classes)
    code = '\n'.join(lines)
    print('Added _HaarFaceDetector class')
else:
    print('_HaarFaceDetector already present')

# Fix _load_face_detector to use the class correctly
old_method = '''    def _load_face_detector(self):
        """Face detector for anonymisation — compatible with all mediapipe versions."""
        try:
            import mediapipe as mp

            # Try old API (mediapipe < 0.10.x)
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                self._face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
                log.info("  \\u2713 Face detector ready (MediaPipe solutions API)")
                return

            # Try new API (mediapipe >= 0.10.x)
            try:
                from mediapipe.tasks import python as mp_tasks
                from mediapipe.tasks.python import vision as mp_vision
                log.info("  \\u2713 Face detector ready (MediaPipe tasks API)")
                self._face_detector = _NewMediaPipeFaceDetector()
                return
            except Exception:
                pass

            # Fallback to OpenCV Haar cascade
            raise ImportError("No compatible MediaPipe API found")

        except Exception as e:
            log.warning(f"  \\u2139  MediaPipe unavailable ({e}) — using OpenCV Haar cascade")
            self._face_detector = _HaarFaceDetector()'''

new_method = '''    def _load_face_detector(self):
        """Face detector — tries MediaPipe first, falls back to OpenCV Haar."""
        import cv2

        # Always prepare Haar as ultimate fallback
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                self._face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
                log.info("  \\u2713 Face detector ready (MediaPipe)")
                return
        except Exception:
            pass

        # Use Haar cascade fallback — works on all systems
        self._face_detector = _HaarFaceDetector()
        log.info("  \\u2713 Face detector ready (OpenCV Haar cascade fallback)")'''

if old_method in code:
    code = code.replace(old_method, new_method)
    print('Replaced _load_face_detector method (exact match)')
elif 'def _load_face_detector' in code:
    import re
    pattern = r'(    def _load_face_detector\(self\):.*?)(\n    def |\nclass |\Z)'
    match = re.search(pattern, code, re.DOTALL)
    if match:
        code = code[:match.start()] + new_method + '\n' + code[match.end(1):]
        print('Replaced _load_face_detector method (regex)')
    else:
        print('WARNING: Could not replace method — appending override')
        code = code.rstrip() + '\n\n' + new_method.replace('    def', 'def') + '\n'
else:
    print('_load_face_detector not found — adding it')
    code = code.rstrip() + '\n\n' + new_method + '\n'

open(path, 'w', encoding='utf-8').write(code)
print('\nDone! Run: python main.py --no-genai')