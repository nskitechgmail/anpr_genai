"""
fix_image_display.py — Fixes image loading and dashboard display.

Problems fixed:
  1. Very large images (4896x4823) crash or show nothing — resize to max 1920px
  2. _restart_pipeline doesn't update canvas for image files
  3. debug_output shows detections work — just dashboard display broken
"""
import re, os

# ══════════════════════════════════════════════════════════════════════════════
# Fix 1: pipeline.py — resize large images before processing
# ══════════════════════════════════════════════════════════════════════════════
pipeline_path = 'core/pipeline.py'
pipeline_code = open(pipeline_path, encoding='utf-8').read()

old_single = '''    def process_single_image(self, image_path: str) -> tuple[np.ndarray, list]:
        """Process a single image (no video loop). Returns (annotated, dets)."""
        self.models.load_all()
        self.recogniser = PlateRecogniser(self.models, self.cfg)
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        detections = self.recogniser.process_frame(frame)
        annotated  = self.annotator.draw(
            frame, detections, FrameStats(vehicles=len(detections)))
        return annotated, detections'''

new_single = '''    def process_single_image(self, image_path: str) -> tuple[np.ndarray, list]:
        """Process a single image (no video loop). Returns (annotated, dets)."""
        self.models.load_all()
        self.recogniser = PlateRecogniser(self.models, self.cfg)
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Resize very large images — keeps plates readable, prevents canvas crash
        h, w = frame.shape[:2]
        max_dim = 1920
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            log.info(f"Resized image: {w}x{h} -> {new_w}x{new_h}")

        detections = self.recogniser.process_frame(frame)
        annotated  = self.annotator.draw(
            frame, detections, FrameStats(vehicles=len(detections)))
        return annotated, detections'''

if old_single in pipeline_code:
    pipeline_code = pipeline_code.replace(old_single, new_single)
    open(pipeline_path, 'w', encoding='utf-8').write(pipeline_code)
    print('pipeline.py: fixed process_single_image with resize')
else:
    # Try regex approach
    import re
    pattern = r'(    def process_single_image\(self.*?return annotated, detections)'
    match = re.search(pattern, pipeline_code, re.DOTALL)
    if match:
        pipeline_code = pipeline_code.replace(match.group(0), new_single)
        open(pipeline_path, 'w', encoding='utf-8').write(pipeline_code)
        print('pipeline.py: fixed via regex')
    else:
        print('pipeline.py: could not patch automatically')
        print('  -> Manually add resize before process_frame in process_single_image')

# ══════════════════════════════════════════════════════════════════════════════
# Fix 2: dashboard.py — rewrite _restart_pipeline and _open_file completely
# ══════════════════════════════════════════════════════════════════════════════
dashboard_path = 'ui/dashboard.py'
dashboard_code = open(dashboard_path, encoding='utf-8').read()

# New _open_file
new_open_file = '''    def _open_file(self):
        import os
        filepath = filedialog.askopenfilename(
            title="Open Video or Image File",
            filetypes=[
                ("All supported", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"),
                ("Video files",   "*.mp4 *.avi *.mov *.mkv"),
                ("Image files",   "*.jpg *.jpeg *.png *.bmp"),
                ("All files",     "*.*"),
            ]
        )
        if not filepath:
            return
        if not os.path.exists(filepath):
            messagebox.showerror("Not Found", f"File not found:\\n{filepath}")
            return
        self._status_var.set(f"Loading: {os.path.basename(filepath)} ...")
        self.root.update()
        self._process_file(filepath)'''

# New _process_file (handles both images and video)
new_process_file = '''    def _process_file(self, filepath):
        """Handle image or video file — replaces _restart_pipeline for files."""
        import os, cv2, numpy as np
        from PIL import Image, ImageTk

        ext = os.path.splitext(filepath)[1].lower()
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        if ext in image_exts:
            # ── Image file: process immediately and show result ──────────
            try:
                self._status_var.set("Processing image ...")
                self.root.update()

                # Stop existing pipeline
                try:
                    self.pipeline.stop()
                except Exception:
                    pass

                # Rebuild pipeline with genai off for speed
                from core.pipeline import ANPRPipeline
                self.cfg.use_genai = False
                self.pipeline = ANPRPipeline(self.cfg)

                # Process
                annotated, dets = self.pipeline.process_single_image(filepath)

                # Show on canvas
                cw = max(self._video_canvas.winfo_width(),  960)
                ch = max(self._video_canvas.winfo_height(), 540)
                h, w = annotated.shape[:2]
                scale = min(cw / max(w, 1), ch / max(h, 1))
                nw = int(w * scale)
                nh = int(h * scale)
                resized = cv2.resize(annotated, (nw, nh))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self._photo_main = ImageTk.PhotoImage(img)
                self._video_canvas.delete("all")
                self._video_canvas.create_image(
                    cw // 2, ch // 2,
                    image=self._photo_main, anchor="center")

                # Update metrics
                plates = sum(1 for d in dets if d.plate)
                viols  = sum(1 for d in dets if d.has_violation())
                self._metric_vars["vehicles"].set(str(len(dets)))
                self._metric_vars["plates"].set(str(plates))
                self._metric_vars["violations"].set(str(viols))
                self._metric_vars["fps"].set("—")

                # Show first plate
                for d in dets:
                    if d.plate:
                        self._plate_text_var.set(d.plate.normalised())
                        break

                # Log violations in tree
                import time as _t
                for d in dets:
                    if d.plate:
                        ts = _t.strftime("%H:%M:%S")
                        tag = "viol" if d.has_violation() else "ok"
                        self._tree.insert("", "0",
                            values=(ts,
                                    d.plate.normalised(),
                                    d.vehicle_class,
                                    d.violation,
                                    f"{d.plate.confidence:.0%}",
                                    self.cfg.camera_id),
                            tags=(tag,))

                fname = os.path.basename(filepath)
                self._status_var.set(
                    f"{fname}  |  Vehicles: {len(dets)}  "
                    f"Plates: {plates}  Violations: {viols}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("Processing Error", str(e))
                self._status_var.set(f"Error: {e}")

        else:
            # ── Video file: restart live pipeline ────────────────────────
            try:
                self.pipeline.stop()
            except Exception:
                pass
            from core.pipeline import ANPRPipeline
            self.cfg.source = filepath
            self.pipeline = ANPRPipeline(self.cfg)
            self.pipeline.start()
            self._running = True
            self._status_var.set(f"Playing: {os.path.basename(filepath)}")'''

# New _restart_pipeline (kept for webcam/RTSP use)
new_restart = '''    def _restart_pipeline(self, source):
        """Restart pipeline with new source (webcam or RTSP)."""
        import os
        try:
            self.pipeline.stop()
        except Exception:
            pass
        from core.pipeline import ANPRPipeline
        self.cfg.source = source
        self.pipeline   = ANPRPipeline(self.cfg)
        self.pipeline.start()
        self._running = True
        self._status_var.set(f"Source: {source}")'''

# ── Apply patches ─────────────────────────────────────────────────────────────
def replace_method(code, method_name, new_code):
    """Replace a method in a class by finding its def line."""
    pattern = rf'(    def {re.escape(method_name)}\(self.*?)(\n    def |\n    # ──|\Z)'
    match = re.search(pattern, code, re.DOTALL)
    if match:
        code = code[:match.start()] + new_code + '\n' + code[match.end(1):]
        print(f'  Replaced: {method_name}')
    else:
        # Append before _on_close or at end of class
        if '_on_close' in code:
            code = code.replace(
                '    def _on_close(self):',
                new_code + '\n\n    def _on_close(self):'
            )
            print(f'  Inserted before _on_close: {method_name}')
        else:
            code = code.rstrip() + '\n\n' + new_code + '\n'
            print(f'  Appended: {method_name}')
    return code

print('dashboard.py patches:')
dashboard_code = replace_method(dashboard_code, '_open_file',        new_open_file)
dashboard_code = replace_method(dashboard_code, '_restart_pipeline',  new_restart)

# Add _process_file if not present
if '_process_file' not in dashboard_code:
    dashboard_code = replace_method(dashboard_code, '_use_webcam',
        new_process_file + '\n\n    def _use_webcam(self):\n        self._restart_pipeline(0)')
    print('  Added: _process_file')

open(dashboard_path, 'w', encoding='utf-8').write(dashboard_code)

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print('''
Done! All fixes applied.

Run:
  python main.py --no-genai

Then click Open File and select your image.
You should see:
  - Green corner boxes around vehicles
  - Cyan plate text below each plate
  - Violation badges if helmet/belt missing
  - Stats updated in right panel
''')
