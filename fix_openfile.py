"""
fix_openfile.py — Fixes the Open File button in dashboard.py
"""
import re

path = 'ui/dashboard.py'
code = open(path, encoding='utf-8').read()

# Fix 1: _open_file method — make it actually restart with the chosen file
old_open = '''    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open Video / Image",
            filetypes=[("Video/Image",
                        "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            self._restart_pipeline(path)'''

new_open = '''    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open Video / Image",
            filetypes=[
                ("All supported", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"),
                ("Video files",   "*.mp4 *.avi *.mov *.mkv"),
                ("Image files",   "*.jpg *.jpeg *.png *.bmp"),
                ("All files",     "*.*"),
            ]
        )
        if not path:
            return
        path = path.replace("/", "\\\\")
        import os
        if not os.path.exists(path):
            from tkinter import messagebox
            messagebox.showerror("File Not Found", f"Cannot find:\\n{path}")
            return
        self._restart_pipeline(path)'''

# Fix 2: _restart_pipeline — handle image files separately from video
old_restart = '''    def _restart_pipeline(self, source):
        self.pipeline.stop()
        self.cfg.source = source
        self.pipeline = ANPRPipeline(self.cfg)
        self.pipeline.start()
        self._status_var.set(f"Source changed: {source}")'''

new_restart = '''    def _restart_pipeline(self, source):
        import os
        self.pipeline.stop()
        self.cfg.source = source

        # Check if it is an image file
        ext = os.path.splitext(str(source))[1].lower()
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        if ext in image_exts:
            # Process single image and display result
            try:
                self.cfg.use_genai = False
                self.pipeline = ANPRPipeline(self.cfg)
                annotated, dets = self.pipeline.process_single_image(source)

                # Show result in canvas
                import cv2
                import numpy as np
                from PIL import Image, ImageTk

                cw = self._video_canvas.winfo_width()  or 960
                ch = self._video_canvas.winfo_height() or 540
                h, w = annotated.shape[:2]
                scale = min(cw / max(w,1), ch / max(h,1))
                nw, nh = int(w * scale), int(h * scale)
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

                # Show plate text if found
                for d in dets:
                    if d.plate:
                        self._plate_text_var.set(d.plate.normalised())
                        break

                # Log violations
                import time as _time
                for d in dets:
                    if d.has_violation() and d.plate:
                        ts = _time.strftime("%H:%M:%S")
                        self._tree.insert("", "0",
                            values=(ts, d.plate.normalised(),
                                    d.vehicle_class, d.violation,
                                    f"{d.plate.confidence:.0%}",
                                    self.cfg.camera_id),
                            tags=("viol",))

                n_plates = sum(1 for d in dets if d.plate)
                self._status_var.set(
                    f"Image processed: {os.path.basename(source)} | "
                    f"Vehicles: {len(dets)} | Plates: {n_plates} | "
                    f"Violations: {viols}"
                )
            except Exception as e:
                from tkinter import messagebox
                messagebox.showerror("Processing Error", str(e))
                import traceback; traceback.print_exc()
        else:
            # Video file or stream — restart live pipeline
            self.pipeline = ANPRPipeline(self.cfg)
            self.pipeline.start()
            self._running = True
            self._status_var.set(f"Playing: {os.path.basename(str(source))}")'''

changed = []

if old_open in code:
    code = code.replace(old_open, new_open)
    changed.append("_open_file")
else:
    print("WARNING: _open_file not matched exactly — check manually")

if old_restart in code:
    code = code.replace(old_restart, new_restart)
    changed.append("_restart_pipeline")
else:
    print("WARNING: _restart_pipeline not matched exactly — check manually")

open(path, 'w', encoding='utf-8').write(code)

if changed:
    print(f"Fixed: {', '.join(changed)}")
    print("Done! Restart: python main.py --no-genai")
else:
    print("No exact matches found.")
    print("Please upload ui/dashboard.py to the chat for a clean rewrite.")