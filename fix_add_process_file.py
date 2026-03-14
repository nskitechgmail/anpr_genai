"""
fix_add_process_file.py — Adds _process_file method directly to dashboard.py
Run from project root: python fix_add_process_file.py
"""

path = 'ui/dashboard.py'
code = open(path, encoding='utf-8').read()

if '_process_file' in code:
    print('_process_file already exists.')
else:
    new_method = '''
    def _process_file(self, filepath):
        """Process image or video file and display result on canvas."""
        import os, cv2
        from PIL import Image, ImageTk

        ext = os.path.splitext(filepath)[1].lower()
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        if ext in image_exts:
            try:
                self._status_var.set("Processing image ...")
                self.root.update()

                # Stop existing pipeline
                try:
                    self.pipeline.stop()
                except Exception:
                    pass

                # Rebuild pipeline
                from core.pipeline import ANPRPipeline
                self.cfg.use_genai = False
                self.pipeline = ANPRPipeline(self.cfg)

                # Process image
                annotated, dets = self.pipeline.process_single_image(filepath)

                # Draw on canvas
                cw = max(self._video_canvas.winfo_width(),  960)
                ch = max(self._video_canvas.winfo_height(), 540)
                h, w = annotated.shape[:2]
                scale = min(cw / max(w, 1), ch / max(h, 1))
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
                self._metric_vars["fps"].set("—")

                # Show first plate text
                for d in dets:
                    if d.plate:
                        self._plate_text_var.set(d.plate.normalised())
                        break

                # Log in violation tree
                import time as _t
                for d in dets:
                    if d.plate:
                        tag = "viol" if d.has_violation() else "ok"
                        self._tree.insert("", "0",
                            values=(_t.strftime("%H:%M:%S"),
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
                from tkinter import messagebox
                messagebox.showerror("Error", str(e))
                self._status_var.set(f"Error: {e}")
        else:
            # Video file — restart live pipeline
            try:
                self.pipeline.stop()
            except Exception:
                pass
            from core.pipeline import ANPRPipeline
            self.cfg.source = filepath
            self.pipeline = ANPRPipeline(self.cfg)
            self.pipeline.start()
            self._running = True
            self._status_var.set(f"Playing: {os.path.basename(filepath)}")
'''

    # Insert just before _on_close
    if '    def _on_close(self):' in code:
        code = code.replace(
            '    def _on_close(self):',
            new_method + '\n    def _on_close(self):'
        )
        print('Inserted _process_file before _on_close')
    else:
        code = code.rstrip() + '\n' + new_method + '\n'
        print('Appended _process_file at end of file')

    open(path, 'w', encoding='utf-8').write(code)

# Also fix _open_file to call _process_file correctly
code = open(path, encoding='utf-8').read()

# Check if _open_file is still calling _restart_pipeline instead of _process_file
if '_restart_pipeline(path)' in code and '_process_file' in code:
    code = code.replace(
        'self._restart_pipeline(path)',
        'self._process_file(path)'
    )
    open(path, 'w', encoding='utf-8').write(code)
    print('Fixed _open_file to call _process_file')

# Verify
code = open(path, encoding='utf-8').read()
has_process = '_process_file' in code
has_call    = 'self._process_file' in code
print(f'_process_file defined: {has_process}')
print(f'_process_file called:  {has_call}')

# Quick syntax check
import ast
try:
    ast.parse(code)
    print('Syntax: OK')
except SyntaxError as e:
    print(f'Syntax ERROR at line {e.lineno}: {e.msg}')

print('\nDone! Run: python main.py --no-genai')
