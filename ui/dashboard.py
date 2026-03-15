"""
ui/dashboard.py — Real-time ANPR monitoring dashboard (Sprint 2).

Key fixes vs previous version:
  • FrameStats imported from core.pipeline (not core.plate_recogniser)
  • _session_start initialised in __init__ (was missing → NameError)
  • ANPRDashboard(settings) — pipeline created internally
  • _process_file handles both images and video
  • _restart_pipeline for webcam/RTSP
"""
from __future__ import annotations
import csv
import logging
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# FrameStats from pipeline — single source of truth
from core.pipeline        import ANPRPipeline, FrameStats
from core.plate_recogniser import VehicleDetection
from config.settings       import Settings

log = logging.getLogger("Dashboard")

# Colour scheme
BG_DARK    = "#0D1B2A"
BG_PANEL   = "#1B2E45"
ACCENT     = "#4FC3F7"
ACCENT2    = "#7B2FBE"
TEXT_LIGHT = "#E0E8F0"
TEXT_DIM   = "#88AABB"
VIOL_RED   = "#FF4444"
COMPL_GRN  = "#00C864"
WARN_GOLD  = "#FFB400"


class ANPRDashboard:
    """Main Tkinter application window."""

    _REFRESH_MS = 33      # ~30 FPS
    _LOG_LIMIT  = 300

    def __init__(self, settings: Settings):
        self.cfg      = settings
        self.pipeline = ANPRPipeline(settings)
        self._running = False
        self._session_start = time.time()   # ← was missing; caused NameError
        self._session_counts = dict(
            total_vehicles=0, total_plates=0,
            total_violations=0, no_helmet=0, no_belt=0)
        self._photo_main  = None
        self._photo_plate = None

    # ── Launch ────────────────────────────────────────────────────

    def run(self):
        self.root = tk.Tk()
        self.root.title(
            "Smart City ANPR System  ·  SRM Institute of Science & Technology")
        self.root.geometry("1400x860")
        self.root.configure(bg=BG_DARK)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_styles()
        self._build_ui()
        self.pipeline.start()
        self._running = True
        self._update_loop()
        self.root.mainloop()

    # ── Styles ────────────────────────────────────────────────────

    def _build_styles(self):
        st = ttk.Style(self.root)
        st.theme_use("clam")
        st.configure("TFrame",       background=BG_DARK,  foreground=TEXT_LIGHT)
        st.configure("Card.TFrame",  background=BG_PANEL, foreground=TEXT_LIGHT)
        st.configure("TLabel",       background=BG_DARK,  foreground=TEXT_LIGHT,
                     font=("Segoe UI", 10))
        st.configure("Header.TLabel",background=BG_PANEL, foreground=ACCENT,
                     font=("Segoe UI", 11, "bold"))
        st.configure("Metric.TLabel",background=BG_DARK,  foreground=ACCENT,
                     font=("Segoe UI", 22, "bold"))
        st.configure("Sub.TLabel",   background=BG_PANEL, foreground=TEXT_DIM,
                     font=("Segoe UI", 9))
        st.configure("Run.TButton",  background=ACCENT,   foreground=BG_DARK,
                     font=("Segoe UI", 9, "bold"), padding=4)
        st.configure("Danger.TButton", background=VIOL_RED, foreground="white",
                     font=("Segoe UI", 9, "bold"), padding=4)
        st.configure("ANPR.Treeview",
                     background=BG_DARK, foreground=TEXT_LIGHT,
                     rowheight=22, fieldbackground=BG_DARK,
                     font=("Segoe UI", 8))
        st.configure("ANPR.Treeview.Heading",
                     background=BG_PANEL, foreground=ACCENT,
                     font=("Segoe UI", 8, "bold"))
        st.map("ANPR.Treeview",
               background=[("selected", ACCENT2)],
               foreground=[("selected", "#FFFFFF")])

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        tb = tk.Frame(self.root, bg=BG_DARK, height=46)
        tb.pack(fill=tk.X)
        tk.Label(tb, text="🚦  Smart City ANPR  —  Multi-Modal Vehicle Detection & LPR",
                 font=("Segoe UI", 13, "bold"), fg=ACCENT, bg=BG_DARK
                 ).pack(side=tk.LEFT, padx=14, pady=8)
        tk.Label(tb, text="SRM IST · Dept. Computational Intelligence · 2024-25",
                 font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_DARK
                 ).pack(side=tk.RIGHT, padx=14)

        # Toolbar
        toolbar = tk.Frame(self.root, bg=BG_PANEL, height=40)
        toolbar.pack(fill=tk.X)
        self._build_toolbar(toolbar)

        # Body
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = tk.Frame(body, bg=BG_PANEL, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6,0))
        right.pack_propagate(False)

        self._build_video_panel(left)
        self._build_metrics_bar(left)
        self._build_log_panel(right)

        # Status bar
        self._status_var = tk.StringVar(value="Ready — pipeline starting …")
        tk.Label(self.root, textvariable=self._status_var,
                 font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_DARK,
                 anchor=tk.W).pack(fill=tk.X, padx=8, pady=2)

    def _build_toolbar(self, parent):
        controls = [
            ("✦ GenAI", "_genai_var", "toggle", self._toggle_genai),
            ("📁 Open File", None, "btn",    self._open_file),
            ("📷 Webcam",   None, "btn",    self._use_webcam),
            ("📡 RTSP",     None, "btn",    self._open_rtsp),
            ("💾 Export CSV",None,"btn",    self._export_csv),
            ("⏹ Stop",      None, "danger", self._stop_pipeline),
        ]
        self._genai_var = tk.BooleanVar(value=self.cfg.use_genai)
        for label, var, kind, cmd in controls:
            if kind == "toggle":
                tk.Checkbutton(parent, text=label,
                               variable=self._genai_var,
                               command=cmd,
                               fg=ACCENT, bg=BG_PANEL,
                               selectcolor=BG_DARK,
                               activeforeground=ACCENT,
                               activebackground=BG_PANEL,
                               font=("Segoe UI", 9)
                               ).pack(side=tk.LEFT, padx=10, pady=6)
            elif kind == "danger":
                ttk.Button(parent, text=label, command=cmd,
                           style="Danger.TButton"
                           ).pack(side=tk.RIGHT, padx=8, pady=6)
            else:
                ttk.Button(parent, text=label, command=cmd,
                           style="Run.TButton"
                           ).pack(side=tk.LEFT, padx=4, pady=6)

        # Confidence slider
        tk.Label(parent, text="Conf:", fg=TEXT_DIM, bg=BG_PANEL,
                 font=("Segoe UI",9)).pack(side=tk.LEFT, padx=(12,2))
        self._conf_var = tk.DoubleVar(value=self.cfg.conf_thresh)
        tk.Scale(parent, from_=0.10, to=0.90, resolution=0.05,
                 orient=tk.HORIZONTAL, variable=self._conf_var,
                 command=self._update_conf,
                 bg=BG_PANEL, fg=TEXT_LIGHT, troughcolor=BG_DARK,
                 highlightthickness=0, length=120,
                 ).pack(side=tk.LEFT)

    def _build_video_panel(self, parent):
        frame = tk.Frame(parent, bg="#000000", bd=2, relief=tk.SUNKEN)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0,4))
        self._video_canvas = tk.Canvas(frame, bg="#000000",
                                       highlightthickness=0)
        self._video_canvas.pack(fill=tk.BOTH, expand=True)
        self._video_canvas.create_text(
            480, 270, text="Initialising …",
            fill=TEXT_DIM, font=("Segoe UI", 14), tags="placeholder")

    def _build_metrics_bar(self, parent):
        bar = tk.Frame(parent, bg=BG_PANEL, height=72)
        bar.pack(fill=tk.X, pady=(0,4))
        bar.pack_propagate(False)
        self._metric_vars = {}
        for label, key, colour in [
            ("FPS",        "fps",        ACCENT),
            ("Vehicles",   "vehicles",   TEXT_LIGHT),
            ("Plates",     "plates",     ACCENT),
            ("Violations", "violations", VIOL_RED),
        ]:
            col = tk.Frame(bar, bg=BG_PANEL)
            col.pack(side=tk.LEFT, expand=True, padx=10, pady=6)
            var = tk.StringVar(value="0")
            self._metric_vars[key] = var
            tk.Label(col, textvariable=var, font=("Segoe UI",22,"bold"),
                     fg=colour, bg=BG_PANEL).pack()
            tk.Label(col, text=label, font=("Segoe UI",8),
                     fg=TEXT_DIM, bg=BG_PANEL).pack()

    def _build_log_panel(self, parent):
        tk.Label(parent, text="Violation Log",
                 font=("Segoe UI",11,"bold"), fg=ACCENT, bg=BG_PANEL
                 ).pack(anchor=tk.W, padx=10, pady=(10,4))

        cols = ("Time","Plate","Vehicle","Violation","Conf","Camera")
        self._tree = ttk.Treeview(parent, columns=cols,
                                   show="headings", height=16,
                                   style="ANPR.Treeview")
        for col in cols:
            self._tree.heading(col, text=col)
            w = 55 if col in ("Time","Conf","Camera") else 75
            self._tree.column(col, width=w, anchor=tk.W)
        self._tree.tag_configure("viol", foreground=VIOL_RED)
        self._tree.tag_configure("ok",   foreground=COMPL_GRN)
        sb = ttk.Scrollbar(parent, orient=tk.VERTICAL,
                           command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                        padx=(10,0), pady=4)
        sb.pack(side=tk.LEFT, fill=tk.Y, pady=4, padx=(0,6))

        # Plate display
        tk.Label(parent, text="Latest Plate", font=("Segoe UI",9),
                 fg=TEXT_DIM, bg=BG_PANEL).pack(padx=10, pady=(8,2))
        self._plate_canvas = tk.Canvas(parent, bg="#000000",
                                        height=80, width=200,
                                        highlightthickness=0)
        self._plate_canvas.pack(padx=10)
        self._plate_text_var = tk.StringVar(value="—")
        tk.Label(parent, textvariable=self._plate_text_var,
                 font=("Courier",14,"bold"), fg=ACCENT, bg=BG_PANEL
                 ).pack(pady=2)

        # Session summary
        self._summary_var = tk.StringVar(value="No detections yet.")
        tk.Label(parent, textvariable=self._summary_var,
                 font=("Segoe UI",8), fg=TEXT_DIM, bg=BG_PANEL,
                 wraplength=340).pack(padx=10, pady=(8,4))

    # ── Refresh loop ──────────────────────────────────────────────

    def _update_loop(self):
        try:
            frame, stats, dets = self.pipeline.get_latest()
            if frame is not None and _PIL_OK:
                self._update_canvas(frame)
            if stats:
                self._update_metrics(stats)
            if dets is not None:
                self._update_plate_display(dets)
                self._update_violations(dets)
                self._update_summary(dets)
        except Exception as e:
            log.debug(f"Refresh error: {e}")
        finally:
            self.root.after(self._REFRESH_MS, self._update_loop)

    def _update_canvas(self, frame: np.ndarray):
        cw = self._video_canvas.winfo_width()  or 960
        ch = self._video_canvas.winfo_height() or 540
        h, w = frame.shape[:2]
        scale = min(cw/max(w,1), ch/max(h,1))
        nw, nh = int(w*scale), int(h*scale)
        if nw < 1 or nh < 1:
            return
        rgb = cv2.cvtColor(cv2.resize(frame,(nw,nh)), cv2.COLOR_BGR2RGB)
        self._photo_main = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._video_canvas.delete("all")
        self._video_canvas.create_image(cw//2, ch//2,
                                         image=self._photo_main,
                                         anchor=tk.CENTER)

    def _update_metrics(self, stats: FrameStats):
        self._metric_vars["fps"].set(f"{stats.fps:.1f}")
        self._metric_vars["vehicles"].set(str(stats.vehicles))
        self._metric_vars["plates"].set(str(stats.plates_read))
        self._metric_vars["violations"].set(str(stats.violations))

    def _update_plate_display(self, dets: list):
        for d in dets:
            if d.plate and d.plate.plate_crop is not None and _PIL_OK:
                try:
                    crop = d.plate.plate_crop
                    if len(crop.shape) == 2:
                        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                    h, w = crop.shape[:2]
                    tw = self._plate_canvas.winfo_width() or 200
                    scale = min(tw/max(w,1), 80/max(h,1))
                    nw2, nh2 = int(w*scale), int(h*scale)
                    rgb2 = cv2.cvtColor(
                        cv2.resize(crop,(nw2,nh2)), cv2.COLOR_BGR2RGB)
                    self._photo_plate = ImageTk.PhotoImage(
                        Image.fromarray(rgb2))
                    self._plate_canvas.delete("all")
                    self._plate_canvas.create_image(
                        tw//2, 40, image=self._photo_plate, anchor=tk.CENTER)
                except Exception:
                    pass
                self._plate_text_var.set(d.plate.normalised())
                break

    def _update_violations(self, dets: list):
        for d in dets:
            if d.has_violation() and d.plate:
                ts    = time.strftime("%H:%M:%S")
                conf  = f"{d.plate.confidence:.0%}"
                tag   = "viol"
                self._tree.insert("","0",
                    values=(ts, d.plate.normalised(), d.vehicle_class,
                            d.violation, conf, self.cfg.camera_id),
                    tags=(tag,))
                if len(self._tree.get_children()) > self._LOG_LIMIT:
                    self._tree.delete(self._tree.get_children()[-1])

    def _update_summary(self, dets: list):
        for d in dets:
            self._session_counts["total_vehicles"] += 1
            if d.plate:
                self._session_counts["total_plates"] += 1
            if d.violation == "No Helmet":
                self._session_counts["no_helmet"]      += 1
                self._session_counts["total_violations"] += 1
            elif d.violation == "No Seat Belt":
                self._session_counts["no_belt"]        += 1
                self._session_counts["total_violations"] += 1
        c = self._session_counts
        elapsed = int(time.time() - self._session_start)
        m, s    = divmod(elapsed, 60)
        h2, m2  = divmod(m, 60)
        self._summary_var.set(
            f"{h2:02d}:{m2:02d}:{s:02d}  |  "
            f"Vehicles:{c['total_vehicles']}  Plates:{c['total_plates']}  "
            f"Violations:{c['total_violations']} "
            f"(H:{c['no_helmet']} B:{c['no_belt']})")

    # ── Toolbar actions ───────────────────────────────────────────

    def _toggle_genai(self):
        self.cfg.use_genai = self._genai_var.get()
        self._status_var.set(
            f"GenAI {'ENABLED' if self.cfg.use_genai else 'DISABLED'}")

    def _update_conf(self, val):
        self.cfg.conf_thresh = float(val)

    def _open_file(self):
        filepath = filedialog.askopenfilename(
            title="Open Video or Image File",
            filetypes=[
                ("All supported","*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"),
                ("Video files",  "*.mp4 *.avi *.mov *.mkv"),
                ("Image files",  "*.jpg *.jpeg *.png *.bmp"),
                ("All files",    "*.*"),
            ])
        if not filepath:
            return
        if not os.path.exists(filepath):
            messagebox.showerror("Not Found", f"File not found:\n{filepath}")
            return
        self._status_var.set(f"Loading: {os.path.basename(filepath)} …")
        self.root.update()
        self._process_file(filepath)

    def _process_file(self, filepath: str):
        """Route image files to single-frame mode; video to live pipeline."""
        ext       = os.path.splitext(filepath)[1].lower()
        image_ext = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}

        if ext in image_ext:
            try:
                self._status_var.set("Processing image …")
                self.root.update()
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                self.cfg.use_genai = False
                self.pipeline = ANPRPipeline(self.cfg)
                annotated, dets = self.pipeline.process_single_image(filepath)

                cw = max(self._video_canvas.winfo_width(),  960)
                ch = max(self._video_canvas.winfo_height(), 540)
                h, w = annotated.shape[:2]
                scale = min(cw/max(w,1), ch/max(h,1))
                nw, nh = int(w*scale), int(h*scale)
                rgb = cv2.cvtColor(
                    cv2.resize(annotated,(nw,nh)), cv2.COLOR_BGR2RGB)
                self._photo_main = ImageTk.PhotoImage(Image.fromarray(rgb))
                self._video_canvas.delete("all")
                self._video_canvas.create_image(
                    cw//2, ch//2, image=self._photo_main, anchor=tk.CENTER)

                plates = sum(1 for d in dets if d.plate)
                viols  = sum(1 for d in dets if d.has_violation())
                self._metric_vars["vehicles"].set(str(len(dets)))
                self._metric_vars["plates"].set(str(plates))
                self._metric_vars["violations"].set(str(viols))
                self._metric_vars["fps"].set("—")

                for d in dets:
                    if d.plate:
                        self._plate_text_var.set(d.plate.normalised())
                        break

                for d in dets:
                    if d.plate:
                        ts  = time.strftime("%H:%M:%S")
                        tag = "viol" if d.has_violation() else "ok"
                        self._tree.insert("","0",
                            values=(ts, d.plate.normalised(),
                                    d.vehicle_class, d.violation,
                                    f"{d.plate.confidence:.0%}",
                                    self.cfg.camera_id),
                            tags=(tag,))

                fname = os.path.basename(filepath)
                self._status_var.set(
                    f"{fname}  |  Vehicles:{len(dets)} "
                    f"Plates:{plates} Violations:{viols}")
            except Exception as e:
                import traceback; traceback.print_exc()
                messagebox.showerror("Error", str(e))
                self._status_var.set(f"Error: {e}")
        else:
            # Video / RTSP
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.cfg.source = filepath
            self.pipeline   = ANPRPipeline(self.cfg)
            self.pipeline.start()
            self._running = True
            self._status_var.set(f"Playing: {os.path.basename(filepath)}")

    def _use_webcam(self):
        self._restart_pipeline(0)

    def _open_rtsp(self):
        win = tk.Toplevel(self.root)
        win.title("RTSP Stream URL")
        win.geometry("500x120")
        win.configure(bg=BG_DARK)
        tk.Label(win, text="RTSP URL:", bg=BG_DARK, fg=TEXT_LIGHT,
                 font=("Segoe UI",10)).pack(anchor=tk.W, padx=12, pady=8)
        entry = tk.Entry(win, width=55, bg=BG_PANEL, fg=TEXT_LIGHT,
                         font=("Courier",10))
        entry.insert(0, "rtsp://admin:password@192.168.1.100:554/stream1")
        entry.pack(padx=12)
        def _ok():
            url = entry.get().strip()
            if url:
                win.destroy()
                self._restart_pipeline(url)
        ttk.Button(win, text="Connect", command=_ok,
                   style="Run.TButton").pack(pady=8)

    def _restart_pipeline(self, source):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.cfg.source = source
        self.pipeline   = ANPRPipeline(self.cfg)
        self.pipeline.start()
        self._running = True
        self._status_var.set(f"Source: {source}")

    def _stop_pipeline(self):
        self._running = False
        self.pipeline.stop()
        self._status_var.set("Pipeline stopped.")

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")],
            initialfile="violations_export.csv")
        if not path:
            return
        records = self.pipeline.reporter.get_all()
        if not records:
            messagebox.showinfo("Export", "No violations to export.")
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=records[0].keys(),
                                   extrasaction="ignore")
                w.writeheader(); w.writerows(records)
            messagebox.showinfo("Export",
                                f"Saved {len(records)} records to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _on_close(self):
        self.pipeline.stop()
        self.root.destroy()
