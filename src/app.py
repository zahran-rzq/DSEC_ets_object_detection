"""
app.py -- Aplikasi desktop Security Camera (light theme).

Menjalankan:
    python -m src.app                 # pakai webcam default (index 0)
    python -m src.app --source 1      # webcam kedua
    python -m src.app --source video.mp4
    python -m src.app --weights runs/detect/train/weights/best.pt

Dependensi:
    pip install ultralytics opencv-python pillow customtkinter
    # Linux butuh: sudo apt install python3-tk

Antarmuka (light theme) tersusun atas 4 area utama:
    A. Top bar    : identitas aplikasi, indikator LIVE, jam
    B. Kiri       : feed video real-time + banner status / REC
    C. Kanan      : panel status deteksi (per-kelas), FPS, model, log kejadian
    D. Bawah      : tombol kontrol + slider ambang confidence/IoU + preset
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import (  # noqa: E402
    AppConfig, CLASS_COLORS_HEX, COCO_CLASS_IDS, preset)
from src.core.pipeline import State, SurveillancePipeline  # noqa: E402
from src.core.camera import CameraThread, probe_cameras   # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    raise SystemExit("OpenCV belum terpasang: pip install opencv-python")

try:
    import customtkinter as ctk
    from PIL import Image
except ImportError:  # pragma: no cover
    raise SystemExit(
        "Modul UI belum terpasang. Jalankan:\n"
        "    pip install customtkinter pillow\n"
        "dan pastikan python3-tk tersedia (Linux: sudo apt install python3-tk).")


# --------------------------------------------------------------------------
# Palet warna -- LIGHT THEME
# --------------------------------------------------------------------------
PALETTE = {
    "bg": "#F4F6FA",           # latar utama
    "panel": "#FFFFFF",        # kartu / panel
    "border": "#DCE1EA",
    "text": "#1B2430",
    "text_muted": "#6B7688",
    "accent": "#2563EB",       # biru primer
    "accent_soft": "#E7EEFD",
    "ok": "#16A34A",
    "warn": "#D97706",
    "danger": "#DC2626",
    "idle": "#94A3B8",
}


# ==========================================================================
class App(ctk.CTk):
    """Jendela utama aplikasi."""

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self.title("SecureVision - DSEC Security Camera Application")
        self.geometry("1360x820")
        self.minsize(1120, 700)
        self.configure(fg_color=PALETTE["bg"])

        ctk.set_appearance_mode(cfg.appearance)      # wajib: "light"
        ctk.set_default_color_theme("blue")
        try:
            ctk.set_widget_scaling(cfg.ui_scale)
        except Exception:
            pass

        self.pipeline: Optional[SurveillancePipeline] = None
        self.camera: Optional[CameraThread] = None
        self.running = False
        self._start_job: Optional[dict] = None     # hasil buka kamera (dari thread)
        self._probe_pending = None                 # hasil pemindaian kamera
        self._probe_running = False
        self._last_frame: Optional[np.ndarray] = None
        self._rec_start = 0.0
        self._cls_vars: Dict[str, ctk.BooleanVar] = {}
        self._count_labels: Dict[str, ctk.CTkLabel] = {}
        self._count_bars: Dict[str, ctk.CTkProgressBar] = {}
        self._blink = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(60, self._ui_loop)

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_topbar()
        self._build_main()
        self._build_bottombar()

    def _card(self, parent, **kw) -> ctk.CTkFrame:
        return ctk.CTkFrame(parent, fg_color=PALETTE["panel"],
                            border_color=PALETTE["border"], border_width=1,
                            corner_radius=12, **kw)

    def _build_topbar(self):
        bar = ctk.CTkFrame(self, fg_color=PALETTE["panel"], corner_radius=0,
                           border_color=PALETTE["border"], border_width=1,
                           height=64)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_propagate(False)

        ctk.CTkLabel(bar, text="  🛡  SecureVision",
                     font=ctk.CTkFont("Segoe UI", 20, "bold"),
                     text_color=PALETTE["text"]).pack(side="left", padx=(18, 6))
        ctk.CTkLabel(bar, text="DSEC · Security Camera Application",
                     font=ctk.CTkFont("Segoe UI", 12),
                     text_color=PALETTE["text_muted"]).pack(side="left")

        self.clock_lbl = ctk.CTkLabel(bar, text="--:--:--",
                                      font=ctk.CTkFont("Consolas", 15, "bold"),
                                      text_color=PALETTE["text"])
        self.clock_lbl.pack(side="right", padx=18)
        self.live_badge = ctk.CTkLabel(bar, text="● OFFLINE",
                                       font=ctk.CTkFont("Segoe UI", 13, "bold"),
                                       text_color=PALETTE["idle"])
        self.live_badge.pack(side="right", padx=6)

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=16, pady=(14, 8))
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # ---------------- KIRI: feed video ----------------
        left = self._card(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.video_lbl = ctk.CTkLabel(
            left, text="Kamera belum dijalankan\n\nKlik  ▶  Start Camera",
            font=ctk.CTkFont("Segoe UI", 16), text_color=PALETTE["text_muted"],
            fg_color="#111827", corner_radius=10, height=560)
        self.video_lbl.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        # banner status di bawah feed
        self.status_banner = ctk.CTkFrame(left, fg_color=PALETTE["accent_soft"],
                                          corner_radius=8, height=34)
        self.status_banner.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        self.status_banner.grid_propagate(False)
        self.rec_dot = ctk.CTkLabel(self.status_banner, text="●",
                                    font=ctk.CTkFont("Segoe UI", 16, "bold"),
                                    text_color=PALETTE["idle"])
        self.rec_dot.pack(side="left", padx=(14, 4))
        self.status_text = ctk.CTkLabel(self.status_banner,
                                        text="STANDBY — menunggu kamera",
                                        font=ctk.CTkFont("Segoe UI", 13, "bold"),
                                        text_color=PALETTE["text_muted"])
        self.status_text.pack(side="left")
        self.rec_timer = ctk.CTkLabel(self.status_banner, text="",
                                      font=ctk.CTkFont("Consolas", 13, "bold"),
                                      text_color=PALETTE["danger"])
        self.rec_timer.pack(side="right", padx=14)

        # ---------------- KANAN: panel status ----------------
        right = ctk.CTkFrame(main, fg_color="transparent", width=340)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)
        right.grid_rowconfigure(3, weight=1)

        # (1) Status sistem
        c1 = self._card(right)
        c1.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ctk.CTkLabel(c1, text="STATUS SISTEM",
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color=PALETTE["text_muted"]).pack(anchor="w", padx=14, pady=(12, 6))

        # --- pemilih sumber kamera ---
        sel = ctk.CTkFrame(c1, fg_color="transparent")
        sel.pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkLabel(sel, text="Sumber kamera", font=ctk.CTkFont("Segoe UI", 12),
                     text_color=PALETTE["text_muted"]).pack(side="left")
        self.cam_var = ctk.StringVar(value=str(self.cfg.camera.source))
        self.cam_menu = ctk.CTkOptionMenu(
            sel, values=["0", "1", "2", "3", "4"], variable=self.cam_var,
            width=72, font=ctk.CTkFont("Segoe UI", 12),
            command=self._on_camera_change)
        self.cam_menu.pack(side="right")
        self.btn_probe = ctk.CTkButton(
            sel, text="Tes", width=52, height=26,
            font=ctk.CTkFont("Segoe UI", 11),
            fg_color="#E2E8F0", text_color=PALETTE["text"], hover_color="#CBD5E1",
            command=self.probe_camera)
        self.btn_probe.pack(side="right", padx=(0, 6))

        self.info = {}
        for key, val in [("Model", Path(self.cfg.model.weights).name),
                         ("Kelas aktif", ", ".join(self.cfg.model.names)),
                         ("Resolusi", "-"),
                         ("Backend", "-"),
                         ("FPS", "0.0"), ("Inferensi", "0 ms"),
                         ("Total event", "0")]:
            row = ctk.CTkFrame(c1, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=1)
            ctk.CTkLabel(row, text=key, font=ctk.CTkFont("Segoe UI", 12),
                         text_color=PALETTE["text_muted"]).pack(side="left")
            lbl = ctk.CTkLabel(row, text=val, font=ctk.CTkFont("Segoe UI", 12, "bold"),
                               text_color=PALETTE["text"])
            lbl.pack(side="right")
            self.info[key] = lbl
        ctk.CTkFrame(c1, height=10, fg_color="transparent").pack()

        # (2) Deteksi per kelas
        c2 = self._card(right)
        c2.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        head = ctk.CTkFrame(c2, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(12, 4))
        ctk.CTkLabel(head, text="DETEKSI TARGET",
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color=PALETTE["text_muted"]).pack(side="left")
        self.total_lbl = ctk.CTkLabel(head, text="0 objek",
                                      font=ctk.CTkFont("Segoe UI", 12, "bold"),
                                      text_color=PALETTE["accent"])
        self.total_lbl.pack(side="right")

        for name in ["person", "car", "truck", "motorbike"]:
            row = ctk.CTkFrame(c2, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=3)
            var = ctk.BooleanVar(value=True)
            self._cls_vars[name] = var
            cb = ctk.CTkCheckBox(row, text=name.capitalize(), variable=var,
                                 width=104, font=ctk.CTkFont("Segoe UI", 12),
                                 text_color=PALETTE["text"],
                                 command=self._on_class_toggle)
            cb.pack(side="left")
            bar = ctk.CTkProgressBar(row, width=90, height=8,
                                     progress_color=CLASS_COLORS_HEX[name])
            bar.set(0)
            bar.pack(side="left", padx=(6, 8))
            cnt = ctk.CTkLabel(row, text="0", width=26,
                               font=ctk.CTkFont("Segoe UI", 13, "bold"),
                               text_color=CLASS_COLORS_HEX[name])
            cnt.pack(side="left")
            self._count_bars[name] = bar
            self._count_labels[name] = cnt
        ctk.CTkFrame(c2, height=10, fg_color="transparent").pack()

        # (3) Output
        c3 = self._card(right)
        c3.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        ctk.CTkLabel(c3, text="PENYIMPANAN",
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color=PALETTE["text_muted"]).pack(anchor="w", padx=14, pady=(12, 6))
        self.out_lbl = ctk.CTkLabel(c3, text=str(Path(self.cfg.recording.output_dir).resolve()),
                                    font=ctk.CTkFont("Segoe UI", 10),
                                    text_color=PALETTE["text_muted"], wraplength=300,
                                    justify="left")
        self.out_lbl.pack(anchor="w", padx=14)
        self.files_lbl = ctk.CTkLabel(c3, text="Snapshot 0  ·  Klip 0",
                                      font=ctk.CTkFont("Segoe UI", 12, "bold"),
                                      text_color=PALETTE["text"])
        self.files_lbl.pack(anchor="w", padx=14, pady=(6, 12))

        # (4) Log
        c4 = self._card(right)
        c4.grid(row=3, column=0, sticky="nsew")
        ctk.CTkLabel(c4, text="LOG KEJADIAN",
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color=PALETTE["text_muted"]).pack(anchor="w", padx=14, pady=(12, 4))
        self.log_box = ctk.CTkTextbox(c4, font=ctk.CTkFont("Consolas", 11),
                                      fg_color="#FBFCFE", text_color=PALETTE["text"],
                                      border_color=PALETTE["border"], border_width=1,
                                      corner_radius=8)
        self.log_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log_box.configure(state="disabled")

    def _build_bottombar(self):
        bar = self._card(self, height=76)
        bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        bar.grid_propagate(False)

        # tombol utama
        self.btn_start = ctk.CTkButton(bar, text="▶  Start Camera", width=150, height=38,
                                       font=ctk.CTkFont("Segoe UI", 13, "bold"),
                                       fg_color=PALETTE["accent"], hover_color="#1D4ED8",
                                       command=self.toggle_camera)
        self.btn_start.pack(side="left", padx=(14, 8), pady=18)

        self.btn_snap = ctk.CTkButton(bar, text="📷  Snapshot", width=120, height=38,
                                      font=ctk.CTkFont("Segoe UI", 13),
                                      fg_color="#0F766E", hover_color="#115E59",
                                      state="disabled", command=self.manual_snapshot)
        self.btn_snap.pack(side="left", padx=4, pady=18)

        self.auto_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(bar, text="Auto-Record", variable=self.auto_var,
                      font=ctk.CTkFont("Segoe UI", 12, "bold"),
                      text_color=PALETTE["text"],
                      progress_color=PALETTE["danger"],
                      command=self._on_auto_record).pack(side="left", padx=14)

        ctk.CTkFrame(bar, width=1, height=40, fg_color=PALETTE["border"]).pack(side="left", padx=6)

        # slider conf / iou
        self._slider(bar, "Confidence", 0.05, 0.90, self.cfg.model.conf,
                     self._on_conf, "conf")
        self._slider(bar, "IoU (NMS)", 0.10, 0.90, self.cfg.model.iou,
                     self._on_iou, "iou")

        ctk.CTkFrame(bar, width=1, height=40, fg_color=PALETTE["border"]).pack(side="left", padx=6)

        ctk.CTkLabel(bar, text="Preset:", font=ctk.CTkFont("Segoe UI", 12),
                     text_color=PALETTE["text_muted"]).pack(side="left")
        self.preset_menu = ctk.CTkOptionMenu(
            bar, values=["balanced", "max_recall", "low_false_alarm", "fast"],
            width=150, font=ctk.CTkFont("Segoe UI", 12),
            command=self._on_preset)
        self.preset_menu.set("balanced")
        self.preset_menu.pack(side="left", padx=6)

        ctk.CTkButton(bar, text="📁 Folder Output", width=130, height=38,
                      font=ctk.CTkFont("Segoe UI", 12),
                      fg_color="#E2E8F0", text_color=PALETTE["text"],
                      hover_color="#CBD5E1",
                      command=self.open_output).pack(side="right", padx=(8, 14))

    def _slider(self, parent, label, lo, hi, val, cmd, tag):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(side="left", padx=8)
        v = ctk.CTkLabel(f, text=f"{label} {val:.2f}",
                         font=ctk.CTkFont("Segoe UI", 11, "bold"),
                         text_color=PALETTE["text"])
        v.pack(anchor="w")
        s = ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=int((hi - lo) / 0.05),
                          width=130, height=16, command=lambda x, t=tag: cmd(x, t))
        s.set(val)
        s.pack()
        setattr(self, f"_{tag}_slider", s)
        setattr(self, f"_{tag}_label", v)

    # ------------------------------------------------------------- event ---
    def _on_conf(self, value, tag="conf"):
        self.cfg.model.conf = float(value)
        for k in self.cfg.model.per_class_conf:
            self.cfg.model.per_class_conf[k] = float(value)
        if self.pipeline:
            self.pipeline.detector.cfg.conf = float(value)
            self.pipeline.detector.cfg.per_class_conf = dict(self.cfg.model.per_class_conf)
        self._conf_label.configure(text=f"Confidence {float(value):.2f}")

    def _on_iou(self, value, tag="iou"):
        self.cfg.model.iou = float(value)
        if self.pipeline:
            self.pipeline.detector.cfg.iou = float(value)
        self._iou_label.configure(text=f"IoU (NMS) {float(value):.2f}")

    def _on_class_toggle(self):
        active = [n for n, v in self._cls_vars.items() if v.get()]
        if not active:
            # cegah keadaan tanpa kelas sama sekali
            self._cls_vars["person"].set(True)
            active = ["person"]
        self.cfg.model.target_classes = active
        if self.pipeline:
            det = self.pipeline.detector
            det.names = det.__class__ and det.cfg.names
            det.cfg.target_classes = active
            det.class_ids = det.cfg.class_ids
            det.id_to_name = {COCO_CLASS_IDS[n]: n for n in det.cfg.names}
        self.info["Kelas aktif"].configure(text=", ".join(active))

    def _on_auto_record(self):
        on = self.auto_var.get()
        self.cfg.recording.save_video = on
        self._log(f"Auto-Record {'AKTIF' if on else 'NONAKTIF'}")

    def _on_preset(self, name):
        keep = self.running
        if keep:
            self.toggle_camera()
        base = preset(name)
        base.model.weights = self.cfg.model.weights
        base.camera.source = self.cfg.camera.source
        base.appearance = "light"
        self.cfg = base
        self._log(f"Preset diterapkan: {name}")
        self.info["Model"].configure(text=Path(self.cfg.model.weights).name)
        self._conf_slider.set(self.cfg.model.conf)
        self._conf_label.configure(text=f"Confidence {self.cfg.model.conf:.2f}")
        self._iou_slider.set(self.cfg.model.iou)
        self._iou_label.configure(text=f"IoU (NMS) {self.cfg.model.iou:.2f}")

    # ---------------------------------------------------------- kamera ---
    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        elif not self._start_job:
            self.start_camera()

    def _on_camera_change(self, value):
        """Ganti sumber kamera; mulai ulang bila sedang berjalan."""
        self.cfg.camera.source = str(value)
        was_running = self.running
        if was_running:
            self.stop_camera()
        self._log(f"🎥 Sumber kamera: {value}")
        if was_running:
            self.after(300, self.start_camera)

    def probe_camera(self):
        """Pindai indeks kamera 0-4 dan laporkan yang benar-benar mengirim frame."""
        if self._probe_running:
            return
        self._probe_running = True
        self.btn_probe.configure(state="disabled", text="...")
        self._log("🔍 Memindai kamera 0-4 ...")

        def job():
            try:
                res = probe_cameras(max_index=5)
            except Exception as e:
                self._log(f"❌ Pemindaian gagal: {e}")
                res = []
            self._probe_pending = res

        threading.Thread(target=job, daemon=True).start()

    def _apply_probe(self, res: list):
        """Terapkan hasil pemindaian (dipanggil dari main thread)."""
        self._probe_running = False
        self.btn_probe.configure(state="normal", text="Tes")
        if not res:
            self._log("❌ Tidak ada kamera yang mengirim frame.")
            self._log("   → Tutup Zoom/Teams/Meet, cek izin kamera, lalu tes ulang.")
            self.status_text.configure(
                text="Kamera tidak mengirim frame — cek aplikasi lain / izin kamera",
                text_color=PALETTE["danger"])
            return
        vals = [str(f["index"]) for f in res]
        self.cam_menu.configure(values=vals)
        for f in res:
            self._log(f"✅ kamera {f['index']}: {f['backend']} "
                      f"{f['width']}×{f['height']} @ {f['fps']:.0f} fps")
        self.cam_var.set(vals[0])
        self.cfg.camera.source = vals[0]
        self._log(f"➡ Kamera {vals[0]} dipilih. Klik 'Start Camera'.")

    def start_camera(self):
        """Muat model + buka kamera di thread terpisah agar GUI tidak beku."""
        if self.running or self._start_job:
            return
        self._start_job = {"state": "running"}
        self.btn_start.configure(state="disabled", text="⏳  Membuka ...")
        self.status_text.configure(text="Memuat model & membuka kamera ...",
                                   text_color=PALETTE["warn"])

        def job():
            try:
                if self.pipeline is None:
                    self.pipeline = SurveillancePipeline(
                        self.cfg, on_event=self._on_pipeline_event)
            except Exception as e:
                self._start_job = {"state": "error",
                                   "msg": f"Gagal memuat model/bobot: {e}"}
                return
            cam = CameraThread(
                self.cfg.camera.source, self.cfg.camera.width,
                self.cfg.camera.height, self.cfg.camera.fps,
                backend=getattr(self.cfg.camera, "backend", "auto"),
                flip=self.cfg.camera.flip_horizontal)
            if not cam.open():
                self._start_job = {"state": "error",
                                   "msg": cam.status.error or "Kamera gagal dibuka."}
                return
            self._start_job = {"state": "ok", "cam": cam, "status": cam.status}

        threading.Thread(target=job, daemon=True).start()

    def _finish_start(self, job: dict):
        self.btn_start.configure(state="normal")
        if job["state"] == "error":
            msg = job["msg"]
            self.btn_start.configure(text="▶  Start Camera",
                                     fg_color=PALETTE["accent"],
                                     hover_color="#1D4ED8")
            self._log("❌ " + msg.replace("\n", " "))
            self.status_text.configure(text=msg.splitlines()[0],
                                       text_color=PALETTE["danger"])
            return
        cam, st = job["cam"], job["status"]
        self.camera = cam
        self.running = True
        cam.running.set()
        cam.start()
        self.btn_start.configure(text="■  Stop Camera", fg_color=PALETTE["danger"],
                                 hover_color="#B91C1C")
        self.btn_snap.configure(state="normal")
        self.live_badge.configure(text="● LIVE", text_color=PALETTE["ok"])
        self.info["Resolusi"].configure(text=f"{st.width}×{st.height}")
        self.info["Backend"].configure(text=f"{st.backend} · {st.fps:.0f} fps")
        self._log(f"▶ Kamera {self.cfg.camera.source} aktif via {st.backend} "
                  f"({st.width}×{st.height})")

    def stop_camera(self):
        self.running = False
        self._start_job = None
        if self.camera:
            self.camera.stop()
            self.camera = None
        if self.pipeline:
            self.pipeline.close()
        self.btn_start.configure(state="normal", text="▶  Start Camera",
                                 fg_color=PALETTE["accent"], hover_color="#1D4ED8")
        self.btn_snap.configure(state="disabled")
        self.live_badge.configure(text="● OFFLINE", text_color=PALETTE["idle"])
        self.status_text.configure(text="STANDBY — kamera dihentikan",
                                   text_color=PALETTE["text_muted"])
        self.rec_dot.configure(text_color=PALETTE["idle"])
        self.rec_timer.configure(text="")
        self._log("■ Kamera dihentikan")

    def manual_snapshot(self):
        if self._last_frame is None or not self.pipeline:
            return
        p = self.pipeline.snapper.save(self._last_frame, tag="manual")
        if p:
            self._log(f"📷 Snapshot manual: {p.name}")

    def open_output(self):
        path = Path(self.cfg.recording.output_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            self._log(f"Tidak bisa membuka folder: {e}")

    def on_close(self):
        self.stop_camera()
        self.destroy()

    # ----------------------------------------------------------- loop -----
    def _ui_loop(self):
        """Dipanggil tiap ~60 ms dari main thread Tk."""
        self.clock_lbl.configure(text=datetime.now().strftime("%H:%M:%S"))

        # hasil pemindaian kamera (dikerjakan di thread -> diterapkan di sini)
        if self._probe_pending is not None:
            res, self._probe_pending = self._probe_pending, None
            self._apply_probe(res)

        # hasil buka kamera (model + backend probing berjalan di thread)
        job = self._start_job
        if job and job.get("state") != "running":
            self._start_job = None
            self._finish_start(job)

        if self.running and self.camera and self.pipeline:
            frame = self.camera.read()
            if frame is not None:
                self._last_frame = frame
                try:
                    out = self.pipeline.process(frame, draw=True)
                    self._render(out["frame"])
                    self._update_status(out)
                except Exception as e:
                    self._log(f"⚠ {e}")
            elif self.camera.status.stale:
                # kamera terbuka (LED menyala) tapi tidak mengirim frame
                self.status_text.configure(
                    text="KAMERA TIDAK MENGIRIM FRAME — mencoba menyambung ulang...",
                    text_color=PALETTE["danger"])
            if (self.camera and self.camera.status.frames == 30
                    and 0 <= self.camera.status.mean_brightness < 6):
                self._log("⚠️ Frame tampak hitam — lensa tertutup, atau kamera "
                          "sedang dipakai aplikasi lain (Zoom/Teams/Meet).")

        # kedipan indikator REC
        self._blink = not self._blink
        if self.pipeline and self.pipeline.state == State.RECORDING:
            self.rec_dot.configure(
                text_color=PALETTE["danger"] if self._blink else "#FCA5A5")
            dur = time.time() - self._rec_start
            self.rec_timer.configure(text=f"REC {int(dur // 60):02d}:{int(dur % 60):02d}")

        self.after(50, self._ui_loop)

    def _render(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        lbl_w = max(320, self.video_lbl.winfo_width() - 24)
        lbl_h = max(240, self.video_lbl.winfo_height() - 24)
        scale = min(lbl_w / w, lbl_h / h)
        if scale <= 0 or scale > 4:
            return
        img = cv2.resize(frame, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ctk_img = ctk.CTkImage(Image.fromarray(img),
                               size=(img.shape[1], img.shape[0]))
        self.video_lbl.configure(image=ctk_img, text="")
        self.video_lbl.image = ctk_img          # jaga referensi (cegah GC)

    def _update_status(self, out: dict):
        st: State = out["state"]
        res = out["result"]
        p = self.pipeline

        # status text
        if st == State.RECORDING:
            self.status_text.configure(text="MEREKAM — target terdeteksi",
                                       text_color=PALETTE["danger"])
        elif st == State.DETECTING:
            self.status_text.configure(text="MENDETEKSI — menunggu konfirmasi",
                                       text_color=PALETTE["warn"])
        elif st == State.COOLDOWN:
            self.status_text.configure(text="COOLDOWN — jeda antar event",
                                       text_color=PALETTE["text_muted"])
        else:
            self.status_text.configure(text="AMAN — tidak ada target",
                                       text_color=PALETTE["ok"])

        if out["event_started"]:
            self._rec_start = time.time()

        # counter per kelas
        total = 0
        for name, lbl in self._count_labels.items():
            n = res.counts.get(name, 0)
            total += n
            lbl.configure(text=str(n))
            self._count_bars[name].set(min(1.0, n / 5.0))
        self.total_lbl.configure(text=f"{total} objek")

        # info panel
        self.info["FPS"].configure(text=f"{p.stats.fps_ema:.1f}")
        self.info["Inferensi"].configure(text=f"{p.stats.last_inference_ms:.0f} ms")
        self.info["Total event"].configure(text=str(p.stats.events))
        self.files_lbl.configure(
            text=f"Snapshot {len(p.snapper.history)}  ·  Klip {len(p.recorder.history)}")

    def _on_pipeline_event(self, event: str, data: dict):
        icons = {"DETECTED": "👁", "REC_START": "⏺", "REC_STOP": "⏹",
                 "SNAPSHOT": "📷"}
        cls = data.get("classes", "")
        extra = data.get("file", "")
        self._log(f"{icons.get(event, '•')} {event:<10} {cls:<24} {extra}")

    def _log(self, line: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{stamp}] {line}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")


# ==========================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="SecureVision - Security Camera App")
    ap.add_argument("--source", default="0",
                    help="indeks webcam (0/1/2) atau path video/RTSP")
    ap.add_argument("--weights", default="yolo11n.pt",
                    help="bobot YOLO pre-trained (yolo11n.pt / best.pt)")
    ap.add_argument("--output", default="runs/capture", help="folder hasil tangkapan")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--preset", default="balanced",
                    choices=["balanced", "max_recall", "low_false_alarm", "fast"])
    ap.add_argument("--device", default="", help="''=auto, 'cpu', '0'")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = preset(args.preset)
    cfg.model.weights = args.weights
    cfg.model.imgsz = args.imgsz
    cfg.model.device = args.device
    cfg.camera.source = args.source
    cfg.recording.output_dir = args.output
    cfg.appearance = "light"
    App(cfg).mainloop()


if __name__ == "__main__":
    main()
