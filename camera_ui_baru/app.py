"""
app.py -- Antarmuka kamera mandiri SecureVision (UI gelap).

Menjalankan dari root repository:
    python -m camera_ui_baru.app
    python -m camera_ui_baru.app --source 1
    python -m camera_ui_baru.app --source 0 --weights runs/detect/train/weights/best.pt
    python -m camera_ui_baru.app --backend dshow        # WindowsDirectShow
    python -m camera_ui_baru.app --no-auto             # tanpa pemicu otomatis

Fitur
-----
* preview live dari webcam/berkas video (pembacaan di thread background),
* deteksi YOLO **hanya 4 kelas target**: person, car, truck, motorbike,
* **snapshot JPG otomatis** saat target terdeteksi,
* **rekaman MP4 otomatis** (pre-roll 3 s + post-roll 4 s + rotasi berkas),
* kontrol manual: Start/Stop Camera, Snapshot, Rekam manual,
* log kejadian `runs/camera_ui_baru/events.csv`.
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, Optional

import cv2
from PIL import Image, ImageTk

from .camera import CameraReader
from .detector import COCO_CLASS_IDS, PER_CLASS_CONF, SecurityDetector
from .recorder import AutoRecorder, TriggerConfig

OUTPUT_ROOT = Path("runs/camera_ui_baru")

# hex warna per kelas untuk label (diambil dari CLASS_COLORS_BGR)
CLASS_COLORS_HEX: Dict[str, str] = {
    "person": "#20A840",
    "car": "#1A76E8",
    "truck": "#D63636",
    "motorbike": "#E05C18",
}


class CameraUI(tk.Tk):
    """Jendela kamera dengan kontrol sumber, deteksi, snapshot, dan rekam."""

    def __init__(self, source: str, weights: str, backend: str = "auto",
                 imgsz: int = 640, auto: bool = True) -> None:
        super().__init__()
        self.title("SecureVision - Camera Monitoring (DSEC ETS)")
        self.geometry("1220x760")
        self.minsize(1040, 660)
        self.configure(bg="#101827")

        self.source_var = tk.StringVar(value=source)
        self.status_var = tk.StringVar(value="Siap")
        self.object_var = tk.StringVar(value="Objek: 0")
        self.fps_var = tk.StringVar(value="FPS: -")
        self.auto_var = tk.BooleanVar(value=auto)
        self.count_vars: Dict[str, tk.StringVar] = {
            name: tk.StringVar(value="0") for name in COCO_CLASS_IDS}

        self.reader: Optional[CameraReader] = None
        self.current_frame = None
        self.video_writer = None
        self.recording = False
        self.started_at = 0.0

        self.detector = SecurityDetector(weights=weights, imgsz=imgsz)
        self._backend = backend
        self.recorder = AutoRecorder(OUTPUT_ROOT, TriggerConfig(), enabled=auto)

        self._build()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(30, self._tick)

    # ---------------------------------------------------------------- UI ---
    def _build(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=8)

        header = tk.Frame(self, bg="#172235", height=64)
        header.pack(fill="x")
        tk.Label(header, text="  SECUREVISION", fg="#E5EDFF", bg="#172235",
                 font=("Segoe UI", 18, "bold")).pack(side="left", padx=16, pady=14)
        tk.Label(header, text="Camera monitoring workspace", fg="#8EA2C9",
                 bg="#172235", font=("Segoe UI", 10)).pack(side="left")
        tk.Label(header, textvariable=self.status_var, fg="#71E6A4",
                 bg="#172235", font=("Segoe UI", 10, "bold")).pack(side="right",
                                                                   padx=18)

        body = tk.Frame(self, bg="#101827")
        body.pack(fill="both", expand=True, padx=16, pady=16)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.preview = tk.Label(
            body, text="Kamera belum aktif\nTekan Start Camera",
            bg="#050A12", fg="#8EA2C9", font=("Segoe UI", 18))
        self.preview.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        panel = tk.Frame(body, bg="#172235", width=258)
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_propagate(False)

        tk.Label(panel, text="CONTROL PANEL", bg="#172235", fg="#8EA2C9",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(18, 8))
        tk.Label(panel, text="Source (0 / 1 / video)", bg="#172235",
                 fg="#E5EDFF").pack(anchor="w", padx=18)
        tk.Entry(panel, textvariable=self.source_var, bg="#24334C", fg="white",
                 insertbackground="white", relief="flat").pack(fill="x", padx=18,
                                                               pady=(6, 12))
        ttk.Button(panel, text="Start Camera",
                   command=self.start_camera).pack(fill="x", padx=18, pady=4)
        ttk.Button(panel, text="Stop Camera",
                   command=self.stop_camera).pack(fill="x", padx=18, pady=4)
        ttk.Button(panel, text="Snapshot (manual)",
                   command=self.save_snapshot).pack(fill="x", padx=18, pady=4)
        ttk.Button(panel, text="Start / Stop Record (manual)",
                   command=self.toggle_recording).pack(fill="x", padx=18, pady=(4, 14))

        ttk.Checkbutton(panel, text="Auto Snapshot + Record",
                        variable=self.auto_var,
                        command=self._sync_auto).pack(anchor="w", padx=18, pady=(0, 14))

        tk.Label(panel, text="DETEKSI TARGET", bg="#172235", fg="#8EA2C9",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(6, 6))
        for name in ("person", "car", "truck", "motorbike"):
            row = tk.Frame(panel, bg="#172235")
            row.pack(fill="x", padx=18, pady=2)
            tk.Label(row, text="●", fg=CLASS_COLORS_HEX[name], bg="#172235",
                     font=("Segoe UI", 12)).pack(side="left")
            tk.Label(row, text=name, fg="#E5EDFF", bg="#172235",
                     font=("Segoe UI", 10)).pack(side="left", padx=6)
            tk.Label(row, textvariable=self.count_vars[name],
                     fg=CLASS_COLORS_HEX[name], bg="#172235",
                     font=("Segoe UI", 11, "bold")).pack(side="right")

        tk.Label(panel, textvariable=self.object_var, bg="#172235", fg="#71E6A4",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=18, pady=(20, 4))
        tk.Label(panel, textvariable=self.fps_var, bg="#172235",
                 fg="#E5EDFF").pack(anchor="w", padx=18, pady=4)

        tk.Label(panel, text="Status", bg="#172235", fg="#8EA2C9").pack(
            anchor="w", padx=18, pady=(20, 2))
        tk.Label(panel, text=self.detector.error or "YOLO siap / mode kamera",
                 wraplength=214, justify="left", bg="#172235",
                 fg="#F5C36B").pack(anchor="w", padx=18)
        tk.Label(panel, text=f"Output: {OUTPUT_ROOT}", wraplength=214,
                 justify="left", bg="#172235", fg="#8EA2C9",
                 font=("Segoe UI", 8)).pack(anchor="w", padx=18, pady=(14, 0))

    def _sync_auto(self) -> None:
        self.recorder.enabled = bool(self.auto_var.get())

    # ------------------------------------------------------------ kamera ---
    def start_camera(self) -> None:
        self.stop_camera()
        self.reader = CameraReader(self.source_var.get().strip() or "0",
                                   backend=self._backend)
        if self.reader.start():
            info = self.reader.info
            self.status_var.set(f"LIVE {info.width}x{info.height}")
        else:
            self.status_var.set("Gagal membuka kamera")
            self.reader = None
            messagebox.showerror("Kamera",
                                 "Kamera tidak dapat dibuka atau tidak "
                                 "mengirim frame.\n\nTutup aplikasi lain "
                                 "(Zoom/Teams/Meet) dan periksa izin kamera.")

    def stop_camera(self) -> None:
        if self.reader:
            self.reader.stop()
            self.reader = None
        self._stop_writer()
        self.recorder.close()
        self.status_var.set("Siap")

    def _tick(self) -> None:
        if self.reader:
            frame = self.reader.read()
            if frame is not None:
                detections = self.detector.detect(frame)
                SecurityDetector.draw(frame, detections)
                self.current_frame = frame

                counts = SecurityDetector.count_by_class(detections)
                for name, var in self.count_vars.items():
                    var.set(str(counts.get(name, 0)))
                self.object_var.set(f"Objek: {len(detections)}")
                self.fps_var.set(f"FPS: {self.reader.info.fps:.1f}")

                # ---- pemicu OTOMATIS: snapshot + rekam ----
                event = self.recorder.update(frame, detections,
                                             auto=self.auto_var.get())
                if event["started"]:
                    self.status_var.set("MEREKAM — target terdeteksi")
                if event["snapshot"] is not None:
                    self.status_var.set(
                        f"Snapshot: {Path(event['snapshot']).name}")

                if self.video_writer is not None:
                    self.video_writer.write(frame)
                self._show(frame)

        if self.recording:
            self.status_var.set(f"RECORDING {int(time.monotonic() - self.started_at)}s")
        elif self.recorder.recording:
            self.status_var.set(
                f"AUTO-REC {int(self.recorder.elapsed)}s "
                f"| klip {len(self.recorder.recordings)}")
        self.after(30, self._tick)

    def _show(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(320, self.preview.winfo_width() - 10)
        height = max(240, self.preview.winfo_height() - 10)
        image.thumbnail((width, height))
        photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=photo, text="")
        self.preview.image = photo

    # ------------------------------------------------------------ manual ---
    def save_snapshot(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo("Snapshot", "Aktifkan kamera terlebih dahulu.")
            return
        path = self.recorder._snapshot(self.current_frame)
        if path:
            self.status_var.set(f"Snapshot tersimpan: {path.name}")

    def toggle_recording(self) -> None:
        if self.recording:
            self._stop_writer()
            self.status_var.set("Rekaman selesai")
            return
        if self.current_frame is None:
            messagebox.showinfo("Rekaman", "Aktifkan kamera terlebih dahulu.")
            return
        output = OUTPUT_ROOT / "recordings"
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"manual_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        h, w = self.current_frame.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 20.0, (w, h))
        if not writer.isOpened():
            writer.release()
            messagebox.showerror("Rekaman", "Codec video tidak dapat dibuka.")
            return
        self.video_writer = writer
        self.recording = True
        self.started_at = time.monotonic()

    def _stop_writer(self) -> None:
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def close(self) -> None:
        self.stop_camera()
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="SecureVision Camera UI Baru")
    parser.add_argument("--source", default="0",
                        help="Indeks kamera atau path video")
    parser.add_argument("--weights", default="yolo11n.pt",
                        help="Bobot YOLO; kosongkan untuk mode kamera saja")
    parser.add_argument("--backend", default="auto",
                        help="auto | dshow | msmf (Windows)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-auto", action="store_true",
                        help="matikan pemicu snapshot/rekam otomatis")
    args = parser.parse_args()
    CameraUI(args.source, args.weights, backend=args.backend,
             imgsz=args.imgsz, auto=not args.no_auto).mainloop()


if __name__ == "__main__":
    main()
