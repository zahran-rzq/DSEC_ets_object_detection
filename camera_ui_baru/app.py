"""UI kamera mandiri untuk SecureVision.

Jalankan dari root project:
    python -m camera_ui_baru.app
    python -m camera_ui_baru.app --source 1 --weights yolo11n.pt
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from .camera import CameraReader


class ObjectDetector:
    """Detektor YOLO opsional; UI tetap dapat berjalan dalam mode kamera saja."""

    def __init__(self, weights: str) -> None:
        self.model = None
        self.error = ""
        if not weights:
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(weights)
        except Exception as exc:
            self.error = f"Deteksi nonaktif: {exc}"

    def draw(self, frame):
        if self.model is None:
            return frame, 0
        try:
            result = self.model.predict(frame, conf=0.35, verbose=False)[0]
            count = 0
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                name = result.names.get(class_id, str(class_id))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 180, 90), 2)
                cv2.putText(
                    frame,
                    f"{name} {confidence:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (36, 180, 90),
                    2,
                )
                count += 1
            return frame, count
        except Exception as exc:
            self.error = f"Inferensi gagal: {exc}"
            return frame, 0


class CameraUI(tk.Tk):
    """Jendela kamera dengan kontrol sumber, deteksi, snapshot, dan rekam."""

    def __init__(self, source: str, weights: str) -> None:
        super().__init__()
        self.title("SecureVision - Camera UI Baru")
        self.geometry("1180x760")
        self.minsize(900, 600)
        self.configure(bg="#101827")

        self.reader: Optional[CameraReader] = None
        self.detector = ObjectDetector(weights)
        self.current_frame = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        self.started_at = 0.0
        self.source_var = tk.StringVar(value=source)
        self.status_var = tk.StringVar(value="Siap")
        self.object_var = tk.StringVar(value="Objek: 0")
        self.fps_var = tk.StringVar(value="FPS: -")

        self._build()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(30, self._tick)

    def _build(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=8)

        header = tk.Frame(self, bg="#172235", height=64)
        header.pack(fill="x")
        tk.Label(
            header, text="  SECUREVISION", fg="#E5EDFF", bg="#172235",
            font=("Segoe UI", 18, "bold")
        ).pack(side="left", padx=16, pady=14)
        tk.Label(
            header, text="Camera monitoring workspace", fg="#8EA2C9", bg="#172235",
            font=("Segoe UI", 10)
        ).pack(side="left")
        tk.Label(
            header, textvariable=self.status_var, fg="#71E6A4", bg="#172235",
            font=("Segoe UI", 10, "bold")
        ).pack(side="right", padx=18)

        body = tk.Frame(self, bg="#101827")
        body.pack(fill="both", expand=True, padx=16, pady=16)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.preview = tk.Label(
            body, text="Kamera belum aktif\nTekan Start Camera",
            bg="#050A12", fg="#8EA2C9", font=("Segoe UI", 18)
        )
        self.preview.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        panel = tk.Frame(body, bg="#172235", width=250)
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_propagate(False)
        tk.Label(panel, text="CONTROL PANEL", bg="#172235", fg="#8EA2C9",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(20, 8))
        tk.Label(panel, text="Source (0 / 1 / video)", bg="#172235", fg="#E5EDFF").pack(
            anchor="w", padx=18
        )
        tk.Entry(panel, textvariable=self.source_var, bg="#24334C", fg="white",
                 insertbackground="white", relief="flat").pack(fill="x", padx=18, pady=(6, 14))
        ttk.Button(panel, text="Start Camera", command=self.start_camera).pack(
            fill="x", padx=18, pady=4
        )
        ttk.Button(panel, text="Stop Camera", command=self.stop_camera).pack(
            fill="x", padx=18, pady=4
        )
        ttk.Button(panel, text="Snapshot", command=self.save_snapshot).pack(
            fill="x", padx=18, pady=(4, 18)
        )
        ttk.Button(panel, text="Start / Stop Record", command=self.toggle_recording).pack(
            fill="x", padx=18, pady=4
        )
        tk.Label(panel, textvariable=self.object_var, bg="#172235", fg="#71E6A4",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=18, pady=(25, 4))
        tk.Label(panel, textvariable=self.fps_var, bg="#172235", fg="#E5EDFF").pack(
            anchor="w", padx=18, pady=4
        )
        tk.Label(panel, text="Status", bg="#172235", fg="#8EA2C9").pack(
            anchor="w", padx=18, pady=(25, 2)
        )
        tk.Label(panel, text=self.detector.error or "YOLO siap / mode kamera",
                 wraplength=210, justify="left", bg="#172235", fg="#F5C36B").pack(
                     anchor="w", padx=18
                 )

    def start_camera(self) -> None:
        self.stop_camera()
        self.reader = CameraReader(self.source_var.get().strip() or "0")
        if self.reader.start():
            self.status_var.set(f"LIVE {self.reader.info.width}x{self.reader.info.height}")
        else:
            self.status_var.set("Gagal membuka kamera")
            messagebox.showerror("Kamera", self.reader.info.error)

    def stop_camera(self) -> None:
        if self.reader:
            self.reader.stop()
            self.reader = None
        self._stop_writer()
        self.status_var.set("Siap")

    def _tick(self) -> None:
        if self.reader:
            frame = self.reader.read()
            if frame is not None:
                self.current_frame, count = self.detector.draw(frame)
                self.object_var.set(f"Objek: {count}")
                self.fps_var.set(f"FPS: {self.reader.info.fps:.1f}")
                self._show(self.current_frame)
                if self.video_writer:
                    self.video_writer.write(self.current_frame)
        if self.recording:
            self.status_var.set(f"RECORDING {int(time.monotonic() - self.started_at)}s")
        self.after(30, self._tick)

    def _show(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((self.preview.winfo_width() - 10, self.preview.winfo_height() - 10))
        photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=photo, text="")
        self.preview.image = photo

    def save_snapshot(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo("Snapshot", "Aktifkan kamera terlebih dahulu.")
            return
        output = Path("runs/camera_ui_baru/snapshots")
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), self.current_frame)
        self.status_var.set(f"Snapshot tersimpan: {path.name}")

    def toggle_recording(self) -> None:
        if self.recording:
            self._stop_writer()
            self.status_var.set("Rekaman selesai")
            return
        if self.reader is None or self.current_frame is None:
            messagebox.showinfo("Rekaman", "Aktifkan kamera terlebih dahulu.")
            return
        output = Path("runs/camera_ui_baru/recordings")
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        h, w = self.current_frame.shape[:2]
        self.video_writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h)
        )
        if not self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
            messagebox.showerror("Rekaman", "Codec video tidak dapat dibuka.")
            return
        self.recording = True
        self.started_at = time.monotonic()

    def _stop_writer(self) -> None:
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def close(self) -> None:
        self.stop_camera()
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="SecureVision Camera UI Baru")
    parser.add_argument("--source", default="0", help="Indeks kamera atau path video")
    parser.add_argument("--weights", default="yolo11n.pt", help="Bobot YOLO; kosongkan untuk mode kamera")
    args = parser.parse_args()
    CameraUI(args.source, args.weights).mainloop()


if __name__ == "__main__":
    main()
