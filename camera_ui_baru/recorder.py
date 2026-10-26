"""
recorder.py -- Pemicu otomatis snapshot & rekaman video untuk UI kamera mandiri.

Ini menjawab bagian inti Soal 2: "sistem diwajibkan memiliki fungsionalitas
untuk menyimpan gambar serta merekam video secara OTOMATIS ketika objek
terdeteksi."

Logika kondisional (state machine)
----------------------------------
    streak  = jumlah frame berturut-turut yang berisi target
    jika  streak >= min_frames  DAN  sudah lewat cooldown  DAN  belum merekam:
            -> simpan snapshot JPG
            -> mulai rekam MP4 (diawali flush pre-roll)
    jika  sedang merekam dan tidak ada target > post_roll detik:
            -> hentikan rekaman, mulai cooldown
    jika  durasi > max_clip detik:
            -> rotasi ke berkas baru
"""

from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class TriggerConfig:
    min_frames: int = 3            # debounce: 3 frame berturut-turut
    cooldown: float = 5.0          # jeda antar event (detik)
    pre_roll: float = 3.0          # detik sebelum pemicu ikut terekam
    post_roll: float = 4.0         # detik setelah objek hilang
    max_clip: float = 60.0         # rotasi berkas
    snapshot_cooldown: float = 2.0
    fps: float = 20.0


class AutoRecorder:
    """Menyimpan snapshot & klip video secara otomatis saat target terdeteksi."""

    def __init__(self, output_dir: Path, cfg: Optional[TriggerConfig] = None,
                 enabled: bool = True, save_snapshot: bool = True,
                 save_video: bool = True):
        self.cfg = cfg or TriggerConfig()
        self.output_dir = Path(output_dir)
        self.snapshot_dir = self.output_dir / "snapshots"
        self.video_dir = self.output_dir / "recordings"
        for d in (self.output_dir, self.snapshot_dir, self.video_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.enabled = enabled
        self.save_snapshot = save_snapshot
        self.save_video = save_video

        self.buffer: Deque[np.ndarray] = deque(
            maxlen=max(1, int(self.cfg.pre_roll * self.cfg.fps)))
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_file: Optional[Path] = None
        self._record_start = 0.0
        self.streak = 0
        self._last_hit = 0.0
        self._last_stop = 0.0
        self._last_snapshot = 0.0

        self.snapshots: List[Path] = []
        self.recordings: List[Path] = []
        self.log_path = self.output_dir / "events.csv"
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(
                    ["timestamp", "event", "classes", "n_objects", "file"])

    # ------------------------------------------------------------- status --
    @property
    def recording(self) -> bool:
        return self._writer is not None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._record_start if self.recording else 0.0

    # -------------------------------------------------------------- inti ---
    def update(self, frame: np.ndarray, detections: list,
               auto: bool = True) -> Dict[str, object]:
        """
        Panggil sekali per frame.

        Return dict: {"snapshot": Path|None, "started": bool, "stopped": Path|None}
        """
        out: Dict[str, object] = {"snapshot": None, "started": False,
                                  "stopped": None}
        now = time.monotonic()
        hit = bool(detections) and auto and self.enabled

        # pre-roll selalu dijaga (kecuali sedang merekam)
        if not self.recording:
            self.buffer.append(frame.copy())

        self.streak = self.streak + 1 if hit else 0
        if hit:
            self._last_hit = now

        # ---------- mulai ----------
        if (hit and self.streak >= self.cfg.min_frames
                and not self.recording
                and (now - self._last_stop) >= self.cfg.cooldown):
            if self.save_video:
                self._start(frame)
                out["started"] = True
            if self.save_snapshot and (now - self._last_snapshot) >= self.cfg.snapshot_cooldown:
                path = self._snapshot(frame)
                if path:
                    out["snapshot"] = path
                    self._last_snapshot = now
            self._log("DETECTED", detections)

        # ---------- berhenti / rotasi ----------
        elif self.recording:
            self._writer.write(frame)                       # type: ignore[union-attr]
            if now - self._record_start >= self.cfg.max_clip:
                out["stopped"] = self._stop("max_duration")
            elif not hit and (now - self._last_hit) >= self.cfg.post_roll:
                out["stopped"] = self._stop("post_roll")

        return out

    # ------------------------------------------------------------ aksi -----
    def _start(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        path = self.video_dir / f"rec_{_stamp()}_{len(self.recordings) + 1:04d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.cfg.fps, (w, h))
        if not writer.isOpened():
            writer.release()
            return
        self._writer = writer
        self._current_file = path
        self._record_start = time.monotonic()
        self.recordings.append(path)
        for buffered in list(self.buffer):      # flush pre-roll
            self._writer.write(buffered)
        self.buffer.clear()
        self._log("REC_START", [], file=path.name)

    def _stop(self, reason: str = "") -> Optional[Path]:
        if self._writer is not None:
            self._writer.release()
        self._writer = None
        self._last_stop = time.monotonic()
        self.streak = 0
        path = self._current_file
        if path:
            self._log("REC_STOP", [], file=path.name)
        self._current_file = None
        return path

    def _snapshot(self, frame: np.ndarray) -> Optional[Path]:
        path = self.snapshot_dir / f"snap_{_stamp()}.jpg"
        ok = cv2.imwrite(str(path), frame,
                         [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return None
        self.snapshots.append(path)
        self._log("SNAPSHOT", [], file=path.name)
        return path

    def _log(self, event: str, detections: list, file: str = "") -> None:
        names = ",".join(sorted({d.name for d in detections})) if detections else ""
        with self.log_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                     event, names, len(detections), file])

    # ------------------------------------------------------------ tutup ----
    def close(self) -> None:
        if self.recording:
            self._stop("shutdown")
