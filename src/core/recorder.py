"""
recorder.py -- Penyimpanan bukti: snapshot (JPG) + rekaman video (MP4) + log CSV.

Tiga komponen utama:
  1. RingBuffer      : menyimpan N detik frame terakhir secara terus-menerus.
                       Dipakai untuk "pre-roll" -- rekaman tidak dimulai dari
                       detik objek terdeteksi, tetapi beberapa detik SEBELUMnya.
                       (Sangat berguna untuk melihat penyusup datang.)
  2. VideoRecorder   : menulis frame ke file .mp4. Saat trigger aktif, isi
                       RingBuffer lebih dulu di-flush ke file baru (pre-roll),
                       lalu frame baru ditambahkan terus sampai post-roll habis.
  3. SnapshotWriter  : menyimpan 1 frame JPG per event (rate-limited).
  4. EventLogger     : menulis baris ke events.csv untuk audit trail.
"""

from __future__ import annotations

import csv
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _pretty(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
class RingBuffer:
    """Buffer melingkar berkapasitas detik (berdasar FPS) untuk pre-roll."""

    def __init__(self, seconds: float, fps: float):
        self.capacity = max(1, int(round(seconds * fps)))
        self._buf: Deque[np.ndarray] = deque(maxlen=self.capacity)

    def push(self, frame: np.ndarray) -> None:
        self._buf.append(frame)

    def snapshot(self) -> List[np.ndarray]:
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


# --------------------------------------------------------------------------
@dataclass
class RecordedClip:
    path: Path
    started_at: datetime
    ended_at: Optional[datetime] = None
    frames: int = 0
    trigger: str = ""

    @property
    def duration_s(self) -> float:
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def size_mb(self) -> float:
        try:
            return self.path.stat().st_size / 1e6
        except OSError:
            return 0.0


class VideoRecorder:
    """
    Menulis klip MP4.

    Alur pakai:
        rec.start(frame_size, fps, trigger_label)   # otomatis flush pre-roll
        rec.write(frame)                            # tiap frame saat REC aktif
        rec.stop()                                  # -> RecordedClip
    """

    def __init__(self, out_dir: Path, fourcc: str = "mp4v"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fourcc = fourcc
        self._writer = None
        self._current: Optional[RecordedClip] = None
        self.history: List[RecordedClip] = []

    # ---- state ----
    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def current(self) -> Optional[RecordedClip]:
        return self._current

    # ---- lifecycle ----
    def start(self, size: tuple, fps: float, trigger: str = "",
              pre_roll: Optional[List[np.ndarray]] = None) -> Path:
        if self.is_recording:
            return self._current.path
        h, w = int(size[1]), int(size[0])
        fname = self.out_dir / f"rec_{_stamp()}_{len(self.history) + 1:04d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc) if cv2 is not None else 0
        self._writer = cv2.VideoWriter(str(fname), fourcc, fps, (w, h)) if cv2 else None
        self._current = RecordedClip(path=fname, started_at=datetime.now(),
                                     trigger=trigger)
        # flush pre-roll dulu supaya kejadian SEBELUM trigger ikut terekam
        for f in pre_roll or []:
            self.write(f, count=True)
        return fname

    def write(self, frame: np.ndarray, count: bool = True) -> None:
        if not self.is_recording or self._writer is None:
            return
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        self._writer.write(frame)
        if count and self._current:
            self._current.frames += 1

    def stop(self) -> Optional[RecordedClip]:
        if not self.is_recording:
            return None
        self._writer.release()
        self._writer = None
        clip = self._current
        if clip:
            clip.ended_at = datetime.now()
            self.history.append(clip)
        self._current = None
        return clip

    def release(self) -> None:
        self.stop()


# --------------------------------------------------------------------------
class SnapshotWriter:
    """Menyimpan frame JPG dengan jeda minimal antar file (cooldown)."""

    def __init__(self, out_dir: Path, cooldown: float = 2.0, quality: int = 90):
        self.out_dir = Path(out_dir) / "snapshots"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown = cooldown
        self.quality = quality
        self._last = 0.0
        self.history: List[Path] = []

    def save(self, frame: np.ndarray, tag: str = "") -> Optional[Path]:
        now = time.time()
        if now - self._last < self.cooldown:
            return None
        self._last = now
        name = f"snap_{_stamp()}{'_' + tag if tag else ''}.jpg"
        p = self.out_dir / name
        if cv2 is not None:
            cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        else:  # pragma: no cover
            from PIL import Image
            Image.fromarray(frame[..., ::-1]).save(p, quality=self.quality)
        self.history.append(p)
        return p


# --------------------------------------------------------------------------
class EventLogger:
    """Log CSV sederhana + mirror di memori untuk ditampilkan di UI."""

    HEADER = ["timestamp", "event", "classes", "conf", "n_objects", "file"]

    def __init__(self, out_dir: Path, enabled: bool = True):
        self.path = Path(out_dir) / "events.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.rows: List[dict] = []
        self._lock = threading.Lock()
        if enabled and not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADER)

    def log(self, event: str, classes: str = "", conf: float = 0.0,
            n_objects: int = 0, file: str = "") -> None:
        row = {"timestamp": _pretty(), "event": event, "classes": classes,
               "conf": f"{conf:.3f}", "n_objects": n_objects, "file": file}
        with self._lock:
            self.rows.append(row)
            if self.enabled:
                with self.path.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row.values())

    def recent(self, n: int = 12) -> List[dict]:
        return self.rows[-n:][::-1]
