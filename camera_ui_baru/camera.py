"""Pembaca kamera background yang tidak memblokir UI."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraInfo:
    source: str = "0"
    backend: str = "auto"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    error: str = ""


class CameraReader:
    """Membaca webcam/video pada thread terpisah dan menyimpan frame terbaru."""

    def __init__(
        self,
        source: str = "0",
        width: int = 1280,
        height: int = 720,
        backend: str = "auto",
    ) -> None:
        self.source = str(source)
        self.width = width
        self.height = height
        self.backend = backend.lower()
        self.info = CameraInfo(source=self.source, backend=self.backend)
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[cv2.VideoCapture] = None

    def _source(self) -> tuple[object, bool]:
        if self.source.isdecimal():
            return int(self.source), True
        return self.source, False

    def _backend_id(self) -> int:
        if self.backend == "dshow":
            return cv2.CAP_DSHOW
        if self.backend == "msmf":
            return cv2.CAP_MSMF
        return cv2.CAP_ANY

    def open(self) -> bool:
        source, is_camera = self._source()
        capture = cv2.VideoCapture(
            source, self._backend_id()
        ) if is_camera else cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            self.info.error = f"Sumber '{self.source}' tidak dapat dibuka."
            return False

        if is_camera:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        deadline = time.monotonic() + 3.0
        first_frame = None
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                first_frame = frame
                break
            time.sleep(0.05)
        if first_frame is None:
            capture.release()
            self.info.error = "Kamera terbuka tetapi tidak mengirim frame."
            return False

        self._capture = capture
        self.info.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or first_frame.shape[1]
        self.info.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or first_frame.shape[0]
        self.info.fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._put(first_frame)
        return True

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        if not self.open():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def _put(self, frame: np.ndarray) -> None:
        try:
            self._frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            pass

    def _read_loop(self) -> None:
        assert self._capture is not None
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if ok and frame is not None and frame.size:
                self._put(frame)
            else:
                self.info.error = "Frame berhenti diterima dari sumber."
                time.sleep(0.05)

    def read(self) -> Optional[np.ndarray]:
        try:
            return self._frames.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._thread = None

