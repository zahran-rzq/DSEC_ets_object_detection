"""
camera.py -- Pembuka kamera yang tangguh (backend probing + warm-up + watchdog).

Masalah yang umum terjadi di laptop Windows:
  * Lampu indikator webcam MENYALA, tetapi `cap.read()` selalu mengembalikan
    False -> layar aplikasi tetap hitam/kosong.
  * Penyebab paling sering:
      1. OpenCV memakai backend MSMF yang tidak kompatibel dengan webcam
         tersebut -> solusi: paksa backend **CAP_DSHOW**.
      2. Aplikasi lain (Zoom / Teams / Meet / browser) sedang memegang kamera.
      3. Resolusi/FPS yang diminta tidak didukung -> kamera menolak stream.
      4. Izin kamera untuk "desktop apps" dimatikan di Windows
         (Settings > Privacy > Camera > Let desktop apps access your camera).
      5. Indeks kamera salah (0 bukan kamera yang dimaksud).

Solusi di modul ini:
  * Mencoba beberapa backend secara berurutan (DSHOW -> MSMF -> ANY pada
    Windows, V4L2 -> ANY pada Linux, AVFOUNDATION -> ANY pada macOS).
  * "Warm-up": backend baru dianggap berhasil HANYA kalau benar-benar
    menghasilkan minimal satu frame (bukan sekadar isOpened() == True).
  * `probe_cameras()` untuk mendaftar kamera mana yang benar-benar bekerja.
  * Watchdog: mendeteksi kamera yang berhenti mengirim frame (mis. dicabut)
    dan bisa mencoba membuka ulang otomatis.
"""

from __future__ import annotations

import os
import platform
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------
def _candidate_backends(backend: str = "auto") -> List[Tuple[str, int]]:
    """Daftar backend yang dicoba, diurutkan dari yang paling andal."""
    if cv2 is None:
        return []
    sysname = platform.system()
    if sysname == "Windows":
        order = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
    elif sysname == "Darwin":
        order = [("AVFOUNDATION", cv2.CAP_AVFOUNDATION), ("ANY", cv2.CAP_ANY)]
    else:
        order = [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]

    if backend and backend.lower() not in ("auto", "", "any"):
        key = backend.upper()
        forced = [(b, getattr(cv2, f"CAP_{key}")) for b, _ in order if b == key]
        return forced or order
    return order


def _source_to_index(source) -> Tuple[object, bool]:
    """'0' -> (0, True) bila angka; path video/RTSP -> (str, False)."""
    s = str(source)
    if s.isdigit():
        return int(s), True
    return s, False


# --------------------------------------------------------------------------
def _try_open(source, backend_id, backend_name, width, height, fps,
              warmup_attempts: int = 30, warmup_delay: float = 0.1):
    """
    Buka kamera dengan satu backend dan pastikan ia benar-benar mengirim frame.

    Return: (cap, frame_pertama, info) atau (None, None, None) bila gagal.
    """
    idx, is_cam = _source_to_index(source)
    if not is_cam:
        backend_name = "FILE/URL"
    try:
        cap = cv2.VideoCapture(idx, backend_id) if is_cam else cv2.VideoCapture(idx)
    except Exception:
        return None, None, None
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None, None, None

    # Minta resolusi/FPS (best-effort: kegagalan di sini TIDAK boleh fatal)
    if is_cam:
        try:
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            if fps:
                cap.set(cv2.CAP_PROP_FPS, int(fps))
        except Exception:
            pass

    # ---- WARM-UP: harus ada minimal 1 frame yang benar-benar terbaca ----
    for _ in range(max(1, warmup_attempts)):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or (frame.shape[0] if frame is not None else 0))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or (frame.shape[1] if frame is not None else 0))
            f = cap.get(cv2.CAP_PROP_FPS) or 0
            info = {
                "backend": backend_name,
                "width": w or (frame.shape[1] if frame is not None else 0),
                "height": h or (frame.shape[0] if frame is not None else 0),
                "fps": float(f) if f and f > 0 else float(fps or 30),
                "index": idx,
            }
            return cap, frame, info
        time.sleep(warmup_delay)

    cap.release()
    return None, None, None


class _QuietOpenCV:
    """Sementara menaikkan level log OpenCV agar pemindaian tidak berisik."""

    def __enter__(self):
        self.prev = cv2.setLogLevel(0) if cv2 is not None else 0
        if cv2 is not None:
            cv2.setLogLevel(3)      # LOG_LEVEL_ERROR
        return self

    def __exit__(self, *a):
        if cv2 is not None:
            cv2.setLogLevel(self.prev)
        return False


def probe_cameras(max_index: int = 5, width: int = 1280, height: int = 720,
                  fps: int = 30) -> List[dict]:
    """
    Pindai indeks kamera 0..max_index-1 dan kembalikan yang BENAR-BENAR
    menghasilkan frame. Dipakai oleh tombol "Tes Kamera" pada GUI dan oleh
    `python -m src.core.camera`.
    """
    found: List[dict] = []
    with _QuietOpenCV():
        for i in range(max_index):
            for name, bid in _candidate_backends():
                cap, frame, info = _try_open(i, bid, name, width, height, fps,
                                             warmup_attempts=6, warmup_delay=0.08)
                if cap is not None:
                    cap.release()
                    if info:
                        info = dict(info)
                        info["ok"] = True
                        found.append(info)
                    break     # kamera ini jalan, tidak perlu coba backend lain
    return found


# --------------------------------------------------------------------------
@dataclass
class CameraStatus:
    opened: bool = False
    backend: str = "-"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frames: int = 0
    last_frame_time: float = 0.0
    mean_brightness: float = -1.0   # -1 = belum diukur
    error: str = ""

    @property
    def stale(self) -> bool:
        """True bila sudah lebih dari 3 detik tidak ada frame baru."""
        return self.opened and (time.time() - self.last_frame_time) > 3.0


class CameraThread(threading.Thread):
    """
    Pembaca kamera di background thread.

    Perbaikan dibanding versi sebelumnya:
      * backend probing + warm-up (kamera harus mengirim frame saat dibuka),
      * watchdog untuk mendeteksi stream yang macet,
      * dapat membuka ulang kamera secara otomatis (`auto_reopen`).
    """

    def __init__(self, source="0", width=1280, height=720, fps=30,
                 backend: str = "auto", flip: bool = False,
                 auto_reopen: Optional[bool] = None,
                 queue_size: Optional[int] = None):
        super().__init__(daemon=True)
        self.source = source
        self.width, self.height, self.fps = width, height, fps
        self.backend = backend
        self.flip = flip
        # auto_reopen: None -> turunan. Berkas video TIDAK diulang (harus
        # berhenti di akhir), sedangkan kamera/RTSP akan disambung ulang.
        if auto_reopen is None:
            auto_reopen = not os.path.isfile(str(source))
        self.auto_reopen = auto_reopen
        # Live: antrean kecil + buang frame lama (selalu tampilkan keadaan
        # terkini). Berkas video: antrean besar + JANGAN buang frame apa pun
        # (kalau tidak, sebagian besar isi video akan terlewat saat forensik).
        self.drop_oldest = bool(auto_reopen)
        if queue_size is None:
            queue_size = 2 if auto_reopen else 96
        self.cap = None
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_size)
        self.running = threading.Event()
        self.status = CameraStatus()
        self._lock = threading.Lock()

    # ------------------------------------------------------------- buka --
    def open(self) -> bool:
        """Coba buka kamera dengan semua backend yang mungkin."""
        errors: List[str] = []
        last_err = ""
        backends = _candidate_backends(self.backend)
        if self.backend and self.backend.lower() not in ("auto", "", "any"):
            print(f"ℹ️  Backend dipaksa: {self.backend}")
        for name, bid in backends:
            cap, frame, info = _try_open(self.source, bid, name,
                                         self.width, self.height, self.fps)
            if cap is not None:
                self.cap = cap
                self.status.opened = True
                self.status.backend = info["backend"]
                self.status.width = info["width"]
                self.status.height = info["height"]
                self.status.fps = info["fps"]
                self.status.last_frame_time = time.time()
                self.status.error = ""
                # jangan buang frame pertama yang sudah terbaca
                if frame is not None:
                    self._put(self._prepare(frame))
                return True
            last_err = f"backend {name} tidak menghasilkan frame"
            errors.append(last_err)

        self.status.opened = False
        self.status.error = (
            "Kamera terdeteksi (LED menyala) tetapi tidak ada frame yang masuk.\n"
            "  Kemungkinan: dipakai aplikasi lain (Zoom/Teams/Meet), izin kamera\n"
            "  untuk desktop apps nonaktif, atau resolusi tidak didukung.\n"
            "  Coba: tutup aplikasi lain, lalu klik 'Tes Kamera'."
        )
        return False

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if self.flip:
            frame = frame[:, ::-1].copy()
        return frame

    def _put(self, frame) -> None:
        if self.drop_oldest and self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()   # buang yang paling lama
            except queue.Empty:
                pass
        try:
            self.frame_queue.put(frame, timeout=0.05 if self.drop_oldest else 0.5)
        except queue.Full:
            pass

    # ------------------------------------------------------------ thread --
    def run(self) -> None:
        miss = 0
        while self.running.is_set():
            if self.cap is None or not self.cap.isOpened():
                if not (self.auto_reopen and self._reopen()):
                    time.sleep(0.5)
                    continue
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.size > 0:
                miss = 0
                self.status.last_frame_time = time.time()
                self.status.frames += 1
                try:
                    self.status.mean_brightness = float(frame.mean())
                except Exception:
                    pass
                self._put(self._prepare(frame))
            else:
                miss += 1
                # 60 kali gagal berturut-turut (~2 detik) -> coba buka ulang
                if self.auto_reopen and miss >= 60:
                    miss = 0
                    self._reopen()
                else:
                    time.sleep(0.03)

    def _reopen(self) -> bool:
        """Tutup lalu buka ulang kamera (dipanggil bila stream macet)."""
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self.status.opened = False
        time.sleep(0.4)
        return self.open()

    # -------------------------------------------------------------- API --
    def read(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def read_blocking(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.running.clear()
        self.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.status.opened = False


# --------------------------------------------------------------------------
def main() -> int:
    """CLI: `python -m src.core.camera` -> pindai dan laporkan kamera."""
    print("=" * 66)
    print("  PEMINDAIAN KAMERA — SecureVision")
    print("=" * 66)
    print(f"Sistem : {platform.system()} {platform.release()}")
    print(f"OpenCV : {cv2.__version__ if cv2 else 'TIDAK TERPASANG'}")
    print("Backend yang akan dicoba:",
          ", ".join(n for n, _ in _candidate_backends()))
    print("-" * 66)

    found = probe_cameras(max_index=5)
    if not found:
        print("❌ Tidak ada kamera yang menghasilkan frame.")
        print("\nLangkah pemeriksaan:")
        print("  1. Tutup Zoom / Teams / Google Meet / browser yang memakai kamera.")
        if platform.system() == "Windows":
            print("  2. Settings > Privacy & security > Camera:")
            print("       - Camera access = On")
            print("       - Let apps access your camera = On")
            print("       - Let desktop apps access your camera = On")
        else:
            print("  2. Pastikan user punya izin akses /dev/video* "
                  "(sudo usermod -aG video $USER)")
        print("  3. Coba kamera di aplikasi bawaan (Camera/Photo Booth) dulu.")
        print("  4. Pastikan tidak ada kamera virtual (OBS/DroidCam) di indeks 0.")
        return 1

    print(f"✅ Ditemukan {len(found)} kamera yang bekerja:\n")
    for i, f in enumerate(found):
        print(f"  [{i}] index={f['index']}  backend={f['backend']:<12} "
              f"{f['width']}x{f['height']} @ {f['fps']:.1f} fps")
    print("\nGunakan di aplikasi:")
    print(f"  python -m src.app --source {found[0]['index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
