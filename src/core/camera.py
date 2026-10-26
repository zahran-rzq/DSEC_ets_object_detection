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
                cap.set(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3), int(width))
            if height:
                cap.set(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4), int(height))
            if fps:
                cap.set(getattr(cv2, "CAP_PROP_FPS", 5), int(fps))
        except Exception:
            pass

    # ---- WARM-UP: harus ada minimal 1 frame yang benar-benar terbaca ----
    for _ in range(max(1, warmup_attempts)):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            h = int(cap.get(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)) or frame.shape[0])
            w = int(cap.get(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)) or frame.shape[1])
            f = cap.get(getattr(cv2, "CAP_PROP_FPS", 5)) or 0
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


def _log_api():
    """
    Temukan API pengatur level log OpenCV yang tersedia.

        OpenCV 4.x : cv2.setLogLevel(level)
        OpenCV 5.x : cv2.utils.logging.setLogLevel(level)  (+ getLogLevel)

    Return: (setter, getter, level_error). setter bisa None bila tidak ada.
    """
    if cv2 is None:
        return None, None, 3
    setter = getattr(cv2, "setLogLevel", None)
    getter, level_error = None, 3
    if not callable(setter):
        mod = getattr(getattr(cv2, "utils", None), "logging", None)
        setter = getattr(mod, "setLogLevel", None)
        getter = getattr(mod, "getLogLevel", None)
        level_error = getattr(mod, "LOG_LEVEL_ERROR", 3)
    if not callable(setter):
        return None, None, 3
    return setter, getter, int(level_error)


class _QuietOpenCV:
    """
    Sementara menaikkan level log OpenCV agar pemindaian kamera tidak berisik.

    Sepenuhnya defensif: bila API logging tidak tersedia (beda versi OpenCV),
    kelas ini menjadi no-op dan TIDAK pernah melempar exception — bug versi
    sebelumnya justru menghentikan seluruh pemindaian pada OpenCV 5.
    """

    def __enter__(self):
        self._setter, self._prev = None, None
        try:
            setter, getter, level_error = _log_api()
            if setter is None:
                return self
            self._prev = int(getter()) if callable(getter) else None
            setter(level_error)
            self._setter = setter
        except Exception:
            self._setter, self._prev = None, None      # gagal -> diam saja
        return self

    def __exit__(self, *a):
        try:
            if self._setter is not None and self._prev is not None:
                self._setter(self._prev)
        except Exception:
            pass
        return False


_ORPHANS: List[tuple] = []      # (thread, box) percobaan yang ditinggalkan


def _reap_orphans() -> None:
    """Rilis kamera dari percobaan yang ditinggalkan karena timeout."""
    alive = []
    for t, box in _ORPHANS:
        if t.is_alive():
            alive.append((t, box))
            continue
        r = box.get("r")
        if r and r[0] is not None:
            try:
                r[0].release()
            except Exception:
                pass
    _ORPHANS[:] = alive


def _try_open_timeout(source, backend_id, backend_name, width, height, fps,
                      warmup_attempts: int = 30, warmup_delay: float = 0.1,
                      timeout: float = 6.0):
    """
    Jalankan _try_open di thread terpisah dengan batas waktu.

    Di Windows, `cv2.VideoCapture(indeks, CAP_DSHOW)` untuk indeks yang tidak
    ada (atau kamera yang sedang dipakai aplikasi lain) dapat MEMBLOKIR TANPA
    BATAS. Karena thread Python tidak bisa dibunuh, percobaan yang melewati
    batas waktu ditinggalkan (thread daemon) dan kameranya dirilis belakangan
    lewat _reap_orphans().
    """
    box: dict = {}

    def job():
        try:
            box["r"] = _try_open(source, backend_id, backend_name,
                                 width, height, fps,
                                 warmup_attempts, warmup_delay)
        except Exception:
            box["r"] = (None, None, None)

    t = threading.Thread(target=job, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        _ORPHANS.append((t, box))
        return None, None, None
    return box.get("r", (None, None, None))


def probe_cameras(max_index: int = 5, width: int = 1280, height: int = 720,
                  fps: int = 30, timeout: float = 4.0,
                  on_progress=None, verbose: bool = False) -> List[dict]:
    """
    Pindai indeks kamera 0..max_index-1 dan kembalikan yang BENAR-BENAR
    menghasilkan frame.

    Setiap percobaan dibatasi `timeout` detik dan progresnya dilaporkan lewat
    `on_progress(str)` / `verbose=True`, sehingga pemindaian tidak pernah
    terlihat "diam" walaupun ada indeks yang memblokir.
    """
    found: List[dict] = []
    _reap_orphans()

    def say(msg: str):
        if verbose:
            print(msg, flush=True)
        if on_progress:
            try:
                on_progress(msg)
            except Exception:
                pass

    with _QuietOpenCV():
        for i in range(max_index):
            say(f"  · menguji indeks {i} ...")
            for name, bid in _candidate_backends():
                say(f"      - backend {name:<12} (maks {timeout:.0f}s)")
                cap, frame, info = _try_open_timeout(
                    i, bid, name, width, height, fps,
                    warmup_attempts=6, warmup_delay=0.08, timeout=timeout)
                if cap is not None:
                    cap.release()
                    if info:
                        info = dict(info)
                        info["ok"] = True
                        found.append(info)
                    say(f"      ✅ indeks {i} berhasil via {name} "
                        f"({info['width']}x{info['height']})" if info else f"      ✅ indeks {i}")
                    break     # kamera ini jalan, tidak perlu coba backend lain
            else:
                say(f"      ❌ indeks {i} tidak menghasilkan frame")
    _reap_orphans()
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
    def open(self, timeout: float = 8.0, on_progress=None) -> bool:
        """Coba buka kamera dengan semua backend yang mungkin (dibatasi waktu)."""
        errors: List[str] = []
        last_err = ""
        _reap_orphans()
        backends = _candidate_backends(self.backend)
        if self.backend and self.backend.lower() not in ("auto", "", "any"):
            print(f"ℹ️  Backend dipaksa: {self.backend}", flush=True)
        for name, bid in backends:
            if on_progress:
                try:
                    on_progress(f"Mencoba backend {name} (maks {timeout:.0f}s) ...")
                except Exception:
                    pass
            cap, frame, info = _try_open_timeout(
                self.source, bid, name, self.width, self.height, self.fps,
                warmup_attempts=20, warmup_delay=0.1, timeout=timeout)
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
    import argparse
    ap = argparse.ArgumentParser("Pemindai kamera SecureVision")
    ap.add_argument("--max-index", type=int, default=5,
                    help="indeks tertinggi yang dipindai (default 5 -> 0..4)")
    ap.add_argument("--timeout", type=float, default=4.0,
                    help="batas waktu tiap percobaan backend, detik (default 4)")
    args, _ = ap.parse_known_args()

    print("=" * 66)
    print("  PEMINDAIAN KAMERA — SecureVision")
    print("=" * 66)
    print(f"Sistem : {platform.system()} {platform.release()}")
    print(f"OpenCV : {cv2.__version__ if cv2 else 'TIDAK TERPASANG'}")
    print("Backend yang akan dicoba:",
          ", ".join(n for n, _ in _candidate_backends()))
    print("-" * 66)
    try:                       # kompatibilitas OpenCV vs Ultralytics
        if int((cv2.__version__ or "4").split(".")[0]) >= 5:
            print("⚠️  OpenCV 5.x terdeteksi. Ultralytics resmi diuji pada OpenCV 4.x.")
            print("   Bila nanti muncul error aneh saat inferensi, turunkan:")
            print("     pip install \"opencv-python>=4.8,<5.0\"")
            print("-" * 66)
    except Exception:
        pass

    print(f"Batas waktu tiap percobaan: {args.timeout:.0f}s (ganti dengan --timeout)", flush=True)
    found = probe_cameras(max_index=args.max_index,
                          timeout=args.timeout, verbose=True)
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
