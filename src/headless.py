"""
headless.py -- Mode tanpa GUI: jalankan pipeline pada webcam / file video / gambar.

Berguna untuk:
  * menjalankan aplikasi di mesin tanpa monitor (Raspberry Pi, mini-PC),
  * menguji logika trigger/rekaman secara otomatis,
  * memproses video yang sudah direkam (forensik).

Contoh:
    python -m src.headless --source 0
    python -m src.headless --source sample.mp4 --output runs/capture
    python -m src.headless --source tests/data/bus.jpg --save-frame
    python -m src.headless --source 0 --no-display --max-frames 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import preset                       # noqa: E402
from src.core.detector import SecurityDetector           # noqa: E402
from src.core.pipeline import State, SurveillancePipeline  # noqa: E402
from src.core.camera import CameraThread, probe_cameras    # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def open_source(source: str, width: int, height: int, fps: int,
                backend: str = "auto"):
    """
    Buka sumber video dengan backend probing + warm-up.
    Kamera hanya dianggap berhasil bila benar-benar mengirim frame.
    """
    cam = CameraThread(source, width, height, fps, backend=backend)
    if not cam.open():
        raise SystemExit("\n❌ " + (cam.status.error or f"Tidak bisa membuka {source}")
                         + "\n   Coba: python -m src.core.camera   (pemindaian kamera)")
    cam.running.set()
    cam.start()
    return cam


def run_on_image(path: str, cfg, save: bool):
    """Mode gambar tunggal: cetak hasil deteksi ke terminal."""
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"❌ Gambar tidak terbaca: {path}")
    det = SecurityDetector(cfg.model)
    res = det.predict(img)
    print(f"\nGambar        : {path}  ({img.shape[1]}x{img.shape[0]})")
    print(f"Model         : {cfg.model.weights}  imgsz={cfg.model.imgsz}")
    print(f"Kelas target  : {', '.join(cfg.model.names)}  -> ids {cfg.model.class_ids}")
    print(f"Inferensi     : {res.inference_ms:.1f} ms")
    print(f"Objek         : {res.n}")
    for name, n in res.counts.items():
        print(f"   - {name:<10}: {n}")
    for d in res.detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        print(f"   [{d.name:<9}] conf={d.conf:.2f} box=({x1},{y1})-({x2},{y2})")
    if save:
        det.draw(img, res)
        out = Path(cfg.recording.output_dir) / f"detected_{Path(path).name}"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        print(f"Tersimpan     : {out}")


def main():
    ap = argparse.ArgumentParser("SecureVision headless")
    ap.add_argument("--source", default="0")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--output", default="runs/capture")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--preset", default="balanced",
                    choices=["balanced", "max_recall", "low_false_alarm", "fast"])
    ap.add_argument("--device", default="")
    ap.add_argument("--backend", default="auto",
                    help="auto | DSHOW | MSMF | V4L2 | AVFOUNDATION | ANY "
                         "(Windows bermasalah -> coba DSHOW)")
    ap.add_argument("--list-cameras", action="store_true",
                    help="pindai kamera yang tersedia lalu keluar")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save-frame", action="store_true", help="(mode gambar) simpan hasil")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = tanpa batas")
    ap.add_argument("--mock", action="store_true",
                    help="pakai detector palsu (uji logika tanpa model)")
    args = ap.parse_args()

    if args.list_cameras:
        from src.core.camera import main as scan_cameras
        raise SystemExit(scan_cameras())

    cfg = preset(args.preset)
    cfg.model.weights = args.weights
    cfg.model.imgsz = args.imgsz
    cfg.model.device = args.device
    cfg.camera.source = args.source
    cfg.recording.output_dir = args.output

    # ---- mode gambar ----
    if Path(args.source).suffix.lower() in IMAGE_EXT:
        run_on_image(args.source, cfg, args.save_frame)
        return

    detector = None
    if args.mock:
        from src.core.detector import MockDetector
        detector = MockDetector(n_person=1)

    pipe = SurveillancePipeline(cfg, detector=detector,
                                on_event=lambda e, d: print(f"  [{e}] {d}"))
    cam = open_source(args.source, cfg.camera.width, cfg.camera.height,
                      cfg.camera.fps, backend=args.backend)
    fps_src = cam.status.fps or 30.0
    cfg.recording.video_fps = float(fps_src)

    print(f"Menjalankan pipeline pada '{args.source}' "
          f"({cam.status.width}x{cam.status.height} @ {fps_src:.1f} fps, "
          f"backend {cam.status.backend}). Tekan Q untuk berhenti.")
    idx = 0
    misses = 0
    t0 = time.time()
    try:
        while True:
            frame = cam.read_blocking(timeout=1.0)
            if frame is None:
                misses += 1
                if not cam.auto_reopen:
                    print("✅ Sumber video selesai.")
                    break
                if not cam.status.opened or misses > 30:
                    print("⚠️  Kamera berhenti mengirim frame.")
                    break
                continue
            misses = 0
            idx += 1
            out = pipe.process(frame, draw=True)
            if out["event_started"]:
                print(f"⏺  REC dimulai  (frame {idx})")
            if out["event_stopped"] and out["clip"]:
                print(f"⏹  REC selesai  -> {out['clip']}")
            if not args.no_display:
                try:
                    cv2.imshow("SecureVision (headless)", out["frame"])
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                except cv2.error:
                    print("⚠️  Tampilan tidak tersedia (opencv-headless) "
                          "-> lanjut tanpa jendela.")
                    args.no_display = True
            if args.max_frames and idx >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        pipe.close()
        dt = time.time() - t0
        s = pipe.stats
        print("\n================ RINGKASAN ================")
        print(f"Frame diproses : {s.frames} dalam {dt:.1f}s ({s.frames/max(dt,1e-6):.1f} fps)")
        print(f"Objek terdeteksi: {s.detections}")
        print(f"Event rekaman  : {s.events}")
        print(f"Klip tersimpan : {len(pipe.recorder.history)}")
        print(f"Snapshot       : {len(pipe.snapper.history)}")
        print(f"Output         : {Path(cfg.recording.output_dir).resolve()}")
        print("===========================================")


if __name__ == "__main__":
    main()
