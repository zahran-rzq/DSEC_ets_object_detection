"""
test_pipeline.py -- Uji logika trigger & perekaman TANPA model dan TANPA webcam.

Tujuan: memastikan state machine (IDLE -> DETECTING -> RECORDING -> COOLDOWN)
dan mekanisme penyimpanan (pre-roll, snapshot, rotasi klip) benar-benar jalan,
sebelum dipasang ke kamera asli.

Menjalankan:
    python -m tests.test_pipeline
    # atau
    pytest tests -q
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import AppConfig, preset          # noqa: E402
from src.core.detector import Detection, MockDetector  # noqa: E402
from src.core.metrics import iou, evaluate             # noqa: E402
from src.core.pipeline import State, SurveillancePipeline  # noqa: E402

PASS, FAIL = "✅", "❌"
_results = []


def check(name: str, cond: bool, detail: str = ""):
    _results.append(cond)
    print(f"{PASS if cond else FAIL} {name}" + (f"  ({detail})" if detail else ""))


def frame(w=640, h=480, color=40):
    f = np.full((h, w, 3), color, dtype=np.uint8)
    return f


# --------------------------------------------------------------------------
def test_state_machine():
    print("\n[1] State machine: IDLE -> DETECTING -> RECORDING -> COOLDOWN")
    tmp = Path(tempfile.mkdtemp())
    cfg = preset("balanced")
    cfg.recording.output_dir = str(tmp)
    cfg.trigger.min_consecutive_frames = 3
    cfg.trigger.pre_roll_seconds = 1.0
    cfg.trigger.post_roll_seconds = 0.5
    cfg.trigger.cooldown_seconds = 0.5
    cfg.trigger.snapshot_cooldown_seconds = 0.2
    cfg.recording.video_fps = 10.0

    det = MockDetector(n_person=1)
    pipe = SurveillancePipeline(cfg, detector=det)
    det._n_person = 0                       # fase 1: tidak ada objek
    for _ in range(5):
        pipe.process(frame())
    check("tanpa objek -> IDLE", pipe.state == State.IDLE, pipe.state.value)

    det._n_person = 1                        # fase 2: objek muncul
    started = False
    for _ in range(6):
        out = pipe.process(frame())
        started |= out["event_started"]
    check("objek konsisten -> RECORDING", started and pipe.state == State.RECORDING,
          pipe.state.value)

    det._n_person = 0                        # fase 3: objek hilang
    stopped = False
    for _ in range(12):
        out = pipe.process(frame())
        stopped |= out["event_stopped"]
        time.sleep(0.05)
    check("objek hilang -> berhenti merekam", stopped and not pipe.recorder.is_recording)
    pipe.close()

    clips = list(tmp.glob("*.mp4"))
    snaps = list((tmp / "snapshots").glob("*.jpg"))
    check("klip video tersimpan", len(clips) >= 1, f"{len(clips)} file")
    check("snapshot tersimpan", len(snaps) >= 1, f"{len(snaps)} file")
    check("log CSV terisi", (tmp / "events.csv").exists())
    if clips:
        cap = cv2.VideoCapture(str(clips[0]))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        check("klip berisi frame (pre-roll masuk)", n >= cfg.trigger.min_consecutive_frames,
              f"{n} frame")
    shutil.rmtree(tmp, ignore_errors=True)


def test_debounce():
    print("\n[2] Debounce: deteksi 1 frame saja TIDAK memicu rekaman")
    tmp = Path(tempfile.mkdtemp())
    cfg = preset("balanced")
    cfg.recording.output_dir = str(tmp)
    cfg.trigger.min_consecutive_frames = 4
    det = MockDetector(n_person=1)
    pipe = SurveillancePipeline(cfg, detector=det)
    det._n_person = 0
    pipe.process(frame())
    det._n_person = 1
    pipe.process(frame())          # hanya 1 frame
    det._n_person = 0
    pipe.process(frame())
    check("flicker 1 frame tidak merekam", not pipe.recorder.is_recording)
    pipe.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_class_filter_ids():
    print("\n[3] Pemetaan kelas target -> id COCO")
    cfg = preset("balanced")
    check("ids = [0,2,3,7]", cfg.model.class_ids == [0, 2, 3, 7], str(cfg.model.class_ids))
    check("nama kanonik", cfg.model.names == ["person", "car", "truck", "motorbike"],
          str(cfg.model.names))
    cfg2 = AppConfig()
    cfg2.model.target_classes = ["Person", "motorcycle", "MOBIL", "truck"]
    check("alias 'motorcycle'/'MOBIL' dinormalisasi",
          set(cfg2.model.names) == {"person", "motorbike", "car", "truck"},
          str(cfg2.model.names))


def test_iou():
    print("\n[4] IoU")
    check("identik = 1.0", abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9)
    check("geser 50% = 0.333", abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9)
    check("tak bersinggungan = 0", iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0)


def test_metrics():
    print("\n[5] Precision / Recall / mAP")
    gt_boxes = [[[10, 10, 60, 60], [100, 100, 160, 160]], [[20, 20, 70, 70]]]
    gt_classes = [[0, 0], [0]]
    pred_boxes = [[[12, 12, 58, 58], [300, 300, 340, 340]], [[25, 25, 68, 68]]]
    pred_scores = [[0.95, 0.40], [0.80]]
    pred_classes = [[0, 0], [0]]
    rep = evaluate(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes)
    o = rep.overall
    check("TP=2, FP=1, FN=1", (o["tp"], o["fp"], o["fn"]) == (2, 1, 1),
          f"tp{o['tp']} fp{o['fp']} fn{o['fn']}")
    check("Precision = 2/3", abs(o["precision"] - 2 / 3) < 1e-6)
    check("Recall = 2/3", abs(o["recall"] - 2 / 3) < 1e-6)
    check("0 <= mAP50 <= 1", 0.0 <= o["map50"] <= 1.0, f"{o['map50']:.3f}")
    print("\n" + rep.summary())


def test_max_clip_rotation():
    print("\n[6] Rotasi klip saat durasi maksimum tercapai")
    tmp = Path(tempfile.mkdtemp())
    cfg = preset("balanced")
    cfg.recording.output_dir = str(tmp)
    cfg.trigger.min_consecutive_frames = 1
    cfg.trigger.max_clip_seconds = 0.3
    cfg.trigger.post_roll_seconds = 5.0
    cfg.recording.video_fps = 10.0
    det = MockDetector(n_person=1)
    pipe = SurveillancePipeline(cfg, detector=det)
    det._n_person = 1
    for _ in range(40):
        pipe.process(frame())
        time.sleep(0.02)
    pipe.close()
    n = len(list(tmp.glob("*.mp4")))
    check("lebih dari satu klip (rotasi jalan)", n >= 2, f"{n} klip")
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("=" * 60)
    print("   SELF-TEST SECUREVISION (tanpa model, tanpa kamera)")
    print("=" * 60)
    test_state_machine()
    test_debounce()
    test_class_filter_ids()
    test_iou()
    test_metrics()
    test_max_clip_rotation()
    ok, bad = sum(_results), len(_results) - sum(_results)
    print("\n" + "=" * 60)
    print(f"   HASIL: {ok} lulus, {bad} gagal")
    print("=" * 60)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
