"""
detector.py -- Pembungkus (wrapper) model YOLO pre-trained.

Tanggung jawab modul ini:
  1. Memuat model YOLO (Ultralytics) beserta bobot pre-trained.
  2. Meneruskan daftar kelas target ke model sehingga objek di luar
     {person, car, truck, motorbike} TIDAK pernah diproses/dikeluarkan.
  3. Menyaring ulang (defensive filtering) berdasar id kelas + confidence
     per-kelas + luas box + ROI, sehingga aman walau dipakai model custom.
  4. Mengembalikan hasil dalam struktur data sederhana (dataclass Detection)
     agar modul lain tidak bergantung pada format Ultralytics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    COCO_CLASS_IDS,
    CLASS_COLORS_BGR,
    ModelConfig,
    normalize_class_names,
)

try:  # OpenCV hanya diperlukan untuk anotasi visual
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------
# Struktur data hasil deteksi
# --------------------------------------------------------------------------
@dataclass
class Detection:
    """Satu objek yang terdeteksi."""
    xyxy: Tuple[float, float, float, float]   # x1, y1, x2, y2 (piksel)
    conf: float                                # 0..1
    cls_id: int                                # id kelas COCO / model
    name: str                                  # nama kelas kanonik
    track_id: Optional[int] = None             # terisi bila tracker aktif

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def color(self) -> tuple:
        return CLASS_COLORS_BGR.get(self.name, CLASS_COLORS_BGR["unknown"])

    def label(self, with_conf: bool = True) -> str:
        s = self.name.capitalize()
        if self.track_id is not None:
            s += f" #{self.track_id}"
        if with_conf:
            s += f" {self.conf:.2f}"
        return s


@dataclass
class FrameResult:
    """Hasil pemrosesan satu frame."""
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.detections)

    def has(self, name: str) -> bool:
        return any(d.name == name for d in self.detections)


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------
class SecurityDetector:
    """
    Wrapper tipis di atas model YOLO Ultralytics.

    Contoh pakai
    ------------
        from src.core.config import preset
        from src.core.detector import SecurityDetector
        det = SecurityDetector(preset().model)
        res = det.predict(frame_bgr)
        print(res.counts)   # {'person': 2, 'car': 1}
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.names: List[str] = normalize_class_names(cfg.target_classes)
        self.class_ids: List[int] = cfg.class_ids
        # id -> nama kanonik (person/car/truck/motorbike)
        self.id_to_name: Dict[int, str] = {COCO_CLASS_IDS[n]: n for n in self.names}
        self._effective_conf = min(
            [cfg.per_class_conf.get(n, cfg.conf) for n in self.names] or [cfg.conf]
        )
        self.model = self._load()
        self.warmup()

    # ---------------- loader ----------------
    def _load(self):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Paket 'ultralytics' belum terpasang.\n"
                "  Laptop/PC : pip install ultralytics\n"
                "  Colab     : jalankan sel '!pip install ultralytics'"
            ) from e
        try:
            model = YOLO(self.cfg.weights)
        except Exception as e:
            raise RuntimeError(
                f"Gagal memuat bobot '{self.cfg.weights}'. "
                "Pastikan file ada, atau biarkan kosong agar diunduh otomatis."
            ) from e
        return model

    def warmup(self) -> None:
        """Satu inference dummy agar alokasi memori GPU/CPU terjadi di awal."""
        try:
            dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                dummy, imgsz=self.cfg.imgsz, conf=self.cfg.conf,
                classes=self.class_ids, device=self.cfg.device or None,
                verbose=False,
            )
        except Exception:
            pass  # warmup gagal tidak fatal (mis. model custom aneh)

    # ---------------- inferensi ----------------
    def predict(self, frame: np.ndarray, persist: bool = True) -> FrameResult:
        """
        Jalankan inferensi pada satu frame BGR (format OpenCV).

        Alur: letterbox(resize) -> normalisasi -> forward pass -> NMS ->
              filter kelas (di level model) -> filter conf per-kelas -> filter
              area/ROI (di level aplikasi).
        """
        t0 = time.perf_counter()
        preds = self.model.predict(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self._effective_conf,       # ambang terendah, disaring ulang di bawah
            iou=self.cfg.iou,                # IoU threshold NMS
            classes=self.class_ids,          # <-- KUNCI: hanya 4 kelas target
            device=self.cfg.device or None,
            half=self.cfg.half,
            verbose=False,
            persist=persist,                 # dibutuhkan tracker (ByteTrack)
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Detection] = []
        counts: Dict[str, int] = {n: 0 for n in self.names}

        for r in preds:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = None
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                cid = int(clss[i])
                name = self.id_to_name.get(cid)
                if name is None:
                    continue  # kelas non-target (defensive, seharusnya tak terjadi)
                conf = float(confs[i])
                if conf < self.cfg.per_class_conf.get(name, self.cfg.conf):
                    continue  # filter confidence per-kelas
                box = tuple(float(v) for v in xyxy[i])
                if (box[2] - box[0]) * (box[3] - box[1]) < 1:
                    continue
                d = Detection(
                    xyxy=box,
                    conf=conf,
                    cls_id=cid,
                    name=name,
                    track_id=int(ids[i]) if ids is not None else None,
                )
                detections.append(d)
                counts[name] = counts.get(name, 0) + 1

        return FrameResult(detections=detections, inference_ms=elapsed_ms, counts=counts)

    # ---------------- utilitas ----------------
    def filter_by_area(self, res: FrameResult, min_area: int) -> FrameResult:
        res.detections = [d for d in res.detections if d.area >= min_area]
        res.counts = {n: 0 for n in self.names}
        for d in res.detections:
            res.counts[d.name] += 1
        return res

    def filter_by_roi(self, res: FrameResult, polygon: Sequence[Sequence[int]]) -> FrameResult:
        """Simpan hanya deteksi yang titik pusatnya berada di dalam polygon ROI."""
        if cv2 is None or not polygon:
            return res
        poly = np.array(polygon, dtype=np.int32)
        res.detections = [
            d for d in res.detections
            if cv2.pointPolygonTest(poly, d.center, False) >= 0
        ]
        res.counts = {n: 0 for n in self.names}
        for d in res.detections:
            res.counts[d.name] += 1
        return res

    # ---------------- anotasi ----------------
    @staticmethod
    def draw(frame: np.ndarray,
             res: FrameResult,
             *,
             thickness: int = 2,
             font_scale: float = 0.55,
             show_label: bool = True) -> np.ndarray:
        """Gambar bounding box + label ke frame (in-place, mengembalikan frame)."""
        if cv2 is None:
            return frame
        for d in res.detections:
            x1, y1, x2, y2 = map(int, d.xyxy)
            color = d.color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            if show_label:
                text = d.label()
                (tw, th), base = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                y_top = max(0, y1 - th - base - 4)
                cv2.rectangle(frame, (x1, y_top), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, text, (x1 + 3, y1 - base - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), 1, cv2.LINE_AA)
        return frame


class MockDetector:
    """
    Detector palsu untuk pengujian logika tanpa model/GPU.

    Mengembalikan deteksi statis sehingga pipeline (trigger, recorder, UI)
    bisa diuji di komputer mana pun -- berguna saat proses development.
    """

    def __init__(self, n_person: int = 1, names: Optional[List[str]] = None):
        self.names = names or ["person", "car", "truck", "motorbike"]
        self.calls = 0
        self._n_person = n_person

    def predict(self, frame, persist: bool = True) -> FrameResult:
        self.calls += 1
        h, w = frame.shape[:2]
        dets = []
        for i in range(self._n_person):
            x1 = int(w * 0.2) + i * 60
            dets.append(Detection(
                xyxy=(x1, int(h * 0.3), x1 + 90, int(h * 0.3) + 180),
                conf=0.85, cls_id=0, name="person", track_id=i + 1))
        return FrameResult(detections=dets, inference_ms=1.0,
                           counts={"person": self._n_person})

    # API yang sama dengan SecurityDetector agar bisa ditukar (duck typing)
    def filter_by_area(self, res: "FrameResult", min_area: int) -> "FrameResult":
        res.detections = [d for d in res.detections if d.area >= min_area]
        res.counts = {n: 0 for n in self.names}
        for d in res.detections:
            res.counts[d.name] = res.counts.get(d.name, 0) + 1
        return res

    def filter_by_roi(self, res: "FrameResult", polygon) -> "FrameResult":
        return res  # mock: abaikan ROI
