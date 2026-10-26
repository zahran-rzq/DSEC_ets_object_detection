"""
detector.py -- Pembungkus YOLO untuk UI kamera mandiri.

Mengapa berkas ini ada?
-----------------------
Versi awal `camera_ui_baru` menggambar SEMUA kelas COCO (80 kelas). Padahal
Soal 2 secara eksplisit meminta sistem **hanya** memproses empat kelas:
person, car, truck, motorbike. Modul ini menerapkan tiga lapis penyaringan
yang sama dengan `src/core/detector.py`:

    1. filter di level model  -> model.predict(classes=[0, 2, 3, 7])
    2. ambang confidence per kelas
    3. filter luas kotak minimum (buang noise jauh)

Catatan: "motorbike" pada soal = kelas COCO "motorcycle" (id 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ------------------------------------------------------------------ kelas --
COCO_CLASS_IDS: Dict[str, int] = {
    "person": 0,
    "car": 2,
    "motorbike": 3,      # COCO: "motorcycle"
    "truck": 7,
}

# ambang confidence per kelas (keamanan: person paling sensitif)
PER_CLASS_CONF: Dict[str, float] = {
    "person": 0.30,
    "car": 0.40,
    "truck": 0.40,
    "motorbike": 0.35,
}

# warna BGR per kelas (OpenCV)
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "person": (56, 168, 32),      # hijau
    "car": (232, 118, 26),        # biru
    "motorbike": (24, 92, 224),   # oranye
    "truck": (36, 36, 214),       # merah
}

DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
MIN_BOX_AREA = 900                # piksel² (~30x30)


@dataclass
class Detection:
    """Satu objek yang terdeteksi."""
    name: str
    conf: float
    box: Tuple[int, int, int, int]      # x1, y1, x2, y2

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


class SecurityDetector:
    """Detektor YOLO yang hanya mengeluarkan 4 kelas target."""

    def __init__(self, weights: str = "yolo11n.pt", imgsz: int = 640,
                 conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                 device: Optional[str] = None,
                 target_classes: Optional[List[str]] = None,
                 min_box_area: int = MIN_BOX_AREA):
        self.weights = weights or ""
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.min_box_area = min_box_area
        self.names: List[str] = target_classes or list(COCO_CLASS_IDS)
        self.class_ids: List[int] = sorted(COCO_CLASS_IDS[n] for n in self.names)
        self.id_to_name: Dict[int, str] = {COCO_CLASS_IDS[n]: n for n in self.names}
        self.model = None
        self.error = ""
        self._load()

    # ------------------------------------------------------------- loader --
    def _load(self) -> None:
        if not self.weights:
            self.error = "Mode kamera saja (bobot YOLO tidak dimuat)."
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
        except Exception as exc:                     # pragma: no cover
            self.error = f"Deteksi nonaktif: {exc}"
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    # ---------------------------------------------------------- inferensi --
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Kembalikan hanya deteksi 4 kelas target pada frame BGR."""
        if self.model is None:
            return []
        # pakai ambang terendah agar penyaringan per-kelas dilakukan sendiri
        conf_min = min([PER_CLASS_CONF.get(n, self.conf) for n in self.names]
                       + [self.conf])
        try:
            result = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=conf_min,
                iou=self.iou,
                classes=self.class_ids,              # <-- kunci: 4 kelas saja
                device=self.device,
                verbose=False,
            )[0]
        except Exception as exc:                     # pragma: no cover
            self.error = f"Inferensi gagal: {exc}"
            return []

        detections: List[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = self.id_to_name.get(class_id)
            if name is None:                         # lapis 1: kelas
                continue
            confidence = float(box.conf[0])
            if confidence < PER_CLASS_CONF.get(name, self.conf):   # lapis 2
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            det = Detection(name=name, conf=confidence, box=(x1, y1, x2, y2))
            if det.area < self.min_box_area:         # lapis 3: ukuran
                continue
            detections.append(det)
        return detections

    # ------------------------------------------------------------ anotasi --
    @staticmethod
    def draw(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Gambar kotak + label berwarna per kelas (in-place)."""
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = CLASS_COLORS_BGR.get(det.name, (90, 180, 36))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.name} {det.conf:.2f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                             0.55, 1)
            top = max(0, y1 - th - base - 4)
            cv2.rectangle(frame, (x1, top), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - base - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------ utilitas --
    @staticmethod
    def count_by_class(detections: List[Detection]) -> Dict[str, int]:
        counts = {name: 0 for name in COCO_CLASS_IDS}
        for det in detections:
            counts[det.name] = counts.get(det.name, 0) + 1
        return counts


def draw_overlay(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Gambar HUD: jumlah objek + waktu (dipakai saat menyimpan bukti)."""
    SecurityDetector.draw(frame, detections)
    total = len(detections)
    text = f"Objek: {total}   {cv2.getTickCount() and ''}".rstrip()
    cv2.rectangle(frame, (0, 0), (260, 30), (20, 24, 36), -1)
    cv2.putText(frame, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (235, 240, 255), 1, cv2.LINE_AA)
    return frame
