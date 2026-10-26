"""
config.py -- Konfigurasi terpusat untuk Security Camera Application.

Semua parameter yang biasanya di-hardcode (threshold confidence, kelas target,
durasi rekaman, dsb.) dikumpulkan di sini supaya mudah di-tune saat ETS tanpa
menyentuh logika program.

Catatan penting pemetaan kelas:
    COCO (dataset pre-training YOLO) memakai nama "motorcycle" (id 3),
    sedangkan soal meminta kelas "motorbike". Keduanya adalah objek yang sama.
    -> Kita petakan: motorbike == COCO motorcycle == class id 3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# --------------------------------------------------------------------------
# 1. Definisi kelas target
# --------------------------------------------------------------------------

# Nama kelas COCO -> id. Hanya 4 kelas ini yang diproses aplikasi.
#   person     = 0
#   car        = 2
#   motorcycle = 3   (disebut "motorbike" pada soal)
#   truck      = 7
COCO_CLASS_IDS: Dict[str, int] = {
    "person": 0,
    "car": 2,
    "motorbike": 3,   # COCO name: motorcycle
    "truck": 7,
}

# Alias agar konfigurasi tidak error kalau user menulis "motorcycle"
CLASS_ALIASES: Dict[str, str] = {
    "motorcycle": "motorbike",
    "motor": "motorbike",
    "bike": "motorbike",
    "people": "person",
    "human": "person",
    "mobil": "car",
    "truk": "truck",
}

# Warna BGR (OpenCV) untuk tiap kelas -> dipakai saat menggambar bounding box
CLASS_COLORS_BGR: Dict[str, tuple] = {
    "person": (56, 168, 32),      # hijau
    "car": (232, 118, 26),        # biru
    "motorbike": (24, 92, 224),   # oranye
    "truck": (36, 36, 214),       # merah
    "unknown": (128, 128, 128),
}

# Warna hex untuk komponen UI (light theme)
CLASS_COLORS_HEX: Dict[str, str] = {
    "person": "#20A840",
    "car": "#1A76E8",
    "motorbike": "#E05C18",
    "truck": "#D63636",
}


def normalize_class_names(names: Iterable[str]) -> List[str]:
    """Normalisasi ejaan user ('motorcycle', 'Motor ', 'MOBIL') -> nama kanonik."""
    out: List[str] = []
    for n in names:
        key = str(n).strip().lower()
        key = CLASS_ALIASES.get(key, key)
        if key in COCO_CLASS_IDS and key not in out:
            out.append(key)
    return out


def class_ids(names: Iterable[str]) -> List[int]:
    """Ambil daftar id kelas COCO (diurutkan) untuk dipassing ke YOLO(classes=[...])."""
    return sorted(COCO_CLASS_IDS[n] for n in normalize_class_names(names))


# --------------------------------------------------------------------------
# 2. Konfigurasi
# --------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Konfigurasi model pre-trained."""
    # Bisa berupa 'yolo11n.pt', 'yolov8n.pt', atau path ke best.pt hasil fine-tune
    weights: str = "yolo11n.pt"
    imgsz: int = 640              # ukuran input letterbox (640 = standar YOLO)
    conf: float = 0.35            # confidence threshold global
    iou: float = 0.45             # IoU threshold untuk NMS
    device: str = ""              # "" = auto (cuda kalau ada), atau "cpu", "0"
    half: bool = False            # FP16 (butuh GPU)
    # Threshold per-kelas (opsional). Kalau kosong -> pakai ModelConfig.conf.
    # Keamanan: person biasanya dibuat lebih sensitif (recall tinggi).
    per_class_conf: Dict[str, float] = field(default_factory=lambda: {
        "person": 0.30,
        "car": 0.40,
        "truck": 0.40,
        "motorbike": 0.35,
    })
    # Kelas yang diproses. Selain ini -> dibuang di level model (paling cepat).
    target_classes: List[str] = field(default_factory=lambda: [
        "person", "car", "truck", "motorbike",
    ])
    # Tracker untuk menjaga ID antar-frame (bytetrack / botsort). "" = nonaktif.
    tracker: str = "bytetrack.yaml"

    @property
    def class_ids(self) -> List[int]:
        return class_ids(self.target_classes)

    @property
    def names(self) -> List[str]:
        return normalize_class_names(self.target_classes)


@dataclass
class TriggerConfig:
    """Logika kondisional kapan rekaman/snapshot dipicu."""
    # Jumlah frame berturut-turut berisi target sebelum trigger aktif
    # (= debounce, menekan false alarm akibat flicker 1 frame).
    min_consecutive_frames: int = 3
    # Waktu jeda minimal antar dua event (detik) -> mencegah spam file.
    cooldown_seconds: float = 5.0
    # Rekaman menyimpan N detik SEBELUM trigger (pre-roll buffer).
    pre_roll_seconds: float = 3.0
    # Rekaman berhenti setelah N detik tanpa ada target lagi (post-roll).
    post_roll_seconds: float = 4.0
    # Batas keras durasi 1 file rekaman (detik), lalu rotasi ke file baru.
    max_clip_seconds: float = 60.0
    # Jeda minimal antar snapshot (detik).
    snapshot_cooldown_seconds: float = 2.0
    # Abaikan deteksi dengan box lebih kecil dari ini (px^2) -> kurangi noise jauh.
    min_box_area: int = 900       # ~30x30 px
    # Opsional: zona deteksi (polygon). Kosong = seluruh frame.
    roi_polygon: List[List[int]] = field(default_factory=list)
    # Simpan hanya jika ada kelas tertentu? Kosong = semua kelas target.
    trigger_on_classes: List[str] = field(default_factory=list)


@dataclass
class RecordingConfig:
    """Konfigurasi output file."""
    output_dir: str = "runs/capture"
    save_snapshot: bool = True
    save_video: bool = True
    video_fps: float = 20.0
    # FourCC: mp4v paling kompatibel; avc1 (H.264) lebih kecil butuh ffmpeg.
    fourcc: str = "mp4v"
    jpeg_quality: int = 90
    # Simpan frame ASLI (tanpa overlay box) atau frame yang sudah dianotasi?
    # Untuk barang bukti: simpan yang dianotasi (mudah dibaca manusia).
    annotate_snapshot: bool = True
    annotate_video: bool = True
    write_csv_log: bool = True


@dataclass
class CameraConfig:
    """Konfigurasi sumber frame."""
    source: str = "0"             # "0" = webcam, atau path video/gambar, atau URL RTSP
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Jalankan inferensi setiap N frame (1 = tiap frame). Naikkan kalau CPU lambat.
    frame_skip: int = 1
    flip_horizontal: bool = False


@dataclass
class AppConfig:
    """Gabungan seluruh konfigurasi."""
    model: ModelConfig = field(default_factory=ModelConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    # Tema UI: "light" (wajib, sesuai soal), tersedia juga "dark"/"system".
    appearance: str = "light"
    ui_scale: float = 1.0

    # -- serialisasi -------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model=ModelConfig(**data.get("model", {})),
            trigger=TriggerConfig(**data.get("trigger", {})),
            recording=RecordingConfig(**data.get("recording", {})),
            camera=CameraConfig(**data.get("camera", {})),
            appearance=data.get("appearance", "light"),
            ui_scale=data.get("ui_scale", 1.0),
        )


# --------------------------------------------------------------------------
# 3. Preset siap pakai (tinggal dipilih saat demo / ujian)
# --------------------------------------------------------------------------
def preset(name: str = "balanced") -> AppConfig:
    """
    Preset parameter.

    balanced  : default demo, seimbang antara recall & precision.
    max_recall: prioritas utama "jangan sampai ada penyusup lolos" (recall↑, FA↑).
    low_false_alarm: untuk area ramai, menekan alarm palsu (precision↑).
    fast      : laptop tanpa GPU, inference lebih ringan (imgsz 416, skip frame).
    """
    cfg = AppConfig()

    if name == "max_recall":
        cfg.model.conf = 0.20
        cfg.model.per_class_conf = {k: 0.20 for k in COCO_CLASS_IDS}
        cfg.model.iou = 0.50
        cfg.trigger.min_consecutive_frames = 2
        cfg.trigger.cooldown_seconds = 3.0
        cfg.trigger.post_roll_seconds = 6.0
        cfg.trigger.min_box_area = 400
    elif name == "low_false_alarm":
        cfg.model.conf = 0.55
        cfg.model.per_class_conf = {"person": 0.50, "car": 0.60,
                                    "truck": 0.60, "motorbike": 0.55}
        cfg.trigger.min_consecutive_frames = 6
        cfg.trigger.cooldown_seconds = 10.0
        cfg.trigger.min_box_area = 2500
    elif name == "fast":
        cfg.model.weights = "yolo11n.pt"
        cfg.model.imgsz = 416
        cfg.camera.frame_skip = 2
        cfg.camera.width, cfg.camera.height = 640, 480
        cfg.recording.video_fps = 12.0

    return cfg


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    c = preset(name)
    print(f"--- preset: {name} ---")
    print(f"target classes : {c.model.names}")
    print(f"COCO class ids : {c.model.class_ids}   <-- diteruskan ke model(classes=[...])")
    print(json.dumps(c.to_dict(), indent=2)[:600])
