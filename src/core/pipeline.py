"""
pipeline.py -- State machine yang menghubungkan kamera -> detektor -> perekam.

Ini adalah "otak" logika kondisional yang diminta soal:

    AMATI (tiap frame)
      ├─ ada target?  -> streak++            ; tidak -> streak = 0
      ├─ streak >= min_consecutive_frames DAN tidak sedang rekam DAN
      │  sudah lewat cooldown  ==> START RECORD  (+ simpan snapshot)
      ├─ sedang rekam & ada target  ==> perbarui last_hit (post-roll di-reset)
      ├─ sedang rekam & tidak ada target > post_roll_seconds ==> STOP RECORD
      └─ durasi > max_clip_seconds ==> rotasi file (stop lalu start baru)

State: IDLE  ->  DETECTING  ->  RECORDING  ->  (COOLDOWN)  ->  IDLE
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import AppConfig, preset
from .detector import Detection, FrameResult, SecurityDetector
from .recorder import EventLogger, RingBuffer, SnapshotWriter, VideoRecorder


class State(str, Enum):
    IDLE = "IDLE"
    DETECTING = "DETECTING"
    RECORDING = "RECORDING"
    COOLDOWN = "COOLDOWN"


@dataclass
class PipelineStats:
    frames: int = 0
    detections: int = 0
    events: int = 0
    fps_ema: float = 0.0
    last_inference_ms: float = 0.0
    last_counts: Dict[str, int] = field(default_factory=dict)


class SurveillancePipeline:
    """
    Contoh pakai (tanpa GUI):

        pipe = SurveillancePipeline(cfg)
        for frame in camera_frames:
            out = pipe.process(frame)
            if out["event_started"]: print("REC mulai:", out["clip"])
        pipe.close()
    """

    def __init__(self, cfg: Optional[AppConfig] = None,
                 detector=None,
                 on_event: Optional[Callable[[str, dict], None]] = None):
        self.cfg = cfg or preset("balanced")
        self.detector = detector or SecurityDetector(self.cfg.model)
        self.on_event = on_event

        out = Path(self.cfg.recording.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.recorder = VideoRecorder(out, self.cfg.recording.fourcc)
        self.snapper = SnapshotWriter(out,
                                      self.cfg.trigger.snapshot_cooldown_seconds,
                                      self.cfg.recording.jpeg_quality)
        self.logger = EventLogger(out, self.cfg.recording.write_csv_log)
        self.buffer = RingBuffer(self.cfg.trigger.pre_roll_seconds,
                                 self.cfg.recording.video_fps)

        # --- state machine ---
        self.state = State.IDLE
        self.streak = 0
        self.last_hit_time = 0.0
        self.record_start_time = 0.0
        self.last_stop_time = 0.0
        self._frame_idx = 0
        self._prev_tick = time.time()
        self.stats = PipelineStats()

    # ------------------------------------------------------------------
    def _passes_trigger(self, res: FrameResult) -> bool:
        """Apakah frame ini memenuhi syarat untuk memicu/memanjangkan rekaman?"""
        if res.n == 0:
            return False
        only = self.cfg.trigger.trigger_on_classes
        if only:
            res.detections = [d for d in res.detections if d.name in only]
            if not res.detections:
                return False
        return True

    def _should_start(self, res: FrameResult, now: float) -> bool:
        enough_streak = self.streak >= self.cfg.trigger.min_consecutive_frames
        cooled_down = (now - self.last_stop_time) >= self.cfg.trigger.cooldown_seconds
        return (enough_streak and cooled_down and not self.recorder.is_recording
                and self._passes_trigger(res))

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray,
                draw: bool = True) -> dict:
        """
        Proses satu frame. Mengembalikan dict berisi:
            frame         : frame (sudah dianotasi bila draw=True)
            result        : FrameResult
            state         : State
            event_started : bool   -> True pada frame pertama rekaman dimulai
            event_stopped : bool
            clip          : Path klip yang baru selesai
            snapshot      : Path snapshot baru (bila ada)
        """
        now = time.time()
        self._frame_idx += 1
        out = {"frame": frame, "result": FrameResult(), "state": self.state,
               "event_started": False, "event_stopped": False,
               "clip": None, "snapshot": None}

        # ---- FPS pengukur (exponential moving average) ----
        dt = now - self._prev_tick
        self._prev_tick = now
        if dt > 0:
            inst = 1.0 / dt
            self.stats.fps_ema = (0.9 * self.stats.fps_ema + 0.1 * inst
                                  if self.stats.fps_ema else inst)
        self.stats.frames += 1

        # ---- (0) selalu simpan frame ke pre-roll buffer (kecuali saat REC) ----
        if not self.recorder.is_recording:
            self.buffer.push(frame.copy())

        # ---- (1) INFERENSI (bisa dilewati sesuai frame_skip) ----
        res = FrameResult()
        if self.cfg.camera.frame_skip <= 1 or self._frame_idx % self.cfg.camera.frame_skip == 0:
            res = self.detector.predict(frame)
            # filter tambahan: luas box minimum & ROI
            min_area = self.cfg.trigger.min_box_area
            res.detections = [d for d in res.detections if d.area >= min_area]
            self.detector.filter_by_roi(res, self.cfg.trigger.roi_polygon)
            res.counts = {k: 0 for k in res.counts}
            for d in res.detections:
                res.counts[d.name] = res.counts.get(d.name, 0) + 1
            self.stats.last_inference_ms = res.inference_ms
            self.stats.last_counts = res.counts
            self.stats.detections += res.n
        else:
            res = FrameResult(counts=dict(self.stats.last_counts))
        out["result"] = res

        # ---- (2) update streak (debounce) ----
        hit = self._passes_trigger(res)
        self.streak = self.streak + 1 if hit else 0
        if hit:
            self.last_hit_time = now

        # ---- (3) TRANSISI STATE ----
        if self.recorder.is_recording:
            # rotasi paksa bila durasi maksimum tercapai
            if now - self.record_start_time >= self.cfg.trigger.max_clip_seconds:
                self._stop(now, reason="max_duration")
                out["event_stopped"] = True
                out["clip"] = self.recorder.history[-1].path if self.recorder.history else None
                if hit:  # masih ada objek -> langsung mulai klip baru
                    self._start(frame, res, now)
                    out["event_started"] = True
            elif not hit and (now - self.last_hit_time) >= self.cfg.trigger.post_roll_seconds:
                self._stop(now, reason="post_roll")
                out["event_stopped"] = True
                out["clip"] = self.recorder.history[-1].path if self.recorder.history else None
                self.state = State.COOLDOWN
            else:
                # frame untuk rekaman: pakai versi teranotasi bila diminta
                self.recorder.write(out["frame"] if not draw else out["frame"])
        else:
            if self._should_start(res, now):
                self._start(frame, res, now)
                out["event_started"] = True
                if self.cfg.recording.save_snapshot:
                    snap_frame = out["frame"] if not draw else out["frame"]
                    tag = "_".join(sorted(res.counts))
                    p = self.snapper.save(snap_frame, tag=tag)
                    if p:
                        out["snapshot"] = p
                        self._emit("SNAPSHOT", {"file": p.name})
                self._emit("DETECTED", {
                    "classes": ",".join(k for k, v in res.counts.items() if v),
                    "conf": max((d.conf for d in res.detections), default=0.0),
                    "n": res.n,
                })
            else:
                self.state = State.DETECTING if hit else (
                    State.COOLDOWN
                    if (now - self.last_stop_time) < self.cfg.trigger.cooldown_seconds
                    else State.IDLE)

        # ---- (4) gambar overlay & HUD ----
        if draw:
            out["frame"] = self._annotate(out["frame"], res)

        out["state"] = self.state
        return out

    # ------------------------------------------------------------------
    def _start(self, frame, res: FrameResult, now: float) -> None:
        self.state = State.RECORDING
        self.record_start_time = now
        pre = self.buffer.snapshot() if self.cfg.recording.save_video else None
        if self.cfg.recording.save_video:
            h, w = frame.shape[:2]
            self.recorder.start((w, h), self.cfg.recording.video_fps,
                                trigger=",".join(sorted(res.counts)),
                                pre_roll=pre)
            self.stats.events += 1
            self._emit("REC_START", {
                "classes": ",".join(k for k, v in res.counts.items() if v),
                "conf": max((d.conf for d in res.detections), default=0.0),
                "n": res.n,
                "file": self.recorder.current.path.name if self.recorder.current else "",
            })
        self.buffer.clear()

    def _stop(self, now: float, reason: str = "") -> None:
        clip = self.recorder.stop()
        self.last_stop_time = now
        self.streak = 0
        if clip:
            self._emit("REC_STOP", {
                "classes": clip.trigger, "n": clip.frames,
                "file": clip.path.name,
            })

    def _emit(self, event: str, data: dict) -> None:
        self.logger.log(event, data.get("classes", ""), data.get("conf", 0.0),
                        data.get("n", 0), data.get("file", ""))
        if self.on_event:
            try:
                self.on_event(event, data)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _annotate(self, frame: np.ndarray, res: FrameResult) -> np.ndarray:
        try:
            import cv2
        except ImportError:
            return frame
        SecurityDetector.draw(frame, res)
        # HUD: state + FPS + waktu
        h, w = frame.shape[:2]
        banner_h = 34
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (28, 30, 36), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        state = self.state.value
        color = {"IDLE": (200, 200, 200), "DETECTING": (60, 190, 250),
                 "RECORDING": (60, 60, 240), "COOLDOWN": (120, 200, 120)}.get(state, (255, 255, 255))
        txt = f"{state}   FPS {self.stats.fps_ema:.1f}   {datetime.now():%H:%M:%S}"
        cv2.putText(frame, txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if self.state == State.RECORDING:
            blink = int(time.time() * 2) % 2 == 0
            if blink:
                cv2.circle(frame, (w - 26, banner_h // 2), 8, (60, 60, 240), -1)
            cv2.putText(frame, "REC", (w - 78, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (60, 60, 240), 2, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.recorder.is_recording:
            self._stop(time.time(), reason="shutdown")
        self.recorder.release()
