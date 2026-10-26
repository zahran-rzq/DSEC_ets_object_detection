"""
metrics.py -- Implementasi metrik evaluasi deteksi objek dari nol.

Ditulis manual (tanpa bergantung pada library metrik) supaya:
  * jelas secara konseptual saat presentasi / ujian,
  * bisa dijalankan di notebook Colab untuk memverifikasi angka yang
    dikeluarkan oleh Ultralytics (mAP50, mAP50-95).

Berisi:
  iou()                 : Intersection over Union untuk 2 box
  match_detections()    : pencocokan prediksi <-> ground truth (greedy, per kelas)
  classification_counts(): TP / FP / FN pada satu ambang IoU
  precision_recall()    : Precision & Recall
  average_precision()   : AP (11-point & all-point interpolation)
  mean_average_precision(): mAP@IoU_thr dan mAP@0.50:0.95 (standar COCO)
  evaluate()            : rekap lengkap + kurva PR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Box = Sequence[float]          # [x1, y1, x2, y2]


# ==========================================================================
# 1. Intersection over Union (IoU)
# ==========================================================================
def iou(box_a: Box, box_b: Box) -> float:
    """
    IoU = |A ∩ B| / |A ∪ B|   (alias Jaccard Index)

    Nilai 0..1. Makin besar = prediksi makin cocok dengan ground truth.
    """
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(preds: Sequence[Box], gts: Sequence[Box]) -> np.ndarray:
    """Matriks IoU berukuran (n_pred, n_gt)."""
    return np.array([[iou(p, g) for g in gts] for p in preds], dtype=float) \
        if preds and gts else np.zeros((len(preds), len(gts)), dtype=float)


# ==========================================================================
# 2. Pencocokan prediksi <-> ground truth
# ==========================================================================
def match_detections(pred_boxes: Sequence[Box],
                     pred_scores: Sequence[float],
                     pred_classes: Sequence[int],
                     gt_boxes: Sequence[Box],
                     gt_classes: Sequence[int],
                     iou_threshold: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    Greedy matching per kelas, diurutkan berdasar confidence tertinggi.

    Aturan (standar PASCAL VOC / COCO):
      * prediksi & GT harus kelas yang sama,
      * IoU(pred, GT) >= threshold,
      * satu GT hanya boleh diklaim sekali (sisanya jadi False Positive),
      * GT yang tak terklaim = False Negative.

    Return: (indeks GT yang cocok utk tiap pred, -1 bila FP,
             indeks pred yang cocok utk tiap GT, -1 bila FN)
    """
    n_p, n_g = len(pred_boxes), len(gt_boxes)
    pred_match = [-1] * n_p
    gt_match = [-1] * n_g
    if n_p == 0 or n_g == 0:
        return pred_match, gt_match

    order = np.argsort(-np.asarray(pred_scores, dtype=float))
    for pi in order:
        best_iou, best_gi = iou_threshold, -1
        for gi in range(n_g):
            if gt_match[gi] != -1:               # GT sudah diklaim
                continue
            if pred_classes[pi] != gt_classes[gi]:
                continue
            v = iou(pred_boxes[pi], gt_boxes[gi])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0:
            pred_match[pi] = best_gi
            gt_match[best_gi] = pi
    return pred_match, gt_match


# ==========================================================================
# 3. Precision, Recall, F-score
# ==========================================================================
def precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    """Precision = TP/(TP+FP)  ;  Recall = TP/(TP+FN)."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec


def f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    """
    F-beta score. beta > 1 memberi bobot lebih pada Recall.
    Untuk sistem keamanan, F2 (beta=2) lebih tepat daripada F1,
    karena harga False Negative (penyusup lolos) jauh lebih mahal
    daripada False Positive (alarm palsu).
    """
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def classification_counts(pred_boxes, pred_scores, pred_classes,
                          gt_boxes, gt_classes,
                          iou_threshold: float = 0.5,
                          conf_threshold: float = 0.001) -> Dict[str, int]:
    """Hitung TP, FP, FN pada satu operating point."""
    keep = [i for i, s in enumerate(pred_scores) if s >= conf_threshold]
    pb = [pred_boxes[i] for i in keep]
    ps = [pred_scores[i] for i in keep]
    pc = [pred_classes[i] for i in keep]
    pred_match, gt_match = match_detections(pb, ps, pc, gt_boxes, gt_classes,
                                            iou_threshold)
    tp = sum(1 for m in pred_match if m >= 0)
    fp = sum(1 for m in pred_match if m < 0)
    fn = sum(1 for m in gt_match if m < 0)
    return {"tp": tp, "fp": fp, "fn": fn}


# ==========================================================================
# 4. Average Precision (AP) & mean Average Precision (mAP)
# ==========================================================================
def pr_curve(pred_boxes, pred_scores, pred_classes,
             gt_boxes, gt_classes,
             iou_threshold: float = 0.5,
             class_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kurva Precision-Recall untuk satu kelas (atau semua kelas bila class_id=None).
    Dikembalikan sebagai (recall[], precision[]) yang sudah terurut ascending.
    """
    if class_id is not None:
        idx = [i for i, c in enumerate(pred_classes) if c == class_id]
        pred_boxes = [pred_boxes[i] for i in idx]
        pred_scores = [pred_scores[i] for i in idx]
        pred_classes = [pred_classes[i] for i in idx]
        gidx = [i for i, c in enumerate(gt_classes) if c == class_id]
        gt_boxes = [gt_boxes[i] for i in gidx]
        gt_classes = [gt_classes[i] for i in gidx]

    n_gt = len(gt_boxes)
    if n_gt == 0:
        return np.array([0.0, 1.0]), (np.array([1.0, 0.0]) if not pred_boxes
                                      else np.array([0.0, 0.0]))
    if not pred_boxes:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    order = np.argsort(-np.asarray(pred_scores, dtype=float))
    tp = np.zeros(len(order))
    fp = np.zeros(len(order))
    used = set()

    for rank, pi in enumerate(order):
        best_iou, best_gi = iou_threshold, -1
        for gi in range(n_gt):
            if gi in used or pred_classes[pi] != gt_classes[gi]:
                continue
            v = iou(pred_boxes[pi], gt_boxes[gi])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0:
            tp[rank] = 1
            used.add(best_gi)
        else:
            fp[rank] = 1

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # monotonic decreasing (standar VOC): p[i] = max(p[i:])
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return np.concatenate(([0.0], rec, [1.0])), np.concatenate(([1.0], prec, [0.0]))


def average_precision(pred_boxes, pred_scores, pred_classes,
                      gt_boxes, gt_classes,
                      iou_threshold: float = 0.5,
                      class_id: Optional[int] = None,
                      method: str = "all") -> float:
    """
    AP = luas di bawah kurva Precision-Recall.

    method='all'      : all-point interpolation (VOC 2010+, dipakai COCO)
    method='11point'  : interpolasi 11 titik (VOC 2007)
    """
    rec, prec = pr_curve(pred_boxes, pred_scores, pred_classes,
                         gt_boxes, gt_classes, iou_threshold, class_id)
    if method == "11point":
        pts = np.linspace(0, 1, 11)
        return float(np.mean([prec[rec >= t].max() if np.any(rec >= t) else 0.0
                              for t in pts]))
    # all-point: integral berdasar renggang recall
    order = np.argsort(rec)
    r, p = rec[order], prec[order]
    return float(np.sum(np.diff(r) * p[1:]))


def mean_average_precision(pred_boxes, pred_scores, pred_classes,
                           gt_boxes, gt_classes,
                           iou_thresholds: Sequence[float] = (0.5,),
                           class_ids: Optional[Iterable[int]] = None,
                           verbose: bool = False) -> Dict[str, float]:
    """
    mAP = rata-rata AP terhadap seluruh kelas (dan seluruh ambang IoU).

    mAP@0.50      -> satu ambang "longgar", fokus pada keberadaan deteksi.
    mAP@0.50:0.95 -> rata-rata atas 10 ambang IoU (0.50, 0.55, ..., 0.95),
                     metrik utama COCO; ketat pada kualitas lokalisasi.
    """
    all_cls = sorted(set(list(gt_classes) + list(pred_classes))) \
        if class_ids is None else list(class_ids)
    out: Dict[str, float] = {}
    aps_per_thr = []
    for thr in iou_thresholds:
        aps = [average_precision(pred_boxes, pred_scores, pred_classes,
                                 gt_boxes, gt_classes, thr, c)
               for c in all_cls]
        aps = [a for a in aps if not np.isnan(a)]
        val = float(np.mean(aps)) if aps else 0.0
        out[f"AP@IoU={thr:.2f}"] = val
        out[f"AP50" if abs(thr - 0.5) < 1e-9 else f"AP{int(round(thr*100))}"] = val
        aps_per_thr.append(val)
    out["mAP"] = float(np.mean(aps_per_thr)) if aps_per_thr else 0.0
    if verbose:
        for c in all_cls:
            print(f"  class {c}: AP@0.50 = "
                  f"{average_precision(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, 0.5, c):.4f}")
    return out


COCO_IOU_THRESHOLDS = tuple(np.arange(0.50, 1.00, 0.05))


# ==========================================================================
# 5. Rekap evaluasi
# ==========================================================================
@dataclass
class EvalReport:
    per_class: Dict[int, dict] = field(default_factory=dict)
    overall: Dict[str, float] = field(default_factory=dict)
    curve: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=========================================================",
                 "                 LAPORAN EVALUASI DETEKSI",
                 "=========================================================",
                 f"{'Kelas':<12}{'GT':>6}{'Pred':>7}{'TP':>6}{'FP':>6}{'FN':>6}"
                 f"{'Prec':>9}{'Recall':>9}{'AP50':>9}{'AP50-95':>10}",
                 "---------------------------------------------------------"]
        for c, d in self.per_class.items():
            lines.append(
                f"{str(c):<12}{d['gt']:>6}{d['pred']:>7}{d['tp']:>6}{d['fp']:>6}"
                f"{d['fn']:>6}{d['precision']:>9.3f}{d['recall']:>9.3f}"
                f"{d['ap50']:>9.3f}{d['ap5095']:>10.3f}")
        lines.append("---------------------------------------------------------")
        o = self.overall
        lines.append(f"mAP@0.50      : {o.get('map50', 0):.4f}")
        lines.append(f"mAP@0.50:0.95 : {o.get('map5095', 0):.4f}")
        lines.append(f"Precision     : {o.get('precision', 0):.4f}")
        lines.append(f"Recall        : {o.get('recall', 0):.4f}")
        lines.append(f"F1            : {o.get('f1', 0):.4f}")
        lines.append(f"F2 (recall-berbobot, metrik utama keamanan) : "
                     f"{o.get('f2', 0):.4f}")
        lines.append("=========================================================")
        return "\n".join(lines)


def evaluate(pred_boxes, pred_scores, pred_classes,
             gt_boxes, gt_classes,
             conf_threshold: float = 0.25,
             class_ids: Optional[Iterable[int]] = None) -> EvalReport:
    """
    Hitung laporan lengkap: per kelas + keseluruhan.

    pred_* / gt_* adalah list sepanjang N yang disejajarkan per gambar:
      pred_boxes[i], pred_scores[i], pred_classes[i] -> prediksi pada gambar i
      gt_boxes[i],   gt_classes[i]                  -> ground truth gambar i
    Cara sederhana: gabungkan seluruh dataset jadi satu "gambar besar"
    (cara ini identik dengan perhitungan micro-average dan cukup untuk
    memahami metrik; COCO sendiri menghitung per gambar lalu dirata-rata).
    """
    # flatten
    PB, PS, PC, GB, GC, IMG = [], [], [], [], [], []
    for i in range(len(gt_boxes)):
        for b, s, c in zip(pred_boxes[i], pred_scores[i], pred_classes[i]):
            if s >= conf_threshold:
                PB.append(b); PS.append(float(s)); PC.append(int(c)); IMG.append(i)
        for b, c in zip(gt_boxes[i], gt_classes[i]):
            GB.append(b); GC.append(int(c))

    all_cls = sorted(set(GC) | set(PC)) if class_ids is None else list(class_ids)
    rep = EvalReport()

    tot = {"tp": 0, "fp": 0, "fn": 0}
    ap50s, ap5095s = [], []
    for c in all_cls:
        gt_n = sum(1 for x in GC if x == c)
        pr_n = sum(1 for x in PC if x == c)
        r = classification_counts(PB, PS, PC, GB, GC, 0.5, conf_threshold) \
            if False else _class_counts(PB, PS, PC, GB, GC, c, 0.5)
        prec, rec = precision_recall(r["tp"], r["fp"], r["fn"])
        ap50 = average_precision(PB, PS, PC, GB, GC, 0.5, c)
        ap5095 = float(np.mean([average_precision(PB, PS, PC, GB, GC, t, c)
                                for t in COCO_IOU_THRESHOLDS]))
        rec_c, prec_c = pr_curve(PB, PS, PC, GB, GC, 0.5, c)
        rep.per_class[c] = {"gt": gt_n, "pred": pr_n, **r,
                            "precision": prec, "recall": rec,
                            "ap50": ap50, "ap5095": ap5095}
        rep.curve[str(c)] = (rec_c, prec_c)
        tot["tp"] += r["tp"]; tot["fp"] += r["fp"]; tot["fn"] += r["fn"]
        ap50s.append(ap50); ap5095s.append(ap5095)

    p, rc = precision_recall(tot["tp"], tot["fp"], tot["fn"])
    rep.overall = {
        "tp": tot["tp"], "fp": tot["fp"], "fn": tot["fn"],
        "precision": p, "recall": rc,
        "f1": f_beta(p, rc, 1.0),
        "f2": f_beta(p, rc, 2.0),
        "map50": float(np.mean(ap50s)) if ap50s else 0.0,
        "map5095": float(np.mean(ap5095s)) if ap5095s else 0.0,
    }
    return rep


def _class_counts(pred_boxes, pred_scores, pred_classes,
                  gt_boxes, gt_classes, class_id: int,
                  iou_threshold: float) -> Dict[str, int]:
    """TP/FP/FN untuk satu kelas (prediksi kelas lain dianggap tidak ada)."""
    pi = [i for i, c in enumerate(pred_classes) if c == class_id]
    gi = [i for i, c in enumerate(gt_classes) if c == class_id]
    pb = [pred_boxes[i] for i in pi]
    ps = [pred_scores[i] for i in pi]
    pc = [pred_classes[i] for i in pi]
    gb = [gt_boxes[i] for i in gi]
    gc = [gt_classes[i] for i in gi]
    pm, gm = match_detections(pb, ps, pc, gb, gc, iou_threshold)
    return {"tp": sum(1 for m in pm if m >= 0),
            "fp": sum(1 for m in pm if m < 0),
            "fn": sum(1 for m in gm if m < 0)}


# ==========================================================================
# 6. Demo / self-test
# ==========================================================================
if __name__ == "__main__":
    # Contoh mainan: 2 gambar, kelas 0 = person
    gt_boxes = [[[10, 10, 60, 60], [100, 100, 160, 160]],
                [[20, 20, 70, 70]]]
    gt_classes = [[0, 0], [0]]
    pred_boxes = [[[12, 12, 58, 58, ], [300, 300, 340, 340]],
                  [[25, 25, 68, 68]]]
    pred_scores = [[0.95, 0.40], [0.80]]
    pred_classes = [[0, 0], [0]]

    print("IoU contoh:", round(iou([0, 0, 10, 10], [0, 0, 10, 10]), 3),
          "| setengah geser:", round(iou([0, 0, 10, 10], [5, 0, 15, 10]), 3),
          "| tak bersinggungan:", round(iou([0, 0, 10, 10], [50, 50, 60, 60]), 3))

    rep = evaluate(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes)
    print(rep.summary())
