"""
make_figures.py -- Menghasilkan seluruh ilustrasi untuk laporan PDF.

Menjalankan:
    python tools/make_figures.py

Keluaran (di docs/gambar/):
    fig_arsitektur.png   diagram arsitektur sistem berlapis
    fig_alur_kerja.png   diagram alur (flowchart) pemrosesan frame
    fig_state.png        diagram transisi state pemicu rekaman
    fig_iou.png          ilustrasi Intersection over Union
    fig_kurva_pr.png     kurva Precision-Recall dan luas AP
    fig_tradeoff.png     precision/recall vs confidence threshold + F2
    fig_peta_metrik.png  peta hubungan IoU -> TP/FP/FN -> Precision/Recall/mAP
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "gambar"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#FFFFFF"
ACCENT = "#2563EB"
ACCENT_S = "#E7EEFD"
DARK = "#1B2430"
MUTED = "#6B7688"
BORDER = "#CBD5E1"
GREEN = "#16A34A"
ORANGE = "#D97706"
RED = "#DC2626"


def _box(ax, x, y, w, h, title, lines, color=ACCENT_S, edge=ACCENT,
         tcolor=DARK, fs=9.5, tfs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
                                linewidth=1.6, edgecolor=edge, facecolor=color,
                                mutation_scale=1, zorder=2))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top",
            fontsize=tfs, fontweight="bold", color=tcolor, zorder=3)
    if lines:
        ax.text(x + w / 2, y + h - 0.42, "\n".join(lines), ha="center", va="top",
                fontsize=fs, color=MUTED, zorder=3, linespacing=1.5)


def _arrow(ax, x1, y1, x2, y2, color=ACCENT, style="-|>", lw=1.8, label=None,
           ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, linewidth=lw,
                                 color=color, linestyle=ls, zorder=4,
                                 shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2 + 0.02, (y1 + y2) / 2, label, fontsize=8.5,
                color=color, ha="left", va="center", zorder=5)


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=170)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    return fig, ax


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor=BG, pad_inches=0.16)
    plt.close(fig)
    print(f"   ✅ {name:<24} {path.stat().st_size/1024:6.0f} KB")


# =====================================================================
def fig_arsitektur():
    fig, ax = _canvas(10, 9.2)
    ax.text(5, 9.75, "Arsitektur Sistem SecureVision", ha="center",
            fontsize=15, fontweight="bold", color=DARK)

    layers = [
        ("LAPISAN AKUISISI", ["Webcam / CCTV RTSP  ->  OpenCV VideoCapture",
                              "frame BGR 1280x720, pembacaan di thread terpisah"],
         "#E7EEFD", ACCENT),
        ("LAPISAN INFERENSI (YOLO)",
         ["letterbox 640 -> normalisasi -> forward pass -> NMS (IoU 0,45)",
          "FILTER KELAS classes=[0,2,3,7] -> conf per kelas -> filter area/ROI"],
         "#DCFCE7", GREEN),
        ("LAPISAN LOGIKA (State Machine)",
         ["IDLE -> DETECTING -> RECORDING -> COOLDOWN",
          "debounce 3 frame | cooldown 5 s | post-roll 4 s | rotasi klip 60 s"],
         "#FEF3C7", ORANGE),
        ("LAPISAN PENYIMPANAN BUKTI",
         ["snapshot .jpg  |  klip .mp4 (pre-roll 3 s)  |  events.csv"],
         "#FEE2E2", RED),
        ("LAPISAN ANTARMUKA (GUI)",
         ["feed video + kotak berwarna | status deteksi | indikator REC",
          "log kejadian | tombol kontrol | slider Confidence / IoU"],
         "#E7EEFD", ACCENT),
    ]
    y = 8.85
    heights = [1.30, 1.52, 1.52, 1.10, 1.30]
    for (title, lines, c, e), hh in zip(layers, heights):
        _box(ax, 0.45, y - hh, 9.10, hh, title, lines, color=c, edge=e)
        y -= hh
        if y > 1.2:
            _arrow(ax, 5.0, y + 0.02, 5.0, y - 0.22)

    ax.text(5, 0.55, "Gambar 1 — Arsitektur berlapis aplikasi kamera keamanan",
            ha="center", fontsize=9.5, color=MUTED, style="italic")
    _save(fig, "fig_arsitektur.png")


# =====================================================================
def fig_alur_kerja():
    fig, ax = _canvas(10, 12.4)
    ax.text(5, 12.1, "Alur Kerja Sistem (per frame)", ha="center",
            fontsize=15, fontweight="bold", color=DARK)

    steps = [
        ("(1) Akuisisi frame", "cap.read() di thread background, antrean 2 frame"),
        ("(2) Ring buffer pre-roll", "3 detik terakhir selalu disimpan di memori"),
        ("(3) Preprocessing", "resize + letterbox 640x640, normalisasi [0,1]"),
        ("(4) Inferensi YOLO", "forward pass -> NMS (IoU 0,45) -> classes=[0,2,3,7]"),
        ("(5) Post-filter", "conf per kelas, luas kotak >= 900 px, ROI"),
        ("(6) Debounce (streak)", "hit streak++ bila ada target, else streak = 0"),
    ]
    y = 11.55
    for title, sub in steps:
        _box(ax, 0.55, y - 0.92, 8.90, 0.92, title, [sub],
             color=ACCENT_S, edge=ACCENT, tfs=10.5, fs=9)
        y -= 0.92
        _arrow(ax, 5.0, y + 0.02, 5.0, y - 0.16)

    # decision
    dy = y - 1.05
    ax.add_patch(FancyBboxPatch((2.15, dy), 5.70, 1.05,
                                boxstyle="round,pad=0.01,rounding_size=0.01",
                                linewidth=1.8, edgecolor=ORANGE,
                                facecolor="#FEF3C7", zorder=2))
    ax.text(5.0, dy + 0.72, "(7) Apakah streak >= 3", ha="center",
            fontsize=10.5, fontweight="bold", color=DARK)
    ax.text(5.0, dy + 0.34, "dan sudah lewat cooldown 5 detik?", ha="center",
            fontsize=9, color=MUTED)
    _arrow(ax, 5.0, dy + 1.05, 5.0, dy + 0.16)

    # kiri: tidak
    _arrow(ax, 2.15, dy + 0.52, 0.95, dy + 0.52)
    ax.text(1.0, dy + 0.78, "TIDAK", fontsize=9, fontweight="bold", color=RED,
            ha="center")
    ax.text(1.0, dy + 0.30, "kembali\nbaca frame", fontsize=8.2, color=MUTED,
            ha="center", linespacing=1.4)
    _arrow(ax, 0.95, dy + 0.10, 0.95, 0.55, ls="--")
    _arrow(ax, 0.95, 0.55, 5.0, 0.55, ls="--")
    _arrow(ax, 5.0, 0.55, 5.0, 0.80, ls="--")

    y2 = dy - 0.55
    _arrow(ax, 5.0, dy, 5.0, y2 + 0.90)
    ax.text(5.25, dy - 0.22, "YA", fontsize=9, fontweight="bold", color=GREEN)

    actions = [
        ("(8) Simpan snapshot JPG", "beranotasi + timestamp"),
        ("(9) Mulai rekam MP4", "flush pre-roll, lalu tulis frame baru"),
        ("(10) Hentikan setelah 4 s tanpa objek", "tutup berkas, catat events.csv"),
    ]
    for title, sub in actions:
        _box(ax, 0.55, y2 - 0.92, 8.90, 0.92, title, [sub],
             color="#DCFCE7", edge=GREEN, tfs=10.5, fs=9)
        y2 -= 0.92
        if title != actions[-1][0]:
            _arrow(ax, 5.0, y2 + 0.02, 5.0, y2 - 0.16)

    ax.text(5, y2 - 0.35, "Gambar 2 — Diagram alur pemrosesan frame dan logika "
                          "kondisional pemicu rekaman",
            ha="center", fontsize=9.5, color=MUTED, style="italic")
    _save(fig, "fig_alur_kerja.png")


# =====================================================================
def fig_state():
    fig, ax = _canvas(10, 5.6)
    ax.text(5, 5.35, "Diagram Transisi State Pemicu Rekaman", ha="center",
            fontsize=15, fontweight="bold", color=DARK)

    states = [
        (0.55, 3.05, "IDLE", "tidak ada target", ACCENT_S, ACCENT),
        (3.05, 3.05, "DETECTING", "target < 3 frame", "#FEF3C7", ORANGE),
        (5.55, 3.05, "RECORDING", "simpan JPG + MP4", "#FEE2E2", RED),
        (8.05, 3.05, "COOLDOWN", "jeda 5 detik", "#DCFCE7", GREEN),
    ]
    for x, y, name, sub, c, e in states:
        _box(ax, x, y, 1.85, 1.25, name, [sub], color=c, edge=e, tfs=11, fs=8.6)

    _arrow(ax, 2.40, 3.68, 3.05, 3.68, label="objek muncul")
    _arrow(ax, 4.90, 3.68, 5.55, 3.68, label="streak >= 3", color=RED)
    _arrow(ax, 7.40, 3.68, 8.05, 3.68, label="objek hilang > 4 s", color=GREEN)
    # kembali dari COOLDOWN ke RECORDING (event baru)
    ax.add_patch(FancyArrowPatch((8.97, 3.05), (6.47, 3.05),
                                 connectionstyle="arc3,rad=-0.55",
                                 arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.8, color=RED, zorder=4))
    ax.text(7.72, 2.42, "target muncul lagi\n(setelah cooldown)", fontsize=8.2,
            color=RED, ha="center", va="top", linespacing=1.4)
    # COOLDOWN -> IDLE
    ax.add_patch(FancyArrowPatch((8.05, 3.30), (2.40, 3.30),
                                 connectionstyle="arc3,rad=0.42",
                                 arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=ACCENT, ls="--", zorder=4))
    ax.text(5.22, 4.62, "cooldown habis, tidak ada target", fontsize=8.4,
            color=ACCENT, ha="center")
    # DETECTING -> IDLE
    _arrow(ax, 3.05, 3.42, 2.40, 3.42, color=MUTED, ls=":")
    ax.text(2.72, 3.62, "hilang", fontsize=7.8, color=MUTED, ha="center")
    # rotasi berkas
    ax.add_patch(FancyArrowPatch((6.05, 4.30), (7.05, 4.30),
                                 connectionstyle="arc3,rad=0.7",
                                 arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.5, color=ORANGE, zorder=4))
    ax.text(6.55, 4.78, "rotasi berkas > 60 s", fontsize=8.2, color=ORANGE,
            ha="center")

    ax.text(5, 0.35, "Gambar 3 — State machine: IDLE -> DETECTING -> RECORDING "
                     "-> COOLDOWN -> IDLE",
            ha="center", fontsize=9.5, color=MUTED, style="italic")
    _save(fig, "fig_state.png")


# =====================================================================
def fig_iou():
    cases = [([0, 0, 10, 10], [0, 0, 10, 10], "identik"),
             ([0, 0, 10, 10], [5, 0, 15, 10], "geser 50%"),
             ([0, 0, 10, 10], [8, 0, 18, 10], "geser 80%"),
             ([0, 0, 10, 10], [50, 50, 60, 60], "tidak bersinggungan")]

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua else 0.0

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.5), dpi=170)
    fig.patch.set_facecolor(BG)
    for ax, (a, b, label) in zip(axes, cases):
        ax.set_facecolor(BG)
        ax.add_patch(plt.Rectangle((a[0], a[1]), a[2]-a[0], a[3]-a[1],
                                   fill=False, edgecolor=ACCENT, lw=2.6,
                                   label="Prediksi"))
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                                   fill=False, edgecolor=RED, lw=2.6,
                                   ls="--", label="Ground truth"))
        v = iou(a, b)
        ax.set_title(f"IoU = {v:.2f}", fontsize=12, fontweight="bold", color=DARK)
        ax.set_xlabel(label, fontsize=9.5, color=MUTED)
        ax.set_xlim(-3, 24 if label != "tidak bersinggungan" else 68)
        ax.set_ylim(-3, 15 if label != "tidak bersinggungan" else 68)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.grid(alpha=.18)
    fig.suptitle("Ilustrasi Intersection over Union (IoU)",
                 fontsize=14, fontweight="bold", color=DARK, y=1.03)
    fig.text(0.5, -0.03,
             "Gambar 4 — IoU = |A ∩ B| / |A ∪ B|. Deteksi dihitung benar (TP) "
             "bila IoU >= 0,50 dan kelas sama.",
             ha="center", fontsize=9.5, color=MUTED, style="italic")
    _save(fig, "fig_iou.png")


# =====================================================================
def fig_kurva_pr():
    rng = np.random.default_rng(7)
    # kurva PR sintetis yang realistis
    rec = np.linspace(0, 1, 200)
    prec = 0.97 - 0.55 * rec ** 3.1 + rng.normal(0, 0.006, rec.size)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    prec = np.clip(prec, 0, 1)
    ap = float(np.sum(np.diff(np.concatenate(([0], rec))) * prec))

    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=170)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.plot(rec, prec, color=ACCENT, lw=2.6, label="Kurva Precision–Recall")
    ax.fill_between(rec, prec, alpha=0.16, color=ACCENT)
    ax.axhline(0.5, color=ORANGE, ls="--", lw=1.2)
    ax.text(0.02, 0.52, "contoh operating point (Recall 0,92)", fontsize=8.5,
            color=ORANGE)
    ax.plot([0.92], [0.62], "o", color=RED, ms=8, zorder=5)
    ax.annotate("Recall = 0,92\n(pilihan untuk keamanan)", xy=(0.92, 0.62),
                xytext=(0.55, 0.86), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlabel("Recall  =  TP / (TP + FN)", fontsize=11)
    ax.set_ylabel("Precision  =  TP / (TP + FP)", fontsize=11)
    ax.set_title(f"Kurva Precision–Recall dan Average Precision\nAP@0,50 = {ap:.3f}",
                 fontsize=12.5, fontweight="bold", color=DARK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.04)
    ax.grid(alpha=.25); ax.legend(fontsize=9.5, loc="lower left")
    fig.text(0.5, -0.02,
             "Gambar 5 — AP = luas di bawah kurva PR; mAP = rata-rata AP "
             "seluruh kelas.",
             ha="center", fontsize=9.5, color=MUTED, style="italic")
    _save(fig, "fig_kurva_pr.png")


# =====================================================================
def fig_tradeoff():
    thr = np.linspace(0.05, 0.90, 120)
    precision = 1 - np.exp(-4.2 * (thr - 0.02))
    precision = np.clip(precision / precision.max(), 0, 1)
    recall = np.clip(1.02 - 1.10 * thr ** 1.25, 0, 1)
    b2 = 4.0
    f2 = np.where(precision + recall > 0,
                  (1 + b2) * precision * recall / (b2 * precision + recall), 0)
    f1 = np.where(precision + recall > 0,
                  2 * precision * recall / (precision + recall), 0)

    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=170)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.plot(thr, precision, "-", color=ACCENT, lw=2.4, label="Precision")
    ax.plot(thr, recall, "-", color=GREEN, lw=2.4, label="Recall")
    ax.plot(thr, f1, "--", color=ORANGE, lw=2.0, label="F1 (seimbang)")
    ax.plot(thr, f2, "-.", color=RED, lw=2.2, label="F2 (bobot recall 4x)")
    i2 = int(np.argmax(f2))
    ax.axvline(thr[i2], color=RED, ls=":", lw=1.6)
    ax.plot([thr[i2]], [f2[i2]], "o", color=RED, ms=8)
    ax.annotate(f"conf* = {thr[i2]:.2f}\nRecall = {recall[i2]:.2f}",
                xy=(thr[i2], f2[i2]), xytext=(thr[i2] + 0.06, 0.30),
                fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlabel("Confidence threshold", fontsize=11)
    ax.set_ylabel("Nilai metrik", fontsize=11)
    ax.set_title("Trade-off Precision vs Recall terhadap Confidence Threshold",
                 fontsize=12.5, fontweight="bold", color=DARK)
    ax.set_xlim(0.05, 0.90); ax.set_ylim(0, 1.05)
    ax.grid(alpha=.25); ax.legend(fontsize=9.5, loc="center right")
    fig.text(0.5, -0.03,
             "Gambar 6 — Untuk kamera keamanan dipilih threshold yang "
             "memaksimalkan F2, bukan F1.",
             ha="center", fontsize=9.5, color=MUTED, style="italic")
    _save(fig, "fig_tradeoff.png")


# =====================================================================
def fig_peta_metrik():
    fig, ax = _canvas(11, 6.4)
    ax.text(5, 6.15, "Peta Hubungan Antar Metrik Evaluasi", ha="center",
            fontsize=15, fontweight="bold", color=DARK)

    _box(ax, 0.35, 4.55, 2.55, 1.15, "IoU", ["tumpang-tindih kotak\nprediksi vs ground truth"],
         color=ACCENT_S, edge=ACCENT, tfs=12, fs=8.4)
    _box(ax, 3.72, 4.55, 2.55, 1.15, "TP / FP / FN", ["IoU >= 0,50 & kelas sama -> TP",
                                                       "selain itu -> FP, GT tak terdeteksi -> FN"],
         color="#FEF3C7", edge=ORANGE, tfs=12, fs=8.4)
    _box(ax, 7.09, 4.55, 2.55, 1.15, "Precision & Recall", ["P = TP/(TP+FP)",
                                                             "R = TP/(TP+FN)"],
         color="#DCFCE7", edge=GREEN, tfs=12, fs=8.6)

    _arrow(ax, 2.90, 5.12, 3.72, 5.12)
    _arrow(ax, 6.27, 5.12, 7.09, 5.12)

    _box(ax, 0.35, 2.75, 2.55, 1.05, "Kurva PR", ["diurutkan berdasar\nconfidence"],
         color=ACCENT_S, edge=ACCENT, tfs=11, fs=8.4)
    _box(ax, 3.72, 2.75, 2.55, 1.05, "AP (per kelas)", ["luas di bawah\nkurva PR"],
         color="#FEF3C7", edge=ORANGE, tfs=11, fs=8.4)
    _box(ax, 7.09, 2.75, 2.55, 1.05, "mAP", ["rata-rata AP\nseluruh kelas"],
         color="#DCFCE7", edge=GREEN, tfs=11, fs=8.6)

    _arrow(ax, 1.62, 4.55, 1.62, 3.80)
    _arrow(ax, 2.90, 3.27, 3.72, 3.27)
    _arrow(ax, 6.27, 3.27, 7.09, 3.27)

    _box(ax, 0.35, 1.05, 9.29, 1.10,
         "Metrik paling krusial untuk kamera keamanan: RECALL",
         ["Recall = TP / (TP + FN) — satu-satunya metrik yang berbanding "
          "terbalik langsung dengan penyusup yang LOLOS (False Negative).",
          "mAP dipakai untuk MEMILIH model; Precision dijaga lewat ROI dan "
          "debounce agar petugas tidak mengalami alarm fatigue."],
         color="#FEE2E2", edge=RED, tfs=11.5, fs=8.8)
    _arrow(ax, 5.0, 2.75, 5.0, 2.15, color=RED)

    ax.text(5, 0.42, "Gambar 7 — Alur turunan metrik: IoU -> TP/FP/FN -> "
                     "Precision/Recall -> AP -> mAP",
            ha="center", fontsize=9.5, color=MUTED, style="italic")
    _save(fig, "fig_peta_metrik.png")


if __name__ == "__main__":
    print("Membuat ilustrasi untuk laporan ...")
    fig_arsitektur()
    fig_alur_kerja()
    fig_state()
    fig_iou()
    fig_kurva_pr()
    fig_tradeoff()
    fig_peta_metrik()
    print(f"Selesai -> {OUT}")
