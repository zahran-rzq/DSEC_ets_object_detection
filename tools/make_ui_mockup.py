"""
make_ui_mockup.py -- Menghasilkan mockup antarmuka aplikasi (light theme).

Menjalankan:
    python tools/make_ui_mockup.py

Keluaran: mockup/ui_mockup_light.png

Mockup digambar secara terprogram (PIL) sehingga seluruh teks terbaca jelas
dan proporsinya persis sama dengan implementasi `src/app.py`.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- palet ---
BG        = (244, 246, 250)
PANEL     = (255, 255, 255)
BORDER    = (220, 225, 234)
TEXT      = (27, 36, 48)
MUTED     = (107, 118, 136)
ACCENT    = (37, 99, 235)
ACCENT_S  = (231, 238, 253)
OK        = (22, 163, 74)
WARN      = (217, 119, 6)
DANGER    = (220, 38, 38)
IDLE      = (148, 163, 184)
VIDEO_BG  = (17, 24, 39)
LOG_BG    = (251, 252, 254)
SEPARATOR = (226, 232, 240)

C_PERSON  = (32, 168, 64)
C_CAR     = (26, 118, 232)
C_TRUCK   = (214, 54, 54)
C_MOTO    = (224, 92, 24)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def F(name: str, size: int):
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


F_TITLE  = F("DejaVuSans-Bold.ttf", 20)
F_SUB    = F("DejaVuSans.ttf", 12)
F_H2     = F("DejaVuSans-Bold.ttf", 11)
F_BODY   = F("DejaVuSans.ttf", 12)
F_BOLD   = F("DejaVuSans-Bold.ttf", 12)
F_SMALL  = F("DejaVuSans.ttf", 10)
F_MONO   = F("DejaVuSansMono.ttf", 11)
F_MB     = F("DejaVuSansMono-Bold.ttf", 13)
F_BTN    = F("DejaVuSans-Bold.ttf", 13)
F_CHIP   = F("DejaVuSans-Bold.ttf", 11)
F_HUD    = F("DejaVuSans-Bold.ttf", 13)


# ------------------------------------------------------------ utilitas ---
def rr(d: ImageDraw.ImageDraw, box, r=10, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def shadow(d, box, r=12, offset=3, color=(226, 231, 239)):
    x0, y0, x1, y1 = box
    d.rounded_rectangle((x0, y0 + offset, x1, y1 + offset), radius=r, fill=color)


def card(d, box, r=12):
    shadow(d, box, r)
    rr(d, box, r, fill=PANEL, outline=BORDER, width=1)


def txt(d, xy, s, font, fill=TEXT, anchor="la"):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def gradient(d, box, c0, c1, vertical=True):
    x0, y0, x1, y1 = box
    n = (y1 - y0) if vertical else (x1 - x0)
    for i in range(max(1, n)):
        t = i / max(1, n - 1)
        c = tuple(int(c0[k] + (c1[k] - c0[k]) * t) for k in range(3))
        if vertical:
            d.line([(x0, y0 + i), (x1, y0 + i)], fill=c)
        else:
            d.line([(x0 + i, y0), (x0 + i, y1)], fill=c)


# ------------------------------------------------------------- kanvas ---
W, H = 1440, 880
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# =========================================================== TOP BAR ====
rr(d, (0, 0, W, 66), r=0, fill=PANEL)
d.line([(0, 65), (W, 65)], fill=BORDER, width=1)

# logo (perisai sederhana)
lx, ly = 26, 16
d.rounded_rectangle((lx, ly, lx + 34, ly + 34), radius=8, fill=ACCENT)
d.polygon([(lx + 17, ly + 7), (lx + 28, ly + 12), (lx + 28, ly + 20),
           (lx + 17, ly + 28), (lx + 6, ly + 20), (lx + 6, ly + 12)], fill=PANEL)
txt(d, (96, 22), "SecureVision", F_TITLE, TEXT)
txt(d, (96, 44), "DSEC · Security Camera Application", F_SUB, MUTED)

txt(d, (W - 26, 33), "20:34:12", F_MB, TEXT, anchor="ra")
txt(d, (W - 116, 33), "●  LIVE", F_BOLD, OK, anchor="ra")

# ======================================================= KARTU VIDEO ====
CX, CY, CW, CH = 18, 82, 1000, 652
card(d, (CX, CY, CX + CW, CY + CH))
VX, VY, VW, VH = CX + 12, CY + 12, CW - 24, 576
rr(d, (VX, VY, VX + VW, VY + VH), r=8, fill=VIDEO_BG)

# ---- isi video: adegan CCTV sintetis ----
clip = img.crop((VX, VY, VX + VW, VY + VH))
cd = ImageDraw.Draw(clip)
horizon = int(VH * 0.46)
gradient(cd, (0, 0, VW, horizon), (150, 186, 214), (214, 232, 245))
gradient(cd, (0, horizon, VW, VH), (150, 155, 161), (92, 97, 105))

# gedung latar
for bx, bw, bh, c in [(40, 130, 150, (168, 181, 196)), (190, 90, 110, (158, 172, 188)),
                      (300, 160, 190, (176, 188, 202)), (700, 120, 130, (162, 175, 191)),
                      (840, 100, 165, (172, 184, 199))]:
    cd.rectangle((bx, horizon - bh, bx + bw, horizon), fill=c)

# jalan raya + marka
cd.rectangle((0, horizon, VW, VH), fill=(108, 113, 120))
gradient(cd, (0, horizon, VW, VH), (128, 133, 140), (86, 91, 98))
for i in range(6):
    y = horizon + 40 + i * 62
    cd.rectangle((0, y, VW, y + 5), fill=(196, 199, 203))

# trotoar
cd.rectangle((0, horizon + 8, VW, horizon + 26), fill=(163, 167, 172))


def draw_person(x, y, h, color=(38, 44, 58)):
    """Siluet orang sederhana: kepala, badan, dua kaki."""
    hh = int(h * 0.16)
    cd.ellipse((x - hh // 2, y, x + hh // 2, y + hh), fill=color)
    bw = int(h * 0.26)
    cd.rounded_rectangle((x - bw // 2, y + hh, x + bw // 2, y + int(h * 0.52)),
                         radius=6, fill=color)
    cd.rounded_rectangle((x - bw // 2, y + int(h * 0.52), x - 3, y + h),
                         radius=3, fill=color)
    cd.rounded_rectangle((x + 3, y + int(h * 0.52), x + bw // 2, y + h),
                         radius=3, fill=color)


def draw_car(x, y, w, body=(38, 84, 168), glass=(196, 216, 238)):
    h = int(w * 0.42)
    cd.rounded_rectangle((x, y + int(h * 0.34), x + w, y + h), radius=8, fill=body)
    cd.rounded_rectangle((x + int(w * 0.16), y, x + int(w * 0.84), y + int(h * 0.42)),
                         radius=8, fill=body)
    cd.rounded_rectangle((x + int(w * 0.21), y + int(h * 0.06), x + int(w * 0.79),
                          y + int(h * 0.36)), radius=5, fill=glass)
    cd.ellipse((x + int(w * 0.10), y + h - 8, x + int(w * 0.30), y + h + 10), fill=(28, 30, 36))
    cd.ellipse((x + int(w * 0.70), y + h - 8, x + int(w * 0.90), y + h + 10), fill=(28, 30, 36))
    cd.rectangle((x + int(w * 0.88), y + int(h * 0.44), x + w, y + int(h * 0.60)),
                 fill=(246, 232, 168))


draw_person(int(VW * 0.24), int(VH * 0.34), int(VH * 0.40))
draw_person(int(VW * 0.36), int(VH * 0.44), int(VH * 0.24), (46, 52, 66))
draw_car(int(VW * 0.56), int(VH * 0.46), int(VW * 0.24))
img.paste(clip, (VX, VY))


# ---- kotak deteksi + label ----
def bbox(x0, y0, x1, y1, color, label, conf=None):
    x0, y0, x1, y1 = VX + x0, VY + y0, VX + x1, VY + y1
    rr(d, (x0, y0, x1, y1), r=3, outline=color, width=3)
    text = f"{label} {conf:.2f}" if conf else label
    tw = d.textbbox((0, 0), text, font=F_CHIP)[2]
    lh = 22
    top = y0 - lh - 3
    if top < VY:
        top = y0 + 3
    rr(d, (x0, top, x0 + tw + 12, top + lh), r=3, fill=color)
    txt(d, (x0 + 6, top + 4), text, F_CHIP, (255, 255, 255))


bbox(int(VW * 0.18), int(VH * 0.32), int(VW * 0.30), int(VH * 0.755), C_PERSON, "PERSON", 0.87)
bbox(int(VW * 0.33), int(VH * 0.42), int(VW * 0.40), int(VH * 0.685), C_PERSON, "PERSON", 0.61)
bbox(int(VW * 0.545), int(VH * 0.44), int(VW * 0.815), int(VH * 0.70), C_CAR, "CAR", 0.72)

# ---- HUD di dalam video ----
hud_h = 34
d.rectangle((VX, VY, VX + VW, VY + hud_h), fill=(30, 33, 40))
txt(d, (VX + 12, VY + 8), "RECORDING     FPS 24.1     20:34:12", F_HUD, DANGER)
# titik REC berkedip
d.ellipse((VX + VW - 92, VY + 12, VX + VW - 76, VY + 28), fill=DANGER)
txt(d, (VX + VW - 68, VY + 8), "REC", F_HUD, DANGER)

# ---- banner status di bawah feed ----
BX, BY, BW_, BH_ = VX, VY + VH + 16, VW, 36
rr(d, (BX, BY, BX + BW_, BY + BH_), r=8, fill=(254, 226, 226))
d.ellipse((BX + 14, BY + 12, BX + 28, BY + 26), fill=DANGER)
txt(d, (BX + 36, BY + 10), "MEREKAM — target terdeteksi", F_BOLD, DANGER)
txt(d, (BX + BW_ - 16, BY + 10), "REC 00:14", F_MB, DANGER, anchor="ra")

# ================================================== KOLOM KANAN ==========
RX, RW = 1030, 392
y = 82


def panel_title(x0, yy, title, right=None, right_color=ACCENT):
    txt(d, (x0 + 14, yy + 12), title, F_H2, MUTED)
    if right:
        txt(d, (x0 + RW - 14, yy + 10), right, F_BOLD, right_color, anchor="ra")


# ---- (1) Status sistem ----
h1 = 170
card(d, (RX, y, RX + RW, y + h1))
panel_title(RX, y, "STATUS SISTEM")
rows = [("Model", "yolo11n.pt"), ("Kelas aktif", "person, car, truck, motorbike"),
        ("Resolusi", "1280 × 720"), ("FPS", "24.1"), ("Inferensi", "12 ms"),
        ("Total event", "7")]
ry = y + 38
for k, v in rows:
    txt(d, (RX + 14, ry), k, F_SMALL if len(k) > 12 else F_BODY, MUTED)
    txt(d, (RX + RW - 14, ry), v,
        F_SMALL if len(v) > 22 else F_BOLD, TEXT, anchor="ra")
    ry += 22
y += h1 + 12

# ---- (2) Deteksi target ----
h2 = 180
card(d, (RX, y, RX + RW, y + h2))
panel_title(RX, y, "DETEKSI TARGET", "3 objek")
ry = y + 38
for name, n, col in [("Person", 2, C_PERSON), ("Car", 1, C_CAR),
                     ("Truck", 0, C_TRUCK), ("Motorbike", 0, C_MOTO)]:
    # checkbox
    cb = (RX + 14, ry + 1, RX + 30, ry + 17)
    rr(d, cb, r=4, fill=col, outline=col, width=1)
    d.line([(RX + 18, ry + 9), (RX + 22, ry + 14), (RX + 27, ry + 5)],
           fill=(255, 255, 255), width=2)
    txt(d, (RX + 38, ry + 1), name, F_BODY, TEXT)
    # progress bar
    bx0 = RX + 150
    rr(d, (bx0, ry + 6, bx0 + 150, ry + 15), r=4, fill=(233, 236, 242))
    if n:
        rr(d, (bx0, ry + 6, bx0 + int(150 * min(1.0, n / 5.0)), ry + 15), r=4, fill=col)
    txt(d, (RX + RW - 16, ry + 1), str(n), F_BOLD, col, anchor="ra")
    ry += 32
y += h2 + 12

# ---- (3) Penyimpanan ----
h3 = 92
card(d, (RX, y, RX + RW, y + h3))
panel_title(RX, y, "PENYIMPANAN")
txt(d, (RX + 14, y + 36), "runs/capture", F_SMALL, MUTED)
txt(d, (RX + 14, y + 56), "Snapshot 12  ·  Klip 7", F_BOLD, TEXT)
y += h3 + 12

# ---- (4) Log kejadian ----
h4 = CY + CH - y
card(d, (RX, y, RX + RW, y + h4))
panel_title(RX, y, "LOG KEJADIAN")
lx0, ly0 = RX + 12, y + 32
rr(d, (lx0, ly0, lx0 + RW - 24, ly0 + (h4 - 46)), r=8, fill=LOG_BG, outline=BORDER)
logs = [(DANGER, "20:34:12", "REC_START   person"),
        (MUTED,  "20:34:12", "SNAPSHOT    person_0.87"),
        (OK,     "20:33:58", "REC_STOP    rec_0006.mp4"),
        (MUTED,  "20:33:41", "DETECTED    car"),
        (DANGER, "20:33:02", "REC_START   car"),
        (MUTED,  "20:32:20", "CAMERA      started")]
ly = ly0 + 10
for col, ts, msg in logs:
    txt(d, (lx0 + 10, ly), f"[{ts}]", F_MONO, MUTED)
    txt(d, (lx0 + 92, ly), msg, F_MONO, col if col != MUTED else TEXT)
    ly += 20

# ================================================== BARIS KONTROL ========
KY, KH = 750, 110
card(d, (CX, KY, CX + W - 2 * CX, KY + KH))


def button(x0, yy, w, h, label, fg, textc=(255, 255, 255), r=8):
    rr(d, (x0, yy, x0 + w, yy + h), r=r, fill=fg)
    txt(d, (x0 + w // 2, yy + h // 2), label, F_BTN, textc, anchor="mm")


btn_y = KY + 26
button(CX + 18, btn_y, 168, 42, "■  Stop Camera", DANGER)
button(CX + 196, btn_y, 132, 42, "Snapshot", (15, 118, 110))

# switch Auto-Record
sx = CX + 348
txt(d, (sx, btn_y + 4), "Auto-Record", F_BOLD, TEXT)
rr(d, (sx, btn_y + 24, sx + 46, btn_y + 48), r=12, fill=DANGER)
d.ellipse((sx + 26, btn_y + 26, sx + 44, btn_y + 46), fill=(255, 255, 255))

# separator
d.line([(CX + 448, KY + 20), (CX + 448, KY + KH - 20)], fill=SEPARATOR, width=1)


def slider(x0, yy, w, label, value, pos=0.42):
    txt(d, (x0, yy), f"{label}  {value:.2f}", F_BODY, TEXT)
    sy = yy + 22
    rr(d, (x0, sy - 3, x0 + w, sy + 3), r=3, fill=(233, 236, 242))
    rr(d, (x0, sy - 3, x0 + int(w * pos), sy + 3), r=3, fill=ACCENT)
    d.ellipse((x0 + int(w * pos) - 8, sy - 8, x0 + int(w * pos) + 8, sy + 8), fill=ACCENT)


slider(CX + 466, btn_y, 190, "Confidence", 0.35, 0.37)
slider(CX + 694, btn_y, 190, "IoU (NMS)", 0.45, 0.44)

# dropdown Preset
px = CX + 912
txt(d, (px, btn_y), "Preset:", F_BODY, MUTED)
rr(d, (px + 52, btn_y + 18, px + 52 + 150, btn_y + 46), r=8, fill=ACCENT_S,
   outline=ACCENT, width=1)
txt(d, (px + 62, btn_y + 24), "balanced", F_BODY, ACCENT)
txt(d, (px + 190, btn_y + 24), "▾", F_BODY, ACCENT)

# tombol folder output
ox = CX + W - 2 * CX - 168
button(ox, btn_y, 150, 42, "Folder Output", (226, 232, 240), TEXT)

# ---------------------------------------------------------------- simpan --
out = Path(__file__).resolve().parents[1] / "mockup" / "ui_mockup_light.png"
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out, optimize=True)
print(f"✅ Mockup tersimpan: {out}  ({img.size[0]}x{img.size[1]}, "
      f"{out.stat().st_size/1024:.0f} KB)")
