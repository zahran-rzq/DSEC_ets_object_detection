"""
md_to_pdf.py -- Mengubah laporan Markdown menjadi PDF siap kumpul.

Menjalankan:
    python tools/md_to_pdf.py docs/LAPORAN_Soal2.md

Keluaran: docs/LAPORAN_Soal2.pdf

Fitur
-----
* Halaman sampul + tabel identitas otomatis dari tabel pertama dokumen.
* Daftar isi otomatis dengan nomor halaman (reportlab TableOfContents).
* Header/footer dan nomor halaman pada setiap halaman (kecuali sampul).
* Heading, paragraf rata kanan-kiri, daftar berbutir/bernomor, tabel
  bergaris, blok kode berlatar, kutipan, dan gambar ber-caption.
* Rumus LaTeX ($...$ dan $$...$$) dirender dengan matplotlib mathtext
  lalu disisipkan sebagai gambar, sehingga tampil rapi di PDF.
"""

from __future__ import annotations

import html
import re
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                ListFlowable, ListItem, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)
try:                                     # reportlab >= 4
    from reportlab.platypus.tableofcontents import TableOfContents
except ImportError:                      # reportlab 3.x
    from reportlab.platypus import TableOfContents

# ------------------------------------------------------------------ warna --
NAVY = colors.HexColor("#172235")
ACCENT = colors.HexColor("#2563EB")
ACCENT_S = colors.HexColor("#E7EEFD")
DARK = colors.HexColor("#1B2430")
MUTED = colors.HexColor("#6B7688")
BORDER = colors.HexColor("#CBD5E1")
CODE_BG = colors.HexColor("#F5F7FB")
PANEL = colors.HexColor("#FBFBFD")

PW, PH = A4
LM = RM = 2.0 * cm
TM = 2.2 * cm
BM = 2.0 * cm
CW = PW - LM - RM          # lebar konten


# ==========================================================================
#  Utilitas teks
# ==========================================================================
_CHAR_FIX = {
    "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
    "\u2265": ">=", "\u2264": "<=", "\u00d7": "x", "\u2192": "->",
    "\u2022": "-",
}


def clean(text: str) -> str:
    """Normalisasi karakter agar aman untuk font Type1 (WinAnsi)."""
    for a, b in _CHAR_FIX.items():
        text = text.replace(a, b)
    # buang karakter di luar Latin-1 yang tidak bisa dirender
    return "".join(ch if ord(ch) < 256 else "" for ch in text)


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def plain(text: str) -> str:
    """Buang markup inline Markdown (untuk entri daftar isi)."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"__([^_]*)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return clean(text).strip()


def short_caption(caption: str, label: str) -> str:
    """'Gambar 3 — State machine ...' -> 'Gambar 3  State machine ...'"""
    m = re.match(rf"^{label}\s*(\d+)\s*[—–\-:]\s*(.+)$", clean(caption))
    if m:
        return f"{label} {m.group(1)}   {m.group(2).strip().rstrip('.')}"
    return plain(caption)


# ==========================================================================
#  Render rumus
# ==========================================================================
class MathRenderer:
    """Mengubah ekspresi LaTeX menjadi PNG lewat matplotlib mathtext."""

    def __init__(self, workdir: Path):
        self.dir = workdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._n = 0

    def render(self, tex: str, fontsize: int = 12, display: bool = False):
        tex = tex.strip()
        self._n += 1
        path = self.dir / f"math_{self._n}.png"
        fs = fontsize + 3 if display else fontsize
        try:
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, f"${tex}$", fontsize=fs)
            fig.savefig(path, dpi=320, transparent=True, format="png",
                        bbox_inches="tight", pad_inches=0.03)
            plt.close(fig)
        except Exception:
            plt.close("all")
            return None
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            w, h = im.size
        scale = (fs / 12.0) ** -1 * 0.80          # px -> point (reportlab)
        return path, w * 0.80, h * 0.80


# ==========================================================================
#  Inline formatting
# ==========================================================================
class Inliner:
    """Mengubah markup inline Markdown menjadi markup reportlab."""

    # urutan: pola yang paling spesifik lebih dahulu
    PATTERNS = [
        (re.compile(r"\$\$(.+?)\$\$", re.S), "math"),
        (re.compile(r"\$([^\n$]+?)\$"), "math"),
        (re.compile(r"`([^`]+)`"), "code"),
        (re.compile(r"\*\*\*(.+?)\*\*\*"), "boldi"),
        (re.compile(r"\*\*(.+?)\*\*", re.S), "bold"),
        (re.compile(r"__(.+?)__", re.S), "bold"),
        (re.compile(r"\*([^*\n]+?)\*"), "ital"),
        (re.compile(r"(?<![\w*])_([^_\n]+?)_(?!\w)"), "ital"),
        (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), "link"),
    ]

    def __init__(self, math: MathRenderer):
        self.math = math

    def __call__(self, text: str, depth: int = 0) -> str:
        text = clean(str(text))
        out, pos = [], 0
        while pos < len(text):
            best = None
            for rx, kind in self.PATTERNS:
                m = rx.search(text, pos)
                if m and (best is None or m.start() < best[0].start()):
                    best = (m, kind)
            if best is None:
                out.append(esc(text[pos:]))
                break
            m, kind = best
            out.append(esc(text[pos:m.start()]))
            out.append(self._render(m, kind, depth))
            pos = m.end()
        return "".join(out)

    def _render(self, m, kind: str, depth: int) -> str:
        inner = m.group(1)
        if kind == "math":
            got = self.math.render(inner, fontsize=11)
            if got:
                path, w, h = got
                return (f'<img src="{path}" width="{w:.1f}" height="{h:.1f}" '
                        f'valign="-3"/>')
            return f"<i>{esc(inner)}</i>"
        if kind == "code":
            return ('<font face="Courier" size="8.5" color="#B91C1C">'
                    f"{esc(inner)}</font>")
        if kind == "link":
            return (f'<a href="{m.group(2)}" color="#2563EB"><u>'
                    f"{esc(inner)}</u></a>")
        # bold / italic / bold-italic -> proses isinya secara rekursif
        body = esc(inner) if depth >= 3 else self(inner, depth + 1)
        if kind == "bold":
            return f"<b>{body}</b>"
        if kind == "boldi":
            return f"<b><i>{body}</i></b>"
        return f"<i>{body}</i>"


# ==========================================================================
#  Parser Markdown -> blok
# ==========================================================================
def parse_blocks(lines: list[str]) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # blok kode
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", "\n".join(buf)))
            continue

        # rumus tampilan $$ ... $$
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            blocks.append(("math", stripped.strip("$")))
            i += 1
            continue
        if stripped.startswith("$$"):
            buf = [stripped[2:]]
            i += 1
            while i < n and not lines[i].strip().endswith("$$"):
                buf.append(lines[i])
                i += 1
            if i < n:
                buf.append(lines[i].strip()[:-2])
                i += 1
            blocks.append(("math", "\n".join(buf)))
            continue

        # garis pemisah
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            blocks.append(("hr", None))
            i += 1
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            blocks.append((f"h{len(m.group(1))}", m.group(2).strip()))
            i += 1
            continue

        # gambar
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        if m:
            blocks.append(("image", (m.group(1), m.group(2))))
            i += 1
            continue

        # tabel
        if stripped.startswith("|") and stripped.endswith("|"):
            buf = []
            while i < n and lines[i].strip().startswith("|"):
                buf.append(lines[i].strip())
                i += 1
            rows = []
            for r in buf:
                cells = [c.strip() for c in r.strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    continue
                rows.append(cells)
            blocks.append(("table", rows))
            continue

        # kutipan
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(("quote", " ".join(buf)))
            continue

        # daftar
        m = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", line)
        if m:
            ordered = bool(re.match(r"\d+[.)]", m.group(2)))
            items = []
            while i < n:
                mm = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", lines[i])
                if not mm:
                    # kelanjutan baris
                    if lines[i].startswith("  ") and items:
                        items[-1] += " " + lines[i].strip()
                        i += 1
                        continue
                    break
                items.append(mm.group(3).strip())
                i += 1
            blocks.append(("list", ("ordered" if ordered else "bullet", items)))
            continue

        # paragraf
        if stripped:
            buf = [stripped]
            i += 1
            while i < n and lines[i].strip() and not re.match(
                    r"^(#{1,6}\s|\s*[-*+]\s|\d+[.)]\s|>|!\[|\||```|\$\$)", lines[i]):
                buf.append(lines[i].strip())
                i += 1
            blocks.append(("para", " ".join(buf)))
            continue

        i += 1
    return blocks


# ==========================================================================
#  Gaya
# ==========================================================================
def styles():
    base = dict(fontName="Helvetica", fontSize=10.3, leading=15.2,
                textColor=DARK, alignment=TA_JUSTIFY, spaceAfter=6)
    s = {}
    s["Body"] = ParagraphStyle("Body", **base)
    s["H1"] = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=16,
                             leading=20, textColor=ACCENT, spaceBefore=6,
                             spaceAfter=12, alignment=TA_LEFT)
    s["H2"] = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=12.6,
                             leading=16, textColor=NAVY, spaceBefore=12,
                             spaceAfter=7)
    s["H3"] = ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=11,
                             leading=14.5, textColor=DARK, spaceBefore=9,
                             spaceAfter=5)
    s["H4"] = ParagraphStyle("H4", fontName="Helvetica-BoldOblique",
                             fontSize=10.3, leading=14, textColor=DARK,
                             spaceBefore=7, spaceAfter=4)
    s["Caption"] = ParagraphStyle("Caption", fontName="Helvetica-Oblique",
                                  fontSize=8.8, leading=12, textColor=MUTED,
                                  alignment=TA_CENTER, spaceBefore=3,
                                  spaceAfter=10)
    s["Quote"] = ParagraphStyle("Quote", fontName="Helvetica-Oblique",
                                fontSize=10, leading=14.5, textColor=DARK,
                                leftIndent=8, rightIndent=6, spaceAfter=8)
    s["TableCap"] = ParagraphStyle("TableCap", fontName="Helvetica-Bold",
                                   fontSize=9, leading=12.5, textColor=NAVY,
                                   spaceBefore=8, spaceAfter=3)
    s["Cell"] = ParagraphStyle("Cell", fontName="Helvetica", fontSize=8.4,
                               leading=11.2, textColor=DARK)
    s["CellH"] = ParagraphStyle("CellH", fontName="Helvetica-Bold",
                                fontSize=8.4, leading=11.2, textColor=NAVY)
    s["Code"] = ParagraphStyle("Code", fontName="Courier", fontSize=7.9,
                               leading=10.2, textColor=colors.HexColor("#0F172A"))
    s["Title"] = ParagraphStyle("Title", fontName="Helvetica-Bold",
                                fontSize=21, leading=26, textColor=NAVY,
                                alignment=TA_CENTER, spaceAfter=10)
    s["SubTitle"] = ParagraphStyle("SubTitle", fontName="Helvetica",
                                   fontSize=12.5, leading=17, textColor=DARK,
                                   alignment=TA_CENTER, spaceAfter=6)
    s["Cover3"] = ParagraphStyle("Cover3", fontName="Helvetica-BoldOblique",
                                 fontSize=10.5, leading=15, textColor=ACCENT,
                                 alignment=TA_CENTER, spaceAfter=4)
    s["TocH"] = ParagraphStyle("TocH", fontName="Helvetica-Bold", fontSize=15,
                               leading=19, textColor=NAVY, spaceAfter=12,
                               alignment=TA_CENTER)
    return s


# ==========================================================================
#  Dokumen
# ==========================================================================
class TaggedTable(Table):
    """Tabel yang tetap membawa penanda daftar tabel walau terpecah halaman."""

    def split(self, availWidth, availHeight):
        parts = super().split(availWidth, availHeight)
        tag = getattr(self, "_lot_text", None)
        for part in parts:
            if tag and not getattr(part, "_lot_text", None):
                part._lot_text = tag
        return parts


class ListOfX(TableOfContents):
    """Daftar Gambar / Daftar Tabel: abaikan entri ganda akibat tabel pecah."""

    def addEntry(self, level, text, pageNum, key=None):
        if any(e[1] == text for e in self._entries):
            return
        super().addEntry(level, text, pageNum, key)


PAGE_TITLE = "Laporan ETS DSEC - Aplikasi Desktop Kamera Keamanan"


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, **kw):
        BaseDocTemplate.__init__(self, filename, pagesize=A4,
                                 leftMargin=LM, rightMargin=RM,
                                 topMargin=TM, bottomMargin=BM,
                                 title="Laporan ETS DSEC - Security Camera App",
                                 author="SecureVision / DSEC",
                                 subject=PAGE_TITLE, **kw)
        frame = Frame(LM, BM, CW, PH - TM - BM, id="body")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame]),
            PageTemplate(id="main", frames=[frame], onPage=self._decorate),
        ])
        self._bm = 0

    # ---------------------------- header / footer -------------------------
    def _decorate(self, canv, doc):
        canv.saveState()
        if doc.page > 2:
            canv.setFont("Helvetica", 7.6)
            canv.setFillColor(MUTED)
            canv.drawString(LM, PH - TM + 0.55 * cm, PAGE_TITLE)
            canv.setStrokeColor(BORDER)
            canv.setLineWidth(0.5)
            canv.line(LM, PH - TM + 0.38 * cm, PW - RM, PH - TM + 0.38 * cm)
        # footer
        canv.setFont("Helvetica", 8)
        canv.setFillColor(MUTED)
        canv.setStrokeColor(BORDER)
        canv.setLineWidth(0.5)
        canv.line(LM, BM - 0.42 * cm, PW - RM, BM - 0.42 * cm)
        canv.drawCentredString(PW / 2.0, BM - 1.15 * cm, f"Halaman {doc.page}")
        canv.restoreState()

    # ---------------------------- TOC -------------------------------------
    def afterFlowable(self, flowable):
        lof = getattr(flowable, "_lof_text", None)
        if lof:
            self.notify("LOFEntry", (0, lof, self.page))
            return
        lot = getattr(flowable, "_lot_text", None)
        if lot:
            self.notify("LOTEntry", (0, lot, self.page))
            return
        if not isinstance(flowable, Paragraph):
            return
        name = flowable.style.name
        text = plain(flowable.getPlainText())
        if name == "H1":
            self._bm += 1
            key = f"h1-{self._bm}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, 0, 0)
            self.notify("TOCEntry", (0, text, self.page))
        elif name == "H2":
            self._bm += 1
            key = f"h2-{self._bm}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, 1, 0)
            self.notify("TOCEntry", (1, text, self.page))
        elif name == "H3":
            self.notify("TOCEntry", (2, text, self.page))


# ==========================================================================
#  Bangun PDF
# ==========================================================================
def build(md_path: Path, out_path: Path | None = None) -> Path:
    md_path = Path(md_path)
    src = md_path.read_text(encoding="utf-8")
    lines = src.split("\n")
    blocks = parse_blocks(lines)
    S = styles()

    workdir = Path(tempfile.mkdtemp(prefix="dsec_pdf_"))
    math = MathRenderer(workdir)
    inline = Inliner(math)

    # ---- ambil tabel identitas (tabel pertama yang memuat "Mata Kuliah") ----
    identity = None
    for kind, payload in blocks:
        if kind == "table" and any("Mata Kuliah" in c for row in payload
                                   for c in row):
            identity = payload
            break

    # ---- jadikan paragraf singkat sebelum tabel sebagai judul tabel --------
    def _judul_dari_heading(teks: str) -> str:
        # "6.1 Prinsip Perancangan" -> "Prinsip Perancangan"
        return re.sub(r"^\d+(\.\d+)*\.?\s*", "", str(teks)).strip()

    prepped: list = []
    for idx, (kind, payload) in enumerate(blocks):
        if kind == "table" and payload is not identity and prepped:
            pk, pv = prepped[-1]
            if pk == "para":
                t = str(pv).strip()
                if len(t) <= 150 and (t.endswith(":") or t.endswith(".")):
                    prepped[-1] = ("tablecap", t.rstrip(".:").strip())
            elif pk in ("h2", "h3", "h4", "h5", "h6"):
                j = _judul_dari_heading(pv)
                if j:
                    prepped.append(("tablecap", j))
        prepped.append((kind, payload))
    blocks = prepped

    story: list = []
    in_toc_skip = False
    first_h2 = True
    prologue = identity is not None   # lewati blok pembuka (sudah jadi sampul)
    pending_cap = ""
    table_no = [0]

    def add_image(rel: str, caption: str):
        p = (md_path.parent / rel).resolve()
        if not p.exists():
            print(f"   !! gambar tidak ditemukan: {p}")
            return
        try:
            from PIL import Image as PILImage
            with PILImage.open(p) as im:
                iw, ih = im.size
        except Exception:
            iw, ih = 1200, 700
        max_w, max_h = CW, PH - TM - BM - 3.2 * cm
        scale = min(max_w / iw, max_h / ih, 1.0)
        w, h = iw * scale, ih * scale
        img = Image(str(p), width=w, height=h)
        img.hAlign = "CENTER"
        img._lof_text = short_caption(caption, "Gambar")
        cap = Paragraph(clean(caption), S["Caption"])
        story.append(Spacer(1, 4))
        story.append(KeepTogether([img, cap]))

    for kind, payload in blocks:
        # lewati blok pembuka: judul, subjudul, catatan (sudah menjadi sampul)
        if prologue:
            if kind == "table" and payload is identity:
                prologue = False
            continue
        # lewati tabel identitas (sudah masuk sampul)
        if kind == "table" and payload is identity:
            continue
        # lewati daftar isi manual (diganti TOC otomatis)
        if kind == "h2" and "DAFTAR ISI" in str(payload).upper():
            in_toc_skip = True
            continue
        if in_toc_skip:
            if kind == "h2":
                in_toc_skip = False
            else:
                continue

        if kind == "h1":
            continue                                  # judul -> sampul
        if kind == "h2":
            if not first_h2:
                story.append(PageBreak())
            first_h2 = False
            story.append(Paragraph(inline(str(payload)), S["H1"]))
        elif kind == "h3":
            story.append(Paragraph(inline(str(payload)), S["H2"]))
        elif kind == "h4":
            story.append(Paragraph(inline(str(payload)), S["H3"]))
        elif kind in ("h5", "h6"):
            story.append(Paragraph(inline(str(payload)), S["H4"]))

        elif kind == "para":
            story.append(Paragraph(inline(str(payload)), S["Body"]))

        elif kind == "quote":
            txt = inline(str(payload))
            p = Paragraph(txt, S["Quote"])
            t = Table([[p]], colWidths=[CW])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("LINEBEFORE", (0, 0), (0, -1), 2.6, ACCENT),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(Spacer(1, 2))
            story.append(t)
            story.append(Spacer(1, 4))

        elif kind == "code":
            code = str(payload)
            wrapped = []
            for ln in code.split("\n"):
                ln = ln.replace("\t", "    ")
                while len(ln) > 104:
                    cut = ln.rfind(" ", 0, 104)
                    cut = cut if cut > 40 else 104
                    wrapped.append(ln[:cut])
                    ln = "    " + ln[cut:].lstrip()
                wrapped.append(ln)
                if len(wrapped) > 60:
                    break
            pre = Preformatted(clean("\n".join(wrapped)), S["Code"])
            t = Table([[pre]], colWidths=[CW])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(Spacer(1, 2))
            story.append(t)
            story.append(Spacer(1, 6))

        elif kind == "list":
            kind_list, items = payload
            flow = ListFlowable(
                [ListItem(Paragraph(inline(it), S["Body"]), leftIndent=16)
                 for it in items],
                bulletType="1" if kind_list == "ordered" else "bullet",
                bulletFontSize=8, leftIndent=16, bulletOffsetY=1)
            story.append(flow)
            story.append(Spacer(1, 3))

        elif kind == "tablecap":
            pending_cap = str(payload)

        elif kind == "table":
            rows = payload
            if not rows:
                continue
            table_no[0] += 1
            if pending_cap:
                cap_txt = f"Tabel {table_no[0]} — {pending_cap}"
            else:
                cap_txt = f"Tabel {table_no[0]}"
            pending_cap = ""
            story.append(Spacer(1, 6))
            story.append(Paragraph(inline(cap_txt), S["TableCap"]))
            ncol = max(len(r) for r in rows)
            rows = [r + [""] * (ncol - len(r)) for r in rows]
            header, body = rows[0], rows[1:]
            data = [[Paragraph(inline(c), S["CellH"]) for c in header]]
            for r in body:
                data.append([Paragraph(inline(c), S["Cell"]) for c in r])
            colw = [CW / ncol] * ncol
            t = TaggedTable(data, colWidths=colw, repeatRows=1, hAlign="CENTER")
            t._lot_text = plain(cap_txt)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT_S),
                ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#FAFBFE")]),
            ]))
            story.append(Spacer(1, 4))
            story.append(t)
            story.append(Spacer(1, 8))

        elif kind == "image":
            caption, rel = payload
            add_image(rel, caption)

        elif kind == "math":
            got = math.render(str(payload), fontsize=13, display=True)
            if got:
                path, w, h = got
                maxw = CW * 0.92
                if w > maxw:
                    h *= maxw / w
                    w = maxw
                im = Image(str(path), width=w, height=h)
                im.hAlign = "CENTER"
                story.append(Spacer(1, 4))
                story.append(im)
                story.append(Spacer(1, 8))

        elif kind == "hr":
            story.append(Spacer(1, 4))
            t = Table([[""]], colWidths=[CW], rowHeights=[1])
            t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.6, BORDER)]))
            story.append(t)
            story.append(Spacer(1, 6))

    # ============================== SAMPUL ==============================
    cover: list = []
    cover.append(Spacer(1, 2.4 * cm))
    cover.append(Paragraph("LAPORAN TUGAS", S["SubTitle"]))
    cover.append(Paragraph("EVALUASI TENGAH SEMESTER", S["Title"]))
    cover.append(Spacer(1, 0.25 * cm))
    cover.append(Paragraph("RANCANGAN DAN IMPLEMENTASI<br/>"
                           "APLIKASI DESKTOP KAMERA KEAMANAN", S["SubTitle"]))
    cover.append(Spacer(1, 0.15 * cm))
    cover.append(Paragraph("Deteksi Real-Time <b>person</b>, <b>car</b>, "
                           "<b>truck</b>, dan <b>motorbike</b><br/>"
                           "menggunakan Model YOLO Pre-Trained<br/>"
                           "dengan Auto-Snapshot dan Auto-Recording", S["Cover3"]))
    cover.append(Spacer(1, 0.9 * cm))

    # emblem
    try:
        from reportlab.graphics.shapes import Drawing, Circle, String
        d = Drawing(CW, 92)
        d.add(Circle(CW / 2, 46, 34, fillColor=ACCENT, strokeColor=None))
        d.add(String(CW / 2, 30, "SV", fontName="Helvetica-Bold", fontSize=26,
                     fillColor=colors.white, textAnchor="middle"))
        cover.append(d)
    except Exception:
        pass
    cover.append(Spacer(1, 0.9 * cm))

    if identity:
        rows = identity
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        data = []
        for k, r in enumerate(rows):
            st = S["Cell"] if k else S["CellH"]
            data.append([Paragraph(inline(c), st) for c in r])
        t = Table(data, colWidths=[CW * 0.30, CW * 0.70], hAlign="CENTER")
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT_S),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        cover.append(t)

    cover.append(Spacer(1, 1.1 * cm))
    cover.append(Paragraph("Program Studi ................................<br/>"
                           "Institut Teknologi Sepuluh Nopember<br/>"
                           "Surabaya, 2025", S["SubTitle"]))
    cover.append(PageBreak())

    # ============================== TOC ==============================
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOCL0", fontName="Helvetica-Bold", fontSize=11,
                       leading=17, textColor=NAVY, leftIndent=0,
                       firstLineIndent=-0.1, spaceBefore=4),
        ParagraphStyle("TOCL1", fontName="Helvetica", fontSize=9.8,
                       leading=14.5, textColor=DARK, leftIndent=18),
        ParagraphStyle("TOCL2", fontName="Helvetica", fontSize=9.2,
                       leading=13, textColor=MUTED, leftIndent=34),
    ]
    toc.dotsMinLevel = 0

    lof = ListOfX(notifyKind="LOFEntry")
    lof.levelStyles = [ParagraphStyle("LOF", fontName="Helvetica",
                                      fontSize=9.6, leading=13.5,
                                      textColor=DARK, leftIndent=0)]
    lot = ListOfX(notifyKind="LOTEntry")
    lot.levelStyles = [ParagraphStyle("LOT", fontName="Helvetica",
                                      fontSize=9.6, leading=13.5,
                                      textColor=DARK, leftIndent=0)]

    toc_page = [Paragraph("DAFTAR ISI", S["TocH"]), toc, PageBreak(),
                Paragraph("DAFTAR GAMBAR", S["TocH"]), lof, PageBreak(),
                Paragraph("DAFTAR TABEL", S["TocH"]), lot, PageBreak()]

    out_path = Path(out_path) if out_path else md_path.with_suffix(".pdf")
    doc = ReportDoc(str(out_path))

    full = ([*cover, *toc_page,
             Paragraph('<para></para>', S["Body"])] + story)
    # halaman 1-2 = sampul + TOC (tanpa header), sisanya template utama
    from reportlab.platypus import NextPageTemplate
    full = [*cover, NextPageTemplate("main"), *toc_page, *story]

    doc.multiBuild(full)
    shutil.rmtree(workdir, ignore_errors=True)
    return out_path


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["docs/LAPORAN_Soal2.md"]:
        p = Path(arg)
        o = build(p)
        print(f"OK  {p}  ->  {o}  ({o.stat().st_size/1024:.0f} KB)")
