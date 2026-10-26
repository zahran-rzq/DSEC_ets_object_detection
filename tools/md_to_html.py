"""
md_to_html.py -- Mengubah laporan Markdown menjadi HTML siap cetak (A4).

Menjalankan:
    python tools/md_to_html.py docs/LAPORAN_Soal2.md

Keluaran: docs/LAPORAN_Soal2.html
Buka di browser lalu Ctrl+P -> "Save as PDF" untuk menghasilkan berkas PDF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown

CSS = """
:root{
  --bg:#ffffff; --text:#1b2430; --muted:#6b7688; --accent:#2563eb;
  --border:#dce1ea; --panel:#f7f9fc; --code:#f4f6fa;
}
*{box-sizing:border-box}
body{
  font-family:"Segoe UI","DejaVu Sans",system-ui,-apple-system,sans-serif;
  color:var(--text); background:var(--bg);
  line-height:1.7; font-size:11.5pt;
  max-width:210mm; margin:0 auto; padding:18mm 16mm;
}
h1{font-size:22pt; line-height:1.25; border-bottom:3px solid var(--accent);
   padding-bottom:8px; margin:0 0 6px}
h1+h3{color:var(--muted); font-weight:400; font-size:12pt; margin-top:0}
h2{font-size:15.5pt; margin-top:26px; padding:6px 0 6px 12px;
   border-left:5px solid var(--accent); background:var(--panel);
   page-break-after:avoid; break-after:avoid}
h3{font-size:13pt; margin-top:20px; color:#0f172a; page-break-after:avoid}
h4{font-size:11.8pt; margin-top:16px; color:#334155; page-break-after:avoid}
p{margin:8px 0; text-align:justify}
a{color:var(--accent); text-decoration:none}
table{border-collapse:collapse; width:100%; margin:14px 0; font-size:10pt;
      page-break-inside:avoid}
th,td{border:1px solid var(--border); padding:6px 9px; text-align:left;
      vertical-align:top}
th{background:var(--panel); font-weight:600}
tr:nth-child(even) td{background:#fcfdff}
code{background:var(--code); border:1px solid var(--border); border-radius:4px;
     padding:1px 5px; font-family:"DejaVu Sans Mono",Consolas,monospace;
     font-size:9.5pt; color:#b91c1c}
pre{background:#0f172a; color:#e2e8f0; padding:12px 14px; border-radius:8px;
    overflow-x:auto; font-size:9pt; line-height:1.5; page-break-inside:avoid}
pre code{background:none; border:none; color:inherit; font-size:9pt; padding:0}
blockquote{border-left:4px solid var(--accent); background:#eff6ff;
           margin:12px 0; padding:8px 14px; border-radius:0 6px 6px 0}
blockquote p{margin:4px 0}
hr{border:none; border-top:1px solid var(--border); margin:22px 0}
img{max-width:100%; border:1px solid var(--border); border-radius:8px;
    display:block; margin:14px auto; page-break-inside:avoid}
ul,ol{padding-left:22px}
li{margin:3px 0}
em{color:var(--muted)}
strong{color:#0f172a}
.pagebreak{page-break-before:always}
@media print{
  body{padding:0; max-width:none; font-size:10.5pt}
  h2{page-break-before:auto}
  pre,table,img,blockquote{page-break-inside:avoid}
  @page{size:A4; margin:18mm 16mm}
}
"""

MATHJAX = """
<script>
  window.MathJax = {
    tex: {inlineMath: [['$','$'], ['\\\\(','\\\\)']],
          displayMath: [['$$','$$'], ['\\\\[','\\\\]']]},
    options: {skipHtmlTags: ['script','noscript','style','textarea','pre','code']}
  };
</script>
<script id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
"""


def convert(md_path: Path) -> Path:
    src = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        src,
        extensions=["tables", "fenced_code", "codehilite", "toc", "attr_list",
                    "sane_lists", "nl2br"],
        extension_configs={"codehilite": {"guess_lang": False, "use_pygments": False}},
    )
    out_path = md_path.with_suffix(".html")
    html = (
        "<!DOCTYPE html>\n<html lang=\"id\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{md_path.stem}</title>\n"
        f"<style>{CSS}</style>\n{MATHJAX}</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["docs/LAPORAN_Soal2.md"]:
        p = Path(arg)
        o = convert(p)
        print(f"✅ {p} -> {o}  ({o.stat().st_size/1024:.0f} KB)")
