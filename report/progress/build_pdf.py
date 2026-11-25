"""Build the PDF version of the progress report from content.py (two columns, via Chrome).

Run: python3 report/progress/build_pdf.py
Writes progress_paper.html and projectprogress-report-cs4212-abdoullah-ndao.pdf.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import content as C  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TOKEN = re.compile(r"(\[[^\]]+\]\([^)]+\)|\*\*.+?\*\*|`.+?`|\*.+?\*)")


def md(text: str) -> str:
    out = []
    for part in TOKEN.split(text):
        if not part:
            continue
        m = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", part)
        if m:
            out.append(f'<a href="{html.escape(m.group(2))}">{html.escape(m.group(1))}</a>')
        elif part.startswith("**"):
            out.append(f"<b>{html.escape(part[2:-2])}</b>")
        elif part.startswith("`"):
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
        elif part.startswith("*"):
            out.append(f"<i>{html.escape(part[1:-1])}</i>")
        else:
            out.append(html.escape(part))
    return "".join(out)


def table(header, rows, cap, bold_rows=(), bold_first_col=False) -> str:
    th = "".join(f"<th>{md(h)}</th>" for h in header)
    body = []
    for r, row in enumerate(rows):
        tds = []
        for c, v in enumerate(row):
            numeric = c > 0 and re.fullmatch(r"[\d.,%]+", v)
            cell = md(v)
            if r in bold_rows or (bold_first_col and c == 0):
                cell = f"<b>{cell}</b>"
            cls = ' class="n"' if numeric else ""
            tds.append(f"<td{cls}>{cell}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead><tbody>'
            f'{"".join(body)}</tbody></table><div class="caption">{md(cap)}</div></div>')


parts = []
for block in C.BLOCKS:
    kind = block[0]
    if kind == "h1":
        parts.append(f"<h2>{html.escape(block[1]).replace('  ', '&nbsp;&nbsp;')}</h2>")
    elif kind == "h2":
        parts.append(f"<h3>{html.escape(block[1])}</h3>")
    elif kind == "p":
        parts.append(f"<p>{md(block[1])}</p>")
    elif kind == "bullets":
        parts.append("<ul>" + "".join(f"<li>{md(x)}</li>" for x in block[1]) + "</ul>")
    elif kind == "table":
        _, header, rows, _w, cap, bold_rows = block
        parts.append(table(header, rows, cap, bold_rows))
    elif kind == "figure":
        parts.append(f'<figure><img src="{block[1]}" alt="System overview">'
                     f'<div class="caption">{md(block[2])}</div></figure>')
    elif kind == "eq":
        eq = "".join((f"<i>{html.escape(t)}</i>" if it else html.escape(t)) if not sub
                     else f"<sub>{html.escape(t)}</sub>" for t, it, sub in block[1])
        parts.append(f'<div class="eq">{eq}</div>')
    elif kind == "lit":
        parts.append(f"<p>{md(C.LIT['intro'])}</p>")
        for g in C.LIT["groups"]:
            parts.append(f"<h3>{html.escape(g['title'])}</h3>")
            for p in g["papers"]:
                name = html.escape(p["name"])
                parts.append(f'<p class="lit"><b>{name}</b> '
                             f'({html.escape(p["cite"])}, <a href="{html.escape(p["url"])}">'
                             f'{html.escape(p["link_text"])}</a>). <i>What it did:</i> '
                             f'{html.escape(p["did"])} <i>How mine differs:</i> '
                             f'{html.escape(p["differs"])}</p>')
        parts.append(f"<p>{md(C.LIT['summary_intro'])}</p>")
        parts.append(table(["Aspect", "Prior work", "My project"], C.LIT["summary"],
                           "**Table 1.** How my project differs from what I read.",
                           bold_first_col=True))

refs = "".join(f"<li>{md(r)}</li>" for r in C.REFS)
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(C.TITLE)}</title>
<style>
@page {{ size: Letter; margin: 0.75in 0.7in 0.8in; }}
html {{ font-family: "Times New Roman", Times, serif; font-size: 10pt; line-height: 1.32; color: #111; }}
body {{ margin: 0; }}
.title {{ text-align: center; margin-bottom: 14pt; }}
.title h1 {{ font-size: 17pt; line-height: 1.2; margin: 0 0 6pt; }}
.title .sub {{ font-size: 11.5pt; font-style: italic; margin-bottom: 8pt; }}
.title .meta {{ font-size: 9pt; color: #444; margin-top: 3pt; }}
.abstract {{ margin: 0 0.35in 12pt; font-size: 9.5pt; text-align: justify; }}
.cols {{ column-count: 2; column-gap: 0.28in; text-align: justify; hyphens: auto; }}
h2 {{ font-size: 11pt; margin: 10pt 0 4pt; break-after: avoid; }}
h3 {{ font-size: 10pt; margin: 8pt 0 3pt; font-style: italic; font-weight: normal; break-after: avoid; }}
p {{ margin: 0 0 6pt; }}
ul {{ margin: 0 0 6pt; padding-left: 14pt; }} li {{ margin-bottom: 2pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 4pt 0 2pt; }}
th, td {{ padding: 2.2pt 3pt; text-align: left; vertical-align: top; }}
thead th {{ border-top: 1.2px solid #000; border-bottom: 0.7px solid #000; }}
tbody tr:last-child td {{ border-bottom: 1.2px solid #000; }}
td.n {{ text-align: right; }}
.caption {{ font-size: 8.5pt; margin: 3pt 0 9pt; text-align: justify; }}
figure {{ margin: 4pt 0 8pt; break-inside: avoid; }} figure img {{ width: 100%; display: block; }}
code {{ font-family: Menlo, Consolas, monospace; font-size: 8.3pt; }}
.eq {{ text-align: center; margin: 4pt 0 6pt; }}
a {{ color: #1a4fb5; text-decoration: none; }}
.lit {{ margin-bottom: 4pt; }}
.refs {{ font-size: 8.3pt; }} .refs ol {{ padding-left: 16pt; }} .refs li {{ margin-bottom: 1.5pt; text-align: left; }}
</style></head><body>
<div class="title"><h1>{html.escape(C.TITLE)}</h1><div class="sub">{html.escape(C.SUBTITLE)}</div>
<div>{html.escape(C.AUTHOR)}</div><div class="meta">{html.escape(C.COURSE)} &middot; {html.escape(C.DATE)}</div></div>
<div class="abstract"><b>Abstract.</b> {md(C.ABSTRACT)}</div>
<div class="cols">
{chr(10).join(parts)}
<div class="refs"><h2>References</h2><ol>{refs}</ol></div>
</div></body></html>"""
(HERE / "progress_paper.html").write_text(page)
pdf = HERE / "projectprogress-report-cs4212-abdoullah-ndao.pdf"
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", (HERE / "progress_paper.html").as_uri()],
               check=True, capture_output=True)
print("wrote", pdf)
