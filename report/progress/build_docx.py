"""Build the Word version of the progress report from content.py.

Needs python-docx. Run: python report/progress/build_docx.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import content as C  # noqa: E402

FONT, MONO = "Times New Roman", "Courier New"
OUT = HERE / "projectprogress-report-cs4212-abdoullah-ndao.docx"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(1))
normal = doc.styles["Normal"]
normal.font.name, normal.font.size = FONT, Pt(11)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15
for name, size in (("Heading 1", 13), ("Heading 2", 11.5)):
    st = doc.styles[name]
    st.font.name, st.font.size, st.font.bold = FONT, Pt(size), True
    st.font.color.rgb = RGBColor(0, 0, 0)
    rf = st.element.rPr.rFonts
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    st.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
    st.paragraph_format.space_after = Pt(4)
    st.paragraph_format.keep_with_next = True

TOKEN = re.compile(r"(\[[^\]]+\]\([^)]+\)|\*\*.+?\*\*|`.+?`|\*.+?\*)")


def add_link(par, text: str, url: str, size: float | None = None) -> None:
    """Clickable hyperlink run (python-docx has no public API for this)."""
    r_id = par.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1A4FB5")
    rpr.append(color)
    if size:
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), str(int(size * 2)))
        rpr.append(sz)
    run.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    run.append(t)
    link.append(run)
    par._p.append(link)


def add_runs(par, text: str, size: float | None = None) -> None:
    for part in TOKEN.split(text):
        if not part:
            continue
        m = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", part)
        if m:
            add_link(par, m.group(1), m.group(2), size)
            continue
        if part.startswith("**"):
            run = par.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("`"):
            run = par.add_run(part[1:-1])
            run.font.name = MONO
            rf = run._element.get_or_add_rPr().get_or_add_rFonts()
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rf.set(qn(attr), MONO)
            run.font.size = Pt((size or 11) - 1)
            continue
        elif part.startswith("*"):
            run = par.add_run(part[1:-1])
            run.italic = True
        else:
            run = par.add_run(part)
        if size:
            run.font.size = Pt(size)


def para(text: str, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size: float | None = None):
    p = doc.add_paragraph()
    p.alignment = align
    add_runs(p, text, size)
    return p


def caption(text: str) -> None:
    para(text, size=9.5).paragraph_format.space_after = Pt(10)


def shade(cell, fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def table(header, rows, widths, cap, bold_rows=(), bold_first_col=False) -> None:
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, values in enumerate([header] + list(rows)):
        for c, value in enumerate(values):
            cell = t.cell(r, c)
            cell.width = Inches(widths[c])
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            numeric = r > 0 and c > 0 and re.fullmatch(r"[\d.,%]+", value)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if numeric else WD_ALIGN_PARAGRAPH.LEFT
            add_runs(p, value, 9.5)
            if r == 0 or (r - 1) in bold_rows or (bold_first_col and c == 0):
                for run in p.runs:
                    run.bold = True
            if r == 0:
                shade(cell, "E8EAF0")
    caption(cap)


t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run(C.TITLE)
r.bold, r.font.size = True, Pt(17)
para(f"*{C.SUBTITLE}*", WD_ALIGN_PARAGRAPH.CENTER, 12)
para(C.AUTHOR, WD_ALIGN_PARAGRAPH.CENTER, 11.5)
para(f"{C.COURSE}  |  {C.DATE}", WD_ALIGN_PARAGRAPH.CENTER, 10)
doc.add_heading("Abstract", 1)
para(C.ABSTRACT)

for block in C.BLOCKS:
    kind = block[0]
    if kind == "h1":
        doc.add_heading(block[1], 1)
    elif kind == "h2":
        doc.add_heading(block[1], 2)
    elif kind == "p":
        para(block[1])
    elif kind == "bullets":
        for item in block[1]:
            add_runs(doc.add_paragraph(style="List Bullet"), item)
    elif kind == "table":
        _, header, rows, widths, cap, bold_rows = block
        table(header, rows, widths, cap, bold_rows)
    elif kind == "figure":
        doc.add_picture(str(HERE / block[1]), width=Inches(5.6))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption(block[2])
    elif kind == "eq":
        eq = doc.add_paragraph()
        eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for text, italic, sub in block[1]:
            run = eq.add_run(text)
            run.italic, run.font.subscript = italic, sub
    elif kind == "lit":
        para(C.LIT["intro"])
        for group in C.LIT["groups"]:
            doc.add_heading(group["title"], 2)
            for paper in group["papers"]:
                para(f"**{paper['name']}** ({paper['cite']}, [{paper['link_text']}]({paper['url']})). "
                     f"*What it did:* {paper['did']} *How mine differs:* {paper['differs']}")
        para(C.LIT["summary_intro"])
        table(["Aspect", "Prior work", "My project"], [list(x) for x in C.LIT["summary"]],
              [1.4, 2.3, 2.8], "**Table 1.** How my project differs from what I read.",
              bold_first_col=True)

doc.add_heading("References", 1)
for i, ref in enumerate(C.REFS, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.35)
    p.paragraph_format.space_after = Pt(2)
    add_runs(p, f"[{i}]\t{ref}", 10)

doc.save(OUT)
print("wrote", OUT)
