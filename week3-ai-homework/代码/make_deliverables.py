"""把交付用的 markdown 转成 PDF 和截图（作业要求：一页结论 PDF + 截图）。

用本机 Chrome 无头模式打印 PDF / 截图——它对中文字体支持最好，无需额外安装依赖。

用法：
    python make_deliverables.py
输出：
    outputs/pdf/A_一页结论.pdf、B_一页结论.pdf、A_10题逐题记录.pdf
    outputs/screenshots/B_四格对照表.png
"""

from __future__ import annotations

import html
import re
import subprocess
import tempfile
from pathlib import Path

import config

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PDF_DIR = config.OUT_DIR / "pdf"

CSS = """
body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin: 28px 34px;
       color: #1a1a1a; line-height: 1.75; font-size: 14px; }
h1 { font-size: 22px; border-bottom: 3px solid #2563eb; padding-bottom: 8px; }
h2 { font-size: 17px; margin-top: 22px; border-left: 4px solid #2563eb; padding-left: 8px; }
h3 { font-size: 15px; margin-top: 16px; }
table { border-collapse: collapse; margin: 12px 0; width: 100%; font-size: 13px; }
th, td { border: 1px solid #cbd5e1; padding: 6px 9px; text-align: left; }
th { background: #eff6ff; }
tr:nth-child(even) td { background: #f8fafc; }
code { background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 12.5px; }
blockquote { border-left: 4px solid #f5b301; background: #fffbeb; margin: 10px 0;
             padding: 8px 14px; color: #444; }
ul { padding-left: 22px; }
hr { border: none; border-top: 1px dashed #cbd5e1; margin: 18px 0; }
strong { color: #0f172a; }
"""


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def md_to_html(md: str) -> str:
    out, in_table, in_list = [], False, False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue                                  # 分隔行
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list and not line.startswith("  "):
            out.append("</ul>")
            in_list = False
        if line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{inline(line[2:])}</blockquote>")
        elif line.startswith("---"):
            out.append("<hr>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def to_pdf(md_path: Path, pdf_path: Path) -> None:
    body = md_to_html(md_path.read_text("utf-8"))
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>" \
               f"</head><body>{body}</body></html>"
    tmp = Path(tempfile.gettempdir()) / (pdf_path.stem + ".html")
    tmp.write_text(html_doc, "utf-8")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", tmp.as_uri()],
                   check=True, capture_output=True, timeout=180)
    print(f"  {md_path.name} → {pdf_path.name}（{pdf_path.stat().st_size/1024:.0f} KB）")


def to_png(md_path: Path, png_path: Path, width: int = 1500, height: int = 1600) -> None:
    body = md_to_html(md_path.read_text("utf-8"))
    style = CSS + "body{width:" + str(width - 68) + "px;background:#fff;}"
    html_doc = ("<!doctype html><html><head><meta charset='utf-8'><style>" + style
                + "</style></head><body>" + body + "</body></html>")
    tmp = Path(tempfile.gettempdir()) / (png_path.stem + ".html")
    tmp.write_text(html_doc, "utf-8")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={width},{height}",
                    f"--screenshot={png_path}", tmp.as_uri()],
                   check=True, capture_output=True, timeout=180)
    print(f"  {md_path.name} → {png_path.name}（{png_path.stat().st_size/1024:.0f} KB）")


def main() -> None:
    print("生成 PDF：")
    to_pdf(config.OUT_DIR / "A_一页结论.md", PDF_DIR / "A_一页结论.pdf")
    to_pdf(config.OUT_DIR / "B_一页结论.md", PDF_DIR / "B_一页结论.pdf")
    to_pdf(config.OUT_DIR / "A_10题逐题记录.md", PDF_DIR / "A_10题逐题记录.pdf")
    to_pdf(config.OUT_DIR / "A_深度三_图谱全景题.md", PDF_DIR / "A_深度三_图谱全景题.pdf")
    print("生成截图：")
    to_png(config.OUT_DIR / "B_四格对照.md",
           config.OUT_DIR / "screenshots" / "B_四格对照表.png")
    to_png(config.OUT_DIR / "A_判分表.md",
           config.OUT_DIR / "screenshots" / "A_10题判分表.png", width=1200)


if __name__ == "__main__":
    main()
