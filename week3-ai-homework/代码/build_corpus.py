"""半年报 PDF → 逐页正文 + 按行列还原的表格 → 带元数据的文本块。

半年报一半的信息在表格里（利润表、分渠道收入表、产销量表），
所以这里不是简单抽文字：先用 PyMuPDF 的 find_tables() 把每张表按行列取出来，
再和正文一起切块，并在块的元数据里记下公司/章节/页码，检索时才能分清是谁的数。

用法：
    python build_corpus.py
输出：
    data/pages/{code}.jsonl      每页的正文和表格
    data/chunks.jsonl            全部文本块（索引和检索的输入）
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

import config

SECTION_RE = re.compile(r"^\s*第[一二三四五六七八九十百]+节\s*\S{2,30}$")


def clean(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def table_to_text(rows: list[list], caption: str) -> str:
    """把表格按行列还原成「表名 + 表头 + 每行一条」的文本。

    切成一行行是为了让「直销 52,006,683,587.88 元」这种数字也能被检索到，
    同时保留表头和表名，避免出现「只剩数字、不知道是哪张表」的碎块。
    """
    lines = [f"【表】{caption}"]
    for row in rows:
        cells = [clean(str(c)) if c is not None else "" for c in row]
        if not any(cells):
            continue
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def guess_caption(page_text: str, y_top: float) -> str:
    """表格上方最近的一行文字通常就是表名，比如「主营业务分行业情况」。

    只认「含中文且不像纯数字」的短行——第一版取到的是表格里溢出的数字
    （比如「【表】25,001,435,220.18」），当表名用会误导后面的问答生成。
    """
    skip_prefix = ("单位", "币种", "金额单位", "本期", "上期", "项目", "科目",
                   "—", "-", ":", "：", "√", "□", "附注", "合计", "序号")
    best = ""
    for line in page_text.splitlines():
        line = clean(line)
        if not line or len(line) > 30 or len(line) < 4:
            continue
        if line.startswith(skip_prefix):
            continue
        zh = len(re.findall(r"[\u4e00-\u9fff]", line))
        digits = len(re.findall(r"[\d,.]", line))
        if zh >= 4 and zh >= digits:      # 以中文为主，才像表名/小标题
            best = line
    return best or "未命名表格"


def extract_pdf(pdf_path: Path, meta: dict) -> list[dict]:
    doc = fitz.open(pdf_path)
    section = ""
    pages = []
    for pno in range(doc.page_count):
        page = doc[pno]
        raw = page.get_text("text")
        text = clean(raw)

        # 更新章节：半年报正文是「第三节 管理层讨论与分析」这种一级标题
        for line in text.splitlines():
            if SECTION_RE.match(line):
                section = clean(line)
                break

        tables = []
        try:
            finder = page.find_tables()
            for tb in finder.tables:
                rows = tb.extract()
                if not rows or len(rows) < 2:
                    continue
                tables.append(table_to_text(rows, guess_caption(text, tb.bbox[1])))
        except Exception:  # 某些页表格识别会抛错，跳过即可
            pass

        pages.append({
            "code": meta["code"], "company": meta["name"], "doc": meta["title"],
            "period": config.REPORT_PERIOD, "page": pno + 1,
            "section": section, "text": text, "tables": tables,
        })
    doc.close()
    return pages


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """按句末收尾的滑窗切块，避免把小结从中间劈开。"""
    text = clean(text)
    if len(text) <= size:
        return [text] if text else []
    pieces, start = [], 0
    enders = "。；！？\n"
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(c) for c in enders)
            if cut > size * 0.5:
                end = start + cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def build_chunks(pages: list[dict]) -> list[dict]:
    """把每一页的正文和表格分别切块，块上带全量元数据。"""
    chunks = []
    for pg in pages:
        units = [("正文", pg["text"])] if pg["text"] else []
        units += [("表格", t) for t in pg["tables"]]
        for kind, body in units:
            for piece in split_text(body, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                if len(piece) < config.MIN_CHUNK and chunks and chunks[-1]["page"] == pg["page"]:
                    chunks[-1]["text"] += "\n" + piece      # 太短就并到上一块
                    chunks[-1]["n_chars"] = len(chunks[-1]["text"])
                    continue
                chunks.append({
                    "code": pg["code"], "company": pg["company"], "doc": pg["doc"],
                    "period": pg["period"], "page": pg["page"], "section": pg["section"],
                    "kind": kind, "text": piece, "n_chars": len(piece),
                })
    for i, c in enumerate(chunks):
        c["id"] = f"{c['code']}-{i:05d}"     # 稳定 ID，回答里的角标就指向它
    return chunks


def main() -> None:
    manifest = json.loads((config.PDF_DIR / "manifest.json").read_text("utf-8"))
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    for code, meta in manifest.items():
        pdf_path = config.ROOT / meta["pdf"]
        pages = extract_pdf(pdf_path, meta)
        out = config.PAGE_DIR / f"{code}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for p in pages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        chunks = build_chunks(pages)
        all_chunks.extend(chunks)
        n_table = sum(1 for c in chunks if c["kind"] == "表格")
        print(f"{code} {meta['name']}: {len(pages)} 页, {len(chunks)} 块（表格块 {n_table}）")

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n合计 {len(all_chunks)} 个文本块 → {config.INDEX_DIR / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
