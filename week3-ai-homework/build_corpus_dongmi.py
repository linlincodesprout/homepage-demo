"""B 方向语料：茅台 11 份财报 → 逐页正文/表格 → 800/120 切块（和 A 完全同一套动作）。

多带两个字段：报告期（2021A/2023H1…）和报告全名，后面生成问答对时要用它们标出处。

用法：
    python build_corpus_dongmi.py
输出：
    data/dongmi/pages/{period}.jsonl
    data/dongmi/chunks.jsonl
"""

from __future__ import annotations

import json

import config
from build_corpus import build_chunks, extract_pdf

DONGMI_DIR = config.DATA_DIR / "dongmi"


def main() -> None:
    manifest = json.loads((DONGMI_DIR / "pdf" / "manifest.json").read_text("utf-8"))
    page_dir = DONGMI_DIR / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for period in sorted(manifest, reverse=True):          # 新的排前面
        meta = manifest[period]
        meta = {**meta, "title": meta["title"]}            # extract_pdf 用 meta["title"] 当报告名
        pages = extract_pdf(config.ROOT / meta["pdf"], meta)
        with open(page_dir / f"{period}.jsonl", "w", encoding="utf-8") as f:
            for p in pages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        chunks = build_chunks(pages)
        for c in chunks:
            c["period"] = period                            # 报告期标签，出处要写它
            c["id"] = f"{period}-{c['id'].split('-')[-1]}"
        all_chunks.extend(chunks)
        n_table = sum(1 for c in chunks if c["kind"] == "表格")
        print(f"{period:7s} {meta['title'][:28]:30s} {len(pages):4d} 页  "
              f"{len(chunks):5d} 块（表格 {n_table}）")

    out = DONGMI_DIR / "chunks.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n合计 {len(all_chunks)} 块 → {out}")


if __name__ == "__main__":
    main()
