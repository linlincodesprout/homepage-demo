"""重建「页码 → 章节」映射。

当初切块时章节识别漏了：半年报的标题常被排版拆成两行（"第三节" 一行、
"管理层讨论与分析" 下一行），单行正则匹配不到，所以 4,793/5,842 块标成"未标注章节"。

重切块会让 11,047 块的 id 变化、整个向量库和已有评测全失效——代价太大。
所以这里只做"补标签"：从逐页文本里重新识别章节，得到 page→section 映射，
给报告和查询时显示用，不动索引。

用法：
    python build_section_map.py
输出：
    data/index/section_map.json   {"600519": {"6": "第三节 管理层讨论与分析", ...}}
"""

from __future__ import annotations

import json
import re

import config

HEAD = re.compile(r"^第([一二三四五六七八九十]+)节$")
TOC = re.compile(r"\.{4,}")


def doc_sections(path) -> dict[str, str]:
    pages = [json.loads(l) for l in open(path, encoding="utf-8")]
    out, current = {}, "第一节"
    for p in pages:
        lines = [x.strip() for x in p["text"].splitlines() if x.strip()]
        found = None
        for i, line in enumerate(lines):
            if TOC.search(line):                 # 目录页，跳过
                continue
            m = HEAD.match(line)
            if m and i + 1 < len(lines) and not HEAD.match(lines[i + 1]):
                title = lines[i + 1][:24]
                if not TOC.search(title):
                    found = f"第{m.group(1)}节 {title}"
        if found:
            current = found
        out[str(p["page"])] = current
    return out


def main() -> None:
    mapping = {}
    for d in (config.PAGE_DIR, config.DATA_DIR / "dongmi" / "pages"):
        for f in sorted(d.glob("*.jsonl")):
            code = f.stem
            mapping[code] = doc_sections(f)
            labeled = sum(1 for v in mapping[code].values() if v)
            print(f"{code}: {len(mapping[code])} 页，识别到章节 {labeled}")
    (config.INDEX_DIR / "section_map.json").write_text(
        json.dumps(mapping, ensure_ascii=False), "utf-8")
    print(f"→ {config.INDEX_DIR / 'section_map.json'}")

    # 抽查
    m = mapping.get("600519", {})
    for pg in ("4", "6", "20", "60", "80"):
        print(f"  茅台 p{pg} → {m.get(pg)}")


if __name__ == "__main__":
    main()
