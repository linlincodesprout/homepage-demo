"""建图第一步的选材：从 12 家白酒公司的 2026 半年报里挑出"讲得出关系"的块。

课件的半年报图谱只建了第二、三、五节（主要财务指标 / 管理层讨论与分析 / 重要事项），
83 份 = 1,528 块。我们 12 家，按同样的口径：管理层讨论与分析的叙述段（行业、竞争、
风险、应对、渠道、产能、项目、政策）+ 主要会计数据表。

（不重新切块：重切会让 11,047 块的 id 变化，整个向量库和已有评测都会失效，
所以这里只做"选"，改的是选材范围，不是切法。）

用法：
    python graph/select_blocks.py --per-company 50
输出：
    data/graph/blocks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

GRAPH_DIR = config.DATA_DIR / "graph"

NARRATIVE = ["行业", "竞争", "风险", "应对", "渠道", "经销商", "产能", "项目",
             "政策", "产品", "价格", "市场", "库存", "动销", "品牌", "产能扩张",
             "消费", "需求", "调整", "直销", "批发", "电商", "团购"]
METRIC = ["主要会计数据", "营业收入", "归属于上市公司股东的净利润", "基本每股收益"]


def score(c: dict) -> float:
    text = c["text"]
    s = sum(1.5 for k in NARRATIVE if k in text)
    s += sum(3.0 for k in METRIC if k in text)
    if c["kind"] == "正文":
        s += 1
    if len(text) < 200:
        s -= 2
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-company", type=int, default=50)
    args = ap.parse_args()

    chunks = [json.loads(l) for l in
              open(config.INDEX_DIR / "chunks.jsonl", encoding="utf-8")]
    by_code: dict[str, list[dict]] = {}
    for c in chunks:
        if c["period"] != "2026H1":
            continue
        by_code.setdefault(c["code"], []).append(c)

    picked = []
    for code, items in by_code.items():
        ranked = sorted(items, key=score, reverse=True)
        keep = [c for c in ranked if score(c) > 0][:args.per_company]
        picked.extend(keep)
        print(f"{code} {items[0]['company']:6s} 共 {len(items):4d} 块 → 选 {len(keep)}")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_DIR / "blocks.jsonl", "w", encoding="utf-8") as f:
        for c in picked:
            f.write(json.dumps({k: c[k] for k in
                                ("id", "code", "company", "doc", "period", "section",
                                 "page", "kind", "text")}, ensure_ascii=False) + "\n")
    print(f"\n合计 {len(picked)} 块 → {GRAPH_DIR / 'blocks.jsonl'}")
    print("估算抽取成本：每块一轮调用，共 %d 次调用" % (len(picked) * 2))


if __name__ == "__main__":
    main()
