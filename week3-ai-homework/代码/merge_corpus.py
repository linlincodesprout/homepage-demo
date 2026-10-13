"""把 B 方向的茅台历年财报并进 A 方向的知识库。

为什么并：作业 B 最后要「和 A 的知识库加同一个小模型比一比」。
如果 A 的库里只有茅台 2026 半年报，那道 2021 年的题本来就没材料，比出来的结果没意义。
所以把茅台 2021–2025 的 5 份年报也放进同一个库，两边共用一份知识底座。

去重规则：2026H1 茅台已在 A 库里，不再重复加入。

用法：
    python merge_corpus.py
输出：
    data/index/chunks.jsonl（覆盖，块数从 5,848 变成约 11,063）
    之后必须重跑 build_index 的向量部分（GPU）和 BM25。
"""

from __future__ import annotations

import json

import config


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main() -> None:
    main_chunks = load(config.INDEX_DIR / "chunks.jsonl")
    dongmi = load(config.DATA_DIR / "dongmi" / "chunks.jsonl")
    extra = [c for c in dongmi if c["period"] != "2026H1"]

    merged = main_chunks + extra
    merged.sort(key=lambda c: (c["code"], c["period"], c["page"]))
    for i, c in enumerate(merged):
        c["id"] = f"{c['code']}-{i:05d}"          # 重新编号，保证角标唯一
    with open(config.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in merged:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_doc: dict[str, int] = {}
    for c in merged:
        by_doc[c["doc"]] = by_doc.get(c["doc"], 0) + 1
    print(f"A 库 {len(main_chunks)} 块 + 茅台历年 {len(extra)} 块 = {len(merged)} 块")
    print(f"覆盖 {len({c['code'] for c in merged})} 家公司、{len(by_doc)} 份报告")
    print("下一步：重跑 GPU 向量编码、阿里向量、BM25")


if __name__ == "__main__":
    main()
