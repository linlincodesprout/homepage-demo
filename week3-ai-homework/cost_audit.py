"""算一下这一轮 DeepSeek 到底花在哪：按本地缓存的字符数估 token。

估算口径：中文 DeepSeek 大致 1 token ≈ 1.5 个汉字（课程里也按这个量级讲），
所以 token ≈ 字符数 / 1.5。只算输入输出，不含系统提示词的固定开销（会低估一点）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

CHARS_PER_TOKEN = 1.5


def tok(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN)


def size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def lines_chars(p: Path, fields: list[str]) -> tuple[int, int]:
    """返回 (记录数, 指定字段的总字符数)。"""
    if not p.exists():
        return 0, 0
    n, chars = 0, 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            n += 1
            rec = json.loads(line)
            for k in fields:
                v = rec.get(k)
                if isinstance(v, str):
                    chars += len(v)
                elif isinstance(v, list):
                    chars += sum(len(json.dumps(x, ensure_ascii=False)) for x in v)
    return n, chars


def main() -> None:
    D = config.DATA_DIR
    rows = []

    # ---- 建图：抽取（600 块 × 2 轮）----
    blocks = D / "graph" / "blocks.jsonl"
    n, chars = lines_chars(blocks, ["text"])
    out_n, out_chars = lines_chars(D / "graph" / "extract_cache.jsonl",
                                   ["entities", "relations"])
    rows.append(("图谱·抽实体与关系（600 块 ×2 轮）", n * 2, chars * 2 + out_chars, out_chars))

    # ---- 建图：合并描述 ----
    merge_in = 0
    if (D / "graph" / "graph.json").exists():
        g = json.loads((D / "graph" / "graph.json").read_text("utf-8"))
        for v in g["nodes"].values():
            if len(v["mentions"]) >= 3 and v.get("description"):
                merge_in += 1400          # 每个实体约 6 段 × 260 字
    rows.append(("图谱·合并实体描述（386 个实体）", 65, merge_in, merge_in // 4))

    # ---- 建图：社区报告 ----
    rep_n, rep_out = lines_chars(D / "graph" / "community_reports.jsonl",
                                 ["summary", "key_facts"])
    rep_in = rep_n * 6000                 # 每次调用约 60 实体 + 60 关系，约 6k 字符
    rows.append(("图谱·社区报告（549 份）", rep_n, rep_in, rep_out))

    # ---- B 方向：造问答 ----
    qa_n, qa_out = lines_chars(D / "dongmi" / "qa_cache.jsonl", ["answer"])
    qa_in = qa_n * 1800
    rows.append(("B·生成投资者问答（660 组）", qa_n, qa_in, qa_out))

    # ---- B 方向：换问法 + 重排 ----
    para = size(D / "dongmi" / "paraphrase_cache.json")
    rows.append(("B·改写换问法（约 80 条，含缓存）", 80, para, para // 2))
    r_n, _ = lines_chars(D / "dongmi" / "rerank_cache.jsonl", [])
    rows.append(("B·重排挑块（160 题）", r_n, r_n * 7000, r_n * 40))

    # ---- A 方向：评测与补救实验 ----
    rows.append(("A·10 题评测 + 全景题补救（约 40 次作答）", 40, 40 * 6000, 40 * 700))

    print(f"{'环节':38s}{'调用数':>8s}{'输入 token':>14s}{'输出 token':>14s}")
    tot_in = tot_out = tot_calls = 0
    for name, calls, cin, cout in rows:
        print(f"{name:38s}{calls:>8d}{tok(cin):>14,d}{tok(cout):>14,d}")
        tot_in += tok(cin)
        tot_out += tok(cout)
        tot_calls += calls
    print("-" * 74)
    print(f"{'合计':38s}{tot_calls:>8d}{tot_in:>14,d}{tot_out:>14,d}")


if __name__ == "__main__":
    main()
