"""把 B 的低分拆开：多少是「检索没取到」，多少是「取到了但模型没答对」。

做法：对每条评测题，先在知识库里找出真正含答案数字的那一块（金块），
再看它在融合排序里的名次——名次 ≤6 却被答错，才是模型的问题；
名次 >6 就是 top-6 的锅，把 k 调大就能救。
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import config
from dongmi_score import hit_ratio
from eval_a import numbers_in
from retrieval import get_retriever


def gold_ids(reference: str, chunks: list[dict], period: str, page: int) -> list[str]:
    """含参考答案全部关键数字、且报告期/页码对得上的块。"""
    need = [n for n in numbers_in(reference) if abs(n) >= 1000]
    out = []
    for c in chunks:
        if c.get("period") != period:
            continue
        have = numbers_in(c["text"])
        if all(any(abs(h - n) <= max(abs(n) * 0.01, 0.02) for h in have) for n in need):
            out.append(c["id"])
    return out


def main() -> None:
    evals = [json.loads(l) for l in
             open(config.DATA_DIR / "dongmi" / "eval_with_context.jsonl", encoding="utf-8")]
    results = [json.loads(l) for l in open(config.OUT_DIR / "B_四格结果.jsonl", encoding="utf-8")]
    kbf = {(r["question"], r["variant"]): r for r in results if r["config"] == "kb_finetuned"}
    kbu = {(r["question"], r["variant"]): r for r in results if r["config"] == "kb_untrained"}

    retriever = get_retriever("qwen3")
    chunks = retriever.chunks

    rows = []
    for e in evals:
        golds = gold_ids(e["reference"], chunks, e["period"], e["page"])
        ranked = retriever.search(e["question"], mode="hybrid", k=60,
                                  company="贵州茅台", period=e["period"], auto_filter=False)
        order = [h["chunk"]["id"] for h in ranked]
        rank = min((order.index(g) + 1 for g in golds if g in order), default=None)
        got = kbf.get((e["question"], e["variant"]))
        got_u = kbu.get((e["question"], e["variant"]))
        rows.append({
            "pool": e["pool"], "variant": e["variant"], "question": e["question"],
            "gold_count": len(golds), "gold_rank": rank,
            "in_top6": rank is not None and rank <= 6,
            "in_top12": rank is not None and rank <= 12,
            "in_top30": rank is not None and rank <= 30,
            "correct_ft": bool(got and hit_ratio(e["reference"], got["output"])[0] >= 0.999),
            "correct_untrained": bool(got_u and hit_ratio(e["reference"], got_u["output"])[0] >= 0.999),
        })

    n = len(rows)
    print(f"评测 {n} 条")
    print("\n【金块在融合排序里的名次分布】")
    for name, cond in (("top6 内", lambda r: r["in_top6"]),
                       ("top12 内", lambda r: r["in_top12"]),
                       ("top30 内", lambda r: r["in_top30"]),
                       ("top60 内", lambda r: r["gold_rank"] is not None),
                       ("压根找不到", lambda r: r["gold_rank"] is None)):
        print(f"  {name}: {sum(1 for r in rows if cond(r))}/{n}")

    print("\n【微调后，按金块名次分桶的答对率】")
    for name, cond in (("金块在 top6", lambda r: r["in_top6"]),
                       ("金块在 7-12", lambda r: r["gold_rank"] is not None and 7 <= r["gold_rank"] <= 12),
                       ("金块在 13-30", lambda r: r["gold_rank"] is not None and 13 <= r["gold_rank"] <= 30),
                       ("金块在 30 名外/找不到", lambda r: r["gold_rank"] is None or r["gold_rank"] > 30)):
        sub = [r for r in rows if cond(r)]
        ok = sum(1 for r in sub if r["correct_ft"])
        print(f"  {name}: {len(sub)} 条，答对 {ok} 条（{ok/len(sub)*100:.0f}%）" if sub else f"  {name}: 0 条")

    print("\n【按池子×问法】")
    for key in [("seen", "original"), ("seen", "paraphrase"),
                ("unseen", "original"), ("unseen", "paraphrase")]:
        sub = [r for r in rows if (r["pool"], r["variant"]) == key]
        ok = sum(1 for r in sub if r["correct_ft"])
        top6 = sum(1 for r in sub if r["in_top6"])
        top12 = sum(1 for r in sub if r["in_top12"])
        print(f"  {key[0]:6s} {key[1]:10s} {len(sub):3d} 条 | 金块在 top6 {top6:3d} | "
              f"top12 {top12:3d} | 微调答对 {ok}")

    (config.OUT_DIR / "B_低分归因.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n明细 → {config.OUT_DIR / 'B_低分归因.json'}")


if __name__ == "__main__":
    main()
