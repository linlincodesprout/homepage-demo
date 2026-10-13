"""评测上下文 v2：把 top-6 放宽到 top-12，并标出每块的口径（合并 / 母公司）。

为什么要改：
  1) 归因发现 160 条里只有 110 条的金块进了 top-6，30 条在 60 名内都找不到——上限被检索停滞；
  2) 材料里有答案却没答对的，典型错法是抄了「母公司」那张几乎一样的表
     （问"销售商品收到的现金"，参考是合并 1,406.9 亿，模型答成母公司 606.3 亿）——
     这正是课程讲义讲的『口径陷阱』。

用法：
    python prepare_eval_context2.py            # 默认 k=12
"""

from __future__ import annotations

import argparse
import json
import re

import config
from retrieval import get_retriever

DONGMI = config.DATA_DIR / "dongmi"

CALIBER_RE = {
    "母公司报表": re.compile(r"母公司(资产负债表|利润表|现金流量表|所有者权益变动表)"),
    "合并报表": re.compile(r"合并(资产负债表|利润表|现金流量表|所有者权益变动表)"),
}


def caliber_of(text: str) -> str:
    for label, pat in CALIBER_RE.items():
        if pat.search(text):
            return label
    return ""


def build(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        c = h["chunk"]
        cal = caliber_of(c["text"])
        head = (f"[{i}] {c['company']}｜{c['doc']}｜{c['section'] or '未标注章节'}"
                f"｜第 {c['page']} 页｜{c['kind']}"
                + (f"｜口径：{cal}" if cal else ""))
        parts.append(f"{head}\n{c['text']}")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=12)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DONGMI / "eval_set.jsonl", encoding="utf-8")]
    retriever = get_retriever("qwen3")
    out, tagged = [], 0
    for i, r in enumerate(rows, 1):
        hits = retriever.search(r["eval_question"], mode="hybrid", k=args.k)
        ctx = build(hits)
        tagged += sum(1 for h in hits if caliber_of(h["chunk"]["text"]))
        out.append({
            "pool": r["pool"], "variant": r["variant"],
            "question": r["eval_question"], "reference": r["answer"],
            "period": r["period"], "page": r["page"],
            "context": ctx, "hits": [h["chunk"]["id"] for h in hits],
        })
        if i % 40 == 0:
            print(f"  {i}/{len(rows)}")
    path = DONGMI / f"eval_with_context_k{args.k}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写好 {len(out)} 条（每条约 {len(out[0]['context'])} 字）→ {path}")
    print(f"上下文里共 {tagged} 个块被标出口径")

    from eval_a import numbers_in
    hit = 0
    for r in out:
        need = [n for n in numbers_in(r["reference"]) if abs(n) >= 1000]
        have = numbers_in(r["context"])
        if all(any(abs(h - n) <= max(abs(n) * 0.01, 0.02) for h in have) for n in need):
            hit += 1
    print(f"其中 {hit}/{len(out)} 条的上下文含参考答案全部关键数字（v1 是 100/160）")


if __name__ == "__main__":
    main()
