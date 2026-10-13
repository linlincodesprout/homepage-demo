"""四格评测的检索部分在本机算好，GPU 那边只负责生成。

评测集 160 条（60 条训练内 + 20 条未见，各含原问法和换问法），
每条用知识库（11,047 块，Qwen3-Embedding-0.6B 向量 + BM25 融合）取前 6 块，
把上下文拼好写进 eval_with_context.jsonl，省得在 GPU 机上再下一遍向量模型。
"""

from __future__ import annotations

import json

import config
from rag import build_context
from retrieval import get_retriever

DONGMI = config.DATA_DIR / "dongmi"


def main() -> None:
    rows = [json.loads(l) for l in open(DONGMI / "eval_set.jsonl", encoding="utf-8")]
    retriever = get_retriever("qwen3")
    out = []
    for i, r in enumerate(rows, 1):
        hits = retriever.search(r["eval_question"], mode="hybrid", k=config.TOP_K_FINAL)
        out.append({
            "pool": r["pool"], "variant": r["variant"],
            "question": r["eval_question"], "original_question": r["question"],
            "reference": r["answer"], "period": r["period"], "page": r["page"],
            "context": build_context(hits),
            "hits": [h["chunk"]["id"] for h in hits],
        })
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}")
    path = DONGMI / "eval_with_context.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写好 {len(out)} 条 → {path}")

    # 顺带统计：知识库里到底有没有参考答案里的数字（判断是检索没找到还是模型没用好）
    from eval_a import numbers_in
    n_ctx_hit = 0
    for r in out:
        need = numbers_in(r["reference"])
        have = numbers_in(r["context"])
        need_big = [n for n in need if abs(n) >= 1000]
        if all(any(abs(h - n) <= max(abs(n) * 0.01, 0.02) for h in have) for n in need_big):
            n_ctx_hit += 1
    print(f"其中 {n_ctx_hit}/{len(out)} 条的上下文里已包含参考答案的全部关键数字"
          f"（其余是检索没取到）")


if __name__ == "__main__":
    main()
