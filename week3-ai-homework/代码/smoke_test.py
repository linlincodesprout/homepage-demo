"""冒烟测试：三条检索路在同一道课上原题上的表现。

题目直接沿用课件里的演示问题：「茅台上半年直销渠道收入是多少、占营业收入多少？」
"""

from __future__ import annotations

import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from retrieval import brief, get_retriever  # noqa: E402

QUESTION = "贵州茅台2026年上半年通过直销渠道实现的收入是多少，占营业收入的比重是多少？"


def show(title: str, hits: list[dict]) -> None:
    print(f"\n===== {title} =====")
    for row in brief(hits)[:6]:
        print(f"[{row['n']}] {row['id']} {row['company']} p{row['page']} {row['kind']} "
              f"rrf={row['rrf']} v={row['vector_rank']} b={row['bm25_rank']}")
        print(f"    {row['head']}")


if __name__ == "__main__":
    for vec in ("ali", "qwen3"):
        t0 = time.time()
        r = get_retriever(vec)
        print(f"\n######## 向量 = {vec}（加载 {time.time()-t0:.1f}s）")
        show(f"{vec} 纯向量 top6", r.search(QUESTION, mode="vector"))
        show(f"{vec} 纯BM25 top6", r.search(QUESTION, mode="bm25"))
        show(f"{vec} 融合 top6", r.search(QUESTION, mode="hybrid"))
