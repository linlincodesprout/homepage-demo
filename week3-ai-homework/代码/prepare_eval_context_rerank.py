"""评测上下文 v3：先召回 30 块，再让大模型从中**挑出最该看的 4 块**。

依据：课程讲义实验一的结果是「只用向量分数 R@1=0.20，本机重排 0.60，DeepSeek 一次挑 0.75，
Jev 0.85」——重排是提升检索精度最有效的杠杆，而且它顺带解决我们的另一个痛点：
窗口里放入三张长得一样的表时，小模型会取错口径；首先不相关的表挑掉，它就没得取错。

用法：
    python prepare_eval_context_rerank.py --candidates 30 --keep 4
输出：
    data/dongmi/eval_with_context_rerank.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from rag import chat
from retrieval import get_retriever

DONGMI = config.DATA_DIR / "dongmi"
CACHE = DONGMI / "rerank_cache.jsonl"

CALIBER_RE = {
    "合并报表": re.compile(r"合并(资产负债表|利润表|现金流量表|所有者权益变动表)"),
    "母公司报表": re.compile(r"母公司(资产负债表|利润表|现金流量表|所有者权益变动表)"),
}

PICK_SYSTEM = (
    "你是检索助手。用户会给你一个问题和一批候选文本块（每块有编号）。"
    "请挑出**最可能包含答案**的若干块，按重要性排序。\n"
    "判断要点：\n"
    "1) 优先选和问题口径、期间一致的块（问题问上半年就不要选年报，问合并口径就不要选母公司报表）；\n"
    "2) 优先选含具体数字的表格块或附注块，不要选只讲业务概述的段落；\n"
    "3) 只输出编号，用逗号分隔，不要解释。例如：3,7,1,12"
)


def caliber_of(text: str) -> str:
    for label, pat in CALIBER_RE.items():
        if pat.search(text):
            return label
    return ""


def head_of(c: dict, i: int) -> str:
    cal = caliber_of(c["text"])
    return (f"[{i}] {c['doc']}｜{c['section'] or '未标注章节'}｜第 {c['page']} 页｜{c['kind']}"
            + (f"｜口径：{cal}" if cal else ""))


def pick(question: str, cands: list[dict], keep: int) -> list[int]:
    listing = "\n".join(f"{head_of(c, i)}\n{c['text'][:300]}" for i, c in enumerate(cands, 1))
    user = f"【问题】\n{question}\n\n【候选块】\n{listing}\n\n请挑出最该看的 {keep} 块，输出编号。"
    for attempt in range(4):
        try:
            raw = chat(PICK_SYSTEM, user, temperature=0.0)
            nums = [int(x) for x in re.findall(r"\d+", raw)]
            seen, order = set(), []
            for n in nums:
                if 1 <= n <= len(cands) and n not in seen:
                    seen.add(n)
                    order.append(n - 1)
            if order:
                return order[:keep]
            raise ValueError("no numbers")
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return list(range(min(keep, len(cands))))          # 兜底：用原始前几名


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=30)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DONGMI / "eval_set.jsonl", encoding="utf-8")]
    retriever = get_retriever("qwen3")

    cand_lists = []
    for r in rows:
        hits = retriever.search(r["eval_question"], mode="hybrid", k=args.candidates)
        cand_lists.append([h["chunk"] for h in hits])
    print(f"召回完成：{len(rows)} 题 × {args.candidates} 候选")

    cache = {}
    if CACHE.exists():
        for line in open(CACHE, encoding="utf-8"):
            rec = json.loads(line)
            cache[rec["question"]] = rec["picked"]
    todo = [(i, r) for i, r in enumerate(rows) if r["eval_question"] not in cache]
    print(f"需要重排 {len(todo)} 题（缓存 {len(cache)}）")

    lock = threading.Lock()
    done = 0
    t0 = time.time()
    with open(CACHE, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(pick, r["eval_question"], cand_lists[i], args.keep): (i, r)
                for i, r in todo}
        for fut in as_completed(futs):
            i, r = futs[fut]
            picked = fut.result()
            with lock:
                cache[r["eval_question"]] = picked
                f.write(json.dumps({"question": r["eval_question"], "picked": picked},
                                   ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(todo)}，{(time.time()-t0)/60:.1f} 分钟")

    out = []
    for i, r in enumerate(rows):
        idxs = cache.get(r["eval_question"], list(range(args.keep)))
        blocks = [cand_lists[i][j] for j in idxs if j < len(cand_lists[i])]
        ctx = "\n\n".join(f"{head_of(c, k+1)}\n{c['text']}" for k, c in enumerate(blocks))
        out.append({"pool": r["pool"], "variant": r["variant"],
                    "question": r["eval_question"], "reference": r["answer"],
                    "period": r["period"], "page": r["page"],
                    "context": ctx, "hits": [c["id"] for c in blocks]})
    path = DONGMI / f"eval_with_context_rerank{args.keep}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写好 {len(out)} 条（每条约 {len(out[0]['context'])} 字）→ {path}")

    from eval_a import numbers_in
    hit = 0
    for r in out:
        need = [n for n in numbers_in(r["reference"]) if abs(n) >= 1000]
        have = numbers_in(r["context"])
        if all(any(abs(h - n) <= max(abs(n) * 0.01, 0.02) for h in have) for n in need):
            hit += 1
    print(f"上下文含参考答案全部关键数字：{hit}/{len(out)}（k6 版本是 100/160）")


if __name__ == "__main__":
    main()
