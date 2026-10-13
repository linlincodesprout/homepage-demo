"""深度三收口：拿三道全景题，把「向量 RAG / 指标表 / GraphRAG」各条路线都执行一次。

三道题（前两道是作业 A 原来答不出的那两道，第三道是课程讲义中半年报的 global 例题）：
  Q9  2026 上半年这 12 家白酒公司里，营业收入同比增速最高和最低的分别是谁、各是多少？
  Q10 2026 上半年这 12 家里，哪些公司的归母净利润同比下滑、各下滑多少？
  Q11 这 12 家公司上半年普遍提到哪些行业挑战？各自的应对措施是什么？（关系型全景题）

五条路线：
  A 向量 RAG（融合 top-6 直接给）
  B 向量 RAG + 重排（召回 30 → DeepSeek 挑 4）
  C 离线指标表（作业 A 里已经做过的那张跨公司指标表）
  D GraphRAG · global（读社区报告，map-reduce）
  E GraphRAG · 指标聚合（顺着图谱里「公司→指标=数值（期间/口径）」的边把数字提取出来）

用法：
    python graph/run_panorama.py            # 跑全部（带缓存）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from metrics import load_table_text
from rag import build_context, chat
from retrieval import get_retriever

GRAPH_DIR = config.DATA_DIR / "graph"
CACHE = GRAPH_DIR / "panorama_cache.json"

SYS = ("你是白酒行业分析师。只能依据给到的材料回答，逐条列出证据；"
       "材料没覆盖的明确写「材料未覆盖」，不要推测。")

QUESTIONS = [
    {"id": "Q9",
     "q": "2026年上半年，这12家白酒公司里营业收入同比增速最高和最低的分别哪一家，各是多少？",
     "expect": {"五粮液": 20.87, "迎驾贡酒": 8.08, "贵州茅台": 1.47, "今世缘": -7.41,
                "山西汾酒": -12.18, "老白干酒": -15.23, "舍得酒业": -15.34,
                "口子窖": -22.89, "古井贡酒": -27.01, "水井坊": -27.78,
                "洋河股份": -28.76, "泸州老窖": -36.35}},
    {"id": "Q10",
     "q": "2026年上半年，这12家白酒公司里哪些公司的归母净利润同比是下滑的？分别下滑多少？",
     "expect": {"水井坊": -105.90, "舍得酒业": -67.26, "口子窖": -48.97, "泸州老窖": -43.37,
                "古井贡酒": -40.89, "洋河股份": -40.10, "山西汾酒": -24.29,
                "老白干酒": -18.47, "今世缘": -6.60, "贵州茅台": -1.95,
                "五粮液": 89.30, "迎驾贡酒": 2.79}},
    {"id": "Q11",
     "q": "2026年上半年，这12家白酒公司普遍提到哪些行业挑战？各家分别怎么应对？",
     "expect": {}},
]

COMPANIES = ["贵州茅台", "五粮液", "泸州老窖", "洋河股份", "山西汾酒", "古井贡酒",
             "今世缘", "口子窖", "迎驾贡酒", "舍得酒业", "水井坊", "老白干酒"]


def nums(text: str) -> list[float]:
    out = []
    for raw in re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", "")):
        try:
            out.append(float(raw.rstrip(".")))
        except ValueError:
            pass
    return out


def coverage(answer: str, expect: dict) -> tuple[int, list[str]]:
    hit, missed = 0, []
    for company, value in expect.items():
        near = company in answer or company[:3] in answer
        val = any(abs(abs(v) - abs(value)) <= max(abs(value) * 0.02, 0.05)
                  for v in nums(answer))
        if near and val:
            hit += 1
        else:
            missed.append(company)
    return hit, missed


def company_coverage(answer: str) -> int:
    return sum(1 for c in COMPANIES if c in answer)


# ---------------------------------------------------------------------------
def route_vector(q: str, rerank: bool) -> tuple[str, str]:
    r = get_retriever("qwen3")
    if not rerank:
        hits = r.search(q, mode="hybrid", k=config.TOP_K_FINAL)
        ctx = build_context(hits)
    else:
        cands = r.search(q, mode="hybrid", k=30)
        listing = "\n".join(
            f"[{i}] {h['chunk']['company']}｜{h['chunk']['doc']}｜{h['chunk']['section'] or '未标注章节'}"
            f"｜第 {h['chunk']['page']} 页｜{h['chunk']['kind']}\n{h['chunk']['text'][:300]}"
            for i, h in enumerate(cands, 1))
        pick = chat("你是检索助手。挑出最可能包含答案的 4 块，只输出编号，逗号分隔。",
                    f"【问题】{q}\n【候选】\n{listing}", temperature=0.0)
        idx = [int(x) for x in re.findall(r"\d+", pick)][:4]
        hits = [cands[i - 1] for i in idx if 1 <= i <= len(cands)] or cands[:4]
        ctx = build_context(hits)
    ans = chat(SYS, f"【材料】\n{ctx}\n\n【问题】\n{q}\n\n请作答，逐句带角标。")
    return ans, ctx


def route_metrics_table(q: str) -> tuple[str, str]:
    table = load_table_text()
    ctx = f"【跨公司指标表（离线抽取）】\n{table}"
    ans = chat(SYS, f"{ctx}\n\n【问题】\n{q}\n\n请作答；先逐条列全 12 家公司（公司+数字），再给结论。")
    return ans, ctx


def route_graph_global(q: str) -> tuple[str, str]:
    from graph.query_graph import global_answer
    res = global_answer(q, level=0)
    ctx = "\n".join(f"- （{p.get('score')}分）{p.get('text')}" for p in res["points"][:40])
    return res["answer"], ctx


def route_graph_metrics(q: str) -> tuple[str, str]:
    from graph.query_graph import metrics_table
    rows = metrics_table()
    lines = [f"- {r['company']} {r['period']} {r['metric']} = {r['value']}"
             f"（{r['caliber']}）〔块 {r['chunk_id']}〕" for r in rows]
    ctx = "【从知识图谱的『公司→指标=数值（期间/口径）』边聚合出来的数据】\n" + "\n".join(lines)
    ans = chat(SYS, f"{ctx}\n\n【问题】\n{q}\n\n请作答；先逐条列全 12 家公司（公司+数字），再给结论。")
    return ans, ctx


ROUTES = [
    ("A 向量RAG·top6", lambda q: route_vector(q, False)),
    ("B 向量RAG·重排4块", lambda q: route_vector(q, True)),
    ("C 离线指标表", route_metrics_table),
    ("D GraphRAG·global读社区报告", route_graph_global),
    ("E GraphRAG·指标聚合", route_graph_metrics),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    cache = json.loads(CACHE.read_text("utf-8")) if CACHE.exists() else {}

    for item in QUESTIONS:
        if args.only and item["id"] != args.only:
            continue
        print("\n" + "=" * 90)
        print(item["id"], item["q"])
        for name, fn in ROUTES:
            key = f"{item['id']}|{name}"
            if key in cache:
                ans, ctx = cache[key]["answer"], cache[key]["context"]
            else:
                t0 = time.time()
                try:
                    ans, ctx = fn(item["q"])
                except Exception as exc:  # noqa: BLE001
                    ans, ctx = f"（这条路跑失败：{type(exc).__name__}: {exc}）", ""
                cache[key] = {"answer": ans, "context": ctx,
                              "seconds": round(time.time() - t0, 1)}
                CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), "utf-8")
            if item["expect"]:
                hit, missed = coverage(ans, item["expect"])
                print(f"  {name:26s} 覆盖 {hit:2d}/12 家  用时 {cache[key]['seconds']}s"
                      f"  未覆盖：{'、'.join(missed) or '无'}")
            else:
                print(f"  {name:26s} 提到 {company_coverage(ans):2d}/12 家  "
                      f"用时 {cache[key]['seconds']}s  字数 {len(ans)}")
    print(f"\n缓存 → {CACHE}")


if __name__ == "__main__":
    main()
