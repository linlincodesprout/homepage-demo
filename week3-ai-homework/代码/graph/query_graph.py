"""建图第六步 + 查询：向量化入库，然后按课程规定的 local / global 两种方式问图。

· 入库（第⑥步）：把实体描述和社区报告都算成向量，问题来了先找入口。
· local 查询：问题向量 → 找最像的实体 → 沿边走一两跳 → 关系句 + 原文块 → 作答
              （课程讲义：局部问题查节点邻域）
· global 查询：不找实体，把社区报告分批装进窗口 → 每批答一遍并给要点打分（map）
              → 按分汇总去重（reduce）（课程讲义：全景问题读社区摘要）
· 指标聚合：图里保留了「公司 --关系--> 指标=数值（期间/口径）」这条边，
  所以数值类全景问题可以直接顺着边把 12 家的数字提取出来——这是课程讲义强调的
  「数值节点必须带口径」带来的好处。

用法：
    python graph/query_graph.py --build                    # 先算实体/报告向量（入库）
    python graph/query_graph.py --mode global --q "问题"
    python graph/query_graph.py --mode local  --q "问题"
    python graph/query_graph.py --metrics                  # 打印从图里聚合出的指标表
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag import chat
from retrieval import get_encoder

GRAPH_DIR = config.DATA_DIR / "graph"
ENT_VEC = GRAPH_DIR / "graph_vectors_entity.npy"
ENT_KEYS = GRAPH_DIR / "graph_vectors_entity_keys.json"
REP_VEC = GRAPH_DIR / "graph_vectors_report.npy"
REP_KEYS = GRAPH_DIR / "graph_vectors_report_keys.json"


def load_graph() -> dict:
    return json.loads((GRAPH_DIR / "graph.json").read_text("utf-8"))


def load_reports() -> list[dict]:
    return [json.loads(l) for l in
            open(GRAPH_DIR / "community_reports.jsonl", encoding="utf-8")]


# ---------------------------------------------------------------------------
# ⑥ 向量化入库
# ---------------------------------------------------------------------------
def build_vectors() -> None:
    """本机 CPU 编码太慢（2 分钟 32 条），实际用的是 GPU 上的 gpu/embed_generic.py：
    先在节点上跑出 graph_vectors_entity.npy / report.npy，再下载回 data/graph/。"""
    if ENT_VEC.exists() and REP_VEC.exists():
        print("向量已存在（来自 GPU 端的 embed_generic.py），跳过本机编码")
        return
    g = load_graph()
    reports = load_reports()
    enc = get_encoder("qwen3")

    names = list(g["nodes"])
    ent_texts = [f"{n}（{g['nodes'][n]['type']}）：{g['nodes'][n]['description'][:400]}"
                 for n in names]
    rep_texts = [f"{r['title']}｜{r['summary']}｜" + "；".join(r["key_facts"][:6])
                 for r in reports]
    t0 = time.time()
    ent_vec = enc.encode(ent_texts) if ent_texts else np.zeros((0, 1024), "float32")
    rep_vec = enc.encode(rep_texts) if rep_texts else np.zeros((0, 1024), "float32")
    np.save(ENT_VEC, ent_vec)
    np.save(REP_VEC, rep_vec)
    print(f"入库完成：实体 {ent_vec.shape[0]} 个、社区报告 {rep_vec.shape[0]} 份，"
          f"用时 {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# local 查询
# ---------------------------------------------------------------------------
LOCAL_SYSTEM = ("你是财报分析助手。只能依据【图谱证据】回答，每句话标注来源编号。"
                "材料不足就说明缺什么，不要推测。")


def local(question: str, hops: int = 1, top_entities: int = 12) -> dict:
    g = load_graph()
    names = json.loads(ENT_KEYS.read_text("utf-8"))
    ent_vec = np.load(ENT_VEC)
    qv = get_encoder("qwen3").encode([question])[0]
    sims = ent_vec @ qv
    order = np.argsort(-sims)[:top_entities]
    seeds = [names[i] for i in order]

    reached = set(seeds)
    for _ in range(hops):
        for r in g["relations"]:
            if r["head"] in reached:
                reached.add(r["tail"])
            if r["tail"] in reached:
                reached.add(r["head"])
    rels = [r for r in g["relations"] if r["head"] in reached and r["tail"] in reached]
    rels.sort(key=lambda r: -r["strength"])
    rels = rels[:60]

    lines = ["【入口实体】"]
    for n in seeds:
        lines.append(f"- {n}（{g['nodes'][n]['type']}）：{g['nodes'][n]['description'][:200]}")
    lines.append("\n【关系清单】")
    for i, r in enumerate(rels, 1):
        lines.append(f"[{i}] {r['head']} --{r['relation']}（{r['why']}，强度{r['strength']}）"
                     f"--> {r['tail']}（出处块 {r['chunk_id']}，{r['doc']} 第{r['page']}页）")
    ctx = "\n".join(lines)
    ans = chat(LOCAL_SYSTEM, f"{ctx}\n\n【问题】\n{question}\n\n请作答，逐句带 [编号]。")
    return {"mode": "local", "answer": ans, "seeds": seeds,
            "relations": rels, "context": ctx}


# ---------------------------------------------------------------------------
# global 查询（map-reduce）
# ---------------------------------------------------------------------------
MAP_SYSTEM = ("你在读一批财报知识图谱的社区报告。请回答给定的问题，"
              "只输出与问题有关的要点，每条要点一个 0–10 的分（越相关越高）。"
              '只输出 JSON：{"points":[{"text":"","score":8,"reports":[编号]}]}。'
              "这批报告若与问题无关，就返回空数组。")

REDUCE_SYSTEM = ("你是财报分析师。下面是从全库社区报告里筛出来的要点（带相关度分数）。"
                 "请据此回答问题：先给结论，再逐条列证据；每条证据后标注 [报告编号]。"
                 "要点之间冲突时以分数高的为准；材料没覆盖的部分明确说未覆盖。")


def global_answer(question: str, batch: int = 12, level: int = 0,
                  workers: int = 6) -> dict:
    reports = [r for r in load_reports() if r["level"] == level]
    if not reports:
        return {"mode": "global", "answer": "没有社区报告", "points": []}
    batches = [reports[i:i + batch] for i in range(0, len(reports), batch)]
    print(f"global 查询：{len(reports)} 份报告分 {len(batches)} 批（map 并发）")

    def do(b):
        lines = []
        for r in b:
            lines.append(f"【报告 {r['index']}】{r['title']}\n{r['summary']}\n要点："
                         + "；".join(r["key_facts"][:8]))
        try:
            raw = re.sub(r"^```(json)?|```$", "",
                         chat(MAP_SYSTEM, f"【问题】{question}\n\n" + "\n\n".join(lines),
                              temperature=0.0).strip(), flags=re.M).strip()
            return json.loads(raw).get("points", [])
        except Exception:  # noqa: BLE001
            return []

    t0 = time.time()
    points = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for got in pool.map(do, batches):
            points.extend(got)
    print(f"map 完成：{len(points)} 条候选要点，用时 {time.time()-t0:.0f}s")
    points.sort(key=lambda p: -float(p.get("score", 0)))
    top = points[:60]
    body = "\n".join(f"- （{p.get('score')}分，报告{p.get('reports')}）{p.get('text')}"
                     for p in top)
    ans = chat(REDUCE_SYSTEM, f"【问题】{question}\n\n【要点】\n{body}")
    return {"mode": "global", "answer": ans, "points": top,
            "n_reports": len(reports), "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# 从图里聚合指标（数值节点带口径的红利）
# ---------------------------------------------------------------------------
def metrics_table() -> list[dict]:
    g = load_graph()
    nodes = g["nodes"]
    rows: dict[tuple[str, str], dict] = {}
    for r in g["relations"]:
        htype = nodes.get(r["head"], {}).get("type")
        ttype = nodes.get(r["tail"], {}).get("type")
        if htype != "上市公司" or ttype not in ("数值", "指标"):
            continue
        m = re.match(r"^(20\d\d年(?:上半年|半年度|度))(.+?)=(.*?)(（.*）)?$", r["tail"])
        if not m:
            continue
        period, metric, value, caliber = m.groups()
        key = (r["head"], metric)
        rows[key] = {"company": r["head"], "period": period, "metric": metric,
                     "value": value, "caliber": (caliber or "").strip("（）"),
                     "chunk_id": r["chunk_id"],
                     "doc": r.get("doc", r.get("company", "")),
                     "page": r.get("page")}
    out = sorted(rows.values(), key=lambda x: (x["metric"], x["company"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--mode", choices=["local", "global"])
    ap.add_argument("--q", default="")
    ap.add_argument("--level", type=int, default=0)
    args = ap.parse_args()

    if args.build:
        build_vectors()
    if args.metrics:
        rows = metrics_table()
        print(f"从图里聚合出 {len(rows)} 条指标")
        for r in rows[:40]:
            print(f"  {r['company']:6s} {r['period']} {r['metric']} = "
                  f"{r['value']}（{r['caliber']}）")
    if args.mode:
        fn = local if args.mode == "local" else (
            lambda q: global_answer(q, level=args.level))
        res = fn(args.q)
        print("\n" + res["answer"])


if __name__ == "__main__":
    main()
