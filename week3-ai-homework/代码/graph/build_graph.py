"""建图第三、四步：合并实体描述 + Leiden 社区聚类。

③合并描述：同一个实体在几十块里被提到，把它们合成一段连贯描述（课件说"按先后写"）。
       为了控成本，只对出现 ≥3 次的实体做，且一次调用合并 6 个实体。
④社区聚类：课件用 Leiden（我们装了 igraph + leidenalg，用的就是它）。
       先递归切到"叶子社区 ≤10 个实体"，再把社区当节点继续聚，形成层级。

用法：
    python graph/build_graph.py --min-mentions 3
输出：
    data/graph/graph.json        实体、关系、社区、层级
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag import chat

GRAPH_DIR = config.DATA_DIR / "graph"

# 课件踩的坑：同一家公司抽成两个节点。这里做一层规范化。
ALIAS = {
    "茅台": "贵州茅台", "贵州茅台酒股份有限公司": "贵州茅台", "茅台集团": "中国贵州茅台酒厂(集团)有限责任公司",
    "五粮液股份": "五粮液", "宜宾五粮液股份有限公司": "五粮液",
    "泸州老窖股份": "泸州老窖", "泸州老窖股份有限公司": "泸州老窖",
    "洋河": "洋河股份", "江苏洋河酒厂股份有限公司": "洋河股份",
    "汾酒": "山西汾酒", "山西杏花村汾酒厂股份有限公司": "山西汾酒",
    "古井": "古井贡酒", "安徽古井贡酒股份有限公司": "古井贡酒",
    "今世缘酒业": "今世缘", "江苏今世缘酒业股份有限公司": "今世缘",
    "口子酒业": "口子窖", "安徽口子酒业股份有限公司": "口子窖",
    "迎驾": "迎驾贡酒", "安徽迎驾贡酒股份有限公司": "迎驾贡酒",
    "舍得": "舍得酒业", "舍得酒业股份有限公司": "舍得酒业",
    "四川水井坊股份有限公司": "水井坊", "老白干": "老白干酒",
    "河北衡水老白干酒业股份有限公司": "老白干酒",
}

MERGE_SYSTEM = (
    "你在为知识图谱合并同一个实体的多段描述。把每一段描述写成一段连贯的中文说明："
    "按来源顺序（期间早的在前），保留关键数字和口径，不要编造，不要加评价。"
    "输出 JSON：{\"实体名\": \"合并后的描述\", ...}，只输出 JSON。"
)


def canon(name: str) -> str:
    name = re.sub(r"[\s\u3000]+", "", str(name))
    return ALIAS.get(name, name)


def merge_descriptions(nodes: dict, min_mentions: int, workers: int) -> None:
    todo = [n for n, v in nodes.items() if len(v["mentions"]) >= min_mentions]
    todo.sort(key=lambda n: -len(nodes[n]["mentions"]))
    print(f"需要合并描述的实体：{len(todo)} 个（提及 ≥{min_mentions} 次）")
    batches = [todo[i:i + 6] for i in range(0, len(todo), 6)]

    def do(batch):
        lines = []
        for n in batch:
            snippets = [f"({m['period']} {m['doc']} 第{m['page']}页) {m['text'][:260]}"
                        for m in nodes[n]["mentions"][:6]]
            lines.append(f"【实体】{n}\n" + "\n".join(f"  - {s}" for s in snippets))
        user = "请为下面每个实体，把它的多段描述合并成一段：\n\n" + "\n\n".join(lines)
        try:
            raw = re.sub(r"^```(json)?|```$", "", chat(MERGE_SYSTEM, user, temperature=0.2).strip(),
                         flags=re.M).strip()
            return batch, json.loads(raw)
        except Exception:  # noqa: BLE001
            return batch, {}

    done = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(do, b) for b in batches]
        for fut in as_completed(futs):
            batch, got = fut.result()
            with lock:
                for n in batch:
                    if n in got and isinstance(got[n], str) and len(got[n]) > 20:
                        nodes[n]["description"] = got[n]
                done += 1
                if done % 20 == 0:
                    print(f"  合并 {done}/{len(batches)} 批", flush=True)
    n_desc = sum(1 for v in nodes.values() if v.get("description"))
    print(f"合并完成：{n_desc} 个实体有合并描述")


def leiden_levels(nodes: dict, edges: list[tuple[str, str, int]]) -> tuple[list[list[str]], list[list[list[str]]]]:
    """递归 Leiden：先切到叶子社区（≤10 实体），再逐层聚合。返回 (叶子社区, 各层)。"""
    import igraph as ig
    import leidenalg

    nodes = list(nodes)
    index = {n: i for i, n in enumerate(nodes)}

    def partition(sub: list[str], resolution: float) -> list[list[str]]:
        if len(sub) <= 10:
            return [sub]
        g = ig.Graph()
        g.add_vertices(len(sub))
        pos = {n: i for i, n in enumerate(sub)}
        g.add_edges([(pos[h], pos[t]) for h, t, _ in edges if h in pos and t in pos])
        if g.ecount() == 0:
            return [sub]
        g.es["weight"] = [w for h, t, w in edges if h in pos and t in pos]
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, weights="weight",
            resolution_parameter=resolution, n_iterations=-1, seed=42)
        groups = [[sub[i] for i in c] for c in part]
        if len(groups) == 1:
            return [sub]
        out = []
        for grp in groups:
            if len(grp) > 10:
                out.extend(partition(grp, resolution * 1.6))
            else:
                out.append(grp)
        return out

    leaves = partition(nodes, 1.0)
    print(f"叶子社区：{len(leaves)} 个，最大 {max(len(c) for c in leaves)} 个实体")

    levels = [leaves]
    current = leaves
    while len(current) > 1:
        # 把社区当节点，社区之间有边就相连，再跑一次 Leiden
        cid = {n: i for i, c in enumerate(current) for n in c}
        cedges = defaultdict(int)
        for h, t, w in edges:
            if h in cid and t in cid and cid[h] != cid[t]:
                cedges[(cid[h], cid[t])] += w
        if not cedges:
            break
        g = ig.Graph(n=len(current))
        keys = list(cedges)
        g.add_edges(keys)
        g.es["weight"] = [cedges[k] for k in keys]
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                        weights="weight", resolution_parameter=1.0,
                                        n_iterations=-1, seed=42)
        nxt = [[e for c in grp for e in current[c]] for grp in part]
        if len(nxt) == len(current):
            break
        levels.append(nxt)
        current = nxt
    print(f"层级：{len(levels)} 层，自上而下规模 " +
          " → ".join(str(len(l)) for l in reversed(levels)))
    return leaves, levels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mentions", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(GRAPH_DIR / "extract_cache.jsonl", encoding="utf-8")]
    block_text = {json.loads(l)["id"]: json.loads(l)["text"]
                  for l in open(GRAPH_DIR / "blocks.jsonl", encoding="utf-8")}
    print(f"抽取记录 {len(recs)} 块")

    nodes: dict[str, dict] = {}
    rels: list[dict] = []
    for r in recs:
        for e in r["entities"]:
            name = canon(e["name"])
            if not name:
                continue
            node = nodes.setdefault(name, {"type": e["type"], "mentions": []})
            node["mentions"].append({"chunk_id": r["chunk_id"], "period": r["period"],
                                     "doc": r["doc"], "page": r["page"],
                                     "text": block_text.get(r["chunk_id"], "")})
        for rel in r["relations"]:
            h, t = canon(rel["head"]), canon(rel["tail"])
            if h in nodes and t in nodes and h != t:
                rels.append({**rel, "head": h, "tail": t, "chunk_id": r["chunk_id"],
                             "code": r["code"], "company": r["company"],
                             "period": r["period"], "page": r["page"]})
    print(f"实体 {len(nodes)} 个（规范化后），关系 {len(rels)} 条")

    merge_descriptions(nodes, args.min_mentions, args.workers)

    weights = defaultdict(int)
    for rel in rels:
        key = tuple(sorted((rel["head"], rel["tail"])))
        weights[key] = max(weights[key], rel["strength"])
    edges = [(h, t, w) for (h, t), w in weights.items()]
    leaves, levels = leiden_levels(nodes, edges)

    out = {
        "nodes": {n: {"type": v["type"], "description": v.get("description", ""),
                      "mentions": [(m["chunk_id"], m["period"], m["doc"], m["page"])
                                   for m in v["mentions"]]}
                  for n, v in nodes.items()},
        "relations": rels,
        "leaf_communities": leaves,
        "levels": levels,
    }
    (GRAPH_DIR / "graph.json").write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    print(f"图谱 → {GRAPH_DIR / 'graph.json'}"
          f"（{len(nodes)} 实体 / {len(rels)} 关系 / {len(leaves)} 叶子社区）")


if __name__ == "__main__":
    main()
