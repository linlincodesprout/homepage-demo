"""把图谱画成一张图（对应课程讲义中那张"半年报图谱全景"）。

取度数最高的若干个实体，节点大小=连接数，颜色=实体类型，边=关系（取两端都在图中的）。
纯数字/会计科目类节点太多且没有讲头，画图时按课程规定的做法排除掉一部分。

用法：
    python graph/render.py --top 160
输出：
    outputs/screenshots/A_图谱全景.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

GRAPH_DIR = config.DATA_DIR / "graph"

SKIP_TYPES = {"数值"}
COLORS = {
    "上市公司": "#f5b301", "子公司或关联方": "#4f9cf9", "产品或业务": "#5dd39e",
    "行业或赛道": "#b07cf0", "地区或市场": "#38bdf8", "客户或供应商": "#f97362",
    "项目或产能": "#a3e635", "事件": "#fb923c", "政策或法规": "#22d3ee",
    "指标": "#c084fc", "风险因素": "#f472b6", "人物": "#94a3b8", "其他": "#64748b",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=160)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib import font_manager

    # matplotlib 默认字体没有汉字，会画成方框；挂上系统的雅黑
    for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                 r"C:\Windows\Fonts\simsun.ttc"):
        if Path(cand).exists():
            font_manager.fontManager.addfont(cand)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    g = json.loads((GRAPH_DIR / "graph.json").read_text("utf-8"))
    nodes, rels = g["nodes"], g["relations"]

    deg = Counter()
    for r in rels:
        deg[r["head"]] += 1
        deg[r["tail"]] += 1
    keep = [n for n, _ in deg.most_common()
            if nodes.get(n, {}).get("type") not in SKIP_TYPES][:args.top]
    keep = set(keep)

    G = nx.Graph()
    for n in keep:
        G.add_node(n, type=nodes[n]["type"])
    for r in rels:
        if r["head"] in keep and r["tail"] in keep:
            G.add_edge(r["head"], r["tail"], weight=r["strength"])

    pos = nx.spring_layout(G, k=0.55, iterations=160, seed=42, weight="weight")
    sizes = [40 + 26 * deg[n] for n in G.nodes]
    colors = [COLORS.get(G.nodes[n]["type"], "#64748b") for n in G.nodes]

    fig, ax = plt.subplots(figsize=(16, 11), facecolor="#0f1420")
    ax.set_facecolor("#0f1420")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.16, width=0.6, edge_color="#7c8db5")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors,
                           alpha=0.92, linewidths=0.4, edgecolors="#0f1420")
    labels = {n: n for n in G.nodes if deg[n] >= 12 and len(str(n)) <= 16}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8,
                            font_color="#e8edf7",
                            font_family=["Microsoft YaHei", "SimHei", "sans-serif"])
    ax.set_title(f"白酒行业半年报知识图谱（12 家公司 · {len(nodes)} 实体 · {len(rels)} 关系）",
                 color="#e8edf7", fontsize=16, pad=14)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                          markerfacecolor=c, markeredgecolor="none", label=t)
               for t, c in COLORS.items() if any(G.nodes[n]["type"] == t for n in G.nodes)]
    ax.legend(handles=handles, loc="lower right", facecolor="#131a29",
              labelcolor="#e8edf7", framealpha=0.9, fontsize=10)
    ax.axis("off")

    out = config.OUT_DIR / "screenshots" / "A_图谱全景.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="#0f1420")
    print(f"图谱配图 → {out}")


if __name__ == "__main__":
    main()
