"""把要入库的文本导出成一个 jsonl，交给 GPU 机器编码（本机 CPU 太慢）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

GRAPH_DIR = config.DATA_DIR / "graph"


def main() -> None:
    g = json.loads((GRAPH_DIR / "graph.json").read_text("utf-8"))
    reports = [json.loads(l) for l in
               open(GRAPH_DIR / "community_reports.jsonl", encoding="utf-8")]

    with open(GRAPH_DIR / "embed_texts.jsonl", "w", encoding="utf-8") as f:
        for n, v in g["nodes"].items():
            f.write(json.dumps({
                "kind": "entity", "key": n,
                "text": f"{n}（{v['type']}）：{v['description'][:400]}",
            }, ensure_ascii=False) + "\n")
        for r in reports:
            f.write(json.dumps({
                "kind": "report", "key": f"{r['level']}-{r['index']}",
                "text": f"{r['title']}｜{r['summary']}｜" + "；".join(r["key_facts"][:6]),
            }, ensure_ascii=False) + "\n")
    n_ent = len(g["nodes"])
    print(f"导出 {n_ent} 个实体 + {len(reports)} 份报告 → {GRAPH_DIR / 'embed_texts.jsonl'}")


if __name__ == "__main__":
    main()
