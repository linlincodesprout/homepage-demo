"""定向补边：把 12 家公司的关键指标（营收/净利 + 同比）确诊性地补进图谱。

为什么需要：通用抽取只对"关系"负责，它对同一张主要会计数据表可能抽成
"公司→营业收入=…"却不抽"营业收入同比增减"，于是全景题顺着边捞数时缺人。
这不是方法不行，是**建图覆盖度**问题——所以补法不是重跑全图，而是：
对每家的「主要会计数据」块用一份**定死的 schema** 再抽一次，把缺的边补上。

用法：
    python graph/patch_metrics.py
输出：
    data/graph/graph.json（原地补边，社区结构不动）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag import chat

GRAPH_DIR = config.DATA_DIR / "graph"

SYSTEM = """你在从一份上市公司半年报的「主要会计数据」表里抽结构化数据。
只输出 JSON，不要解释。schema：
{
  "company": "公司规范简称（如 贵州茅台）",
  "rows": [
    {"metric": "营业收入", "period": "2026年上半年", "value": "90,703,260,964.48元",
     "caliber": "合并", "yoy": "1.47%"},
    {"metric": "归属于上市公司股东的净利润", "period": "2026年上半年",
     "value": "44,516,880,421.86元", "caliber": "合并", "yoy": "-1.95%"}
  ]
}
要求：
1) 只抽这四个指标：营业收入、归属于上市公司股东的净利润、扣非归母净利润、经营活动产生的现金流量净额；
2) value 必须是原文数字原样（含单位），yoy 是原文给的同比百分比（没有就填 null）；
3) period 写成「2026年上半年」这样；
4) caliber 填「合并」或「母公司」，原文没写就填「未标注」。"""


def main() -> None:
    chunks = [json.loads(l) for l in
              open(config.INDEX_DIR / "chunks.jsonl", encoding="utf-8")]
    targets = {}
    for c in chunks:
        if c["period"] != "2026H1" or c["code"] in targets:
            continue
        if "主要会计数据" in c["text"] and "本报告期" in c["text"]:
            targets[c["code"]] = c
    print(f"找到 {len(targets)} 家公司的「主要会计数据」块")

    g = json.loads((GRAPH_DIR / "graph.json").read_text("utf-8"))
    nodes, rels = g["nodes"], g["relations"]
    added = 0

    for code, c in sorted(targets.items()):
        user = f"【公司】{c['company']}\n【原文】\n{c['text']}"
        for attempt in range(4):
            try:
                raw = re.sub(r"^```(json)?|```$", "",
                             chat(SYSTEM, user, temperature=0.0).strip(), flags=re.M).strip()
                data = json.loads(raw)
                break
            except Exception:  # noqa: BLE001
                data = None
        if not data:
            print(f"  {c['company']} 抽取失败")
            continue
        comp = data.get("company") or c["company"]
        comp = comp if comp in nodes else c["company"]
        nodes.setdefault(comp, {"type": "上市公司", "description": "", "mentions": []})
        for row in data.get("rows", []):
            metric, period = row.get("metric"), row.get("period") or "2026年上半年"
            value, cal = str(row.get("value")), str(row.get("caliber") or "未标注")
            yoy = row.get("yoy")
            key = f"{period}{metric}={value}（{cal}）"
            nodes.setdefault(key, {"type": "数值", "description": "", "mentions": []})
            rel = {"head": comp, "tail": key, "relation": f"{period}{metric}",
                   "why": "来自主要会计数据表（定向补边）", "strength": 9,
                   "chunk_id": c["id"], "code": code, "company": c["company"],
                   "period": c["period"], "page": c["page"]}
            if not any(r["head"] == comp and r["tail"] == key for r in rels):
                rels.append(rel)
                added += 1
            if yoy and str(yoy).lower() != "null":
                ykey = f"{period}{metric}同比增减={yoy}"
                nodes.setdefault(ykey, {"type": "指标", "description": "", "mentions": []})
                rel2 = {**rel, "tail": ykey, "relation": f"{period}{metric}同比增减"}
                if not any(r["head"] == comp and r["tail"] == ykey for r in rels):
                    rels.append(rel2)
                    added += 1
        print(f"  {c['company']}: 抽出 {len(data.get('rows', []))} 行")

    (GRAPH_DIR / "graph.json").write_text(json.dumps(g, ensure_ascii=False), "utf-8")
    print(f"补进 {added} 条边 → 现在 {len(nodes)} 实体 / {len(rels)} 关系")


if __name__ == "__main__":
    main()
