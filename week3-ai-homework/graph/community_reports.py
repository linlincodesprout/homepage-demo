"""建图第五步：给每个社区写一份报告（课件的"橙色三步"之三）。

课件的说法：「每个社区每一层写一份」——叶子社区给细节，上层社区给行业全景。
全景问题（global 查询）读的就是这些报告，不读原文块。

报告里必须保留出处（块 id），因为课件强调「图谱是通往分散证据的地图，不是证据替代品」。

用法：
    python graph/community_reports.py --workers 8
输出：
    data/graph/community_reports.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag import chat

GRAPH_DIR = config.DATA_DIR / "graph"
CACHE = GRAPH_DIR / "report_cache.jsonl"

SYSTEM = """你在为一个财报知识图谱的社区写报告。
给你一个社区里的实体（含合并描述）和它们之间的关系（含强度）。
请写一份简洁的报告，供"回答行业全景问题"使用：

1) title：一句话小标题（不超过 20 字）；
2) summary：3–6 句概括，把这个社区里最重要的事实和关系讲清楚；
3) key_facts：3–8 条要点，**每条必须带具体数字和口径**（如"泸州老窖2026年上半年营业收入
   10,472,224,575.78元，同比-36.35%（合并口径）"），没有数字的要点不要写；
4) companies：涉及的公司名列表。

只依据给到的材料，不要推测。只输出 JSON：
{"title":"","summary":"","key_facts":[""],"companies":[""]}"""


def build_prompt(nodes: dict, rels: list[dict], members: list[str], degree: dict) -> str:
    members = sorted(members, key=lambda n: -degree.get(n, 0))       # 大社区取连接最多的
    lines = ["【本社区实体】"]
    for n in members[:60]:
        v = nodes.get(n, {})
        desc = (v.get("description") or "")[:220]
        lines.append(f"- {n}（{v.get('type', '')}）：{desc}")
    inner = [r for r in rels if r["head"] in members and r["tail"] in members]
    inner.sort(key=lambda r: -r["strength"])
    lines.append("\n【本社区内部关系（按强度排序，最多 60 条）】")
    for r in inner[:60]:
        lines.append(f"- {r['head']} --{r['relation']}（强度{r['strength']}，"
                     f"{r['why']}）--> {r['tail']} [块 {r['chunk_id']}]")
    return "\n".join(lines)


def one(level: int, idx: int, members: list[str], nodes: dict, rels: list[dict],
        degree: dict) -> dict:
    user = build_prompt(nodes, rels, members, degree)
    for attempt in range(3):
        try:
            raw = re.sub(r"^```(json)?|```$", "",
                         chat(SYSTEM, user, temperature=0.3).strip(), flags=re.M).strip()
            obj = json.loads(raw)
            return {"level": level, "index": idx, "members": members,
                    "title": str(obj.get("title", ""))[:60],
                    "summary": str(obj.get("summary", "")),
                    "key_facts": [str(x) for x in obj.get("key_facts", [])][:10],
                    "companies": [str(x) for x in obj.get("companies", [])][:20],
                    "sources": sorted({r["chunk_id"] for r in rels
                                       if r["head"] in members and r["tail"] in members})[:40]}
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return {"level": level, "index": idx, "members": members, "title": "（生成失败）",
            "summary": "", "key_facts": [], "companies": [], "sources": []}


def member_key(members: list[str]) -> str:
    """用成员集合当缓存键：社区内容没变就不重写报告，省调用。

    图谱补抽了 286 块之后社区会变，但总有一部分社区的成员没动——
    按 (层号, 序号) 当键的话它们也会被当成新的重写一遍，白花钱。
    """
    return hashlib.md5("|".join(sorted(members)).encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    g = json.loads((GRAPH_DIR / "graph.json").read_text("utf-8"))
    nodes, rels = g["nodes"], g["relations"]
    degree: dict[str, int] = {}
    for r in rels:
        degree[r["head"]] = degree.get(r["head"], 0) + 1
        degree[r["tail"]] = degree.get(r["tail"], 0) + 1

    jobs = []
    for li, level in enumerate(g["levels"]):
        for ci, members in enumerate(level):
            if len(members) < 3:
                continue
            # 叶子层：绝大多数是"只出现一次的数值节点"凑成的社区，没有讲头，跳过
            if li == 0 and not any(nodes.get(m, {}).get("type") not in ("数值", "指标")
                                   for m in members):
                continue
            jobs.append((li, ci, members))
    print(f"要写 {len(jobs)} 份社区报告（{len(g['levels'])} 层，已过滤无内容的数值社区）")

    reused = {}
    if (GRAPH_DIR / "community_reports.jsonl").exists():
        for line in open(GRAPH_DIR / "community_reports.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("members") and rec.get("key_facts"):
                reused[member_key(rec["members"])] = rec
    todo, kept = [], []
    for li, ci, members in jobs:
        hit = reused.get(member_key(members))
        if hit:
            kept.append({**hit, "level": li, "index": ci})
        else:
            todo.append((li, ci, members))
    print(f"可复用旧报告 {len(kept)} 份，待写 {len(todo)} 份")

    if not todo:
        out = kept
        with open(GRAPH_DIR / "community_reports.jsonl", "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"社区报告 {len(out)} 份（全部复用）")
        return

    lock = threading.Lock()
    n = 0
    t0 = time.time()
    with open(CACHE, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, li, ci, m, nodes, rels, degree): (li, ci)
                for li, ci, m in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 20 == 0:
                    print(f"  {n}/{len(todo)}，{(time.time()-t0)/60:.1f} 分钟", flush=True)

    # 汇总成最终文件
    all_recs = [json.loads(l) for l in open(CACHE, encoding="utf-8")]
    seen = {}
    for r in all_recs:
        if r.get("key_facts"):
            seen[member_key(r["members"])] = r
    out = list(kept)
    for li, ci, members in jobs:
        rec = seen.get(member_key(members))
        if rec:
            out.append({**rec, "level": li, "index": ci})
    with open(GRAPH_DIR / "community_reports.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"社区报告 {len(out)} 份 → {GRAPH_DIR / 'community_reports.jsonl'}")


if __name__ == "__main__":
    main()
