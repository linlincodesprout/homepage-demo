"""建图第二步：逐块抽实体与关系（课程规定的"橙色三步"之一，要调大模型）。

课程规定的规格（第三课深度三页）：
  · 半年报用 13 类实体：上市公司 / 子公司或关联方 / 人物 / 产品或业务 / 行业或赛道 /
    地区或市场 / 客户或供应商 / 项目或产能 / 事件 / 政策或法规 / 指标 / 风险因素 / 数值
  · 每条关系带一句「为什么相关」和 1–10 的强度分——后面合并描述、聚类、写社区报告都靠这两样
  · 抽完再问一轮补漏
  · 全部保留 source（块 id），因为图谱是通往证据的地图，不是证据替代品

课程讲义中提示的两个注意事项，我们直接规避：
  1) 实体重复：「茅台」和「贵州茅台」变成两个节点 → 提示词里强制用规范全称；
  2) 数值节点丢失口径：21,547 个数值只剩一个数 → 强制写成「指标=数值（期间/口径）」。

用法：
    python graph/extract.py --workers 8
输出：
    data/graph/extract_cache.jsonl   （每块的原始抽取结果）
"""

from __future__ import annotations

import argparse
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
CACHE = GRAPH_DIR / "extract_cache.jsonl"

ENTITY_TYPES = ["上市公司", "子公司或关联方", "人物", "产品或业务", "行业或赛道",
                "地区或市场", "客户或供应商", "项目或产能", "事件", "政策或法规",
                "指标", "风险因素", "数值"]

SYSTEM = f"""你是知识图谱抽取助手。从给定的半年报原文里抽取实体和关系，输出 JSON。

实体类型只能用这 13 类：{('、'.join(ENTITY_TYPES))}

硬性要求：
1) 公司名一律用规范全称（"贵州茅台酒股份有限公司"可以规范成"贵州茅台"，
   但绝不要同时出现"茅台"和"贵州茅台"两个节点）；同一实体只出现一次。
2) 数值必须带口径和期间，写成「指标=数值（期间/口径）」这样的名字，
   例如「2026年上半年营业收入=90,703,260,964.48元（合并）」。绝对不要只留一个光秃秃的数字。
3) 关系写成短句，说清性质；有变化的写成「从……到……」；
   每条关系都要给 1–10 的强度分 strength，并用 why 小结说明为什么相关。
4) 关系两端必须是你抽出来的实体名（head / tail）。
5) 只输出 JSON，不要解释：
{{"entities":[{{"name":"","type":""}}],
  "relations":[{{"head":"","tail":"","relation":"","why":"","strength":8}}]}}"""

SUPPLEMENT = """下面是一段财报原文，以及已经抽出来的实体和关系。
请**补充**漏掉的实体和关系（已经有的不要重复）。仍然只输出 JSON，格式同上。"""


def norm(name: str) -> str:
    return re.sub(r"[\s\u3000]+", "", str(name)).strip("、。，,")


def parse(raw: str) -> dict:
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
    obj = json.loads(raw)
    ents = []
    for e in obj.get("entities", []):
        n = norm(e.get("name", ""))
        t = str(e.get("type", "")).strip()
        if n and len(n) <= 60:
            ents.append({"name": n, "type": t if t in ENTITY_TYPES else "其他"})
    rels = []
    for r in obj.get("relations", []):
        h, t = norm(r.get("head", "")), norm(r.get("tail", ""))
        if h and t and h != t:
            try:
                s = int(float(r.get("strength", 5)))
            except (TypeError, ValueError):
                s = 5
            rels.append({"head": h, "tail": t,
                         "relation": str(r.get("relation", ""))[:80],
                         "why": str(r.get("why", ""))[:120],
                         "strength": max(1, min(10, s))})
    return {"entities": ents, "relations": rels}


def extract_one(c: dict) -> dict:
    user = (f"【来源块 id】{c['id']}\n【出处】{c['company']}｜{c['doc']}｜"
            f"{c['section'] or '未标注章节'}｜第 {c['page']} 页\n【原文】\n{c['text']}")
    ent, rel = [], []
    for rnd in range(2):
        prompt = user if rnd == 0 else (
            f"{user}\n\n【已抽取】{json.dumps({'entities': ent, 'relations': rel}, ensure_ascii=False)}"
            f"\n\n{SUPPLEMENT}")
        system = SYSTEM if rnd == 0 else SYSTEM
        for attempt in range(5):
            try:
                got = parse(chat(system, prompt, temperature=0.2))
                ent += [e for e in got["entities"] if e not in ent]
                rel += [r for r in got["relations"] if r not in rel]
                break
            except Exception:  # noqa: BLE001
                time.sleep(2.0 * (attempt + 1))      # 接口不稳定时延长等待后重试
    return {"chunk_id": c["id"], "code": c["code"], "company": c["company"],
            "doc": c["doc"], "period": c["period"], "section": c["section"],
            "page": c["page"], "kind": c["kind"],
            "entities": ent, "relations": rel}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    blocks = [json.loads(l) for l in open(GRAPH_DIR / "blocks.jsonl", encoding="utf-8")]
    if args.limit:
        blocks = blocks[:args.limit]
    done = {}
    if CACHE.exists():
        for line in open(CACHE, encoding="utf-8"):
            rec = json.loads(line)
            done[rec["chunk_id"]] = rec
    # 空结果不算做完：第一轮跑的时候接口一度不稳定，返回空结果的记录被误记为"已完成"，
    # 结果 600 块里 286 块是空的（4 家公司整家缺席），图谱覆盖度显著下降。
    done = {k: v for k, v in done.items() if v.get("entities") or v.get("relations")}
    todo = [c for c in blocks if c["id"] not in done]
    print(f"共 {len(blocks)} 块，已完成 {len(done)}，待抽 {len(todo)}（每块 2 轮调用）")

    lock = threading.Lock()
    n = 0
    t0 = time.time()
    with open(CACHE, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(extract_one, c): c for c in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 20 == 0:
                    rate = n / max(time.time() - t0, 1)
                    print(f"  {n}/{len(todo)} 块，{rate*60:.0f} 块/分钟，"
                          f"预计还需 {(len(todo)-n)/max(rate,1e-9)/60:.1f} 分钟", flush=True)
    print(f"抽取完成，用时 {(time.time()-t0)/60:.1f} 分钟 → {CACHE}")


if __name__ == "__main__":
    main()
