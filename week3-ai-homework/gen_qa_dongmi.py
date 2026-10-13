"""B 方向第一步：让大模型从财报原文生成「投资者问 / 董秘答」问答对。

课件原话：「让大模型从原文生成一批问答（投资者问、董秘答，答案注明出自哪份报告哪一页）」。
做法：
  1. 按「事实密度」给 5,701 个块打分（数字多、表格、经营/财务章节优先），每份报告取前 40 块；
  2. 另外留 20 块从没进过训练池的块，专门做「从未见过的事实」对照；
  3. 每条块让 DeepSeek 生成 1 个问答，答案必须原样引用数字并标注（来源：报告名 第 X 页）。

用法：
    python gen_qa_dongmi.py --per-report 40 --workers 8
输出：
    data/dongmi/qa_train.jsonl   （训练池，未抽检）
    data/dongmi/qa_unseen.jsonl  （对照池，绝不进训练）
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from rag import chat

DONGMI_DIR = config.DATA_DIR / "dongmi"
CACHE = DONGMI_DIR / "qa_cache.jsonl"

GEN_SYSTEM = (
    "你是贵州茅台酒股份有限公司的董事会秘书，正在业绩说明会上回答投资者提问。"
    "请依据我给你的财报原文，写一组投资者问答。\n"
    "硬性要求：\n"
    "1) 问题要像真实投资者问的：自然、口语化，围绕经营或财务事实，带上年份和口径；"
    "严禁出现「第 X 页」「表格中」「这一行」「原文」这类元信息，也不要问「某行四个数字分别是多少」；\n"
    "2) 如果原文只是表格碎片、拼不出一个自然的投资者问题，直接输出 {\"skip\": true}；\n"
    "3) 回答只使用原文信息，数字与原文完全一致（可带单位），不许换算、推算或补充原文没有的内容；\n"
    "4) 回答 60-160 字，先说结论再给数字，最后以「（来源：<报告全名> 第 X 页）」结尾；\n"
    "5) 不要出现「这段材料」「根据上文」这类话；\n"
    "6) 严格输出 JSON：{\"question\": \"...\", \"answer\": \"...\"} 或 {\"skip\": true}，"
    "不要任何多余文字。"
)


def fact_score(c: dict) -> float:
    text = c["text"]
    digits = len(re.findall(r"\d", text))
    ratio = digits / max(len(text), 1)
    score = min(digits / 10.0, 20)
    if c["kind"] == "表格":
        has_caption = "【表】" in text and len(re.findall(r"[\u4e00-\u9fff]", text)) > 20
        score += 22 if has_caption else 4
    if ratio > 0.35:                    # 纯数字碎片（表格被切开后只剩数字）
        score -= 25
    sec = c["section"] or ""
    if any(k in sec for k in ("经营", "讨论", "财务", "会计", "重要事项", "股东")):
        score += 12
    if "□适用√不适用" in c["text"] and c["kind"] == "正文":
        score -= 8
    if len(text) < 250:
        score -= 6
    return score


def pick_chunks(chunks: list[dict], per_report: int, unseen: int) -> tuple[list, list]:
    by_report: dict[str, list[dict]] = {}
    for c in chunks:
        by_report.setdefault(c["period"], []).append(c)

    train_pool, unseen_pool = [], []
    for period, items in by_report.items():
        ranked = sorted(items, key=fact_score, reverse=True)
        train_pool.extend(ranked[:per_report])
        leftovers = ranked[per_report:]
        if len(leftovers) > unseen:
            unseen_pool.extend(random.Random(42).sample(leftovers, unseen))
        else:
            unseen_pool.extend(leftovers)
    return train_pool, unseen_pool


def load_cache() -> dict[str, dict]:
    done = {}
    if CACHE.exists():
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["chunk_id"]] = rec
    return done


def gen_one(c: dict) -> dict:
    user = (f"【报告】{c['doc']}（报告期 {c['period']}）\n"
            f"【位置】{c['section'] or '未标注章节'} 第 {c['page']} 页 {c['kind']}\n"
            f"【原文】\n{c['text']}\n\n请生成一组投资者问答。")
    for attempt in range(4):
        try:
            raw = chat(GEN_SYSTEM, user, temperature=0.7).strip()
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
            obj = json.loads(raw)
            if obj.get("skip"):
                return {"chunk_id": c["id"], "skipped": True}
            q, a = obj["question"].strip(), obj["answer"].strip()
            if len(q) < 8 or len(a) < 20:
                raise ValueError("too short")
            return {
                "chunk_id": c["id"], "period": c["period"], "doc": c["doc"],
                "page": c["page"], "section": c["section"], "kind": c["kind"],
                "source_text": c["text"], "question": q, "answer": a,
            }
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                return {"chunk_id": c["id"], "error": f"{type(exc).__name__}: {exc}"}
            time.sleep(1.5 * (attempt + 1))
    return {"chunk_id": c["id"], "error": "unknown"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-report", type=int, default=40)
    ap.add_argument("--unseen", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(DONGMI_DIR / "chunks.jsonl", encoding="utf-8")]
    train_pool, unseen_pool = pick_chunks(chunks, args.per_report, args.unseen)
    todo = train_pool + unseen_pool
    print(f"训练池 {len(train_pool)} 块，对照池 {len(unseen_pool)} 块，共 {len(todo)} 块")

    cache = load_cache()
    print(f"缓存已有 {len(cache)} 条")
    pending = [c for c in todo if c["id"] not in cache]

    lock = threading.Lock()
    done_n = 0
    t0 = time.time()
    with open(CACHE, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(gen_one, c): c for c in pending}
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                cache[rec["chunk_id"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done_n += 1
                if done_n % 25 == 0 or done_n == len(pending):
                    rate = done_n / max(time.time() - t0, 1)
                    print(f"  {done_n}/{len(pending)} 条，{rate*60:.0f} 条/分钟，"
                          f"预计还需 {(len(pending)-done_n)/max(rate,1e-9)/60:.1f} 分钟")

    def usable(rec: dict) -> bool:
        return "error" not in rec and not rec.get("skipped") and bool(rec.get("question"))

    train_out = [cache[c["id"]] for c in train_pool if c["id"] in cache and usable(cache[c["id"]])]
    unseen_out = [cache[c["id"]] for c in unseen_pool if c["id"] in cache and usable(cache[c["id"]])]
    (DONGMI_DIR / "qa_train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_out), "utf-8")
    (DONGMI_DIR / "qa_unseen.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in unseen_out), "utf-8")
    bad = sum(1 for c in todo if c["id"] in cache and "error" in cache[c["id"]])
    skipped = sum(1 for c in todo if c["id"] in cache and cache[c["id"]].get("skipped"))
    print(f"\n训练池 {len(train_out)} 条，对照池 {len(unseen_out)} 条，"
          f"失败 {bad} 条，原文不适合出题跳过 {skipped} 条")


if __name__ == "__main__":
    main()
