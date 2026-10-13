"""B 方向第二步：抽检 + 切分训练/评测集。

本作业要求「人工抽检删掉编造的」。这里做两层：
  第一层（机器）：把答案里的每个数字回到原文里核对，对不上的标成可疑；
                 同时检查出处是否写成「（来源：报告全名 第 X 页）」并且和块元数据一致。
  第二层（人工）：导出一张抽检表 outputs/dongmi_抽检表.md，随机抽 40 条并排显示
                 问题 / 答案 / 原文片段，人眼过一遍，把不实的行标掉。

用法：
    python dongmi_check.py            # 生成 qa_checked.jsonl 和抽检表
    python dongmi_check.py --apply    # 读回抽检结果（qa_checked.jsonl 里 drop=true 的剔除）
"""

from __future__ import annotations

import argparse
import json
import random
import re

import config
from rag import chat

DONGMI_DIR = config.DATA_DIR / "dongmi"
OUT = config.OUT_DIR


def numbers(text: str) -> list[str]:
    """抓出「值得核对」的数字：去掉千分位，跳过年份和小整数。

    第一版把所有数字都拿去核，结果年份「2026」和页码全被判成"原文里没有"，
    440 条里误报 378 条。财报核对只关心金额/比例这类实体数字。
    """
    out = []
    for raw in re.findall(r"\d[\d,]*\.?\d*", text):
        s = raw.replace(",", "").rstrip(".")
        digits = s.replace(".", "")
        if len(digits) <= 2:                       # 1、12 这种小数字不核
            continue
        if len(digits) == 4 and 1900 <= int(float(s)) <= 2099:
            continue                                # 年份不核
        out.append(s)
    return out


def found_in(n: str, src: str, page: int) -> bool:
    """数字是否出现在原文里；顺带容忍单位换算（元↔万元↔亿元）。"""
    if n in src:
        return True
    try:
        v = float(n)
    except ValueError:
        return False
    if abs(v - page) < 1e-9:
        return True
    for scale in (1e4, 1e8, 1e-4, 1e-8):
        for fmt in (f"{v*scale:.2f}", f"{v*scale:.4f}"):
            s = fmt.rstrip("0").rstrip(".")
            if len(s.replace(".", "")) >= 3 and s in src:
                return True
    return False


def check_one(rec: dict) -> dict:
    src = rec["source_text"].replace(",", "")
    ans_nums = numbers(rec["answer"])
    missing = [n for n in ans_nums if not found_in(n, src, rec["page"])]
    cite_ok = bool(re.search(r"（来源：.+?第\s*\d+\s*页）", rec["answer"]))
    page_ok = f"第 {rec['page']} 页" in rec["answer"] or f"第{rec['page']}页" in rec["answer"]
    flags = []
    if missing:
        flags.append("数字未在原文找到:" + ",".join(missing[:5]))
    if not cite_ok:
        flags.append("缺出处")
    if not page_ok:
        flags.append("页码与原文块不符")
    rec["check"] = {"missing_numbers": missing, "cite_ok": cite_ok,
                    "page_ok": page_ok, "flags": flags, "drop": bool(flags)}
    return rec


def make_review_sheet(records: list[dict], n: int = 40) -> None:
    rng = random.Random(7)
    sample = rng.sample(records, min(n, len(records)))
    lines = ["# 董秘问答抽检表（B 方向）", "",
             "规则：答案里的每个数字都必须能在原文里找到；出处页码必须和原文块一致。",
             "人工过一遍，把不实的行在「结论」一栏标 `删`。", "",
             "| # | 报告 | 页 | 投资者问 | 董秘答 | 原文片段 | 机器检查 | 结论 |",
             "|---|---|---|---|---|---|---|---|"]
    for i, rec in enumerate(sample, start=1):
        snippet = rec["source_text"][:160].replace("\n", " ").replace("|", "/")
        q = rec["question"].replace("|", "/")
        a = rec["answer"].replace("|", "/")
        chk = "OK" if not rec["check"]["flags"] else "；".join(rec["check"]["flags"])
        lines.append(f"| {i} | {rec['period']} | {rec['page']} | {q} | {a} | {snippet} | {chk} |  |")
    (OUT / "dongmi_抽检表.md").write_text("\n".join(lines), "utf-8")


PARA_CACHE = DONGMI_DIR / "paraphrase_cache.json"


def paraphrase(question: str) -> str:
    """换问法：同一件事换一种问法，用来做四格对照的「换问法」那一格。带缓存。"""
    cache = json.loads(PARA_CACHE.read_text("utf-8")) if PARA_CACHE.exists() else {}
    if question in cache:
        return cache[question]
    new = chat(
        "把下面这个投资者问题换一种说法，意思完全不变，但用词和句式要明显不同。"
        "只输出改写后的小结。",
        question, temperature=0.9).strip().splitlines()[0].strip()
    cache[question] = new
    PARA_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")
    return new


def build_eval_sets(train: list[dict], unseen: list[dict],
                    n_seen: int = 60, n_unseen: int = 20) -> None:
    rng = random.Random(11)
    seen = rng.sample(train, min(n_seen, len(train)))
    unseen = rng.sample(unseen, min(n_unseen, len(unseen)))
    eval_records = []
    for pool, tag in ((seen, "seen"), (unseen, "unseen")):
        for rec in pool:
            eval_records.append({**rec, "pool": tag, "variant": "original",
                                 "eval_question": rec["question"]})
            eval_records.append({**rec, "pool": tag, "variant": "paraphrase",
                                 "eval_question": paraphrase(rec["question"])})
    path = DONGMI_DIR / "eval_set.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in eval_records), "utf-8")
    print(f"评测集：{len(eval_records)} 条（{len(seen)} 条训练内 + {len(unseen)} 条未见，"
          f"各含原问法和换问法）→ {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    train = [json.loads(l) for l in open(DONGMI_DIR / "qa_train.jsonl", encoding="utf-8")]
    unseen = [json.loads(l) for l in open(DONGMI_DIR / "qa_unseen.jsonl", encoding="utf-8")]
    print(f"训练池 {len(train)} 条，对照池 {len(unseen)} 条")

    checked = [check_one(r) for r in train]
    n_bad = sum(1 for r in checked if r["check"]["drop"])
    print(f"机器抽检：{n_bad} 条有问题（数字对不上/缺出处/页码不符）")
    (DONGMI_DIR / "qa_checked.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in checked), "utf-8")

    unseen = [check_one(r) for r in unseen]
    (DONGMI_DIR / "qa_unseen_checked.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in unseen), "utf-8")

    make_review_sheet(checked)
    print(f"抽检表 → {OUT / 'dongmi_抽检表.md'}")

    clean = [r for r in checked if not r["check"]["drop"]]
    clean_unseen = [r for r in unseen if not r["check"]["drop"]]
    print(f"抽检后可用：训练 {len(clean)} 条，对照 {len(clean_unseen)} 条")
    build_eval_sets(clean, clean_unseen)


if __name__ == "__main__":
    main()
