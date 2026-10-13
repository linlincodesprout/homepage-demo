"""B 方向：把四格评测结果算成分数表。

判分口径（和 A 用同一套）：
  · 参考答案里的关键数字（≥3 位、排除年份）都出现在回答里 → 这道题算对；
  · 另外统计「有没有标出处」——课件里董秘答必须带（来源：报告名 第 X 页）。

用法：
    python dongmi_score.py            # 读 outputs/B_四格结果.jsonl
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import config

RESULT = config.OUT_DIR / "B_四格结果.jsonl"            # 全部四格（微调那两格是 3 轮）
RESULT6 = config.OUT_DIR / "B_四格结果_6轮.jsonl"        # 微调那两格用 6 轮，作为正式结果
CONFIGS = [("untrained", "没微调"), ("finetuned", "微调过"),
           ("kb_untrained", "知识库+没微调"), ("kb_finetuned", "知识库+微调")]


def numbers(text: str) -> list[float]:
    out = []
    for raw in re.findall(r"-?\d[\d,]*\.?\d*", text):
        s = raw.replace(",", "").rstrip(".")
        digits = s.replace(".", "").replace("-", "")
        if len(digits) < 3:
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        if len(digits) == 4 and 1900 <= v <= 2099:
            continue
        out.append(v)
    return out


def hit_ratio(ref: str, out: str) -> tuple[float, int, int]:
    need = numbers(ref)
    if not need:
        return 1.0, 0, 0
    have = numbers(out)
    ok = 0
    for n in need:
        tol = max(abs(n) * 0.01, 0.02)
        if any(abs(h - n) <= tol for h in have):
            ok += 1
    return ok / len(need), ok, len(need)


def has_citation(text: str) -> bool:
    return bool(re.search(r"(来源[:：]|第\s*\d+\s*页)", text))


def main() -> None:
    rows = [json.loads(l) for l in open(RESULT, encoding="utf-8")]
    rows3 = [r for r in rows if r["config"] in ("finetuned", "kb_finetuned")]
    if RESULT6.exists():
        rows = [r for r in rows if r["config"] not in ("finetuned", "kb_finetuned")]
        rows += [json.loads(l) for l in open(RESULT6, encoding="utf-8")]
    for r in rows:
        ratio, ok, need = hit_ratio(r["reference"], r["output"])
        r["score"] = {"hit_rate": round(ratio, 3), "hit": ok, "need": need,
                      "correct": ratio >= 0.999, "citation": has_citation(r["output"])}

    agg: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        agg[(r["config"], r["pool"], r["variant"])].append(r)

    lines = ["# 作业 B · 四格对照结果", "",
             "判分：参考答案的关键数字（≥3 位、排除年份）全部答到算对；容差 1%。", "",
             "微调那两格用的是 **6 轮** 的 LoRA 权重（见下面的轮数对比）。", "",
             "| 配置 | 题池 | 问法 | 题数 | 完全答对 | 平均数字命中率 | 带出处 |",
             "|---|---|---|---|---|---|---|"]
    order = {c: i for i, (c, _) in enumerate(CONFIGS)}
    for key in sorted(agg, key=lambda k: (order.get(k[0], 9), k[1], k[2])):
        items = agg[key]
        n = len(items)
        correct = sum(1 for x in items if x["score"]["correct"])
        avg = sum(x["score"]["hit_rate"] for x in items) / n
        cite = sum(1 for x in items if x["score"]["citation"])
        cfg_label = dict(CONFIGS).get(key[0], key[0])
        pool_label = {"seen": "训练过的事实", "unseen": "从未训练的事实"}[key[1]]
        var_label = {"original": "原问法", "paraphrase": "换问法"}[key[2]]
        lines.append(f"| {cfg_label} | {pool_label} | {var_label} | {n} | "
                     f"{correct}/{n}（{correct/n*100:.0f}%） | {avg*100:.0f}% | {cite}/{n} |")

    (config.OUT_DIR / "B_四格对照.md").write_text("\n".join(lines), "utf-8")

    if RESULT6.exists():
        # 3 轮 vs 6 轮：只比微调那两格
        extra = ["", "## 训练轮数对比（同一份数据、同一个 r=16）", "",
                 "| 轮数 | 题池 | 问法 | 完全答对 | 平均数字命中率 | 带出处 |",
                 "|---|---|---|---|---|---|"]
        for label, data in (("3 轮", rows3),
                            ("6 轮", [json.loads(l) for l in open(RESULT6, encoding="utf-8")])):
            for r in data:
                ratio, ok, need = hit_ratio(r["reference"], r["output"])
                r["_s"] = {"hit_rate": ratio, "correct": ratio >= 0.999,
                           "citation": has_citation(r["output"])}
            bucket = defaultdict(list)
            for r in data:
                bucket[(r["config"], r["pool"], r["variant"])].append(r)
            for key in sorted(bucket, key=lambda k: (order.get(k[0], 9), k[1], k[2])):
                items = bucket[key]
                n = len(items)
                correct = sum(1 for x in items if x["_s"]["correct"])
                avg = sum(x["_s"]["hit_rate"] for x in items) / n
                cite = sum(1 for x in items if x["_s"]["citation"])
                cfg_label = dict(CONFIGS).get(key[0], key[0])
                pool_label = {"seen": "训练过", "unseen": "从未训练"}[key[1]]
                var_label = {"original": "原问法", "paraphrase": "换问法"}[key[2]]
                extra.append(f"| {label} | {cfg_label}·{pool_label} | {var_label} | "
                             f"{correct}/{n}（{correct/n*100:.0f}%） | {avg*100:.0f}% | {cite}/{n} |")
        with open(config.OUT_DIR / "B_四格对照.md", "a", encoding="utf-8") as f:
            f.write("\n".join(extra) + "\n")
    with open(RESULT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(open(config.OUT_DIR / "B_四格对照.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
