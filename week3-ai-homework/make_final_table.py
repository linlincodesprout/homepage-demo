"""生成最终版四格对照表（用目前最好的一套配置）。"""

from __future__ import annotations

import json
from collections import defaultdict

import config
from dongmi_score import has_citation, hit_ratio

NAME = {"untrained": "没微调", "finetuned": "微调过",
        "kb_untrained": "知识库+没微调", "kb_finetuned": "知识库+微调过"}
POOL = {"seen": "训练过", "unseen": "从未训练"}
VAR = {"original": "原问法", "paraphrase": "换问法"}


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in open(config.OUT_DIR / name, encoding="utf-8")]


def main() -> None:
    rows = []
    rows += [r for r in load("B_四格结果.jsonl") if r["config"] == "untrained"]
    rows += [r for r in load("B_四格结果_6轮.jsonl") if r["config"] == "finetuned"]
    rows += load("B_四格结果_重排版.jsonl")          # kb_untrained + kb_finetuned（重排 4 块）

    bucket: dict[tuple, list] = defaultdict(list)
    for r in rows:
        bucket[(r["config"], r["pool"], r["variant"])].append(r)

    lines = ["# 作业 B · 四格对照表（最终版）", "",
             "配置：404 条 × 6 轮的 LoRA 权重 + 「召回 30 块 → DeepSeek 挑 4 块」的重排检索。",
             "判分：参考答案的关键数字（≥3 位、排除年份）全部答到算对，容差 1%。", "",
             "| 配置 | 题池 | 问法 | 题数 | 答对 | 平均数字命中率 | 带出处 |",
             "|---|---|---|---|---|---|---|"]
    for cfg in ("untrained", "finetuned", "kb_untrained", "kb_finetuned"):
        for pool in ("seen", "unseen"):
            for var in ("original", "paraphrase"):
                items = bucket.get((cfg, pool, var), [])
                n = len(items)
                ok = sum(1 for r in items if hit_ratio(r["reference"], r["output"])[0] >= 0.999)
                avg = sum(hit_ratio(r["reference"], r["output"])[0] for r in items) / n
                cite = sum(1 for r in items if has_citation(r["output"]))
                lines.append(f"| {NAME[cfg]} | {POOL[pool]} | {VAR[var]} | {n} | "
                             f"{ok}（{ok/n*100:.0f}%） | {avg*100:.0f}% | {cite}/{n} |")
    text = "\n".join(lines)
    (config.OUT_DIR / "B_四格对照.md").write_text(text, "utf-8")
    print(text)


if __name__ == "__main__":
    main()
