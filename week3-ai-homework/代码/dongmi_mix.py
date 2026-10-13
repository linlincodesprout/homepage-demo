"""把两类训练样本合成一份：不带材料的 404 条 + 带材料的 464 条。

系统提示词不同（一个说"凭定期报告作答"、一个说"只能依据材料"），
模型靠这个信号分辨"这轮有没有材料"，所以要一起训，两个能力都在。
"""

from __future__ import annotations

import json
import random

import config

DONGMI = config.DATA_DIR / "dongmi"


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in open(DONGMI / name, encoding="utf-8")]


def main() -> None:
    base = load("sft_train.jsonl")          # 404 条，不带材料
    material = load("sft_material.jsonl")   # 464 条，带材料
    for r in material:
        r["source"] = "material"
    for r in base:
        r["source"] = "memory"

    rows = base + material
    random.Random(2026).shuffle(rows)
    # 验证集从两类里各抽一点，保证两类都被监控
    val_n = 60
    for i, r in enumerate(rows):
        r["split"] = "val" if i < val_n else "train"

    path = DONGMI / "sft_combined.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_train = len(rows) - val_n
    kinds = {}
    for r in rows:
        kinds[r["source"]] = kinds.get(r["source"], 0) + 1
    print(f"合并 {len(rows)} 条（{kinds}），训练 {n_train} / 验证 {val_n} → {path}")


if __name__ == "__main__":
    main()
