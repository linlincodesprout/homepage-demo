"""B 方向第三步：把 440 条问答渲染成结构化训练文本（SFT 用）。

课件说的「输入回答问题符号结构化文本」就是这个：用 Qwen 的对话模板
（<|im_start|> / <|im_end|>）把 system / user / assistant 三段拼起来，
让模型把「投资者问 → 董秘答（带出处）」的格式和口吻训练进权重。

用法：
    python dongmi_format.py
输出：
    data/dongmi/sft_train.jsonl   每行 {"text": "...", "messages": [...]}
    训练时按 9:1 切验证集
"""

from __future__ import annotations

import json
import random

import config

DONGMI_DIR = config.DATA_DIR / "dongmi"

SYSTEM = ("你是贵州茅台酒股份有限公司的董事会秘书。回答投资者提问时只依据公司已披露的定期报告，"
          "数字原样引用并说明口径，回答末尾标注（来源：报告全名 第 X 页）；"
          "报告里没有的内容，直接说明未披露，不要推测。")

PROMPT = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n{answer}<|im_end|>\n"
)


def main() -> None:
    rows = [json.loads(l) for l in open(DONGMI_DIR / "qa_checked.jsonl", encoding="utf-8")]
    rows = [r for r in rows if not r["check"]["drop"]]
    rng = random.Random(3407)          # 和课程里一致的种子习惯
    rng.shuffle(rows)
    n_val = max(20, int(len(rows) * 0.1))

    out = []
    for i, r in enumerate(rows):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["answer"]},
        ]
        out.append({
            "text": PROMPT.format(system=SYSTEM, question=r["question"], answer=r["answer"]),
            "messages": messages,
            "split": "val" if i < n_val else "train",
            "meta": {"period": r["period"], "page": r["page"], "chunk_id": r["chunk_id"]},
        })

    path = DONGMI_DIR / "sft_train.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_train = len(out) - n_val
    print(f"SFT 样本 {len(out)} 条（训练 {n_train} / 验证 {n_val}）→ {path}")
    print("\n模板示例：\n" + out[0]["text"][:400])


if __name__ == "__main__":
    main()
