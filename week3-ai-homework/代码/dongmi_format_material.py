"""给「董秘」补一类它从没学过的训练样本：**带材料答题**。

为什么必须补：现有 404 条训练样本全是「不给材料、凭记忆答」，
所以到"知识库+微调"那格它是在用没训过的姿势答题，只会取错表、不会在材料里找。

造两类样本（都套同一套对话模板，材料块头部带公司/报告/页码/口径）：
  1. 正样本：材料里**含**答案 → 要求照着材料答、标出处（配 3 个干扰块，模拟真实检索结果）；
  2. 负样本：材料里**不含**答案 → 要求老实说"材料未覆盖"（教它别编）。

用法：
    python dongmi_format_material.py
输出：
    data/dongmi/sft_material.jsonl
"""

from __future__ import annotations

import json
import random
import re

import config
from dongmi_format import SYSTEM as SYSTEM_NO_MATERIAL   # 不给材料时的 system

DONGMI = config.DATA_DIR / "dongmi"

SYSTEM_WITH_MATERIAL = (
    "你是贵州茅台酒股份有限公司的董事会秘书。只能依据【材料】回答，不得使用材料之外的知识。\n"
    "材料里经常同时出现合并报表和母公司报表、或同一张表的多个年份：先判断问题问的是哪个口径、"
    "哪个期间，只采用对应那一份的数据；没特别说明时优先用合并口径，并在回答里写明口径。\n"
    "数字要和材料一模一样，不要换算。回答末尾标注（来源：报告全名 第 X 页）。"
    "材料里查不到就写“材料未覆盖”，不要推测。"
)

PROMPT = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n{answer}<|im_end|>\n"
)

CALIBER_RE = {
    "合并报表": re.compile(r"合并(资产负债表|利润表|现金流量表|所有者权益变动表)"),
    "母公司报表": re.compile(r"母公司(资产负债表|利润表|现金流量表|所有者权益变动表)"),
}


def caliber_of(text: str) -> str:
    for label, pat in CALIBER_RE.items():
        if pat.search(text):
            return label
    return ""


def header(c: dict) -> str:
    cal = caliber_of(c["text"])
    return (f"[{c['doc']}｜{c['section'] or '未标注章节'}｜第 {c['page']} 页｜{c['kind']}"
            + (f"｜口径：{cal}" if cal else "") + "]")


def material_text(blocks: list[dict]) -> str:
    return "\n\n".join(f"{header(c)}\n{c['text']}" for c in blocks)


def main() -> None:
    chunks = [json.loads(l) for l in open(DONGMI / "chunks.jsonl", encoding="utf-8")]
    by_id = {c["id"]: c for c in chunks}
    qa = [json.loads(l) for l in open(DONGMI / "qa_checked.jsonl", encoding="utf-8")]
    qa = [r for r in qa if not r["check"]["drop"]]
    print(f"可用问答 {len(qa)} 条")

    rng = random.Random(3407)
    pool_same = [c for c in chunks if c["kind"] == "表格"]

    out = []
    for r in qa:
        src = by_id.get(r["chunk_id"])
        if src is None:
            continue
        # 3 个干扰块：同公司不同页、同章节不同类型、随机表格块（模拟真实检索里那些"很像"的表）
        same_page = [c for c in chunks if c["period"] == src["period"] and c["page"] != src["page"]]
        same_sec = [c for c in chunks if c["section"] == src["section"] and c["id"] != src["id"]]
        cands = []
        for group in (same_page, same_sec, pool_same):
            if group:
                cands.append(rng.choice(group))
        blocks = [src] + [c for c in cands if c["id"] != src["id"]]
        rng.shuffle(blocks)
        user = f"【材料】\n{material_text(blocks)}\n\n【问题】\n{r['question']}"
        out.append({
            "text": PROMPT.format(system=SYSTEM_WITH_MATERIAL, user=user, answer=r["answer"]),
            "messages": [{"role": "system", "content": SYSTEM_WITH_MATERIAL},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": r["answer"]}],
            "kind": "positive", "split": "train",
        })

    # 负样本：材料里放的全是别的话题（不含答案），正确答案是"材料未覆盖"
    n_neg = 60
    picks = rng.sample(qa, min(n_neg, len(qa)))
    for r in picks:
        src = by_id.get(r["chunk_id"])
        others = [c for c in chunks if c["period"] != src["period"] and c["kind"] == src["kind"]]
        blocks = rng.sample(others, min(3, len(others)))
        user = f"【材料】\n{material_text(blocks)}\n\n【问题】\n{r['question']}"
        answer = (f"材料未覆盖。提供的材料里没有关于该问题的数据"
                  f"（材料中只有{'、'.join(sorted({c['period'] for c in blocks}))}等期间的内容），"
                  f"建议查阅对应期间的定期报告后再作答。")
        out.append({
            "text": PROMPT.format(system=SYSTEM_WITH_MATERIAL, user=user, answer=answer),
            "messages": [{"role": "system", "content": SYSTEM_WITH_MATERIAL},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": answer}],
            "kind": "negative", "split": "train",
        })

    rng.shuffle(out)
    n_val = max(40, int(len(out) * 0.1))
    for i, r in enumerate(out):
        r["split"] = "val" if i < n_val else "train"

    path = DONGMI / "sft_material.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(1 for r in out if r["kind"] == "positive")
    print(f"带材料样本 {len(out)} 条（正 {pos} / 负 {len(out)-pos}；"
          f"训练 {len(out)-n_val} / 验证 {n_val}）→ {path}")


if __name__ == "__main__":
    main()
