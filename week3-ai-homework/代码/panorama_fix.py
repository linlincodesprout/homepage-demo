"""全景题补救实验：为什么两道跨公司题会错，以及要花多大代价才能对。

三种做法：
  A 基线：一次检索取前 6 块（评测里的做法）——只够覆盖 1-2 家公司；
  B 加宽：一次检索取前 60 块——块多了，但窗口里仍以某几家公司为主；
  C 逐家检索：对 12 家公司各检索一次、每家取 2 块，再合并（24 块）。

每题分别交给 DeepSeek 作答，然后数「12 家公司的数字答到了几个」。
"""

from __future__ import annotations

import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import config
from metrics import load_table_text
from rag import answer
from retrieval import COMPANY_ALIASES, get_retriever

QUESTIONS = [
    {"id": "Q9", "q": "2026年上半年，这12家白酒公司里营业收入同比增速最高和最低的分别是哪一家，各是多少？",
     "expect": {"五粮液": 20.87, "迎驾贡酒": 8.08, "贵州茅台": 1.47, "今世缘": -7.41,
                "山西汾酒": -12.18, "老白干酒": -15.23, "舍得酒业": -15.34,
                "口子窖": -22.89, "古井贡酒": -27.01, "水井坊": -27.78,
                "洋河股份": -28.76, "泸州老窖": -36.35}},
    {"id": "Q10", "q": "2026年上半年，这12家白酒公司里哪些公司的归母净利润同比是下滑的？分别下滑多少？",
     "expect": {"水井坊": -105.90, "舍得酒业": -67.26, "口子窖": -48.97, "泸州老窖": -43.37,
                "古井贡酒": -40.89, "洋河股份": -40.10, "山西汾酒": -24.29,
                "老白干酒": -18.47, "今世缘": -6.60, "贵州茅台": -1.95,
                "五粮液": 89.30, "迎驾贡酒": 2.79}},
]


def coverage(ans: str, expect: dict) -> tuple[int, list[str]]:
    """答案里覆盖到几家公司的数字（公司名和数值要同时出现才算覆盖）。"""
    missed = []
    hit = 0
    for company, value in expect.items():
        aliases = COMPANY_ALIASES[company]
        txt = ans.replace(",", "")
        near = any(a in txt for a in aliases)
        val = any(abs(abs(v) - abs(value)) <= max(abs(value) * 0.01, 0.02)
                  for v in _nums(txt))
        if near and val:
            hit += 1
        else:
            missed.append(company)
    return hit, missed


def _nums(text: str) -> list[float]:
    import re
    out = []
    for raw in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            out.append(float(raw.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def main() -> None:
    retriever = get_retriever("qwen3")
    companies = list(COMPANY_ALIASES)
    report = {}
    for item in QUESTIONS:
        print("\n" + "=" * 80)
        print(item["id"], item["q"])
        trials = {}

        # A 基线
        hits = retriever.search(item["q"], k=6)
        ans = answer(item["q"], hits)["answer"]
        trials["A 一次检索 top6"] = (hits, ans)

        # B 加宽
        hits = retriever.search(item["q"], k=60)
        ans = answer(item["q"], hits)["answer"]
        trials["B 一次检索 top60"] = (hits, ans)

        # C 逐家检索
        picked, seen = [], set()
        for company in companies:
            for h in retriever.search(item["q"], k=2, company=company, period="2026H1",
                                      auto_filter=False):
                if h["chunk"]["id"] not in seen:
                    seen.add(h["chunk"]["id"])
                    picked.append(h)
        picked.sort(key=lambda h: -h["rrf"])
        ans = answer(item["q"], picked)["answer"]
        trials["C 逐家检索 24 块"] = (picked, ans)

        # D 知识侧预处理：离线抽好的跨公司指标表 + top6 原文块
        table = load_table_text()
        base = retriever.search(item["q"], k=6)
        from rag import build_context, chat, ANSWER_SYSTEM
        ctx = (f"【跨公司指标表（离线从各家公司半年报『主要会计数据』抽取）】\n{table}\n\n"
               f"【相关原文块】\n{build_context(base)}")
        ans = chat(ANSWER_SYSTEM,
                   f"{ctx}\n\n【问题】\n{item['q']}\n\n请先把12家公司逐条列全（公司+数字），"
                   f"最后再给出结论。逐句带角标。")
        trials["D 指标表+top6"] = (base, ans)

        report[item["id"]] = {}
        for name, (hs, ans) in trials.items():
            hit, missed = coverage(ans, item["expect"])
            report[item["id"]][name] = {"hit": hit, "missed": missed,
                                        "blocks": len(hs), "answer": ans}
            print(f"  {name}: 覆盖 {hit}/12 家，未覆盖 {missed}")

    (config.OUT_DIR / "A_全景题补救实验.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    lines = ["# 作业 A · 全景题补救实验", "",
             "题目一次问 12 家公司，向量检索一次取前 6 块时窗口里只装得下 1-2 家。", "",
             "| 题号 | 做法 | 喂进去的块数 | 覆盖到的公司数 | 没覆盖到的 |",
             "|---|---|---|---|---|"]
    for qid, trials in report.items():
        for name, r in trials.items():
            lines.append(f"| {qid} | {name} | {r['blocks']} | {r['hit']}/12 | "
                         f"{'、'.join(r['missed']) or '全部覆盖'} |")
    (config.OUT_DIR / "A_全景题补救实验.md").write_text("\n".join(lines), "utf-8")
    print(f"\n实验记录 → {config.OUT_DIR / 'A_全景题补救实验.md'}")


if __name__ == "__main__":
    main()
