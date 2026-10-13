"""作业 A 的 10 道题评测：逐题记录召回了哪些块、答对没有、错在哪。

课件要求：「出 10 道题（至少 2 道跨公司的全景题），逐题记录召回了哪些块、答对没有、错在哪。」

用法：
    python eval_a.py --collect     # 只打印每题各路线召回的前 6 块，人工写参考答案用
    python eval_a.py               # 跑完整评测，出 outputs/A_评测记录.md
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time

import config
from rag import answer
from retrieval import brief, get_retriever

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

QUESTIONS_FILE = config.OUT_DIR / "A_问卷.json"
DEFAULT_QUESTIONS = [
    {"id": "Q1", "type": "单公司",
     "q": "贵州茅台2026年上半年通过直销渠道实现的收入是多少，占营业收入的比重是多少？"},
    {"id": "Q2", "type": "单公司",
     "q": "五粮液2026年上半年营业收入和归属于上市公司股东的净利润分别是多少，同比增速如何？"},
    {"id": "Q3", "type": "单公司",
     "q": "泸州老窖2026年上半年营业收入和归属于上市公司股东的净利润是多少？"},
    {"id": "Q4", "type": "单公司",
     "q": "山西汾酒2026年上半年营业收入和归属于上市公司股东的净利润是多少，同比变化如何？"},
    {"id": "Q5", "type": "单公司",
     "q": "洋河股份2026年上半年营业收入和归母净利润同比是增长还是下降，幅度多少？"},
    {"id": "Q6", "type": "单公司",
     "q": "古井贡酒2026年上半年营业收入和归属于上市公司股东的净利润是多少？"},
    {"id": "Q7", "type": "单公司",
     "q": "今世缘2026年上半年营业收入和归属于上市公司股东的净利润是多少？"},
    {"id": "Q8", "type": "单公司",
     "q": "水井坊2026年上半年营业收入和归属于上市公司股东的净利润是多少？"},
    {"id": "Q9", "type": "全景题",
     "q": "2026年上半年，这12家白酒公司里营业收入同比增速最高和最低的分别是哪一家，各是多少？"},
    {"id": "Q10", "type": "全景题",
     "q": "2026年上半年，这12家白酒公司里哪些公司的归母净利润同比是下滑的？分别下滑多少？"},
]


def load_questions() -> list[dict]:
    if QUESTIONS_FILE.exists():
        return json.loads(QUESTIONS_FILE.read_text("utf-8"))
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    QUESTIONS_FILE.write_text(json.dumps(DEFAULT_QUESTIONS, ensure_ascii=False, indent=2), "utf-8")
    return DEFAULT_QUESTIONS


def collect(questions: list[dict]) -> None:
    """打印每条路线召回的内容，方便人工核对参考答案。"""
    for vec in ("qwen3", "ali"):
        r = get_retriever(vec)
        for item in questions:
            print("\n" + "=" * 90)
            print(f"{item['id']} [{item['type']}] {item['q']}   —— 向量路 = {vec}")
            for mode in ("vector", "hybrid"):
                hits = r.search(item["q"], mode=mode)
                print(f"\n---- {mode} ----")
                for row in brief(hits)[:6]:
                    print(f"[{row['n']}] {row['id']} {row['company']} p{row['page']} "
                          f"{row['kind']} v={row['vector_rank']} b={row['bm25_rank']}")
                    print(f"    {row['head']}")


def run(questions: list[dict]) -> None:
    reports = []
    for item in questions:
        rec = {"id": item["id"], "type": item["type"], "question": item["q"],
               "reference": item.get("reference", ""), "routes": {}}
        primary = None
        for vec in ("qwen3", "ali"):
            r = get_retriever(vec)
            routes = {}
            for mode in ("vector", "bm25", "hybrid"):
                t0 = time.time()
                hits = r.search(item["q"], mode=mode)
                routes[mode] = {"seconds": round(time.time() - t0, 2), "hits": brief(hits)}
            rec["routes"][vec] = routes
            primary = r.search(item["q"], mode="hybrid")     # 用 qwen3 的融合结果作答
        t0 = time.time()
        ans = answer(item["q"], primary)
        rec["answer_seconds"] = round(time.time() - t0, 2)
        rec["context_hits"] = ans["hits"]
        rec["answer"] = ans["answer"]
        rec["context"] = ans["context"]
        reports.append(rec)
        print(f"{item['id']} 完成：召回首条 {ans['hits'][0]['id']}，"
              f"检索 {rec['routes']['qwen3']['hybrid']['seconds']}s，"
              f"回答 {rec['answer_seconds']}s")

    score(reports)
    write_summary(reports)

    (config.OUT_DIR / "A_评测结果.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), "utf-8")

    lines = ["# 作业 A · 10 道题评测记录", "",
             "三条检索路线（Qwen3 向量 / 阿里向量 / jieba BM25）各取 30 块候选，"
             "融合（RRF）取前 6 块交给 DeepSeek 作答。", ""]
    for rec in reports:
        lines += [f"## {rec['id']}（{rec['type']}）", "",
                  f"**问题**：{rec['question']}", ""]
        if rec["reference"]:
            lines += [f"**参考答案**：{rec['reference']}", ""]
        for vec in ("qwen3", "ali"):
            for mode in ("vector", "bm25", "hybrid"):
                hits = rec["routes"][vec][mode]["hits"]
                ids = "、".join(f"{h['id']}({h['company']}p{h['page']})" for h in hits[:6])
                lines.append(f"- {vec} · {mode}：{ids}")
            lines.append("")
        lines += ["**回答**：", "", rec["answer"], "",
                  "**给模型看的 6 块**：", ""]
        for h in rec["context_hits"]:
            lines.append(f"- [{h['n']}] {h['id']} {h['company']} p{h['page']} "
                         f"{h['section']} · {h['kind']}")
        lines.append("")
    (config.OUT_DIR / "A_评测记录.md").write_text("\n".join(lines), "utf-8")
    print(f"\n报告 → {config.OUT_DIR / 'A_评测记录.md'}")


def numbers_in(text: str) -> list[float]:
    """答案里出现的所有数字（去掉千分位），用于和题目要求的关键数字比对。"""
    out = []
    for raw in re.findall(r"-?\d[\d,]*\.?\d*", text):
        s = raw.replace(",", "").rstrip(".")
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def matched(v: float, ans_nums: list[float]) -> bool:
    """容差比对：大数字要精确，百分比允许 1% 相对误差。

    这样「-105.9」能对上「-105.90」，「89.31」也能算命中「89.30」（材料给的 89.30 带注）。
    """
    tol = max(abs(v) * 0.01, 0.02)
    return any(abs(a - v) <= tol for a in ans_nums)


def score(reports: list[dict]) -> None:
    """逐题判对错：问题要求的关键数字都答到 = 对。

    判分只看「这道题该答的数字」，不拿参考答案里所有数字去扣——
    第一版把「同比」也算进去，结果 Q3、Q6（问题根本没问同比）被误判为错。
    """
    n_ok = 0
    for rec in reports:
        keys = rec.get("key") or []
        ans = rec["answer"]
        ans_nums = numbers_in(ans)
        missing = [k for k in keys if not matched(k, ans_nums)]
        hit = len(keys) - len(missing)
        rec["score"] = {
            "ref_numbers": len(keys),
            "hit_numbers": hit,
            "hit_rate": round(hit / len(keys), 3) if keys else 0.0,
            "missing": missing,
            "said_not_covered": "未覆盖" in ans,
            "correct": bool(keys) and not missing,
        }
        n_ok += int(rec["score"]["correct"])
    print(f"自动判分：{n_ok}/{len(reports)} 题要求的关键数字全部答到")


def write_summary(reports: list[dict]) -> None:
    lines = ["# 作业 A · 10 道题判分表", "",
             "判分口径：只看**这道题该答的数字**是否答到（大数字精确、百分比允许 1% 误差）。",
             "「覆盖」那一列是命中率，全景题用它看答全了几家公司。", "",
             "| 题号 | 类型 | 关键数字命中 | 覆盖率 | 答里出现「未覆盖」 | 判定 |",
             "|---|---|---|---|---|---|"]
    for rec in reports:
        s = rec["score"]
        hit = f"{s['hit_numbers']}/{s['ref_numbers']}"
        if s["missing"]:
            hit += "（缺 " + "、".join(_fmt(m) for m in s["missing"][:4]) + "）"
        lines.append(f"| {rec['id']} | {rec['type']} | {hit} | {s['hit_rate']*100:.0f}% | "
                     f"{'是' if s['said_not_covered'] else '否'} | "
                     f"{'✅ 对' if s['correct'] else '❌ 不对'} |")
    (config.OUT_DIR / "A_判分表.md").write_text("\n".join(lines), "utf-8")


def _fmt(v: float) -> str:
    return f"{v:,.2f}".rstrip("0").rstrip(".") if abs(v) >= 1 else f"{v}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    qs = load_questions()
    if args.collect:
        collect(qs)
    else:
        run(qs)
