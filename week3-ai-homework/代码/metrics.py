"""知识侧预处理：从每份半年报的「主要会计数据」里抽出一张跨公司指标表。

为什么需要它：全景题（「12 家里谁的增速最高」）靠 top-6 文本块永远答不全——
窗口里只装得下 1-2 家公司。课件里的解法就是「知识侧装一层」：
离线把结构化事实抽出来，检索时先判断这是聚合题，再用表作答。

用法：
    python metrics.py            # 生成 data/index/metrics_2026H1.json
"""

from __future__ import annotations

import json
import re

import config

OUT = config.INDEX_DIR / "metrics_2026H1.json"

# 各家报告的行首标签写法不一样，逐个试
LABELS = {
    "revenue": ["营业收入（元）", "营业收入(元)", "营业收入"],
    "net_profit": ["归属于上市公司股东的净利润（元）", "归属于上市公司股东的净利润",
                   "归属于母公司股东的净利润"],
}


TOKEN = re.compile(r"-?[\d,]+\.\d{2}|-?\d+\.\d+")


def pick_after(flat: str, labels: list[str]) -> dict | None:
    """在标签后面顺次取「本期、上期、同比」三个数。

    表格被切开后千分位和竖线还在，所以在标签后 300 个字符里按位置找数字：
    前两个量级 ≥ 1e6 的是本期/上期，上期之后第一个小于 1000 的就是同比百分比。

    第一版用 findall 取「后面的数字」，结果把大数字的小数部分当成了同比
    （90,703,260,964.48 被读成 964.48%），所以这里必须带位置匹配。
    """
    for label in labels:
        idx = flat.find(label)
        if idx < 0:
            continue
        seg = flat[idx:idx + 300]
        toks = [(m.start(), float(m.group().replace(",", "")))
                for m in TOKEN.finditer(seg)]
        big = [t for t in toks if abs(t[1]) >= 1e6]
        if len(big) < 2:
            continue
        cur, prior = big[0][1], big[1][1]
        after = [t[1] for t in toks if t[0] > big[1][0] and abs(t[1]) < 1000]
        yoy = after[0] if after else None
        if yoy is not None:
            return {"current": cur, "prior": prior, "yoy": yoy}
    return None


def parse_doc(text: str) -> dict:
    flat = re.sub(r"[\s|]+", "", text)
    out = {}
    for key, labels in LABELS.items():
        got = pick_after(flat, labels)
        if got:
            out[key] = got
    return out


def main() -> None:
    chunks = [json.loads(l) for l in
              open(config.INDEX_DIR / "chunks.jsonl", encoding="utf-8")]
    rows = {}
    for c in chunks:
        if c["period"] != "2026H1" or "主要会计数据" not in c["text"]:
            continue
        if c["code"] in rows:
            continue
        got = parse_doc(c["text"])
        if "revenue" not in got or "net_profit" not in got:
            continue
        rows[c["code"]] = {
            "code": c["code"], "company": c["company"], "doc": c["doc"],
            "page": c["page"], "chunk_id": c["id"],
            "revenue": got["revenue"]["current"], "revenue_yoy": got["revenue"]["yoy"],
            "net_profit": got["net_profit"]["current"],
            "net_profit_yoy": got["net_profit"]["yoy"],
        }
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    print(f"抽出 {len(rows)} 家公司的指标 → {OUT}")
    print(f"{'公司':<10}{'营业收入(亿)':>14}{'同比':>10}{'归母净利(亿)':>14}{'同比':>10}")
    for r in sorted(rows.values(), key=lambda x: -x["revenue"]):
        print(f"{r['company']:<10}{r['revenue']/1e8:>14.2f}{r['revenue_yoy']:>9.2f}%"
              f"{r['net_profit']/1e8:>14.2f}{r['net_profit_yoy']:>9.2f}%")


def load_table_text() -> str:
    """把表渲染成给大模型看的一小段文本。"""
    rows = json.loads(OUT.read_text("utf-8"))
    lines = ["| 公司 | 营业收入(元) | 营业收入同比(%) | 归母净利润(元) | 归母净利润同比(%) | 出处 |",
             "|---|---|---|---|---|---|"]
    for r in sorted(rows.values(), key=lambda x: -x["revenue"]):
        lines.append(f"| {r['company']} | {r['revenue']:,.2f} | {r['revenue_yoy']} | "
                     f"{r['net_profit']:,.2f} | {r['net_profit_yoy']} | "
                     f"{r['doc']} 第 {r['page']} 页 |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
