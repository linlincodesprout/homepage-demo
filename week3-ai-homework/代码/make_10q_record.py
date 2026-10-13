"""把 10 道题的评测记录并成一份可以直接交的文档。

每题包含：题目/类型/参考答案 → 三条检索路线各召回什么 → 给模型看的 6 块
→ 模型回答 → 判定（命中多少关键数字）→ 错在哪 / 对在哪。

用法：
    python make_10q_record.py
输出：
    outputs/A_10题逐题记录.md
"""

from __future__ import annotations

import json

import config

OUT = config.OUT_DIR

# 逐题错因/点评（人工写，对着实际回答复盘）
NOTES = {
    "Q1": ("对", 
           "两条正确数字（直销 52,006,683,587.88 元、占营业收入 57.34%）都答到了，"
           "而且主动说明「材料未直接披露该比例，是按两个数相除得到」——这是我们要的严谨。"
           "唯一的小干扰：它同时把母公司口径的 44,955,323.85 元也列了出来，"
           "但明确标了「另一口径」，没有混用，判它对。"),
    "Q2": ("对",
           "营收 284.17 亿（+20.87%）、归母净利 87.53 亿都答到。"
           "小瑕疵：它说「归母净利润同比增速材料未覆盖」——实际上给的块里没有 89.30% 那一行，"
           "它没有编，而是如实说缺，这个行为本身是加分项。"),
    "Q3": ("对", "召回首条就是主要会计数据页，两个数字一次命中，8 秒答完。"),
    "Q4": ("对", "营收 210.44 亿（-12.18%）、归母净利 64.39 亿（-24.29%）全对。"),
    "Q5": ("对", "营收 -28.76%、归母净利 -40.10% 全对，且写明了是「下降」。"),
    "Q6": ("对",
           "营收 101.31 亿、归母净利 21.64 亿全对；还额外指出材料里有母公司口径的 64.71 亿，"
           "并标明「口径为母公司」，没有拿来当答案。"),
    "Q7": ("对", "营收 64.35 亿、归母净利 20.82 亿全对。"),
    "Q8": ("对",
           "水井坊净利润是 -622 万（亏损），负号处理正确，同比 -105.9% 也答到。"),
    "Q9": ("错",
           "**向量 RAG 的错因不是模型，是检索的天花板**：窗口只有 6 块，无法容纳 12 家公司公司的数字，"
           "所以只覆盖 2/12 家（五粮液、泸州老窖），其余 10 家写「材料未覆盖」。\n"
           "   - 换重排（召回 30 挑 4）仍是 2/12；把窗口开到 60 块是 3/12；挨家检索也是 3/12。\n"
           "   - **深度三的补救**：GraphRAG 顺着「公司→指标=数值（期间/口径）」的边取数 → "
           "**12/12（18 秒）**；离线跨公司指标表 → **12/12（5 秒）**。\n"
           "   - 注意：图谱路线第一版只有 9/12，补全指标边之后才满分——"
           "详见 `A_深度三_图谱全景题.md` 第三·补节。"),
    "Q10": ("错",
            "同上：窗口无法容纳 12 家公司，只覆盖 2/12。\n"
            "   - **深度三的补救**：GraphRAG 指标聚合 **12/12（27 秒）**（补全指标边后）、"
            "离线指标表 **12/12（5 秒）**。\n"
            "   - 同时说清一个细节：这道题正确答案是 12 家里 10 家下滑，"
            "只有五粮液（+89.30%）和迎驾贡酒（+2.79%）增长——"
            "这种「谁是例外」的信息，碎块检索尤其容易漏。"),
}


def hits_md(rows: list[dict], limit: int = 6) -> str:
    out = []
    for r in rows[:limit]:
        rrf = f"{r['rrf']:.4f}" if r.get("rrf") is not None else "-"
        v = r.get("vector_rank") or "-"
        b = r.get("bm25_rank") or "-"
        out.append(f"`{r['id']}` {r['company']} p{r['page']} {r['kind']}"
                   f"（向量第{v} / 关键词第{b} / 融合分{rrf}）")
    return "、".join(out)


def main() -> None:
    results = json.loads((OUT / "A_评测结果.json").read_text("utf-8"))
    banks = {q["id"]: q for q in json.loads((OUT / "A_问卷.json").read_text("utf-8"))}
    # 切块时章节识别漏了（标题被排版拆成两行），这里用重建的 page→section 映射补显示
    smap = json.loads((config.INDEX_DIR / "section_map.json").read_text("utf-8"))

    def section_of(hit: dict) -> str:
        if hit.get("section"):
            return hit["section"]
        code = hit.get("code") or str(hit.get("id", "")).split("-")[0]
        for key in (code, hit.get("period", "")):
            sec = smap.get(key, {}).get(str(hit.get("page")), "")
            if sec:
                return sec
        return "未标注章节"

    lines = [
        "# 作业 A · 10 道题逐题记录",
        "",
        "语料：12 家白酒公司 2026 年半年报（另并入贵州茅台 2021–2025 年报），"
        "切成 11,047 块，建了 Qwen3-Embedding-0.6B 与阿里 text-embedding-v4 两套向量 + jieba BM25。",
        "",
        "每题跑三条检索路线（各取 30 块候选），RRF 融合后取前 6 块交给大模型作答；"
        "判分只看**这道题该答的数字**有没有答到（大数字精确、百分比允许 1% 误差）。",
        "",
        "## 总览",
        "",
        "| 题号 | 类型 | 关键数字命中 | 判定 | 错在哪（小结） |",
        "|---|---|---|---|---|",
    ]
    for rec in results:
        s = rec["score"]
        grade, _ = NOTES.get(rec["id"], ("", ""))
        brief = {"Q9": "窗口只装 6 块，12 家无法容纳",
                 "Q10": "同上；「谁是例外」尤其容易漏"}.get(rec["id"], "无")
        if s["correct"]:
            brief = "—"
        hit = f"{s['hit_numbers']}/{s['ref_numbers']}"
        lines.append(f"| {rec['id']} | {rec['type']} | {hit} | "
                     f"{'✅ 对' if s['correct'] else '❌ 错'} | {brief} |")
    n_ok = sum(1 for r in results if r["score"]["correct"])
    lines += ["", f"**8 道单公司题全部答对；2 道跨公司全景题答不出（原因见每题）。**", ""]
    lines += ["", "---", ""]

    for rec in results:
        qid = rec["id"]
        bank = banks[qid]
        s = rec["score"]
        lines += [f"## {qid} · {rec['type']}", "",
                  f"**问题**：{rec['question']}", "",
                  f"**参考答案**：{bank.get('reference', '')}", ""]

        lines += ["### 一、召回了哪些块", ""]
        for vec, label in (("qwen3", "向量：Qwen3-Embedding-0.6B"), ("ali", "向量：阿里 text-embedding-v4")):
            if vec not in rec["routes"]:
                continue
            r = rec["routes"][vec]
            lines.append(f"- **{label} · 纯向量**：{hits_md(r['vector']['hits'])}")
            lines.append(f"- **{label} · BM25**：{hits_md(r['bm25']['hits'])}")
            lines.append(f"- **{label} · 融合**：{hits_md(r['hybrid']['hits'])}")
        lines.append("")
        lines += ["### 二、给模型看的 6 块（Qwen3 融合结果）", ""]
        for h in rec["context_hits"]:
            lines.append(f"- [{h['n']}] `{h['id']}` {h['company']} 第 {h['page']} 页 "
                         f"{h['kind']}｜{section_of(h)}")
        lines.append("")

        lines += ["### 三、模型的回答", "", rec["answer"].strip(), ""]

        lines += ["### 四、判定", ""]
        miss = "、".join(f"{m:,.2f}".rstrip("0").rstrip(".") for m in s["missing"]) or "无"
        lines += [f"- 关键数字命中：**{s['hit_numbers']}/{s['ref_numbers']}**",
                  f"- 缺的数字：{miss}",
                  f"- 结论：{'✅ 答对' if s['correct'] else '❌ 未答对'}", ""]

        grade, note = NOTES.get(qid, ("", ""))
        lines += ["### 五、错在哪 / 对在哪", "", note or "—", "", "---", ""]

    (OUT / "A_10题逐题记录.md").write_text("\n".join(lines), "utf-8")
    print(f"写好 10 题记录（{n_ok}/10 答对）→ {OUT / 'A_10题逐题记录.md'}")


if __name__ == "__main__":
    main()
