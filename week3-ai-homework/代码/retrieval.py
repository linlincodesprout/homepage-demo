"""检索层：两条向量（本机 Qwen3-Embedding-0.6B / 阿里 text-embedding-v4）+ 一条 BM25。

课程里三条路都跑过，这里做成可切换：
    vector=qwen3 | ali      用哪条向量
    mode=vector | bm25 | hybrid
查询和文本必须用同一个 embedding 模型编码，否则两个向量空间对不上。
"""

from __future__ import annotations

import json
import pickle
import re
from functools import lru_cache

import jieba
import numpy as np

import config

EMB_FILES = {
    "qwen3": "emb_qwen3.npy",
    "ali": "emb_ali.npy",
}

# 公司名/简称 → 规范公司名。问题里出现公司名时，直接把它当成硬过滤条件。
# 课程讲义中的说法：「公司名是最强的锚」「每块带公司标签，检索时才分得清是谁的数」。
COMPANY_ALIASES = {
    "贵州茅台": ["贵州茅台", "茅台"],
    "五粮液": ["五粮液"],
    "泸州老窖": ["泸州老窖", "老窖"],
    "洋河股份": ["洋河股份", "洋河"],
    "山西汾酒": ["山西汾酒", "汾酒", "杏花村"],
    "古井贡酒": ["古井贡酒", "古井"],
    "今世缘": ["今世缘"],
    "口子窖": ["口子窖", "口子酒"],
    "迎驾贡酒": ["迎驾贡酒", "迎驾"],
    "舍得酒业": ["舍得酒业", "舍得"],
    "水井坊": ["水井坊"],
    "老白干酒": ["老白干酒", "老白干", "衡水老白干"],
}


def detect_company(query: str) -> str | None:
    """从问题里认公司名。问题里同时出现多家（全景题）时返回 None，不做过滤。"""
    hits = [name for name, alias in COMPANY_ALIASES.items()
            if any(a in query for a in alias)]
    return hits[0] if len(hits) == 1 else None


def detect_period(query: str) -> str | None:
    """从问题里认报告期：『2024年上半年/半年度』→ 2024H1，『2024年年度报告』→ 2024A。

    这是课程讲义中「剔除废止版本」的同类动作——法规库靠时效性字段，我们靠报告期标签。
    """
    m = re.search(r"(20\d\d)\s*年\s*(半年度|上半年|中期|中报)", query)
    if m:
        return f"{m.group(1)}H1"
    m = re.search(r"(20\d\d)\s*年\s*(年度报告|年报|全年)", query)
    if m:
        return f"{m.group(1)}A"
    return None


# ---------------------------------------------------------------------------
# 查询侧编码器
# ---------------------------------------------------------------------------
class Qwen3QueryEncoder:
    """本机 CPU 跑 Qwen3-Embedding-0.6B（只编码问题，每次几秒，够用）。"""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.EMBED_MODEL_QWEN3, padding_side="left")
        self.model = AutoModel.from_pretrained(config.EMBED_MODEL_QWEN3).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        """分批编码。

        一开始把 7,000 多个实体一次性送进去，padding 到 1024 token，
        等于一次前向算 700 万 token——必然卡死。这里按 32 条一批。
        """
        torch = self.torch
        out = []
        for i in range(0, len(texts), 32):
            enc = self.tok(texts[i:i + 32], padding=True, truncation=True,
                           max_length=1024, return_tensors="pt")
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            vec = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)
            out.append(vec.numpy())
            if i % 320 == 0:
                print(f"    编码 {min(i+32, len(texts))}/{len(texts)}", flush=True)
        return np.concatenate(out).astype("float32")


class AliQueryEncoder:
    """阿里百炼 text-embedding-v4（和建库时同一个模型）。"""

    def __init__(self) -> None:
        import keyring
        from openai import OpenAI
        key = keyring.get_password("DashScopeAPI", "default")
        if not key:
            raise SystemExit("没有找到阿里百炼 key，先运行 save_key.py")
        self.client = OpenAI(api_key=key,
                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def encode(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model="text-embedding-v4",
                                             input=texts, dimensions=1024)
        vec = np.array([d.embedding for d in resp.data], dtype="float32")
        return vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8, None)


@lru_cache(maxsize=4)
def get_encoder(name: str):
    if name == "qwen3":
        return Qwen3QueryEncoder()
    if name == "ali":
        return AliQueryEncoder()
    raise ValueError(f"未知向量：{name}")


# ---------------------------------------------------------------------------
# 检索器
# ---------------------------------------------------------------------------
class Retriever:
    def __init__(self, vector: str = "qwen3") -> None:
        self.vector = vector
        with open(config.INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(l) for l in f]
        self.emb = np.load(config.INDEX_DIR / EMB_FILES[vector])
        with open(config.INDEX_DIR / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)["bm25"]

    # ---- 单路 ----
    def _mask(self, company: str | None, period: str | None) -> np.ndarray | None:
        if not company and not period:
            return None
        mask = np.array([(not company or c["company"] == company)
                         and (not period or c.get("period") == period)
                         for c in self.chunks])
        return np.where(mask, 0.0, -np.inf)      # 不满足元数据条件的直接排除

    def vector_search(self, query: str, k: int = config.TOP_K_VECTOR,
                      company: str | None = None,
                      period: str | None = None) -> list[dict]:
        q = get_encoder(self.vector).encode([query])[0]
        sims = self.emb @ q
        mask = self._mask(company, period)
        if mask is not None:
            sims = sims + mask
        order = np.argsort(-sims)[:k]
        return [{"idx": int(i), "vector_score": float(sims[i])} for i in order]

    def bm25_search(self, query: str, k: int = config.TOP_K_BM25,
                    company: str | None = None,
                    period: str | None = None) -> list[dict]:
        toks = [t for t in jieba.lcut_for_search(query) if t.strip()]
        scores = self.bm25.get_scores(toks)
        mask = self._mask(company, period)
        if mask is not None:
            scores = scores + mask
        order = np.argsort(-scores)[:k]
        # 零分候选一个查询词都没命中，只是排序里仅用于补齐数量的，不该去抢融合名次
        return [{"idx": int(i), "bm25_score": float(scores[i])}
                for i in order if scores[i] > 0]

    # ---- 融合 ----
    def search(self, query: str, mode: str = "hybrid",
               k: int = config.TOP_K_FINAL, company: str | None = None,
               period: str | None = None, auto_filter: bool = True) -> list[dict]:
        """RRF 融合：两条路各自按名次计分 1/(K+rank)，只比名次不比量纲。

        auto_filter=True 时从问题里自动认出「公司名 + 报告期」做硬过滤：
        第一版没做这一步，问茅台时 BM25 把洋河、水井坊的块混进来，
        问 2026 上半年时 2022/2024 年的同类表格也挤进前 6，
        第 74 页那张直销收入表因此被挤掉，答案错成「材料未覆盖」。
        """
        if auto_filter:
            company = company or detect_company(query)
            period = period or detect_period(query)
        lists = []
        if mode in ("vector", "hybrid"):
            lists.append(("vector", self.vector_search(query, company=company, period=period)))
        if mode in ("bm25", "hybrid"):
            lists.append(("bm25", self.bm25_search(query, company=company, period=period)))

        # BM25 容易被通用词蹭名次，权重压低一半；向量是主线（课程讲义口径：向量管意思、关键词管字面）
        weights = {"vector": 1.0, "bm25": 0.5}
        merged: dict[int, dict] = {}
        for source, hits in lists:
            for rank, h in enumerate(hits, start=1):
                item = merged.setdefault(h["idx"], {
                    "idx": h["idx"], "rrf": 0.0,
                    "vector_rank": None, "bm25_rank": None})
                item["rrf"] += weights[source] / (config.RRF_K + rank)
                item["company_filter"] = company
                item["period_filter"] = period
                if source == "vector":
                    item["vector_rank"] = rank
                    item["vector_score"] = round(h["vector_score"], 4)
                else:
                    item["bm25_rank"] = rank
                    item["bm25_score"] = round(h["bm25_score"], 3)

        out = sorted(merged.values(), key=lambda x: -x["rrf"])[:k]
        for item in out:
            item["chunk"] = self.chunks[item["idx"]]
        return out


@lru_cache(maxsize=4)
def get_retriever(vector: str = "qwen3") -> Retriever:
    return Retriever(vector)


def brief(hits: list[dict]) -> list[dict]:
    """给评测报告用的精简命中记录。"""
    rows = []
    for i, h in enumerate(hits, start=1):
        c = h["chunk"]
        rows.append({
            "n": i, "id": c["id"], "company": c["company"], "page": c["page"],
            "section": c["section"], "kind": c["kind"], "rrf": round(h["rrf"], 5),
            "vector_rank": h.get("vector_rank"), "bm25_rank": h.get("bm25_rank"),
            "head": c["text"][:120].replace("\n", " "),
        })
    return rows
