"""检索与回答：向量 + BM25 双路召回 → RRF 融合 → 大模型带引用作答。

回答有三个硬约束（对应第三课「有出处的回答」）：
1) 只依据检索到的块作答，块里没有就写「材料未覆盖」；
2) 每句话末尾带 [编号] 角标，编号对应上下文里的块；
3) 涉及金额/比例必须原样引用数字，不许改写口径。
"""

from __future__ import annotations

import json
import os
import pickle
import re
from functools import lru_cache

import jieba
import numpy as np

import config


class Retriever:
    def __init__(self) -> None:
        self.chunks = [json.loads(l) for l in
                       open(config.INDEX_DIR / "chunks.jsonl", encoding="utf-8")]
        self.emb = np.load(config.INDEX_DIR / "emb.npy")
        with open(config.INDEX_DIR / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)["bm25"]
        self._model = None

    # ---------------- 向量路 ----------------
    @property
    def model(self):
        if self._model is None:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.EMBED_MODEL)
            self._model.max_seq_length = 512
        return self._model

    def encode(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True,
                                convert_to_numpy=True)[0]
        return vec.astype("float32")

    def vector_search(self, query: str, k: int = config.TOP_K_VECTOR) -> list[dict]:
        q = self.encode(config.EMBED_QUERY_PREFIX + query)
        sims = self.emb @ q
        order = np.argsort(-sims)[:k]
        return [{"idx": int(i), "vector_score": float(sims[i])} for i in order]

    # ---------------- BM25 路 ----------------
    def bm25_search(self, query: str, k: int = config.TOP_K_BM25) -> list[dict]:
        toks = [t for t in jieba.lcut_for_search(query) if t.strip()]
        scores = self.bm25.get_scores(toks)
        order = np.argsort(-scores)[:k]
        return [{"idx": int(i), "bm25_score": float(scores[i])} for i in order]

    # ---------------- 融合 ----------------
    def search(self, query: str, mode: str = "hybrid",
               k: int = config.TOP_K_FINAL) -> list[dict]:
        """mode: vector | bm25 | hybrid（RRF 融合，只看名次不看分数量纲）。"""
        lists: list[tuple[str, list[dict]]] = []
        if mode in ("vector", "hybrid"):
            lists.append(("vector", self.vector_search(query)))
        if mode in ("bm25", "hybrid"):
            lists.append(("bm25", self.bm25_search(query)))

        merged: dict[int, dict] = {}
        for source, hits in lists:
            for rank, h in enumerate(hits, start=1):
                item = merged.setdefault(h["idx"], {"idx": h["idx"], "rrf": 0.0,
                                                    "vector_rank": None, "bm25_rank": None})
                item["rrf"] += 1.0 / (config.RRF_K + rank)
                if source == "vector":
                    item["vector_rank"] = rank
                    item["vector_score"] = h["vector_score"]
                else:
                    item["bm25_rank"] = rank
                    item["bm25_score"] = h["bm25_score"]

        out = sorted(merged.values(), key=lambda x: -x["rrf"])[:k]
        for item in out:
            item["chunk"] = self.chunks[item["idx"]]
        return out


# ---------------------------------------------------------------------------
# 大模型：密钥从 Windows 凭据管理器读，不落盘
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_client():
    import keyring
    from openai import OpenAI
    key = keyring.get_password(config.KEYRING_SERVICE, config.KEYRING_ACCOUNT)
    if not key:
        raise SystemExit("没有找到 DeepSeek API Key，请先运行 deepseek_save_key.py")
    return OpenAI(api_key=key, base_url=config.LLM_BASE_URL)


def chat(system: str, user: str, model: str | None = None,
         temperature: float | None = None) -> str:
    resp = get_client().chat.completions.create(
        model=model or config.LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        stream=False,
    )
    return resp.choices[0].message.content or ""


ANSWER_SYSTEM = (
    "你是一名严谨的财报分析师。只能依据【材料】回答，不得使用材料之外的任何知识。"
    "材料里没有的信息，直接写「材料未覆盖」。"
    "每小结的末尾都要标注依据的编号，格式如 [1][3]。"
    "金额和比例必须原样引用材料中的数字，并说明口径（合并/母公司、营业收入/营业总收入等）。"
)


def build_context(hits: list[dict], max_chars: int = config.MAX_CONTEXT_CHARS) -> str:
    lines, used = [], 0
    for i, h in enumerate(hits, start=1):
        c = h["chunk"]
        head = (f"[{i}] {c['company']}｜{c['doc']}｜{c['section'] or '未标注章节'}"
                f"｜第 {c['page']} 页｜{c['kind']}｜块 {c['id']}")
        body = c["text"]
        block = f"{head}\n{body}"
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)]
        if not block:
            break
        lines.append(block)
        used += len(block)
    return "\n\n".join(lines)


def answer(question: str, hits: list[dict]) -> dict:
    ctx = build_context(hits)
    text = chat(ANSWER_SYSTEM, f"【材料】\n{ctx}\n\n【问题】\n{question}\n\n请作答，逐句带角标。")
    return {"question": question, "answer": text, "context": ctx,
            "hits": [{"n": i + 1, "id": h["chunk"]["id"], "company": h["chunk"]["company"],
                      "page": h["chunk"]["page"], "section": h["chunk"]["section"],
                      "kind": h["chunk"]["kind"], "rrf": round(h["rrf"], 5),
                      "vector_rank": h.get("vector_rank"), "bm25_rank": h.get("bm25_rank")}
                     for i, h in enumerate(hits)]}


# ---------------------------------------------------------------------------
# 问题侧：按第三课的三种做法改写问题
# ---------------------------------------------------------------------------
def refine_question(query: str) -> str:
    """LLM 压缩：把口语铺垫洗掉，只留检索要用的核心问句。"""
    out = chat("你是检索助手。把用户的口语问题压缩成一句干净的检索问句，"
               "保留公司名、指标名、时间范围，去掉铺垫和无关信息。只输出这一句。",
               query, temperature=0.0)
    return out.strip().splitlines()[0].strip()


def extract_keywords(query: str) -> list[str]:
    """LLM 抽关键词：给 BM25 用，比直接拿口语原句分词干净得多。"""
    out = chat("你是检索助手。从用户问题里抽出 5-12 个最适合做全文检索的关键词，"
               "包含公司名、科目名、年份等专有名词，用中文逗号分隔，只输出关键词。",
               query, temperature=0.0)
    parts = re.split(r"[，,、;；\s]+", out.strip())
    return [p for p in parts if p][:15]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
