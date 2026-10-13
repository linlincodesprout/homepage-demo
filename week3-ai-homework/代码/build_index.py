"""建索引：本机向量模型（BGE 中文）+ BM25 倒排，两条路各建一份。

向量负责「说法不一样但意思一样」（问「自营渠道」能找到报表里的「直销」），
BM25 负责「字面必须出现」（公司名、科目名、年份、条款号）。
两边都建，检索时并排跑再融合——这是第三课讲的「问题侧/知识侧各装一层」。

用法：
    python build_index.py            # 建两份索引（已存在且块数一致则跳过）
    python build_index.py --force    # 强制重建
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import jieba
import numpy as np

import config


def search_text(c: dict) -> str:
    """进索引的文本 = 元数据前缀 + 正文。

    前缀让「公司名/章节」参与匹配，问「茅台上半年直销收入」时才不会提取到别家的表。
    """
    return f"{c['company']}｜{c['section'] or '未标注章节'}｜{c['kind']}｜{c['text']}"


def load_chunks() -> list[dict]:
    path = config.INDEX_DIR / "chunks.jsonl"
    if not path.exists():
        raise SystemExit("先运行 build_corpus.py 生成 data/index/chunks.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_vectors(texts: list[str], force: bool) -> None:
    emb_path = config.INDEX_DIR / "emb.npy"
    meta_path = config.INDEX_DIR / "emb_meta.json"
    if emb_path.exists() and not force:
        meta = json.loads(meta_path.read_text("utf-8"))
        if meta.get("count") == len(texts) and meta.get("model") == config.EMBED_MODEL:
            print(f"[跳过] 向量索引已存在：{meta['count']} 块 × {meta['dim']} 维")
            return

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBED_MODEL)
    model.max_seq_length = 512
    print(f"模型 {config.EMBED_MODEL} 加载完成，开始编码 {len(texts)} 个块……")
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=config.EMBED_BATCH,
        normalize_embeddings=True,      # 归一化后内积就是余弦相似度
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    np.save(emb_path, emb)
    meta_path.write_text(json.dumps({
        "model": config.EMBED_MODEL, "dim": int(emb.shape[1]),
        "count": int(emb.shape[0]), "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), "utf-8")
    print(f"向量索引完成：{emb.shape}，用时 {time.time()-t0:.0f}s → {emb_path}")


def build_bm25(texts: list[str], force: bool) -> None:
    bm25_path = config.INDEX_DIR / "bm25.pkl"
    if bm25_path.exists() and not force:
        with open(bm25_path, "rb") as f:
            obj = pickle.load(f)
        if len(obj["tokens"]) == len(texts):
            print(f"[跳过] BM25 索引已存在：{len(obj['tokens'])} 块，词表 {len(obj['bm25'].idf)}")
            return

    from rank_bm25 import BM25Okapi

    print("jieba 分词……")
    t0 = time.time()
    tokens = []
    for text in texts:
        toks = [t for t in jieba.lcut_for_search(text) if t.strip()]
        tokens.append(toks)
    bm25 = BM25Okapi(tokens)
    with open(bm25_path, "wb") as f:
        pickle.dump({"tokens": tokens, "bm25": bm25}, f)
    print(f"BM25 完成：词表 {len(bm25.idf)}，用时 {time.time()-t0:.0f}s → {bm25_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    chunks = load_chunks()
    texts = [search_text(c) for c in chunks]
    print(f"共 {len(chunks)} 个块，来自 {len({c['code'] for c in chunks})} 家公司")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    build_vectors(texts, args.force)
    build_bm25(texts, args.force)


if __name__ == "__main__":
    main()
