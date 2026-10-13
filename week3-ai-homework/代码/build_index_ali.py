"""阿里百炼 text-embedding-v4 那一路向量（课程实验一里对比用的第二个 1024 维向量）。

课程原话：「阿里 text-embedding-v4 按量付费且接口限流，只在食品饮料 83 家子库上再算一遍」。
我们这批 5,848 块 × 约 800 字，费用约 2-3 元。

用法：
    python build_index_ali.py            # 断点续跑，已算好的批次不重复调
    python build_index_ali.py --force
输出：
    data/index/emb_ali.npy + emb_ali_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import config
from build_index import load_chunks, search_text

KEYRING_SERVICE = "DashScopeAPI"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "text-embedding-v4"
DIM = 1024
BATCH = 10          # 百炼单次最多 10 条（v4 口径）
CACHE = config.INDEX_DIR / "ali_cache.jsonl"


def get_client():
    import keyring
    from openai import OpenAI
    key = keyring.get_password(KEYRING_SERVICE, "default")
    if not key:
        raise SystemExit("没有找到阿里百炼 key，先运行 save_key.py")
    return OpenAI(api_key=key, base_url=BASE_URL)


def load_cache() -> dict[int, list[float]]:
    done: dict[int, list[float]] = {}
    if CACHE.exists():
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["i"]] = rec["v"]
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    chunks = load_chunks()
    texts = [search_text(c) for c in chunks]
    n = len(texts)
    print(f"共 {n} 块")

    if args.force and CACHE.exists():
        CACHE.unlink()
    cache = load_cache()
    print(f"缓存已有 {len(cache)} 块")

    client = get_client()
    todo = [i for i in range(n) if i not in cache]
    t0 = time.time()
    with open(CACHE, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), BATCH):
            idxs = todo[start:start + BATCH]
            for attempt in range(5):
                try:
                    resp = client.embeddings.create(
                        model=MODEL, input=[texts[i] for i in idxs], dimensions=DIM)
                    for k, item in enumerate(resp.data):
                        vec = item.embedding
                        i = idxs[k]
                        cache[i] = vec
                        f.write(json.dumps({"i": i, "v": vec}) + "\n")
                    f.flush()
                    break
                except Exception as exc:  # noqa: BLE001
                    wait = 2 ** attempt
                    print(f"  第 {start//BATCH} 批失败（{attempt+1}/5）："
                          f"{type(exc).__name__} {str(exc)[:120]}，{wait}s 后重试")
                    time.sleep(wait)
            else:
                raise SystemExit("连续失败，停下来检查 key / 限流")
            done = len(cache)
            if (start // BATCH) % 20 == 0:
                rate = done / max(time.time() - t0, 1)
                print(f"  {done}/{n} 块，{rate:.1f} 块/秒，"
                      f"预计还需 {(n-done)/max(rate,0.1)/60:.1f} 分钟")
            time.sleep(0.05)

    emb = np.array([cache[i] for i in range(n)], dtype="float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-8, None)     # 归一化，内积即余弦
    np.save(config.INDEX_DIR / "emb_ali.npy", emb)
    (config.INDEX_DIR / "emb_ali_meta.json").write_text(json.dumps({
        "model": MODEL, "dim": int(emb.shape[1]), "count": int(emb.shape[0]),
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), "utf-8")
    print(f"完成：{emb.shape}，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
