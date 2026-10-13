"""在 AutoDL/Colab 的 GPU 上给文本块算向量（课程同款：Qwen3-Embedding-0.6B，1024 维）。

只依赖 transformers + torch，不用 sentence-transformers——
镜像自带的 torch 2.5.1 与 sentence-transformers 6.x 要求的 transformers 5.x 冲突。
池化用官方口径：左侧 padding，取最后一个 token 的隐状态，再 L2 归一化。

用法（实例里）：
    export HF_ENDPOINT=https://hf-mirror.com
    python embed_chunks.py --chunks chunks.jsonl --out emb_qwen3.npy

产物：
    emb_qwen3.npy        形状 (N, 1024)，已 L2 归一化（内积即余弦）
    emb_qwen3_meta.json  模型名/维度/块数/耗时，本机 build_index 会读它校验
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def search_text(c: dict) -> str:
    """必须和本机 build_index.py 里的规则完全一致，否则索引和查询对不上。"""
    return f"{c['company']}｜{c['section'] or '未标注章节'}｜{c['kind']}｜{c['text']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="chunks.jsonl")
    ap.add_argument("--out", default="emb_qwen3.npy")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024, dest="max_len")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    chunks = [json.loads(line) for line in open(args.chunks, encoding="utf-8")]
    texts = [search_text(c) for c in chunks]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(texts)} 个块，设备 = "
          f"{torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()

    t0 = time.time()
    vecs = []
    for start in range(0, len(texts), args.batch):
        batch = texts[start:start + args.batch]
        enc = tok(batch, padding=True, truncation=True, max_length=args.max_len,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        last = hidden[:, -1]                                   # 左 padding：末位就是句尾
        last = torch.nn.functional.normalize(last.float(), dim=-1)
        vecs.append(last.cpu().numpy())
        done = min(start + args.batch, len(texts))
        if start % (args.batch * 20) == 0 or done == len(texts):
            print(f"  {done}/{len(texts)}  {done / max(time.time() - t0, 1e-6):.1f} 块/秒",
                  flush=True)
    emb = np.concatenate(vecs).astype("float32")
    np.save(args.out, emb)
    meta = {
        "model": args.model, "dim": int(emb.shape[1]), "count": int(emb.shape[0]),
        "seconds": round(time.time() - t0, 1), "device": "cuda",
        "max_seq_length": args.max_len,
    }
    with open(args.out.replace(".npy", "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"完成：{emb.shape}，用时 {meta['seconds']}s")
    print(f"把 {args.out} 和 {args.out.replace('.npy', '_meta.json')} 下载回本机 data/index/")


if __name__ == "__main__":
    main()
