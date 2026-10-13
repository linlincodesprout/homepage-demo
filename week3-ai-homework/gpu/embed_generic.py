"""在 GPU 上把任意 jsonl（字段 kind/key/text）编码成向量。

给知识图谱用：实体描述 + 社区报告。用法：
    python embed_generic.py --in embed_texts.jsonl --out graph_vectors
产物：
    graph_vectors_entity.npy / graph_vectors_report.npy + 各自的 key 列表
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", default="embed_texts.jsonl", dest="inp")
    ap.add_argument("--out", default="graph_vectors")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    print(f"{len(rows)} 条文本，设备 = {torch.cuda.get_device_name(0)}")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16).cuda().eval()

    out = {}
    for kind in ("entity", "report"):
        texts = [r["text"] for r in rows if r["kind"] == kind]
        keys = [r["key"] for r in rows if r["kind"] == kind]
        if not texts:
            continue
        t0 = time.time()
        vecs = []
        for i in range(0, len(texts), args.batch):
            enc = tok(texts[i:i + args.batch], padding=True, truncation=True,
                      max_length=1024, return_tensors="pt").to("cuda")
            with torch.no_grad():
                h = model(**enc).last_hidden_state[:, -1]
            vecs.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
        emb = np.concatenate(vecs).astype("float32")
        np.save(f"{args.out}_{kind}.npy", emb)
        with open(f"{args.out}_{kind}_keys.json", "w", encoding="utf-8") as f:
            json.dump(keys, f, ensure_ascii=False)
        out[kind] = emb.shape
        print(f"{kind}: {emb.shape}，用时 {time.time()-t0:.0f}s")
    print("完成", out)


if __name__ == "__main__":
    main()
