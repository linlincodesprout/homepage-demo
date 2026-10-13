"""作业 A 的页面：语义地图 + 提问 + 带出处的回答。

只用标准库起服务（不用 web 框架），前端在 web/index.html，是一张 canvas 散点图：
  · 11,047 个文本块用主成分分析压到平面，颜色区分公司，一座小岛就是一家公司的报告；
  · 点任意一个点，右边显示这一块的原文（对应课件里「点开看原文」那一步）；
  · 提问后：问题向量用同一套主成分投影落到图上（金色菱形），
    同时高亮召回的前 6 块——「变成向量 → 落到图上 → 召回周围的块 → 生成回答」四步都在页面上；
  · 回答逐句带角标，点任何一个块都能跳回原文。

用法：
    python webapp.py            # 浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import config
from rag import answer
from retrieval import brief, get_encoder, get_retriever

MAP_CACHE = config.INDEX_DIR / "map2d.npz"
_lock = threading.Lock()
MAP: dict | None = None


def build_map(vector: str = "qwen3") -> dict:
    """把全部块压到二维，并把投影矩阵留下，提问时同一个投影把问题向量也落到图上。"""
    emb = np.load(config.INDEX_DIR / f"emb_{vector}.npy")
    chunks = [json.loads(l) for l in
              open(config.INDEX_DIR / "chunks.jsonl", encoding="utf-8")]

    if MAP_CACHE.exists():
        z = np.load(MAP_CACHE)
        if z["coords"].shape[0] == emb.shape[0]:
            coords, mean, comps = z["coords"], z["mean"], z["comps"]
        else:
            coords = None
    else:
        coords = None

    if coords is None:
        mean = emb.mean(axis=0, keepdims=True)
        centered = emb - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        comps = vt[:2]
        coords = centered @ comps.T
        np.savez_compressed(MAP_CACHE, coords=coords.astype("float32"),
                            mean=mean.astype("float32"), comps=comps.astype("float32"))

    return {
        "coords": coords, "mean": mean, "comps": comps,
        "points": [{"id": c["id"], "company": c["company"], "period": c["period"],
                    "page": c["page"], "kind": c["kind"],
                    "doc": c["doc"], "section": c["section"]} for c in chunks],
    }


def project(vec: np.ndarray) -> list[float]:
    z = (vec - MAP["mean"]) @ MAP["comps"].T
    return [round(float(z[0][0]), 3), round(float(z[0][1]), 3)]


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):            # 静音
        pass

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        qs = parse_qs(u.query)

        if u.path == "/":
            html = (config.WEB_DIR / "index.html").read_bytes()
            return self._send(html, "text/html; charset=utf-8")

        if u.path == "/api/map":
            payload = {
                "points": MAP["points"],
                "coords": [[round(float(x), 3), round(float(y), 3)]
                           for x, y in MAP["coords"]],
            }
            return self._send(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        if u.path == "/api/chunk":
            cid = qs.get("id", [""])[0]
            for c in get_retriever("qwen3").chunks:
                if c["id"] == cid:
                    return self._send(json.dumps(c, ensure_ascii=False).encode("utf-8"))
            return self._send(b"{}")

        if u.path == "/api/ask":
            q = qs.get("q", [""])[0]
            vector = qs.get("vector", ["qwen3"])[0]
            mode = qs.get("mode", ["hybrid"])[0]
            retriever = get_retriever(vector)

            t0 = time.time()
            hits = retriever.search(q, mode=mode)
            search_s = round(time.time() - t0, 2)

            t0 = time.time()
            ans = answer(q, hits)
            answer_s = round(time.time() - t0, 2)

            qvec = get_encoder(vector).encode([q])
            with _lock:
                qpt = project(qvec)
            payload = {
                "query_point": qpt, "search_seconds": search_s,
                "answer_seconds": answer_s, "answer": ans["answer"],
                "hits": brief(hits),
            }
            return self._send(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        self.send_error(404)


def main() -> None:
    global MAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=config.WEB_PORT)
    args = ap.parse_args()
    print("正在计算语义地图（首次约 1-2 分钟）……")
    MAP = build_map("qwen3")
    print(f"共 {len(MAP['points'])} 个点")
    print(f"服务地址： http://{config.WEB_HOST}:{args.port}")
    ThreadingHTTPServer((config.WEB_HOST, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
