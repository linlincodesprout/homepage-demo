"""只建 BM25 倒排（jieba 分词），不碰向量。课程同款：全文检索那条路。"""

from build_index import build_bm25, load_chunks, search_text

if __name__ == "__main__":
    chunks = load_chunks()
    build_bm25([search_text(c) for c in chunks], force=False)
