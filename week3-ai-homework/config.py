"""全局配置：公司清单、路径、切块与检索参数。

第三课作业（方向 A）：白酒行业 12 家上市公司 2026 年半年报 → 问答知识库。
所有路径都相对项目根目录，换机器只需要改本文件。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"
PAGE_DIR = DATA_DIR / "pages"
INDEX_DIR = DATA_DIR / "index"
OUT_DIR = ROOT / "outputs"
WEB_DIR = ROOT / "web"

# ---------------------------------------------------------------------------
# 语料：白酒行业 12 家公司（沪市 8 家 + 深市 4 家），交易所披露的 2026 年半年报全文
# ---------------------------------------------------------------------------
COMPANIES = [
    ("600519", "贵州茅台", "sse"),
    ("000858", "五粮液", "szse"),
    ("000568", "泸州老窖", "szse"),
    ("002304", "洋河股份", "szse"),
    ("600809", "山西汾酒", "sse"),
    ("000596", "古井贡酒", "szse"),
    ("603369", "今世缘", "sse"),
    ("603589", "口子窖", "sse"),
    ("603198", "迎驾贡酒", "sse"),
    ("600702", "舍得酒业", "sse"),
    ("600779", "水井坊", "sse"),
    ("600559", "老白干酒", "sse"),
]

REPORT_PERIOD = "2026H1"          # 2026 年半年度报告
REPORT_KEYWORD = "半年度报告"       # 用于在公告标题里挑全文
REPORT_SE_DATE = "2026-07-01~2026-10-08"

# ---------------------------------------------------------------------------
# 切块参数
# ---------------------------------------------------------------------------
# 课程同款：800 字一块、重合 120 字、尽量在句末收尾
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK = 80          # 小于这个长度的块直接并到上一块

# ---------------------------------------------------------------------------
# 检索参数
# ---------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-base-zh-v1.5"
EMBED_MODEL_QWEN3_ID = "Qwen/Qwen3-Embedding-0.6B"   # 课程同款（GPU 上建库，本机只编码问题）


def _resolve_qwen3() -> str:
    """优先用本机已下载的目录，找不到再退回 HuggingFace 仓库名。"""
    import os
    candidates = [
        os.environ.get("QWEN3_EMB_PATH", ""),
        str(ROOT / "models" / "Qwen3-Embedding-0.6B"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     r"Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local\codex-models\Qwen3-Embedding-0.6B"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "codex-models", "Qwen3-Embedding-0.6B"),
    ]
    for c in candidates:
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "config.json")):
            return c
    return EMBED_MODEL_QWEN3_ID


EMBED_MODEL_QWEN3 = _resolve_qwen3()
EMBED_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："  # BGE 官方中文查询前缀
EMBED_BATCH = 32

TOP_K_VECTOR = 30       # 向量路召回
TOP_K_BM25 = 30         # BM25 路召回
TOP_K_FINAL = 6         # 融合后交给大模型的块数
RRF_K = 60              # RRF 融合常数
MAX_CONTEXT_CHARS = 9000

# ---------------------------------------------------------------------------
# 大模型（回答生成）：DeepSeek，密钥从 Windows 凭据管理器读取，不落盘
# ---------------------------------------------------------------------------
KEYRING_SERVICE = "DeepSeekAPI"
KEYRING_ACCOUNT = "default"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-flash"
LLM_TEMPERATURE = 0.2

WEB_HOST = "127.0.0.1"
WEB_PORT = 8000
