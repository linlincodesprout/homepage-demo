# 第三课作业：白酒行业半年报知识库（A）+ 贵州茅台「董秘」微调（B）

两个方向共用同一套语料管线和一个知识库。所有数字都是真跑出来的，评测记录、抽检表、
结论都在 `outputs/` 里，没有手写凑数。

## 目录

```
config.py               公司清单、路径、切块/检索参数（换机器只改这里）
fetch_reports.py        巨潮资讯网下载 12 家白酒公司 2026 半年报
fetch_reports_dongmi.py 下载贵州茅台 2021–2025 年报 + 2021–2026 半年报（共 11 份）
build_corpus.py         PDF → 正文 + 按行列还原的表格 → 800/120 切块
build_corpus_dongmi.py  同上，针对茅台 11 份报告
merge_corpus.py         把茅台历年财报并进同一个知识库（给 B 的对照用）
build_index_ali.py      阿里 text-embedding-v4 向量（断点续跑）
build_bm25.py           jieba + BM25 倒排
retrieval.py            检索层：公司/报告期硬过滤 + 向量/BM25/RRF 融合
rag.py                  DeepSeek 带角标回答；问题精炼、关键词抽取
metrics.py              知识侧预处理：跨公司指标表（全景题用）
webapp.py + web/        语义地图 + 提问 + 带出处回答的页面
eval_a.py               A 的 10 道题评测（含自动判分）
panorama_fix.py         全景题补救实验（top6 / top60 / 逐家检索 / 指标表）
gen_qa_dongmi.py        B：从财报原文生成「投资者问/董秘答」
dongmi_check.py         B：数字回原文核对 + 抽检表 + 冻结评测集
dongmi_format.py        B：渲染成 Qwen 对话模板的 SFT 文本
gpu/                    AutoDL 上跑的脚本（向量编码、LoRA 训练、四格评测）
graph/                  深度三：建图六步 + 全景题五路对比（详见下节）
docs/AutoDL使用指南.md   租卡、传文件、跑脚本的操作说明
data/                   语料、切块、向量、BM25（大文件，可重新生成）
outputs/                交付物：评测记录、判分表、抽检表、补救实验、一页结论、截图
```

## 复现步骤

```powershell
# 0. 环境
..\venv\Scripts\python.exe -m pip install pymupdf jieba rank_bm25 numpy transformers openai keyring

# 1. 取语料（A 十二家 + B 茅台历年）
python fetch_reports.py
python fetch_reports_dongmi.py

# 2. 抽正文/表格 → 切块 → 合并成一个库
python build_corpus.py
python build_corpus_dongmi.py
python merge_corpus.py

# 3. 索引：阿里向量（API，约 4 元）+ BM25；本机 Qwen3 向量走 gpu/embed_chunks.py
python build_index_ali.py
python build_bm25.py

# 4. 知识侧预处理 + 评测 + 页面
python metrics.py
python eval_a.py
python panorama_fix.py
python webapp.py            # http://127.0.0.1:8000
```

向量索引要想完全复现 1,047 万字符那 11,047 个块，需要 GPU（见 `docs/AutoDL使用指南.md`），
脚本是 `gpu/embed_chunks.py`，4090 上 3.5 分钟。

## 交付物清单

| 作业要求 | 对应文件 |
|---|---|
| A 代码仓库 | 本目录 |
| A 页面截图 | `outputs/screenshots/A_页面_初始.png`、`A_页面_提问.png` |
| A 10 道题逐题记录 | `outputs/A_评测记录.md`、`A_判分表.md`、`A_全景题补救实验.md` |
| A 一页结论 | `outputs/A_一页结论.md` |
| B 数据生成脚本 + 抽检记录 | `gen_qa_dongmi.py`、`dongmi_check.py`、`outputs/dongmi_抽检表.md` |
| B 训练脚本 + 权重 | `gpu/train_lora_dongmi.py`；LoRA 权重在 `outputs/dongmi_lora_6轮/`（另存 3 轮版做轮数对比） |
| B 四格对照表 | `outputs/B_四格对照.md`（原始 640+320 条结果：`outputs/B_四格结果*.jsonl`） |
| B 一页结论 | `outputs/B_一页结论.md` |

## 深度三（织成网）：把知识图谱加上去

作业 A 原本只能答单公司题（8/10），两道跨公司全景题只覆盖 2/12 家。
按课件的六步补了知识图谱：选材 600 块 → 抽实体关系（**12,498 实体 / 21,942 关系**）
→ 合并描述（744 个）→ **Leiden** 社区聚类（4 层）→ 社区报告（**1,244 份**）→ 向量化入库。

```powershell
python graph/select_blocks.py --per-company 50   # ① 选材
python graph/extract.py --workers 8              # ② 抽实体与关系（要调大模型）
python graph/build_graph.py                      # ③④ 合并描述 + Leiden 聚类
python graph/community_reports.py                # ⑤ 社区报告（要调大模型）
python graph/export_texts.py                     # ⑥ 导出待编码文本
#    在 GPU 上：python gpu/embed_generic.py --in embed_texts.jsonl --out graph_vectors
python graph/run_panorama.py                     # 三道全景题 × 五条路线
python graph/render.py --top 180                 # 图谱配图
python graph/query_graph.py --mode global --q "…"  # 查图（global）
python graph/query_graph.py --mode local  --q "…"  # 查图（local）
```

结果、五路对比、两个工程坑的复盘见 `outputs/A_深度三_图谱全景题.md`。

## B 方向：从造数据到四格对照的完整链路

```powershell
python fetch_reports_dongmi.py     # 茅台 11 份财报
python build_corpus_dongmi.py      # 5,690 块
python gen_qa_dongmi.py            # 生成 660 组投资者问答（DeepSeek）
python dongmi_check.py             # 数字回原文核对 + 40 条抽检表 + 冻结评测集
python dongmi_format.py            # 渲染成 Qwen 对话模板（结构化文本）
python prepare_eval_context.py     # 本机算好 160 条的检索上下文
```

GPU 端（AutoDL，脚本在 `gpu/`）：

```bash
python train_lora_dongmi.py --backend peft --epochs 6 --rank 16 --batch 2 --grad-accum 4 --out lora_dongmi6
python eval_dongmi.py --configs finetuned,kb_finetuned --adapter lora_dongmi6 --out eval_results6.jsonl
```

回到本机：`python dongmi_score.py` 出四格对照表。

> 环境备注：Qwen3.5-2B 的模型类型是 `qwen3_5`，只有 transformers≥5 认识；而 transformers 5 要求
> torch≥2.6。所以实例上把 torch 升到 2.6.0+cu124 后，用 `peft + bitsandbytes` 做 4bit QLoRA
> （unsloth 的依赖解析会把 torch 换成 CUDA 13 版本，要多下 3GB，因此没用它）。

## 密钥

DeepSeek 和阿里百炼的 key 都放在 Windows 凭据管理器里（`save_key.py`），
代码里没有任何明文密钥，仓库里也不会有。
