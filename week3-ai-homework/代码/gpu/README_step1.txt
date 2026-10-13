第一步：给 5,848 个文本块算向量（Qwen3-Embedding-0.6B，1024 维）
==========================================================

在 AutoDL 实例的 Terminal 里，把下面这几行按顺序粘贴执行，一行一回车。

  cd /root/autodl-tmp
  unzip -o gpu_bundle_step1.zip && cd gpu_bundle

  export HF_ENDPOINT=https://hf-mirror.com
  pip install -q -U sentence-transformers

  python embed_chunks.py --chunks chunks.jsonl --out emb_qwen3.npy

跑完（4090 上大约 1-2 分钟）在 /root/autodl-tmp/gpu_bundle/ 下会得到两个文件：

  emb_qwen3.npy         约 24 MB，5848 × 1024
  emb_qwen3_meta.json   模型名、维度、块数、耗时

把这两个文件下载回本机（JupyterLab 左侧文件区右键 Download），
放进 C:\Users\86198\Documents\ChatGPT\人工智能与数据分析\assignment3_reports_rag\data\index\

判断成功的标志：终端最后一行是 “完成：(5848, 1024)”。
如果报错，把最后 20 行贴给我。
