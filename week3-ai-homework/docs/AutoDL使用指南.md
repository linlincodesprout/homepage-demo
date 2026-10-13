# AutoDL 租卡指南（本作业用）

> 只有两件事需要 GPU：① 用 Qwen3-Embedding-0.6B 给 8,098 个文本块算向量；② 用 Qwen3.5-2B 做 LoRA 微调。
> 这两件事在 **RTX 4090 24GB** 上都能跑，不需要 A100。A100 是老师口头说的 27B 版本才需要。

## 一、注册与充值

1. 打开 https://www.autodl.com ，用手机号注册，完成**实名认证**（要身份证，学生认证还能便宜一点）。
2. 控制台右上角「充值」，**充 20 元**就够整个作业（4090 约 1.6–2.4 元/小时，全程含调试 5–8 小时 ≈ 10–20 元）。
3. 计费规则：**开机才计费，关机不计 GPU 费**；数据保留在实例里。中途不用了随时关机，钱按分钟扣。

## 二、选卡与建实例

1. 左侧「算力市场」→ 选地区（无所谓的，挑有货的）→ GPU 型号：

| 用途 | 推荐卡 | 显存 | 参考价 |
|---|---|---|---|
| 本作业（0.6B 编码 + 2B QLoRA） | **RTX 4090** | 24GB | 1.6–2.4 元/小时 |
| 想顺手试 27B QLoRA | A100-PCIE | 40GB | 4–5 元/小时 |
| 27B 不量化 LoRA | A100-SXM4 | 80GB | 7–9 元/小时 |

2. 点「1 张」→ 选**镜像**：
   - 基础镜像 → `PyTorch` → `2.4.0` 及以上 → `CUDA 12.1/12.4` → `Python 3.11`（默认这套就行，unsloth 会在实例里自己装）。
   - 也可以在「社区镜像」里直接搜 `unsloth`，选下载量高的那个，省一次 pip。
3. 数据盘保持默认（一般 50GB 免费额度），点「立即创建」。

## 三、进入实例

创建好之后在「容器实例」里点「开机」，然后有两种进法：

- **JupyterLab**（推荐）：点「JupyterLab」按钮，浏览器里就能上传文件、跑 Python、看终端。
- **SSH**：控制台会给出这行命令，复制到 PowerShell 直接执行
  ```
  ssh -p 12345 root@connect.xxxx.seetacloud.com
  ```
  密码在控制台实例信息里（一串随机字符）。**如果你愿意把这行 SSH 命令和密码给我，我可以直接登上去把 ①② 全部跑完**，你只需要在控制台开关机。

## 四、传文件（只有一个要点）

**所有数据都放 `/root/autodl-tmp/`**（数据盘，容量大）；`/root/` 是系统盘，只有 30–50GB，容易满。

- JupyterLab：左侧文件区进 `autodl-tmp`，直接把 zip 拖进去，然后在 Terminal 里 `unzip xx.zip`。
- 命令行：`scp -P 端口 本地文件 root@host:/root/autodl-tmp/`

要上传的东西（我会打包好一个 `gpu_bundle.zip`，几 MB）：

```
gpu_bundle.zip
├── chunks.jsonl        # 8,098 个文本块（A 方向）
├── dongmi/pdf/         # 茅台 11 份财报 PDF（B 方向，可选，抽问答对在本机做）
└── scripts/            # 下面这些脚本
```

## 五、实例里跑什么

```bash
cd /root/autodl-tmp
unzip -o gpu_bundle.zip && cd gpu_bundle

# ① 给文本块算向量（Qwen3-Embedding-0.6B，1024 维）——4090 上约 1–2 分钟
pip install -q -U sentence-transformers
python embed_chunks.py --chunks chunks.jsonl --out emb_qwen3.npy

# ② 茅台「董秘」LoRA 微调（Qwen3.5-2B 4bit，unsloth）——约 20–40 分钟
pip install -q -U unsloth
python train_lora_dongmi.py --data dongmi_qa.jsonl --out lora_dongmi
```

跑完把这两个产物下载回本机（JupyterLab 右键 Download，或 `scp` 反向拉）：

- `emb_qwen3.npy` + `emb_qwen3_meta.json` → 本机做检索
- `lora_dongmi/` 里的 `adapter_model.safetensors` + 四格对照的评测输出 → 本机对照表

## 六、常见坑

- **连接断**：关掉浏览器不等于实例关机，但长时间训练要用 `tmux` 或 `nohup`，避免 SSH 掉线把训练带走。
  ```bash
  tmux new -s train     # 进 tmux 后跑训练，Ctrl+B 松开再按 D 退出，SSH 掉线也不影响
  tmux attach -t train  # 重新连上来看
  ```
- **磁盘满**：模型缓存默认在 `/root/.cache`（系统盘）。先 `export HF_HOME=/root/autodl-tmp/hf` 再下载模型。
- **下载慢**：国内实例连 HuggingFace 不稳，统一用镜像站 `export HF_ENDPOINT=https://hf-mirror.com`。
- **不用了就关机**，别直接删实例（删了要重新配环境）。

## 七、多少钱、花多久（实际情况按第一次跑通算）

| 步骤 | 时间 | 花费 |
|---|---|---|
| 建实例 + 传文件 + 装环境 | 20 分钟 | ≈ 1 元 |
| ① 向量编码 | 2 分钟 | ≈ 0.1 元 |
| ② 2B QLoRA 微调（1 轮） | 20–40 分钟 | ≈ 1.5 元 |
| 调试余量 | 2–3 小时 | ≈ 5 元 |
| **合计** | | **≈ 10–20 元** |
