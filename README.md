# homepage-demo

本仓库包含「学会使用大模型」课程的两周作业：第三周作业在 `week3-ai-homework/` 目录，第一周作业（个人主页）为根目录的 `index.html`。

## 第三周作业：财报问答知识库（方向 A）+ 董秘微调（方向 B）

代码与全部交付物在 [`week3-ai-homework/`](./week3-ai-homework) 目录，A、B 分设两个子目录。

| 方向 | 交付物 | 文件 |
|---|---|---|
| **A · 财报问答知识库** | 一页结论（PDF） | [`方向A_财报问答知识库/一页结论.pdf`](./week3-ai-homework/方向A_财报问答知识库/一页结论.pdf) |
| | 页面截图 | [`方向A_财报问答知识库/截图_问答页面_提问.png`](./week3-ai-homework/方向A_财报问答知识库/截图_问答页面_提问.png) |
| | 10 道题逐题记录（召回 / 判定 / 错因） | [`方向A_财报问答知识库/10题逐题记录.pdf`](./week3-ai-homework/方向A_财报问答知识库/10题逐题记录.pdf) |
| | 延伸：知识图谱报告与全景图 | [`方向A_财报问答知识库/延伸_深度三/`](./week3-ai-homework/方向A_财报问答知识库/延伸_深度三) |
| **B · 董秘微调** | 一页结论（PDF） | [`方向B_董秘微调/一页结论.pdf`](./week3-ai-homework/方向B_董秘微调/一页结论.pdf) |
| | 四格对照表（PNG / MD） | [`方向B_董秘微调/四格对照表.png`](./week3-ai-homework/方向B_董秘微调/四格对照表.png) |
| | 抽检记录 | [`方向B_董秘微调/抽检记录.md`](./week3-ai-homework/方向B_董秘微调/抽检记录.md) |
| | 训练笔记本（.ipynb） | [`方向B_董秘微调/训练笔记本.ipynb`](./week3-ai-homework/方向B_董秘微调/训练笔记本.ipynb) |
| | LoRA 权重与训练日志 | [`方向B_董秘微调/LoRA权重/`](./week3-ai-homework/方向B_董秘微调/LoRA权重) |
| **索引** | 交付物总览与逐条对照 | [`week3-ai-homework/交付说明_总览.md`](./week3-ai-homework/交付说明_总览.md) |
| | 全部代码 | [`week3-ai-homework/代码/`](./week3-ai-homework/代码) |

**结论摘要**：同一批财报中，答案集中于个别文本块的单点事实题适合建库（8/8 全对、带出处、秒级返回）；
答案分散在十几家公司之间的全景题仅靠检索无法回答（基线覆盖 2/12 家），需先在知识侧做结构化处理或建图谱
（覆盖提升至 12/12）；微调适合固化口径与格式（带出处率 44/60→60/60），事实仍由知识库提供
（数字命中率 8%→53%）。

---

## 第一周作业：个人主页

一个只用 `index.html` 做出来的个人主页，用来完成「学会使用大模型」第一课的部署作业。

## 文件结构

```
homepage-demo/
├── index.html   # 页面本体（样式与脚本都内联，无外部依赖）
└── README.md
```

## 本地预览

直接双击 `index.html`，或在目录里开一个静态服务器：

```bash
python -m http.server 8000
# 打开 http://localhost:8000
```

## 发布到 Vercel

### 路线 A：网页操作（老师演示的路线，不需要命令行）

1. 打开 <https://github.com/new>，仓库名填 `homepage-demo`，选 **Public**，点 *Create repository*。
2. 空仓库页面点 *creating a new file*，文件名 `index.html`，把本目录 `index.html` 的内容整段贴进去。
3. 点 *Commit changes*，写一句提交说明（例如 `add homepage`）再点 *Commit*。
4. 打开 <https://vercel.com/new>，在列表里找到 `homepage-demo`，点 *Import*。
5. 直接点 *Deploy*，等十几秒出现 *Congratulations*，点开得到 `https://homepage-demo-<你的名字>.vercel.app`。

以后每次在 GitHub 上改 `index.html`，Vercel 会自动重新部署，链接不变、内容更新。

### 路线 B：命令行

```bash
npm i -g vercel
vercel login
vercel --prod
```

## 修改内容

页面里的文字都在 `index.html` 里，直接搜索对应文字即可替换：

- 姓名与定位：`<h1>` 和 `.lede` 段落
- 实习与项目：`#experience`、`#projects` 两节
- 技能条：`style="--p:.92"` 里的数字就是进度（0–1）
- 联系方式：`#contact` 里的邮箱与 GitHub 链接
