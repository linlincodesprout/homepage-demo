# homepage-demo

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

## 第三周作业：财报问答知识库 + 董秘微调

代码与交付物在 [`week3-ai-homework/`](./week3-ai-homework) 目录：

- **方向 A（建库）**：12 家白酒公司财报 → 11,047 个文本块，向量 + BM25 混合检索，
  带出处的问答页面，10 道题逐题记录（含 2 道跨公司全景题）。
  延伸做了深度三（知识图谱）：12,498 实体 / 21,942 关系 / 1,244 份社区报告，
  两道全景题的覆盖从 2/12 家提到 12/12。
- **方向 B（微调）**：贵州茅台 11 份财报 → 660 组「投资者问 / 董秘答」抽检后留 404 条
  → Qwen3.5-2B 4bit QLoRA 微调 → 四格对照（训练过/没训练过 × 原问法/换问法）。
  一页结论 PDF 在 `week3-ai-homework/outputs/pdf/`，截图在同目录 `outputs/screenshots/`。

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
