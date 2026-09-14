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
