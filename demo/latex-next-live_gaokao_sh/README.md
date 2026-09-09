# LaTeX Live Preview with Claude Code Integration

一个实时 LaTeX 编译和预览系统，支持 PDF 文本选择反向定位到 LaTeX 源码位置。

## 功能特性

- 🚀 **实时编译** - 文件变化自动触发 LaTeX 编译
- 🔍 **双向同步** - 支持 PDF 到 LaTeX 源码的反向查找
- 🎯 **精确定位** - 双击或选择 PDF 文本即可定位源码
- 🖥️ **Claude Code 集成** - 支持 AI 辅助编辑
- 📱 **跨平台支持** - 支持 Safari、Chrome、Firefox 等浏览器

## 快速开始

### 系统要求

- **Node.js** >= 18.x
- **TeX Live** (完整安装，包含 latexmk, pdflatex, bibtex)
- **SyncTeX** (通常随 TeX Live 安装)

检查环境：
```bash
node --version      # 应显示 v18.x 或更高
latexmk --version   # 应显示 Latexmk 版本
synctex help        # 应显示 SyncTeX 帮助信息
```

### 1. 克隆项目

```bash
git clone <your-repo-url> latex-next-live
cd latex-next-live
```

### 2. 安装依赖

```bash
npm install
```

### 3. 配置 Overleaf 项目（可选）

如果要与 Overleaf 同步：
```bash
# 删除示例 tex 目录
rm -rf tex

# 从 Overleaf 克隆你的项目
git clone https://git.overleaf.com/<YOUR_PROJECT_ID> tex

# 配置 git 身份
cd tex
git config user.email "your-email@example.com"
git config user.name "Your Name"
cd ..
```

### 4. 配置主文件路径

编辑 `server.js` 顶部的配置（约第 26-31 行）：
```javascript
const TEX_ROOT = path.join(ROOT, 'tex');           // LaTeX 源码目录
const MAIN_TEX = 'main.tex';                       // 主 .tex 文件名（ICLR 2026）
const PDF_PATH = path.join(OUT_DIR, 'main.pdf');   // 输出 PDF 路径
```

### 5. 启动服务

**开发模式**（支持热重载）：
```bash
npm run dev
```

**生产模式**：
```bash
npm run build
npm run start
```

### 6. 访问服务

打开浏览器访问：http://localhost:3000

如果是远程服务器，使用服务器 IP：http://<server-ip>:3000

### 后台运行（服务器部署）

使用 `nohup` 后台运行：
```bash
nohup npm run dev > latex-live.log 2>&1 &
```

使用 `screen` 保持会话：
```bash
screen -S latex-live
npm run dev
# 按 Ctrl+A, D 分离会话
# screen -r latex-live 重新连接
```

使用 `pm2` 进程管理（推荐）：
```bash
npm install -g pm2
pm2 start server.js --name latex-live
pm2 save
pm2 startup  # 开机自启
```

### 使用方法

1. **编译 LaTeX**
   - 点击"一键编译"按钮或使用快捷键 `⌘⇧R`
   - 保存 `.tex` 文件时自动编译

2. **PDF 反向查找**
   - **双击** PDF 中的文字定位到源码位置
   - **选择文本** 查看对应的 LaTeX 源码
   - 支持跨行选择和连字符处理

3. **Claude Code 编辑**
   - 终端输入命令进行 AI 辅助编辑
   - 支持自动补全和智能建议

## 项目结构

```
latex-next-live/
├── app/                    # Next.js 应用代码
│   ├── page.jsx           # 主页面组件
│   ├── globals.css        # 全局样式
│   └── layout.jsx         # 布局组件
├── tex/                   # LaTeX 源文件 (Overleaf git clone)
│   ├── main.tex          # 主文档 (ICLR 2026)
│   ├── math_commands.tex # 数学命令定义
│   ├── iclr2026_conference.bib  # 参考文献
│   ├── chapters/         # 章节文件
│   │   ├── intro.tex, method.tex, experiments.tex,
│   │   ├── related.tex, conclusion.tex, appendix.tex,
│   │   ├── preliminaries.tex, discussion.tex
│   ├── figures/          # 图片资源 (png, pdf)
│   ├── tables/           # 表格文件
│   ├── *.sty             # 样式文件 (iclr2026_conference, fancyhdr, etc.)
│   └── iclr2026_conference.bst  # 参考文献样式
├── build/                 # 编译生成的 PDF
├── public/
│   └── pdfjs/            # PDF.js 库
├── server.js             # Express 服务器
├── package.json          # 项目配置
└── .env.local           # 环境变量配置

```

## 技术栈

- **前端**: Next.js + React
- **PDF 渲染**: PDF.js
- **同步技术**: SyncTeX
- **后端**: Node.js + Express
- **LaTeX 编译**: latexmk + xelatex
- **实时通信**: Server-Sent Events (SSE)

## 配置说明

### 环境变量 (.env.local)

```bash
# Claude Code 配置
CLAUDE_CODE_BIN=claude
CLAUDE_CODE_ARGS=code
PROJECT_ROOT=/path/to/tex/folder

# 可选：WebSocket 连接保护
# CLAUDE_CODE_TOKEN=your-secret-token
```

### LaTeX 编译配置

默认使用 `xelatex` 编译，配置在 `server.js` 中：

```javascript
const compileCmd = 'latexmk -xelatex -interaction=nonstopmode -file-line-error';
```

## API 接口

### POST /api/compile
手动触发 LaTeX 编译

### POST /api/revsearch
PDF 文本反向查找
- 参数：`{ page, h, v, selectedText, endPage, endH, endV }`
- 返回：源码位置信息

### GET /api/sse
Server-Sent Events 连接，接收编译状态更新

### POST /api/claude
Claude Code 命令执行接口

## 故障排查

### 划词选择不工作
1. 清理缓存：`rm -rf .next`
2. 重启服务器：`npm run dev`
3. 检查浏览器控制台错误信息

### 编译失败
1. 检查 LaTeX 环境：`which xelatex`
2. 查看服务器日志中的编译错误
3. 确保 `.tex` 文件语法正确

### 反向查找不准确
1. 确保编译时生成了 `.synctex.gz` 文件
2. 检查 PDF 坐标转换是否正确
3. 验证 SyncTeX 数据完整性

## 备份与恢复

创建备份：
```bash
zip -r latex-next-live-backup-$(date +%Y%m%d-%H%M%S).zip latex-next-live \
  -x "*.next/*" -x "*node_modules/*" -x "*.git/*" -x "*tex/output/*"
```

恢复备份：
```bash
unzip latex-next-live-backup-YYYYMMDD-HHMMSS.zip
cd latex-next-live
npm install
npm run dev
```

## Overleaf 双向同步

本项目支持与 Overleaf 双向同步，可以在本地编辑器和 Overleaf 之间无缝切换。

### 初始设置

#### 1. 从 Overleaf 克隆项目

```bash
# 进入项目目录
cd latex-next-live

# 删除现有 tex 目录（如果有）
rm -rf tex

# 从 Overleaf 克隆（替换为你的项目 ID）
git clone https://git@git.overleaf.com/<YOUR_PROJECT_ID> tex
```

**当前项目 ID**: `<YOUR_PROJECT_ID>` (ICLR 2026)

**获取项目 ID**: 在 Overleaf 项目中，点击左侧菜单 → Git → 复制 git clone 命令中的项目 ID。

#### 2. 配置 Git 身份（首次使用）

```bash
cd tex
git config user.email "your-email@example.com"
git config user.name "Your Name"
```

#### 3. 配置凭据存储（可选，避免重复输入密码）

```bash
git config --global credential.helper store
```

### 日常使用

#### 从 Overleaf 拉取更新

```bash
cd tex
git pull origin master
```

#### 推送本地修改到 Overleaf

```bash
cd tex
git add -A
git commit -m "描述你的修改"
git push origin master
```

#### 一键同步脚本

可以创建便捷脚本 `sync.sh`：

```bash
#!/bin/bash
cd tex

# 拉取 Overleaf 最新修改
echo "📥 Pulling from Overleaf..."
git pull origin master

# 如果有本地修改，提交并推送
if [[ -n $(git status -s) ]]; then
    echo "📤 Pushing local changes..."
    git add -A
    git commit -m "Sync: $(date '+%Y-%m-%d %H:%M')"
    git push origin master
fi

echo "✅ Sync complete!"
```

### 认证方式

Overleaf Git 使用认证令牌（Authentication Token）：

1. 登录 Overleaf → Account Settings → Git Integration
2. 生成 Authentication Token
3. 使用时：
   - 用户名: `git`
   - 密码: 你的 Authentication Token

### 冲突处理

如果本地和 Overleaf 同时修改了同一文件：

```bash
# 拉取时会提示冲突
git pull origin master

# 手动解决冲突后
git add -A
git commit -m "Resolve conflicts"
git push origin master
```

### 注意事项

- **同步频率**: 建议在开始编辑前先 `git pull`，完成后及时 `git push`
- **大文件**: 避免在 Overleaf 项目中存放过大的图片文件
- **编译文件**: `.aux`、`.log` 等编译生成的文件不会被同步（已在 `.gitignore` 中）
- **并发编辑**: 避免同时在本地和 Overleaf 编辑同一文件

## 开发说明

### 前端优化要点

1. **坐标转换**: 使用 `viewport.convertToPdfPoint` 确保精确度
2. **参数传递**: 发送 `h/v` 而非 `coords.x/y`
3. **文本规范化**: 处理连字符断行 `-\n`
4. **事件处理**: 监听 `mouseup/dblclick/pointerup/touchend`
5. **Safari 兼容**: 添加 20ms 延迟等待选区稳定

### 调试技巧

1. 开启详细日志：在浏览器控制台查看
2. 检查网络请求：监控 `/api/revsearch` 请求
3. SyncTeX 调试：`synctex view -i line:col:file.tex -o output.pdf`

## 更新日志

### 2026-01-28
- 切换到 ICLR 2026 论文项目 (Overleaf: `<YOUR_PROJECT_ID>`)
- 更新 tex 目录结构：main.tex 位于根目录，chapters/ 存放章节文件，figures/ 存放图片，tables/ 存放表格
- 使用 iclr2026_conference 样式文件
- 更新文档以反映新的项目配置

### 2026-01-27
- 切换到 ICML 2026 论文项目 (Overleaf: `69748574086b06e7be502304`)
- 更新 tex 目录结构：main.tex 位于根目录，sections/ 存放章节文件
- 更新文档以反映新的项目配置

### 2024-10-04
- 修复 Safari/Chrome 文本选择兼容性问题
- 优化坐标转换精度
- 改进参数传递格式

### 2024-10-03
- 实现 PDF 文本选择反向查找
- 添加 Claude Code 集成
- 支持实时编译

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请在 GitHub 上提交 Issue。