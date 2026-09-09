/* eslint-disable no-console */
require('dotenv').config(); // 加载 .env 文件

const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const express = require('express');
const bodyParser = require('body-parser');
const chokidar = require('chokidar');
const { exec, execFile, execFileSync } = require('child_process');
const WebSocket = require('ws');
const pty = require('node-pty');
const next = require('next');
const multer = require('multer');

// ==========================================
// 安全配置：每次启动生成新的访问 Token
// ==========================================
const WS_ACCESS_TOKEN = crypto.randomBytes(32).toString('hex');
console.log('\n' + '='.repeat(60));
console.log('🔐 WebSocket 终端访问 Token (每次启动重新生成):');
console.log('   ' + WS_ACCESS_TOKEN);
console.log('='.repeat(60) + '\n');

// —— 基本路径配置 —— //
const ROOT = __dirname;
const TEX_ROOT = path.join(ROOT, 'tex');           // 你的 LaTeX 源码目录
const OUT_DIR  = path.join(ROOT, 'build');         // 编译输出目录
const TMP_DIR  = path.join(ROOT, 'tmp');           // 临时文件目录（图片上传）
const MAIN_TEX = 'main.tex';                       // 主文件名
const PDF_PATH = path.join(OUT_DIR, 'main.pdf');

// 反向代理子路径前缀（JupyterHub server-proxy 会在转发前剥掉该前缀，所以路由匹配
// 保持不带前缀；只有"浏览器看到的 URL"（重定向、登录页 fetch、资源链接）才需要它）。
// 例：APP_BASE_PATH=/user/cck/proxy/2219 。不设时为空，行为与原来一致。
const BASE = (process.env.APP_BASE_PATH || '').replace(/\/$/, '');

// Claude 终端启动命令：不再写死模型。默认用账号默认模型（纯 claude）；
// 设了 CLAUDE_MODEL 才追加 --model；也可用 CLAUDE_TERMINAL_CMD 完全自定义。
const CLAUDE_MODEL = (process.env.CLAUDE_MODEL || '').trim();
const CLAUDE_CMD = (process.env.CLAUDE_TERMINAL_CMD || '').trim()
  || ('claude' + (CLAUDE_MODEL ? ` --model ${CLAUDE_MODEL}` : ''));

// 确保输出目录存在
fs.mkdirSync(OUT_DIR, { recursive: true });
fs.mkdirSync(TMP_DIR, { recursive: true });

// 确保能找到 macTeX/ripgrep 以及常见 CLI（包含 Intel Mac 的 /usr/local/bin）
process.env.PATH = `/Library/TeX/texbin:/opt/homebrew/bin:/usr/local/bin:${process.env.PATH || ''}`;

const dev = process.env.NODE_ENV !== 'production';
const appNext = next({ dev });
const handle = appNext.getRequestHandler();

appNext.prepare().then(() => {
  const app = express();
  app.use(bodyParser.json({ limit: '2mb' }));

  // ==========================================
  // 网页级认证保护
  // ==========================================

  // 登录页面 HTML（内联样式，不依赖外部资源）
  const LOGIN_PAGE_HTML = `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LaTeX Live - 访问验证</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .container {
      background: white;
      padding: 40px;
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      max-width: 400px;
      width: 90%;
    }
    h1 { color: #333; margin-bottom: 8px; font-size: 24px; }
    p { color: #666; margin-bottom: 24px; font-size: 14px; }
    .input-group { margin-bottom: 20px; }
    label { display: block; color: #333; margin-bottom: 8px; font-weight: 500; }
    input[type="password"] {
      width: 100%;
      padding: 12px 16px;
      border: 2px solid #e1e1e1;
      border-radius: 8px;
      font-size: 16px;
      font-family: monospace;
      transition: border-color 0.2s;
    }
    input[type="password"]:focus {
      outline: none;
      border-color: #667eea;
    }
    button {
      width: 100%;
      padding: 14px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 16px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102,126,234,0.4); }
    button:active { transform: translateY(0); }
    .error {
      background: #fee;
      color: #c00;
      padding: 12px;
      border-radius: 8px;
      margin-bottom: 20px;
      font-size: 14px;
      display: none;
    }
    .error.show { display: block; }
    .hint { color: #999; font-size: 12px; margin-top: 16px; text-align: center; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🔐 访问验证</h1>
    <p>请输入服务启动时显示的访问 Token</p>
    <div id="error" class="error"></div>
    <form id="loginForm">
      <div class="input-group">
        <label for="token">Access Token</label>
        <input type="password" id="token" name="token" placeholder="输入 64 位 Token" required autocomplete="off">
      </div>
      <button type="submit">验证并进入</button>
    </form>
    <p class="hint">Token 在服务启动的终端中显示</p>
  </div>
  <script>
    document.getElementById('loginForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const token = document.getElementById('token').value.trim();
      const errorEl = document.getElementById('error');

      try {
        const res = await fetch('${BASE}/api/auth', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token })
        });

        if (res.ok) {
          window.location.href = '${BASE}/';
        } else {
          const data = await res.json();
          errorEl.textContent = data.error || '验证失败';
          errorEl.classList.add('show');
        }
      } catch (err) {
        errorEl.textContent = '网络错误，请重试';
        errorEl.classList.add('show');
      }
    });
  </script>
</body>
</html>
`;

  // 登录页面路由（不需要认证）
  app.get('/login', (req, res) => {
    res.type('html').send(LOGIN_PAGE_HTML);
  });

  // Token 验证端点（不需要认证）
  app.post('/api/auth', (req, res) => {
    const { token } = req.body || {};
    if (token === WS_ACCESS_TOKEN) {
      // 设置认证 cookie（httpOnly 保护，有效期 24 小时）
      res.cookie('auth_token', token, {
        httpOnly: true,
        maxAge: 24 * 60 * 60 * 1000, // 24 小时
        sameSite: 'strict'
      });
      res.json({ ok: true });
    } else {
      res.status(401).json({ error: 'Token 无效' });
    }
  });

  // 登出端点
  app.get('/logout', (req, res) => {
    res.clearCookie('auth_token');
    res.redirect(BASE + '/login');
  });

  // 解析 cookie 的辅助函数
  function parseCookies(cookieHeader) {
    const cookies = {};
    if (cookieHeader) {
      cookieHeader.split(';').forEach(cookie => {
        const [name, value] = cookie.trim().split('=');
        if (name && value) cookies[name] = decodeURIComponent(value);
      });
    }
    return cookies;
  }

  // 认证中间件：检查所有请求
  app.use((req, res, next) => {
    // 放行的路径（不需要认证）
    const publicPaths = ['/login', '/api/auth', '/favicon.ico'];
    if (publicPaths.some(p => req.path === p || req.path.startsWith(p + '/'))) {
      return next();
    }

    // 检查 cookie 中的 token
    const cookies = parseCookies(req.headers.cookie);
    if (cookies.auth_token === WS_ACCESS_TOKEN) {
      return next();
    }

    // 对于 API 请求返回 401，对于页面请求重定向到登录页
    if (req.path.startsWith('/api/') || req.path.startsWith('/ws/')) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    // 重定向到登录页
    res.redirect(BASE + '/login');
  });

  // 挂载 pdf.js 官方 viewer 静态资源（借鉴/复用）
  app.use('/pdfjs', express.static(path.join(ROOT, 'public/pdfjs')));

  // —— SSE 客户端集合 —— //
  const sseClients = new Set();
  function broadcast(evt, data) {
    const payload = `event: ${evt}\ndata: ${JSON.stringify(data)}\n\n`;
    console.log(`📢 SSE broadcast: ${evt} to ${sseClients.size} clients`);
    for (const res of sseClients) res.write(payload);
  }

  // —— SSE 事件流：编译完成时通知前端 —— //
  // 支持两个端点以兼容旧代码
  const setupSSE = (req, res) => {
    res.set({
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    });
    res.flushHeaders();
    res.write('event: connected\ndata: ok\n\n');
    sseClients.add(res);
    req.on('close', () => sseClients.delete(res));
  };

  app.get('/events', setupSSE);
  app.get('/api/sse', setupSSE);  // 新的端点，匹配前端代码

  // PDF 版本号（mtime）：供前端轮询触发刷新。反向代理（JupyterHub server-proxy）会
  // 缓冲 SSE 流，轮询这个轻量端点最稳妥。
  app.get('/api/pdf-version', (req, res) => {
    fs.stat(PDF_PATH, (err, st) => {
      res.json({ mtime: err ? 0 : st.mtimeMs });
    });
  });

  // —— 编译（一次性） —— //
  let compiling = false;
  function runLatexmk(tag = 'manual') {
    if (compiling) return;
    compiling = true;

    // 使用 -auxdir 选项，将辅助文件放在单独目录，PDF 直接生成在 build 目录
    const auxDir = path.join(OUT_DIR, '.aux');
    if (!fs.existsSync(auxDir)) {
      fs.mkdirSync(auxDir, { recursive: true });
    }

    const cmd = `latexmk -pdf -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error -auxdir="${auxDir}" -outdir="${OUT_DIR}" "${MAIN_TEX}"`;

    console.log('🔨 Compiling LaTeX...');
    // 设置环境变量让 LaTeX/BibTeX 搜索子目录（使用绝对路径）
    const latexDir = path.join(TEX_ROOT, 'latex');
    const icmlDir = path.join(TEX_ROOT, 'icml_main');
    const texEnv = {
      ...process.env,
      TEXINPUTS: `./icml_main/:${icmlDir}/:./latex/:${latexDir}/:` + (process.env.TEXINPUTS || ''),
      BIBINPUTS: `./icml_main/:${icmlDir}/:./latex/:${latexDir}/:` + (process.env.BIBINPUTS || ''),
      BSTINPUTS: `./icml_main/:${icmlDir}/:./latex/:${latexDir}/:` + (process.env.BSTINPUTS || '')
    };
    exec(cmd, { cwd: TEX_ROOT, maxBuffer: 16 * 1024 * 1024, env: texEnv }, (err, stdout, stderr) => {
      compiling = false;

      const ok = !err && fs.existsSync(PDF_PATH);
      broadcast('compiled', { success: ok, tag, logTail: (stdout + '\n' + stderr).slice(-6000) });

      if (ok) {
        console.log('✅ Compilation successful');
      } else {
        console.error('❌ Compilation failed:\n', stdout, '\n', stderr);
      }
    });
  }

  // 首次编译
  runLatexmk('startup');

  // —— 监听 .tex/.bib/... 变更自动编译 —— //
  chokidar.watch(TEX_ROOT, {
    ignored: /(^|[/\\])(\.git|\.DS_Store)$/i,
    ignoreInitial: true,
    awaitWriteFinish: { stabilityThreshold: 200, pollInterval: 50 }
  }).on('all', (evt, file) => {
    if (/\.(tex|bib|sty|cls|png|jpg|pdf|eps|svg)$/i.test(file)) runLatexmk('watch');
  });

  // —— 提供 PDF 给 pdf.js viewer —— //
  app.get('/pdf', (req, res) => {
    if (!fs.existsSync(PDF_PATH)) return res.status(404).send('PDF not found');
    res.setHeader('Cache-Control', 'no-store');
    res.sendFile(PDF_PATH);
  });

  // —— 一键编译（按钮 + ⌘⇧R）—— //
  app.post('/api/compile', (req, res) => { runLatexmk('manual'); res.json({ ok: true }); });

  // —— 下载 tex 目录压缩包 —— //
  app.get('/api/download-tex', (req, res) => {
    const zipPath = path.join(TMP_DIR, 'tex.zip');

    // 使用 zip 命令压缩 tex 目录
    exec(`cd "${ROOT}" && zip -r "${zipPath}" tex -x "*.aux" "*.log" "*.out" "*.toc" "*.bbl" "*.blg" "*.fls" "*.fdb_latexmk" "*.synctex.gz"`, (err) => {
      if (err) {
        console.error('压缩失败:', err);
        return res.status(500).send('压缩失败');
      }

      // 发送文件
      res.download(zipPath, 'tex.zip', (downloadErr) => {
        if (downloadErr) {
          console.error('下载失败:', downloadErr);
        }
        // 下载完成后删除临时文件
        fs.unlink(zipPath, (unlinkErr) => {
          if (unlinkErr) console.error('删除临时文件失败:', unlinkErr);
        });
      });
    });
  });

  // —— 图片上传接口 —— //
  const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, TMP_DIR);
    },
    filename: (req, file, cb) => {
      const timestamp = Date.now();
      const ext = path.extname(file.originalname) || '.png';
      cb(null, `screenshot-${timestamp}${ext}`);
    }
  });
  const upload = multer({
    storage,
    limits: { fileSize: 10 * 1024 * 1024 } // 10MB 限制
  });

  app.post('/api/upload-image', upload.single('image'), (req, res) => {
    if (!req.file) {
      return res.status(400).send('No image uploaded');
    }
    const filePath = path.join(TMP_DIR, req.file.filename);
    res.json({ path: filePath });
  });

  // ======================
  //  阿里云百炼 翻译代理
  //  model: 通义千问-MT-Plus
  // ======================
  app.post('/api/qwen-translate', async (req, res) => {
    try {
      const apiKey = process.env.DASHSCOPE_API_KEY;
      if (!apiKey) return res.status(500).json({ ok: false, error: 'DASHSCOPE_API_KEY not set' });
      const { segments = [], source_lang = 'en', target_lang = 'zh' } = req.body || {};
      if (!Array.isArray(segments) || segments.length === 0) {
        return res.status(400).json({ ok: false, error: 'segments required' });
      }

      // 逐段调用，保证稳妥（可改为并发队列）
      const results = [];
      for (const seg of segments) {
        const text = String(seg.text || '').trim();
        if (!text) { results.push({ ...seg, translation: '' }); continue; }

        // Qwen-MT 专用翻译接口
        const base_url = process.env.DASHSCOPE_BASE || 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1';
        const model = process.env.DASHSCOPE_MODEL || 'qwen-mt-turbo';
        const url = `${base_url}/chat/completions`;
        const payload = {
          model: model,
          messages: [{ role: 'user', content: text }],
          translation_options: {
            source_lang: source_lang === 'en' ? 'English' : source_lang,
            target_lang: target_lang === 'zh' ? 'Chinese' : target_lang
          }
        };

        const r = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
          },
          body: JSON.stringify(payload)
        });
        if (!r.ok) {
          const t = await r.text();
          results.push({ ...seg, error: `HTTP ${r.status}: ${t}`, translation: '' });
          continue;
        }
        const data = await r.json();
        // Qwen-MT 返回格式: data.choices[0].message.content
        const translated = data?.choices?.[0]?.message?.content || '';

        results.push({ ...seg, translation: translated });
      }
      res.json({ ok: true, results });
    } catch (err) {
      console.error('qwen-translate error', err);
      res.status(500).json({ ok: false, error: String(err?.message || err) });
    }
  });

  // —— 源码片段读取（命中行高亮用）—— //
  // 安全修复: 限制只能访问 tex/ 和 scripts/ 目录，防止路径遍历攻击
  app.get('/api/source', (req, res) => {
    const f = req.query.file;
    const start = Number(req.query.start || 1);
    const end = Number(req.query.end || start);

    if (!f) return res.status(400).json({ error: 'file parameter required' });

    // 路径安全检查：只允许访问 tex/ 和 scripts/ 目录
    const realPath = path.resolve(f);
    const safePaths = [
      path.resolve(TEX_ROOT),
      path.resolve(ROOT, 'scripts')
    ];

    const isAllowed = safePaths.some(safe => realPath.startsWith(safe + path.sep) || realPath === safe);
    if (!isAllowed) {
      console.warn(`🚫 [Security] 阻止路径遍历尝试: ${f} -> ${realPath} (来自 IP: ${req.ip})`);
      return res.status(403).json({ error: 'access denied' });
    }

    if (!fs.existsSync(realPath)) return res.status(404).json({ error: 'file not found' });

    try {
      const lines = fs.readFileSync(realPath, 'utf8').split(/\r?\n/);
      const s = Math.max(1, start), e = Math.min(lines.length, end);
      const snippet = [];
      for (let i = s; i <= e; i++) snippet.push(`${i}: ${lines[i - 1]}`);
      res.json({ snippet: snippet.join('\n') });
    } catch (e) {
      res.status(500).json({ error: 'read failed' });
    }
  });

  // —— 反查接口：只使用 SyncTeX —— //
  app.post('/api/revsearch', async (req, res) => {
    const { selectedText, page, h, v, endPage, endH, endV, useSyncTexOnly } = req.body || {};

    // 必须有页码和坐标
    if (!page || !Number.isFinite(h) || !Number.isFinite(v)) {
      return res.status(400).json({ error: '需要页码和坐标信息' });
    }

    console.log('Received coordinates:', {
      start: `Page ${page}: (${h}, ${v})`,
      end: endPage ? `Page ${endPage}: (${endH}, ${endV})` : 'Not provided',
      selectedText: selectedText?.substring(0, 50)
    });

    if (!fs.existsSync(PDF_PATH)) {
      return res.status(404).json({ error: 'PDF 文件不存在' });
    }

    // 检查 synctex 文件是否存在（从 MAIN_TEX 推断文件名）
    const mainBaseName = path.basename(MAIN_TEX, '.tex');
    const synctexPath = path.join(OUT_DIR, `${mainBaseName}.synctex.gz`);
    if (!fs.existsSync(synctexPath)) {
      return res.status(404).json({ error: 'SyncTeX 文件不存在，请重新编译' });
    }

    // 如果提供了结束坐标，先获取结束位置的行号
    let endLineNumber = null;
    if (endPage && Number.isFinite(endH) && Number.isFinite(endV)) {
      const endArgs = ['edit', '-o', `${endPage}:${endH}:${endV}:${PDF_PATH}`];
      console.log(`SyncTeX end command: synctex ${endArgs.join(' ')}`);

      try {
        const endResult = execFileSync('synctex', endArgs, { cwd: TEX_ROOT, maxBuffer: 2 * 1024 * 1024 });
        const endOutput = endResult.toString();
        const endLineMatch = endOutput.match(/Line:\s*(\d+)/i);
        if (endLineMatch) {
          endLineNumber = parseInt(endLineMatch[1]);
          console.log(`📍 结束行号: ${endLineNumber}`);
        }
      } catch (err) {
        console.error('Failed to get end position:', err);
      }
    }

    // 使用 SyncTeX 查找起始位置
    const args = ['edit', '-o', `${page}:${h}:${v}:${PDF_PATH}`];

    console.log(`SyncTeX start command: synctex ${args.join(' ')}`);

    execFile('synctex', args, { cwd: TEX_ROOT, maxBuffer: 2 * 1024 * 1024 }, (err, stdout, stderr) => {
      if (err) {
        console.error('SyncTeX error:', err, stderr);
        return res.status(500).json({
          error: 'SyncTeX 执行失败',
          details: stderr || err.message
        });
      }

      console.log('SyncTeX output:', stdout);

      // 解析 SyncTeX 输出
      const lines = stdout.split('\n');
      let file = null;
      let line = null;
      let column = null;

      for (const l of lines) {
        const inputMatch = l.match(/Input:\s*(.+\.tex)/i);
        if (inputMatch) {
          file = inputMatch[1].trim();
          // 处理相对路径
          if (!path.isAbsolute(file)) {
            file = path.join(TEX_ROOT, file);
          }
        }

        const lineMatch = l.match(/Line:\s*(\d+)/i);
        if (lineMatch) {
          line = Number(lineMatch[1]);
          console.log(`📍 起始行号: ${line}`);
        }

        const colMatch = l.match(/Column:\s*(\d+)/i);
        if (colMatch) {
          column = Number(colMatch[1]);
        }
      }

      if (file && line) {
        // 如果没有结束行号，使用起始行号
        if (!endLineNumber) {
          endLineNumber = line;
        }

        // 读取上下文
        const snippet = readContext(file, line);

        // 提取对应的 LaTeX 文本
        let latexText = '';
        let lineStart = line;
        let lineEnd = endLineNumber || line;
        let lineRange = undefined;

        try {
          const fileContent = fs.readFileSync(file, 'utf8');
          const lines = fileContent.split(/\r?\n/);

          if (line > 0 && line <= lines.length) {
            // 如果有明确的起始和结束行号，直接提取
            if (endLineNumber && endLineNumber >= line) {
              const startIdx = line - 1;
              const endIdx = Math.min(endLineNumber - 1, lines.length - 1);

              console.log(`📄 提取范围: 第 ${line} 行到第 ${endLineNumber} 行`);

              // 提取文本
              const latexLines = [];
              for (let i = startIdx; i <= endIdx; i++) {
                latexLines.push(lines[i]);
              }

              latexText = latexLines.join('\n').trim();
              lineStart = line;
              lineEnd = endLineNumber;
              lineRange = lineStart !== lineEnd ? `${lineStart}-${lineEnd}` : `${lineStart}`;
            } else {
              // 双击或单行选择情况，返回当前行
              latexText = lines[line - 1] || '';
              lineStart = line;
              lineEnd = line;
            }
          }
        } catch (err) {
          console.error('Error reading LaTeX text:', err);
        }

        const base = {
          method: 'synctex',
          file,
          line,
          column,
          snippet,
          latexText,  // 添加 LaTeX 原文
          lineStart,  // 起始行
          lineEnd,    // 结束行
          lineRange,  // 行号范围
          debug: {
            command: `synctex ${args.join(' ')}`,
            page, h, v
          }
        };

        // 如果只使用 SyncTeX，直接返回结果
        const sel = (selectedText || '').trim();
        if (useSyncTexOnly) {
          return res.json(base);
        }

        // 如果提供了选中文本，但上下文不包含它，则用 Python 候选增强一次
        if (sel && !normText(snippet).includes(normText(sel))) {
          console.log('SyncTeX context doesn\'t contain selected text, trying Python revsearch...');
          runPythonRevsearch({
            PDF_PATH: PDF_PATH,
            TEX_ROOT: TEX_ROOT,
            OUT_DIR: OUT_DIR,
            page,
            h,
            v,
            selectedText: sel
          }).then(r => {
            if (r && r.candidates) {
              // 返回候选列表，添加 LaTeX 文本
              const enrichedCandidates = r.candidates.map(c => {
                const snippet = readContext(c.file, c.line);
                let latexText = '';
                try {
                  const fileContent = fs.readFileSync(c.file, 'utf8');
                  const lines = fileContent.split(/\r?\n/);
                  if (c.line > 0 && c.line <= lines.length) {
                    const startLine = Math.max(0, c.line - 1);
                    const endLine = Math.min(lines.length, c.line + 10);
                    const latexLines = [];
                    for (let i = startLine; i < endLine; i++) {
                      const currentLine = lines[i];
                      latexLines.push(currentLine);
                      if (currentLine.trim() === '' && latexLines.length > 1) break;
                      if (i > startLine && currentLine.match(/^\\(section|subsection|begin|end|item)/)) break;
                    }
                    latexText = latexLines.join('\n').trim();
                  }
                } catch {}
                return { ...c, snippet, latexText };
              });
              return res.json({ candidates: enrichedCandidates });
            } else if (r && r.file && r.line) {
              const snip2 = readContext(r.file, r.line);
              let latexText = '';
              try {
                const fileContent = fs.readFileSync(r.file, 'utf8');
                const lines = fileContent.split(/\r?\n/);
                if (r.line > 0 && r.line <= lines.length) {
                  const startLine = Math.max(0, r.line - 1);
                  const endLine = Math.min(lines.length, r.line + 10);
                  const latexLines = [];
                  for (let i = startLine; i < endLine; i++) {
                    const currentLine = lines[i];
                    latexLines.push(currentLine);
                    if (currentLine.trim() === '' && latexLines.length > 1) break;
                    if (i > startLine && currentLine.match(/^\\(section|subsection|begin|end|item)/)) break;
                  }
                  latexText = latexLines.join('\n').trim();
                }
              } catch {}
              return res.json({
                method: r.method || 'synctex+fuzzy',
                file: r.file,
                line: r.line,
                column: r.column,
                score: r.score,
                snippet: snip2,
                latexText,
                debug: base.debug
              });
            }
            return res.json(base);
          }).catch(() => res.json(base));
          return;
        }
        return res.json(base);
      }

      // 如果没有找到，尝试调整坐标
      console.log('First attempt failed, trying with adjusted coordinates...');

      // 尝试多个点（上下左右偏移）
      // 增加更多偏移点，特别是针对列表项
      const offsets = [
        [0, 0],     // 再试一次原始点
        [0, -3], [0, 3], [-3, 0], [3, 0],     // 很小的偏移
        [0, -5], [0, 5], [-5, 0], [5, 0],     // 小偏移
        [0, -8], [0, 8], [-8, 0], [8, 0],     // 中等偏移
        [0, -10], [0, 10], [-10, 0], [10, 0], // 大偏移
        [-5, -5], [5, 5], [-5, 5], [5, -5],   // 对角偏移
        [0, -15], [0, 15], [-15, 0], [15, 0], // 更大偏移（针对列表缩进）
      ];

      let attempts = 0;
      const tryOffset = () => {
        if (attempts >= offsets.length) {
          return res.status(404).json({
            error: 'SyncTeX 无法定位到源文件',
            debug: { page, h, v, selectedText }
          });
        }

        const [dh, dv] = offsets[attempts++];
        const adjustedH = h + dh;
        const adjustedV = v + dv;
        const adjustedArgs = ['edit', '-o', `${page}:${adjustedH}:${adjustedV}:${PDF_PATH}`];

        execFile('synctex', adjustedArgs, { cwd: TEX_ROOT }, (err2, stdout2) => {
          if (!err2 && stdout2) {
            const inputMatch = stdout2.match(/Input:\s*(.+\.tex)/i);
            const lineMatch = stdout2.match(/Line:\s*(\d+)/i);

            if (inputMatch && lineMatch) {
              let file = inputMatch[1].trim();
              if (!path.isAbsolute(file)) {
                file = path.join(TEX_ROOT, file);
              }
              const line = Number(lineMatch[1]);
              const snippet = readContext(file, line);

              // 提取 latexText（与正常流程保持一致）
              let latexText = '';
              try {
                const fileContent = fs.readFileSync(file, 'utf8');
                const lines = fileContent.split(/\r?\n/);
                if (line > 0 && line <= lines.length) {
                  latexText = lines[line - 1] || '';
                }
              } catch (err) {
                console.error('Error reading LaTeX text:', err);
              }

              return res.json({
                method: 'synctex',
                file,
                line,
                snippet,
                latexText,
                adjusted: true,
                offset: [dh, dv]
              });
            }
          }
          tryOffset();
        });
      };

      tryOffset();
    });
  });

  function readContext(file, line) {
    try {
      const arr = fs.readFileSync(file, 'utf8').split(/\r?\n/);
      const s = Math.max(1, line - 2), e = Math.min(arr.length, line + 2);
      return Array.from({ length: e - s + 1 }, (_, i) => {
        const ln = s + i;
        return `${ln}: ${arr[ln - 1]}`;
      }).join('\n');
    } catch { return ''; }
  }

  function normText(s) {
    // 归一化文本：去除零宽字符，进行 NFKC 归一化
    return (s || '')
      .replace(/[\u200b\u200c\u200d\uFEFF]/g, '')
      .normalize('NFKC')
      .toLowerCase();
  }

  function runPythonRevsearch(payload) {
    return new Promise((resolve) => {
      const py = execFile('python3', [path.join(ROOT, 'scripts', 'revsearch.py')], { cwd: ROOT, maxBuffer: 16 * 1024 * 1024 }, (err, stdout) => {
        if (err || !stdout) return resolve(null);
        try { resolve(JSON.parse(stdout)); } catch { resolve(null); }
      });
      py.stdin.write(JSON.stringify(payload));
      py.stdin.end();
    });
  }

  // —— 交给 Next 处理其余路由（/ 等）—— //
  app.all('*', (req, res) => handle(req, res));

  // —— 启动 —— //
  const PORT = process.env.PORT || 2219;
  const server = app.listen(PORT, () => console.log(`✅ LaTeX Live on http://localhost:${PORT}`));

  // ================================
  // WebSocket 终端 - 使用 noServer 模式手动处理 upgrade
  // ================================
  const wssClaude = new WebSocket.Server({ noServer: true });
  const wssBash = new WebSocket.Server({ noServer: true });
  const wssBash2 = new WebSocket.Server({ noServer: true });

  function handleTerminalConnection(ws, defaultCmd, label) {
    console.log(`📡 [WebSocket] ${label} connected`);
    const shell = process.env.SHELL || (process.platform === 'win32' ? 'powershell.exe' : 'bash');
    const CMD = defaultCmd;
    const CWD = process.env.CLAUDE_CWD || ROOT;

    // 如果 CMD 为空，直接启动交互式 shell
    const args = process.platform === 'win32'
      ? []
      : (CMD ? ['-lc', CMD] : ['-l']);

    const p = pty.spawn(shell, args, {
      name: 'xterm-color',
      cols: 80,
      rows: 24,
      cwd: CWD,
      env: process.env  // 传递所有环境变量
    });

    try { ws.send(JSON.stringify({ type: 'status', data: `spawn: ${CMD || 'bash'} (cwd: ${CWD})` })); } catch {}

    p.onData((data) => {
      try { ws.send(JSON.stringify({ type: 'output', data })); } catch {}
    });
    p.onExit(() => { try { ws.close(); } catch {} });

    ws.on('message', (buf) => {
      let msg; try { msg = JSON.parse(buf.toString()); } catch { return; }
      if (msg?.type === 'input' && typeof msg.data === 'string') p.write(msg.data);
      if (msg?.type === 'resize' && msg.cols && msg.rows) try { p.resize(msg.cols, msg.rows); } catch {}
    });
    ws.on('close', () => { try { p.kill(); } catch {} });
  }

  wssClaude.on('connection', (ws) => handleTerminalConnection(ws, CLAUDE_CMD, 'Claude Terminal'));
  wssBash.on('connection', (ws) => handleTerminalConnection(ws, '', 'Bash Terminal'));
  wssBash2.on('connection', (ws) => handleTerminalConnection(ws, '', 'Bash Terminal 2'));

  // 手动处理 HTTP upgrade 请求 (带 Cookie 认证)
  server.on('upgrade', (request, socket, head) => {
    const url = new URL(request.url, `http://${request.headers.host}`);
    const pathname = url.pathname;

    // 从 cookie 中读取认证 token
    const cookies = parseCookies(request.headers.cookie);
    const authToken = cookies.auth_token;

    // Token 认证：使用 cookie 中的 auth_token
    if (authToken !== WS_ACCESS_TOKEN) {
      console.warn(`🚫 [Security] 拒绝未授权的 WebSocket 连接: ${pathname} (IP: ${request.socket.remoteAddress})`);
      socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
      socket.destroy();
      return;
    }

    if (pathname === '/ws/terminal') {
      console.log(`✅ [WebSocket] Claude Terminal 认证成功 (IP: ${request.socket.remoteAddress})`);
      wssClaude.handleUpgrade(request, socket, head, (ws) => {
        wssClaude.emit('connection', ws, request);
      });
    } else if (pathname === '/ws/terminal-codex') {
      console.log(`✅ [WebSocket] Bash Terminal 认证成功 (IP: ${request.socket.remoteAddress})`);
      wssBash.handleUpgrade(request, socket, head, (ws) => {
        wssBash.emit('connection', ws, request);
      });
    } else if (pathname === '/ws/terminal-bash2') {
      console.log(`✅ [WebSocket] Bash Terminal 2 认证成功 (IP: ${request.socket.remoteAddress})`);
      wssBash2.handleUpgrade(request, socket, head, (ws) => {
        wssBash2.emit('connection', ws, request);
      });
    } else {
      // 其他路径不处理，销毁连接
      socket.destroy();
    }
  });
});