'use client';
import { useEffect, useRef, useState, useMemo } from 'react';
import './globals.css';

// 反向代理子路径前缀（由 next.config.js 的 env 注入）。本地直接访问时为空字符串。
const BASE = process.env.NEXT_PUBLIC_APP_BASE_PATH || '';
// Claude 终端模型：空=用账号默认模型（已去掉写死的 Opus 4.5），可用 CLAUDE_MODEL 覆盖。
const CLAUDE_MODEL = process.env.NEXT_PUBLIC_CLAUDE_MODEL || '';

export default function Home() {
  const iframeRef = useRef(null);
  const leftRef = useRef(null);
  const bottomRef = useRef(null);        // 左下（终端容器）用于持久化高度
  const termHostRef = useRef(null);      // xterm 挂载点 (Claude)
  const termHostRef2 = useRef(null);     // xterm 挂载点 (Codex)
  const termRef = useRef(null);
  const termRef2 = useRef(null);
  const fitRef = useRef(null);
  const fitRef2 = useRef(null);
  const wsRef = useRef(null);
  const wsRef2 = useRef(null);
  // 第三个终端 (Bash 2)
  const termHostRef3 = useRef(null);
  const termRef3 = useRef(null);
  const fitRef3 = useRef(null);
  const wsRef3 = useRef(null);
  const [activeTerminal, setActiveTerminal] = useState(0); // 0: Claude, 1: Bash, 2: Bash 2
  const [cliStatus2, setCliStatus2] = useState('disconnected'); // Bash 终端状态
  const [cliStatus3, setCliStatus3] = useState('disconnected'); // Bash 2 终端状态
  const splitRef = useRef(null);         // 两栏容器
  const dividerRef = useRef(null);       // 分割条
  const draggingRef = useRef(false);
  const startXRef = useRef(0);
  const startWRef = useRef(0);

  // —— 解析与稳定性相关 —— //
  const parserRef = useRef(null);            // "Edit file" 解析状态
  const ansiCarryRef = useRef('');           // 尾部半个 ANSI 序列
  const resizeBusyRef = useRef(false);       // 终端正在 resize，暂停解析
  const resizeBusyTimerRef = useRef(null);
  const parserBufferRef = useRef('');        // 暂存输出，稳定后统一解析
  const finalizeTimerRef = useRef(null);     // 轻微 idle 延迟
  const fitRafRef = useRef(null);            // ResizeObserver 的 rAF 节流
  const editFileDataRef = useRef(null);      // ref备份，防止state被意外清空

  const [status, setStatus] = useState('waiting…');
  const [result, setResult] = useState('选择或双击 PDF 中的文本即可翻译');
  const [candidates, setCandidates] = useState(null);
  const [editFileData, _setEditFileData] = useState(null); // 存储 Edit file 数据

  // 安全的setter：resize期间禁止清空editFileData
  const setEditFileData = (data) => {
    if (data === null && resizeBusyRef.current) {
      console.log('[State] Blocked editFileData clear during resize');
      return; // 拒绝清空
    }
    editFileDataRef.current = data; // 同步更新ref
    _setEditFileData(data);
  };
  const [cliStatus, setCliStatus] = useState('disconnected'); // 终端连接状态
  const [askButton, setAskButton] = useState({ show: false, x: 0, y: 0 }); // Ask Ackleaf 按钮
  const currentResultRef = useRef(''); // 存储当前定位结果用于发送到终端
  const fileInputRef = useRef(null); // 文件上传input引用
  const outputBufferRef = useRef(''); // 累积终端输出
  const outputTimerRef = useRef(null); // 延迟显示定时器
  const [leftWidth, setLeftWidth] = useState(0);  // 左侧像素宽
  const [dragging, setDragging] = useState(false);
  const [terminalHeight, setTerminalHeight] = useState(0); // 终端高度
  const [vDragging, setVDragging] = useState(false); // 垂直拖拽状态
  const [isTranslating, setIsTranslating] = useState(false); // 翻译API请求中
  const [selectedTextTranslation, setSelectedTextTranslation] = useState(null); // 划词翻译结果
  const vDividerRef = useRef(null);
  const vDraggingRef = useRef(false);
  const startYRef = useRef(0);
  const startHRef = useRef(0);

  // ==============================
  //  增量 ANSI 清洗 + Edit file 解析
  // ==============================
  function stripAnsiIncremental(s) {
    // 先把上次没吃完的 ESC 序列拼上
    s = (ansiCarryRef.current || '') + (s || '');
    ansiCarryRef.current = '';
    if (!s) return '';

    // 1) 移除完整 CSI/SGR 序列：ESC [ ... <final>
    s = s.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, '');
    // 2) 移除完整 OSC 序列：ESC ] ... (BEL|ST)
    s = s.replace(/\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g, '');

    // 3) 若末尾出现未闭合的 ESC 开头，则截断为 carry
    const lastEsc = s.lastIndexOf('\x1b');
    if (lastEsc !== -1) {
      const tail = s.slice(lastEsc);
      // 检查是否是完整的ANSI序列：必须匹配完整格式 ESC[参数字母 或 ESC]...BEL/ST
      const hasTerm = /^\[[\d;?]*[ -\/]*[@-~]/.test(tail.slice(1)) ||
                      /^\][^\x07\x1b]*(?:\x07|\x1b\\)/.test(tail.slice(1));
      if (!hasTerm) {
        ansiCarryRef.current = tail;
        s = s.slice(0, lastEsc);
      }
    }

    // 4) 清理被"截断 ESC"遗留的 SGR 碎片，如 "[48;5;72m"、"[39m"
    s = s.replace(/\[[0-9;?]*m/g, '');
    // 5) 去掉其它控制字符（保留 \n \r \t）
    s = s.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '');
    return s;
  }

  function stripBorders(line){
    // 去掉框线/竖线字符，保留内容
    return line.replace(/[│╎╌─━│┃╭╮╰╯┌┐└┘]/g, '').trimEnd();
  }
  function ensureParser(){
    if (!parserRef.current){
      parserRef.current = {
        buf: '',
        capturing: false,
        file: '',
        lines: [],
        hunks: [],
        curHunk: null
      };
    }
    return parserRef.current;
  }
  function pushLineToHunks(state, kind, text){
    if (!state.curHunk) state.curHunk = { minus: [], plus: [] };
    if (kind === '-') state.curHunk.minus.push(text);
    if (kind === '+') state.curHunk.plus.push(text);
  }
  function sealHunkIfAny(state){
    if (state.curHunk && (state.curHunk.minus.length || state.curHunk.plus.length)){
      state.hunks.push(state.curHunk);
    }
    state.curHunk = null;
  }
  function finalizeBlockAndMirror(state){
    // 结束当前 hunk
    sealHunkIfAny(state);
    const file = state.file;

    // —— 关键：没有 hunk 的"空块"直接丢弃，避免覆盖现有结果 —— //
    if (!state.hunks || state.hunks.length === 0) {
      console.log('[Parser] No hunks found, skipping update');
      return;
    }

    // 组合翻译段
    const blocks = [];
    const segments = [];
    state.hunks.forEach((h, idx) => {
      const oldTxt = h.minus.join('\n').trim();
      const newTxt = h.plus.join('\n').trim();
      blocks.push({ idx, oldTxt, newTxt });
      if (oldTxt) segments.push({ idx, tag: '---', text: oldTxt, file });
      if (newTxt) segments.push({ idx, tag: '+++', text: newTxt, file });
    });

    console.log(`[Parser] Finalizing ${state.hunks.length} hunks for file: ${file}`);

    // —— 内容去重：检查是否和现有内容相同 —— //
    const existing = editFileDataRef.current;
    if (existing && existing.file === file && existing.blocks) {
      // 比较blocks内容是否完全相同
      if (existing.blocks.length === blocks.length) {
        const isSame = blocks.every((b, i) => {
          const eb = existing.blocks[i];
          return eb && eb.oldTxt === b.oldTxt && eb.newTxt === b.newTxt;
        });
        if (isSame) {
          console.log('[Parser] Content unchanged, skipping update to avoid re-translation');
          return; // 内容相同，跳过更新
        }
      }
    }

    // 先展示无翻译数据（setEditFileData会自动同步ref）
    const data = { file, blocks, translations: {} };
    setEditFileData(data);
    setResult(null);
    setCandidates(null);

    // 并行请求翻译（每个segment独立请求）
    if (segments.length){
      setIsTranslating(true); // 开始翻译

      // 并行发送所有翻译请求
      const promises = segments.map(seg =>
        translateSegments([seg]).catch(err => {
          console.error(`翻译失败 [${seg.idx}:${seg.tag}]:`, err);
          return {}; // 返回空对象，不中断其他翻译
        })
      );

      Promise.all(promises).then(results => {
        // 合并所有翻译结果
        const map = {};
        results.forEach(m => Object.assign(map, m));
        const dataWithTranslation = { file, blocks, translations: map };
        setEditFileData(dataWithTranslation);
        setIsTranslating(false); // 翻译完成
      }).catch(err => {
        console.error('翻译错误:', err);
        setStatus('翻译失败：' + String(err?.message || err));
        setIsTranslating(false); // 翻译失败也要停止
      });
    }
  }
  function keyOf(idx, tag){ return `${idx}:${tag}`; }

  async function translateSegments(segments){
    const resp = await fetch(`${BASE}/api/qwen-translate`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ segments, source_lang:'en', target_lang:'zh' })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    if (!data?.ok) throw new Error(data?.error || 'translate failed');
    const map = {};
    for (const r of data.results || []){
      map[keyOf(r.idx, r.tag)] = r.translation || '';
    }
    return map;
  }

  function feedClaudeParser(chunk){
    // resize期间：只缓存，完全不碰解析器（完全冻结）
    if (resizeBusyRef.current) {
      parserBufferRef.current += chunk;
      console.log('[Parser] Resize busy, buffering chunk:', chunk.length, 'bytes');
      return; // 直接返回，不进行任何解析
    }

    const state = ensureParser();
    // 使用增量 ANSI 清洗，保留换行
    state.buf += stripAnsiIncremental(chunk);
    let i;
    while ((i = state.buf.indexOf('\n')) >= 0){
      const raw = state.buf.slice(0, i+1);
      state.buf = state.buf.slice(i+1);
      const line = stripBorders(raw.replace(/\r/g,''));
      handleParsedLine(state, line);
    }
  }

  function handleParsedLine(state, line){
    const trimmed = line.trim();
    if (!state.capturing){
      // 进入块：Edit file <path>
      const m = trimmed.match(/^Edit file\s+(.+)\s*$/i);
      if (m){
        console.log('[Parser] Detected Edit file:', m[1]);
        state.capturing = true;
        state.file = m[1].trim();
        state.lines = [];
        state.hunks = [];
        state.curHunk = null;
      }
      return;
    }

    // 判断结束：Do you want ...
    if (/^Do you want to make this edit to/i.test(trimmed)){
      console.log('[Parser] Detected end, scheduling finalize with 120ms delay');
      // 轻微 idle：再等 120ms，吃完可能落后的尾包
      const snap = {
        file: state.file,
        hunks: state.hunks.slice(),
        lines: state.lines.slice()
      };
      if (finalizeTimerRef.current) clearTimeout(finalizeTimerRef.current);
      finalizeTimerRef.current = setTimeout(() => {
        finalizeBlockAndMirror(snap);
      }, 120);
      // 重置状态机
      state.capturing = false;
      state.file = '';
      state.lines = [];
      state.hunks = [];
      state.curHunk = null;
      return;
    }

    // 记录原始行
    state.lines.push(trimmed);

    // 解析 diff 行（Claude 格式）
    // 匹配: "91 - text" 或 "   - text" (续行)
    const mm = trimmed.match(/^\s*(?:\d+\s+)?([+-])\s+(.*)$/);
    if (mm){
      const sign = mm[1];
      const text = mm[2];

      // 检测新 hunk：遇到"行号 + 符号"格式，且与当前 hunk 符号不同
      const hasLineNumber = /^\d+\s+[+-]/.test(trimmed);
      if (hasLineNumber && state.curHunk) {
        // 如果当前是 - 行，但 curHunk 已有 + 行，说明新 hunk 开始
        if (sign === '-' && state.curHunk.plus.length > 0) {
          sealHunkIfAny(state);
        }
      }

      pushLineToHunks(state, sign, text);
      return;
    }

    // git 风格
    if (/^---\s/.test(trimmed)){ sealHunkIfAny(state); state.curHunk = state.curHunk || { minus:[], plus:[] }; return; }
    if (/^\+\+\+\s/.test(trimmed)){ state.curHunk = state.curHunk || { minus:[], plus:[] }; return; }

    // 其他行（行号、空行、公式等）：
    // 只有在遇到新的行号（非 +/- 行）时，才封闭 hunk
    if (/^\d+\s+[^+-]/.test(trimmed) && state.curHunk && (state.curHunk.minus.length > 0 || state.curHunk.plus.length > 0)){
      sealHunkIfAny(state);
    }
  }

  // —— PDF 自动刷新 —— //
  // SSE 在反向代理（JupyterHub server-proxy）下会被整体缓冲而失效，所以以"轮询 PDF
  // 版本号(mtime)"作为稳妥的刷新驱动；SSE 仍保留用于本地即时触发。两者共用同一个
  // 按 mtime 去重的入口 checkAndReloadPdf，避免重复刷新。
  const lastPdfMtimeRef = useRef(0);

  function reloadPdfPreservingState() {
    try {
      const win = iframeRef.current?.contentWindow;
      const app = win?.PDFViewerApplication;
      const currentPage = app?.page || 1;
      const currentScale = app?.pdfViewer?.currentScale || 'page-width';
      const scrollTop = app?.pdfViewer?.container?.scrollTop || 0;
      const scrollLeft = app?.pdfViewer?.container?.scrollLeft || 0;
      if (iframeRef.current) {
        iframeRef.current.src = `${BASE}/pdfjs/web/viewer.html?file=${encodeURIComponent(`${BASE}/pdf`)}&zoom=page-fit&ts=${Date.now()}#pagemode=none`;
        const onIframeLoad = () => {
          setTimeout(() => {
            const newApp = iframeRef.current?.contentWindow?.PDFViewerApplication;
            if (newApp?.pdfViewer) {
              newApp.page = currentPage;
              newApp.pdfViewer.currentScale = currentScale;
              if (newApp.pdfViewer.container) {
                newApp.pdfViewer.container.scrollTop = scrollTop;
                newApp.pdfViewer.container.scrollLeft = scrollLeft;
              }
            }
          }, 500);
          iframeRef.current?.removeEventListener('load', onIframeLoad);
        };
        iframeRef.current.addEventListener('load', onIframeLoad);
      }
    } catch (error) {
      console.error('Error during PDF reload:', error);
      if (iframeRef.current) {
        iframeRef.current.src = `${BASE}/pdfjs/web/viewer.html?file=${encodeURIComponent(`${BASE}/pdf`)}&zoom=page-fit&ts=${Date.now()}#pagemode=none`;
      }
    }
  }

  async function checkAndReloadPdf() {
    try {
      const r = await fetch(`${BASE}/api/pdf-version`, { cache: 'no-store' });
      if (!r.ok) return;
      const { mtime } = await r.json();
      if (!mtime) return;
      if (lastPdfMtimeRef.current === 0) { lastPdfMtimeRef.current = mtime; setStatus('ready ✅'); return; } // 首次仅建立基线
      if (mtime > lastPdfMtimeRef.current) {
        lastPdfMtimeRef.current = mtime;
        setStatus('compiled ✅');
        reloadPdfPreservingState();
      }
    } catch {}
  }

  // 轮询 PDF 版本（在任何代理下都可靠），1.5s 一次
  useEffect(() => {
    checkAndReloadPdf();
    const id = setInterval(() => { checkAndReloadPdf(); }, 1500);
    return () => clearInterval(id);
  }, []);

  // 连接 SSE：编译完成后刷新 PDF（保留当前页）- 支持自动重连（本地即时；代理下由轮询兜底）
  useEffect(() => {
    let es = null;
    let reconnectTimer = null;
    let isClosing = false;

    const connectSSE = () => {
      if (isClosing) return;

      es = new EventSource(`${BASE}/api/sse`);

      // 监听连接成功事件
      es.addEventListener('connected', (ev) => {
        console.log('✅ SSE connected');
        setStatus('ready ✅');
      });

      es.addEventListener('compiled', async (ev) => {
        console.log('📨 Received compiled event:', ev.data);
        const data = JSON.parse(ev.data || '{}');
        setStatus(data.success ? 'compiled ✅' : 'compile error ❌（看 server 日志）');
        if (data.success) {
          // 稍等确保 PDF 落盘，然后走统一的按 mtime 去重的刷新入口
          await new Promise(resolve => setTimeout(resolve, 300));
          checkAndReloadPdf();
        }
      });

      es.onerror = (error) => {
        // 反向代理下 SSE 会被缓冲而持续报错；此时由轮询兜底，这里不再覆盖状态文案
        console.warn('SSE error (polling will cover refresh):', error?.type || error);
        es.close();

        // 自动重连（3秒后）
        if (!isClosing) {
          reconnectTimer = setTimeout(() => {
            console.log('🔄 SSE reconnecting...');
            connectSSE();
          }, 3000);
        }
      };

      console.log('SSE connected to /api/sse');
    };

    connectSSE();

    return () => {
      console.log('SSE closing');
      isClosing = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (es) es.close();
    };
  }, []);

  // 一键编译：按钮 + ⌘⇧R
  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && (e.key === 'R' || e.key === 'r')) {
        e.preventDefault();
        fetch(`${BASE}/api/compile`, { method: 'POST' });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // PDF侧边栏通过URL参数 #pagemode=none 控制，无需额外代码

  // 左侧面板宽度可拖拽，并持久化
  useEffect(() => {
    const el = leftRef.current;
    if (!el) return;
    const w = localStorage.getItem('leftPaneWidth');
    if (w) el.style.width = w;
    const onMouseUp = () => {
      if (leftRef.current) {
        const width = getComputedStyle(leftRef.current).width;
        localStorage.setItem('leftPaneWidth', width);
      }
    };
    window.addEventListener('mouseup', onMouseUp);
    return () => window.removeEventListener('mouseup', onMouseUp);
  }, []);

  // 左下终端尺寸监听（高度固定为50%，不再需要持久化）
  useEffect(() => {
    const onResize = () => {
      // 终端尺寸可能变化，触发 fit 与 resize
      try {
        fitRef.current?.fit();
        sendResize();
      } catch {}
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // 点击页面其他地方时隐藏 Ask Ackleaf 按钮
  useEffect(() => {
    const hideButton = (e) => {
      // 如果点击的不是按钮本身，隐藏按钮
      if (askButton.show && !e.target.closest('button')) {
        setAskButton({ show: false, x: 0, y: 0 });
      }
    };
    window.addEventListener('mousedown', hideButton);
    return () => window.removeEventListener('mousedown', hideButton);
  }, [askButton.show]);

  // iframe 加载后，监听双击和选择事件
  useEffect(() => {
    let isScrolling = false;
    let scrollTimeout = null;

    const onLoad = () => {
      const win = iframeRef.current?.contentWindow;
      const doc = win?.document;
      if (!doc) {
        console.error('Cannot access iframe document');
        return;
      }

      console.log('Setting up event handlers for iframe');

      // 监听滚动事件
      doc.addEventListener('scroll', () => {
        isScrolling = true;
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(() => {
          isScrolling = false;
        }, 150);
      }, true);

      // 处理文本选择事件
      const handleSelection = () => {
        // 如果正在滚动，忽略选择事件
        if (isScrolling) {
          console.log('Ignoring selection during scroll');
          return;
        }

        const sel = win.getSelection();
        if (!sel || sel.isCollapsed || String(sel).trim() === '') return;

        // 获取选中文本
        const text = String(sel).trim();
        if (text.length < 2) return; // 忽略太短的选择

        // 限制选择长度，避免意外选中整页
        if (text.length > 5000) {
          console.log('Selection too long, ignoring:', text.length);
          return;
        }

        // 获取选择的第一个范围
        const range = sel.getRangeAt(0);

        // 获取选择起始和结束位置
        const startContainer = range.startContainer;
        const startOffset = range.startOffset;
        const endContainer = range.endContainer;
        const endOffset = range.endOffset;

        // 创建临时范围来获取起始位置（第一个字符）
        const startRange = doc.createRange();
        startRange.setStart(startContainer, startOffset);
        startRange.setEnd(startContainer, Math.min(startOffset + 1, startContainer.length || startContainer.textContent.length));
        const startRect = startRange.getBoundingClientRect();

        // 创建临时范围来获取结束位置（最后一个字符）
        const endRange = doc.createRange();
        endRange.setStart(endContainer, Math.max(0, endOffset - 1));
        endRange.setEnd(endContainer, endOffset);
        const endRect = endRange.getBoundingClientRect();

        // 找到包含选中内容的 PDF 页面
        let pageEl = startContainer;
        if (pageEl.nodeType === Node.TEXT_NODE) {
          pageEl = pageEl.parentElement;
        }
        while (pageEl && !pageEl.classList?.contains('page')) {
          pageEl = pageEl.parentElement;
        }

        if (!pageEl) {
          console.error('Cannot find page element');
          return;
        }

        const pageNum = parseInt(pageEl.dataset.pageNumber);
        const app = win.PDFViewerApplication;
        const pageView = app?.pdfViewer?.getPageView(pageNum - 1);

        if (!pageView?.viewport) {
          console.error('Cannot get page viewport');
          return;
        }

        // 获取页面相对坐标
        const pageRect = pageEl.getBoundingClientRect();

        // 使用选择起始位置的中心点
        const cx = (startRect.left + startRect.right) / 2;
        const cy = (startRect.top + startRect.bottom) / 2;

        // 计算相对于 PDF 页面的坐标
        const x = cx - pageRect.left;
        const y = cy - pageRect.top;

        // PDF.js convertToPdfPoint 转换屏幕坐标到 PDF 坐标
        let [pdfX, pdfY] = pageView.viewport.convertToPdfPoint(x, y);

        // 获取页面高度
        const pageHeight = pageView.viewport.viewBox[3];

        // SyncTeX 期望的 Y 坐标是从页面底部开始的
        const syncX = Math.max(1, Math.round(pdfX));
        const syncY = Math.max(1, Math.round(pageHeight - pdfY));

        // 同时计算结束位置的坐标
        let endPageNum = pageNum;
        let endPageEl = endContainer;
        if (endPageEl.nodeType === Node.TEXT_NODE) {
          endPageEl = endPageEl.parentElement;
        }
        while (endPageEl && !endPageEl.classList?.contains('page')) {
          endPageEl = endPageEl.parentElement;
        }
        if (endPageEl) {
          endPageNum = parseInt(endPageEl.dataset.pageNumber);
        }

        // 计算结束位置的坐标
        let endSyncX = syncX;
        let endSyncY = syncY;
        if (endPageNum === pageNum && endRect) {
          const endPageView = app?.pdfViewer?.getPageView(endPageNum - 1);
          if (endPageView?.viewport) {
            const endPageRect = endPageEl.getBoundingClientRect();
            const endCx = (endRect.left + endRect.right) / 2;
            const endCy = (endRect.top + endRect.bottom) / 2;
            const endX = endCx - endPageRect.left;
            const endY = endCy - endPageRect.top;
            let [endPdfX, endPdfY] = endPageView.viewport.convertToPdfPoint(endX, endY);
            endSyncX = Math.max(1, Math.round(endPdfX));
            endSyncY = Math.max(1, Math.round(pageHeight - endPdfY));
          }
        }

        console.log('Selection:', text);
        console.log(`Page ${pageNum}: Start (${syncX}, ${syncY}) -> End (${endSyncX}, ${endSyncY})`);

        // 显示正在查找的提示
        setResult(`正在查找选中文本 "${text}" 的源码位置...`);

        const requestData = {
          selectedText: text,
          page: pageNum,
          h: syncX,
          v: syncY,
          endPage: endPageNum,
          endH: endSyncX,
          endV: endSyncY,
          useSyncTexOnly: true
        };

        fetch(`${BASE}/api/revsearch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestData)
        })
        .then(async r => {
          const data = await r.json();
          if (!r.ok) {
            throw new Error(data.error || 'Request failed');
          }
          return data;
        })
        .then(obj => {
          if (obj?.candidates) {
            setCandidates(obj.candidates);
          } else if (obj?.file && obj?.line) {
            // 添加选中的文本到结果对象
            renderResult({ ...obj, selectedText: text });
          } else {
            setResult('未找到对应的 LaTeX 源码');
          }
        })
        .catch((err) => {
          console.error('Error:', err);
          setResult(`错误: ${err.message || '无法定位'}`);
        });
      };

      // 处理双击事件 - 直接查找源码
      const handleDoubleClick = (e) => {
        e.preventDefault();

        // 获取双击位置的单词
        const sel = win.getSelection();
        if (!sel || String(sel).trim() === '') return;

        // 获取双击选中的单词
        let text = String(sel).trim();
        console.log('Double-clicked word:', text);

        // 显示正在查找的提示
        setResult(`正在查找 "${text}" 的源码位置...`);

        // 获取选中单词的范围
        const range = sel.getRangeAt(0);

        // 找到包含选中内容的 PDF 页面
        let pageEl = range.commonAncestorContainer;
        if (pageEl.nodeType === Node.TEXT_NODE) {
          pageEl = pageEl.parentElement;
        }
        while (pageEl && !pageEl.classList?.contains('page')) {
          pageEl = pageEl.parentElement;
        }

        if (!pageEl) {
          console.error('Cannot find page element');
          setResult('无法找到页面元素');
          return;
        }

        const pageNum = parseInt(pageEl.dataset.pageNumber);
        const app = win.PDFViewerApplication;
        const pageView = app?.pdfViewer?.getPageView(pageNum - 1);

        if (!pageView?.viewport) {
          console.error('Cannot get page viewport');
          setResult('无法获取页面视图');
          return;
        }

        // 获取页面相对坐标
        const pageRect = pageEl.getBoundingClientRect();

        // 获取双击的确切位置（不依赖选中文本）
        // 使用点击事件的坐标会更准确
        const clickRect = range.getBoundingClientRect();

        // 获取所有被选中的矩形
        const rects = Array.from(range.getClientRects());

        // 用点击点命中对应的矩形；若找不到就退化为整个选区
        const hit = rects.find(r => (
          e.clientX >= r.left && e.clientX <= r.right &&
          e.clientY >= r.top && e.clientY <= r.bottom
        )) || clickRect;

        // 用命中的矩形中心作为定位点，更稳定
        const cx = (hit.left + hit.right) / 2;
        const cy = (hit.top + hit.bottom) / 2;

        // 计算相对于 PDF 页面的坐标
        const pageScale = pageView.viewport.scale;
        const pageRotation = pageView.viewport.rotation;

        // 获取点击点相对于页面的坐标（像素）
        const x = cx - pageRect.left;
        const y = cy - pageRect.top;

        // PDF.js convertToPdfPoint 转换屏幕坐标到 PDF 坐标
        // 返回的坐标是 PDF 坐标系（原点在左下）
        let [pdfX, pdfY] = pageView.viewport.convertToPdfPoint(x, y);

        // 获取页面高度
        const pageHeight = pageView.viewport.viewBox[3];

        // SyncTeX 期望的 Y 坐标是从页面底部开始的
        // 需要翻转 Y 坐标
        const syncX = Math.max(1, Math.round(pdfX));
        const syncY = Math.max(1, Math.round(pageHeight - pdfY));

        console.log('Click position:', { x: Math.round(x), y: Math.round(y) });
        console.log('Page scale:', pageScale);
        console.log('PDF coordinates:', { pdfX, pdfY });
        console.log('SyncTeX coordinates:', { syncX, syncY });

        console.log(`Page ${pageNum}: Screen(${Math.round(x)}, ${Math.round(y)}) -> PDF(${syncX}, ${syncY})`);
        console.log('Viewport:', pageView.viewport.viewBox);

        const requestData = {
          selectedText: text,
          page: pageNum,
          h: syncX,
          v: syncY,
          useSyncTexOnly: true  // 只使用 SyncTeX
        };
        console.log('Sending request:', requestData);

        fetch(`${BASE}/api/revsearch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestData)
        })
        .then(async r => {
          console.log('Response status:', r.status);
          const data = await r.json();
          console.log('Response data:', data);
          if (!r.ok) {
            throw new Error(data.error || 'Request failed');
          }
          return data;
        })
        .then(obj => {
          console.log('Processing result:', obj);
          if (obj?.candidates) {
            setCandidates(obj.candidates);
          } else if (obj?.file && obj?.line) {
            // 添加选中的文本到结果对象
            renderResult({ ...obj, selectedText: text });
          } else {
            console.error('Invalid response format:', obj);
            setResult('响应格式错误');
          }
        })
        .catch((err) => {
          console.error('Error details:', err);
          setResult(`错误: ${err.message || '未定位到 .tex'}`);
        });
      };

      // 监听双击事件
      doc.addEventListener('dblclick', handleDoubleClick);

      // 监听选择结束事件（鼠标释放）
      doc.addEventListener('mouseup', (e) => {
        // 延迟一下以确保选择完成
        setTimeout(() => {
          // 如果正在滚动，不处理
          if (isScrolling) return;

          handleSelection();
          // 如果有选中文本，显示 Ask Ackleaf 按钮
          const sel = win.getSelection();
          const selText = sel ? String(sel).trim() : '';
          if (selText !== '' && selText.length <= 5000) {
            // 计算按钮位置（相对于视口）
            const range = sel.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            const iframeRect = iframeRef.current.getBoundingClientRect();
            setAskButton({
              show: true,
              x: iframeRect.left + rect.right + 10,
              y: iframeRect.top + rect.top
            });
          } else {
            setAskButton({ show: false, x: 0, y: 0 });
          }
        }, 50);
      });

      console.log('Event handlers attached successfully');
    };

    const iframe = iframeRef.current;
    if (iframe) {
      // 尝试立即设置（如果已经加载）
      setTimeout(() => {
        onLoad();
      }, 1000);

      // 也监听 load 事件
      iframe.addEventListener('load', onLoad);
      return () => {
        iframe.removeEventListener('load', onLoad);
        if (scrollTimeout) clearTimeout(scrollTimeout);
      };
    }
  }, []);

  function renderResult(obj) {
    if (!obj) return;
    const lines = [];

    // Markdown 风格标题
    lines.push('## LaTeX to Modify');

    if (obj.file) {
      // 显示绝对路径
      lines.push(`- **File**: ${obj.file}`);
    }

    // 显示行号或行号范围
    if (obj.lineRange) {
      lines.push(`- **Lines**: ${obj.lineRange}`);
    } else if (obj.line) {
      lines.push(`- **Line**: ${obj.line}`);
    }

    // 显示选中的 PDF 文本
    if (obj.selectedText) {
      lines.push('');
      lines.push('### Selected Text (PDF)');
      lines.push(obj.selectedText);
    }

    // 显示对应的 LaTeX 源码（限制最多200个字符）
    const sourceText = obj.latexText || obj.snippet;
    if (sourceText) {
      lines.push('');
      lines.push('### LaTeX Source');
      const maxLength = 200;
      const displayText = sourceText.length > maxLength
        ? sourceText.substring(0, maxLength) + '...'
        : sourceText;
      lines.push(displayText);
    }

    // 添加修改要求部分
    lines.push('');
    lines.push('### Modification Request');

    const text = lines.join('\n');
    setResult(text);
    // 不再清空 Edit file 数据，两者可以共存
    // setEditFileData(null); // 已删除：避免resize时误清空
    currentResultRef.current = text; // 保存用于发送到终端

    // 创建组合文本用于复制
    let clipboardText = '';
    if (obj.selectedText) {
      clipboardText += `[选中的文本]\n${obj.selectedText}\n\n`;
    }
    if (obj.latexText) {
      clipboardText += `[LaTeX 源码]\n${obj.latexText}`;
    } else if (obj.file && obj.line) {
      clipboardText = `${obj.file}:${obj.line}`;
    }

    if (clipboardText) {
      navigator.clipboard?.writeText(clipboardText.trim()).catch(() => {});
    }

    setCandidates(null);

    // 如果有选中的文本，调用翻译API
    if (obj.selectedText) {
      setEditFileData(null); // 清空 Edit file，显示划词翻译
      const metadata = {
        text: obj.selectedText,
        translation: null,
        file: obj.file,
        line: obj.line,
        lineRange: obj.lineRange
      };
      setSelectedTextTranslation(metadata);
      setIsTranslating(true);

      const segment = { idx: 0, tag: 'selected', text: obj.selectedText };
      translateSegments([segment]).then(map => {
        const translation = map['0:selected'] || '';
        setSelectedTextTranslation({ ...metadata, translation });
        setIsTranslating(false);
      }).catch(err => {
        console.error('划词翻译失败:', err);
        setSelectedTextTranslation({ ...metadata, translation: '翻译失败' });
        setIsTranslating(false);
      });
    } else {
      setSelectedTextTranslation(null); // 没有选中文本，清空划词翻译
    }
  }

  // —— 终端：WebSocket 连接 & xterm 初始化 —— //
  // 认证通过 cookie 自动处理，无需在 URL 中传递 token
  function wsURL(path = '/ws/terminal') {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${location.host}${BASE}${path}`;
  }
  function sendResize() {
    const ws = wsRef.current, term = termRef.current;
    if (!ws || !term) return;
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
      console.log('[Resize] Starting resize window, freezing parser');
      resizeBusyRef.current = true;
      if (resizeBusyTimerRef.current) clearTimeout(resizeBusyTimerRef.current);
      resizeBusyTimerRef.current = setTimeout(() => {
        console.log('[Resize] Resize window ended, unfreezing parser');
        resizeBusyRef.current = false;
        if (parserBufferRef.current) {
          const bufferSize = parserBufferRef.current.length;
          console.log('[Resize] Flushing buffered data:', bufferSize, 'bytes');
          feedClaudeParser(parserBufferRef.current);
          parserBufferRef.current = '';
        } else {
          console.log('[Resize] No buffered data to flush');
        }
      }, 500);
    }
  }
  function sendResize2() {
    const ws = wsRef2.current, term = termRef2.current;
    if (!ws || !term) return;
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
    }
  }
  function reconnectWS() {
    console.log('[WS] Attempting to connect to:', wsURL());
    try { wsRef.current?.close(); } catch {}
    setCliStatus('connecting');
    const ws = new WebSocket(wsURL('/ws/terminal'));
    wsRef.current = ws;
    ws.onopen = () => {
      console.log('[WS] Connected successfully');
      setCliStatus('connected');
      try { fitRef.current?.fit(); } catch {}
      sendResize();
    };
    ws.onmessage = (ev) => {
      console.log('[WS] Received message, length:', ev.data?.length);
      let data = ev.data;
      try {
        const msg = JSON.parse(ev.data);
        if (msg?.type === 'status') return;
        if (msg?.type === 'output') {
          data = msg.data;
          termRef.current?.write(data);
          termRef.current?.scrollToBottom();
        }
      } catch {
        termRef.current?.write(data);
        termRef.current?.scrollToBottom();
      }
      if (data) {
        try {
          feedClaudeParser(data);
        } catch (e) {
          console.error('parser error', e);
        }
      }
    };
    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
      setCliStatus('error');
    };
    ws.onclose = () => {
      console.log('[WS] Connection closed');
      setCliStatus('disconnected');
    };
  }
  function reconnectWS2() {
    console.log('[WS2] Attempting to connect to:', wsURL('/ws/terminal-codex'));
    try { wsRef2.current?.close(); } catch {}
    setCliStatus2('connecting');
    const ws = new WebSocket(wsURL('/ws/terminal-codex'));
    wsRef2.current = ws;
    ws.onopen = () => {
      console.log('[WS2] Connected successfully');
      setCliStatus2('connected');
      try { fitRef2.current?.fit(); } catch {}
      sendResize2();
    };
    ws.onmessage = (ev) => {
      let data = ev.data;
      try {
        const msg = JSON.parse(ev.data);
        if (msg?.type === 'status') return;
        if (msg?.type === 'output') {
          data = msg.data;
          termRef2.current?.write(data);
          termRef2.current?.scrollToBottom();
        }
      } catch {
        termRef2.current?.write(data);
        termRef2.current?.scrollToBottom();
      }
    };
    ws.onerror = (err) => {
      console.error('[WS2] Error:', err);
      setCliStatus2('error');
    };
    ws.onclose = () => {
      console.log('[WS2] Connection closed');
      setCliStatus2('disconnected');
    };
  }
  function sendResize3() {
    const ws = wsRef3.current, term = termRef3.current;
    if (!ws || !term) return;
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
    }
  }
  function reconnectWS3() {
    console.log('[WS3] Attempting to connect to:', wsURL('/ws/terminal-bash2'));
    try { wsRef3.current?.close(); } catch {}
    setCliStatus3('connecting');
    const ws = new WebSocket(wsURL('/ws/terminal-bash2'));
    wsRef3.current = ws;
    ws.onopen = () => {
      console.log('[WS3] Connected successfully');
      setCliStatus3('connected');
      try { fitRef3.current?.fit(); } catch {}
      sendResize3();
    };
    ws.onmessage = (ev) => {
      let data = ev.data;
      try {
        const msg = JSON.parse(ev.data);
        if (msg?.type === 'status') return;
        if (msg?.type === 'output') {
          data = msg.data;
          termRef3.current?.write(data);
          termRef3.current?.scrollToBottom();
        }
      } catch {
        termRef3.current?.write(data);
        termRef3.current?.scrollToBottom();
      }
    };
    ws.onerror = (err) => {
      console.error('[WS3] Error:', err);
      setCliStatus3('error');
    };
    ws.onclose = () => {
      console.log('[WS3] Connection closed');
      setCliStatus3('disconnected');
    };
  }
  function sendInput(s) {
    const ws = activeTerminal === 0 ? wsRef.current : activeTerminal === 1 ? wsRef2.current : wsRef3.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'input', data: s }));
  }

  // 处理 Ask Ackleaf 按钮点击
  function handleAskClaude() {
    const text = currentResultRef.current;
    if (text && text.trim() !== '') {
      // 发送到终端
      sendInput(text + '\n');
      // 隐藏按钮
      setAskButton({ show: false, x: 0, y: 0 });
      // 聚焦终端
      setTimeout(() => {
        termRef.current?.focus();
      }, 100);
    }
  }

  // 处理文件上传
  async function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file || !file.type.startsWith('image/')) {
      return;
    }

    const term = termRef.current;
    if (!term) return;

    try {
      const formData = new FormData();
      formData.append('image', file);
      const res = await fetch(`${BASE}/api/upload-image`, {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const { path } = await res.json();
        // 只发送路径到终端，后面加一个空格方便继续输入
        sendInput(path + ' ');
      } else {
        const error = await res.text();
        // 上传失败时显示错误（通过 shell 的 echo）
        sendInput(`echo "❌ Upload failed: ${error}"\n`);
      }
    } catch (err) {
      sendInput(`echo "❌ Upload error: ${err.message}"\n`);
    }

    // 重置input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }

  // —— 自定义水平拖拽：初始化宽度 + 监听窗口尺寸 —— //
  function clampWidth(px){
    const total = splitRef.current?.clientWidth || window.innerWidth;
    const minLeft = 300;                        // 左侧最小
    const minRight = 360;                       // 右侧最小
    const maxLeft = Math.min(total - minRight, Math.round(total * 0.8));
    return Math.max(minLeft, Math.min(maxLeft, px));
  }
  useEffect(() => {
    const total = splitRef.current?.clientWidth || window.innerWidth;
    const saved = parseFloat(localStorage.getItem('leftPaneWidthPxV2') || 'NaN');
    const initial = clampWidth(Number.isFinite(saved) ? saved : Math.round(total * 0.40));
    setLeftWidth(initial);
  }, []);
  useEffect(() => {
    const onResize = () => setLeftWidth(w => clampWidth(w || 0));
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // —— 自定义垂直拖拽：调整终端高度 —— //
  function clampHeight(px){
    const total = leftRef.current?.clientHeight || window.innerHeight - 56;
    const minTerminal = 160;                    // 终端最小高度
    const minTop = 200;                         // 上方最小高度
    const maxTerminal = Math.max(minTerminal, total - minTop);
    return Math.max(minTerminal, Math.min(maxTerminal, px));
  }
  useEffect(() => {
    const total = leftRef.current?.clientHeight || window.innerHeight - 56;
    const saved = parseFloat(localStorage.getItem('terminalHeightPx') || 'NaN');
    const initial = clampHeight(Number.isFinite(saved) ? saved : Math.round(total * 0.34));
    setTerminalHeight(initial);
  }, []);
  useEffect(() => {
    const onResize = () => setTerminalHeight(h => clampHeight(h || 0));
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // 垂直拖拽事件处理
  let vRafId = null;
  function onVDividerPointerDown(e){
    e.preventDefault();
    const curH = bottomRef.current?.getBoundingClientRect().height || terminalHeight || 0;
    startYRef.current = e.clientY;
    startHRef.current = curH;
    vDraggingRef.current = true;
    setVDragging(true);
    document.body.classList.add('v-dragging');
    vDividerRef.current?.classList.add('dragging');
    window.addEventListener('pointermove', onVDividerPointerMove);
    window.addEventListener('pointerup', onVDividerPointerUp, { once: true });
    window.addEventListener('pointercancel', onVDividerPointerUp, { once: true });
  }
  function onVDividerPointerMove(e){
    if (!vDraggingRef.current) return;
    if (vRafId) return;
    vRafId = requestAnimationFrame(() => {
      vRafId = null;
      const delta = e.clientY - startYRef.current;
      setTerminalHeight(clampHeight(startHRef.current + delta)); // 终端在上：向下拖终端变大
    });
  }
  function onVDividerPointerUp(){
    vDraggingRef.current = false;
    setVDragging(false);
    document.body.classList.remove('v-dragging');
    vDividerRef.current?.classList.remove('dragging');
    const h = bottomRef.current?.getBoundingClientRect().height;
    if (h) localStorage.setItem('terminalHeightPx', String(Math.round(h)));
    window.removeEventListener('pointermove', onVDividerPointerMove);
  }

  // 开始/进行/结束拖拽（使用 Pointer 事件 + rAF 节流）
  let rafId = null;
  function onDividerPointerDown(e){
    e.preventDefault();
    const curW = leftRef.current?.getBoundingClientRect().width || leftWidth || 0;
    startXRef.current = e.clientX;
    startWRef.current = curW;
    draggingRef.current = true;
    setDragging(true);
    document.body.classList.add('dragging');
    dividerRef.current?.classList.add('dragging');
    window.addEventListener('pointermove', onDividerPointerMove);
    window.addEventListener('pointerup', onDividerPointerUp, { once: true });
    window.addEventListener('pointercancel', onDividerPointerUp, { once: true });
  }
  function onDividerPointerMove(e){
    if (!draggingRef.current) return;
    if (rafId) return;
    rafId = requestAnimationFrame(() => {
      rafId = null;
      const delta = e.clientX - startXRef.current;
      setLeftWidth(clampWidth(startWRef.current - delta));  // 反转：向右拖面板变窄
    });
  }
  function onDividerPointerUp(){
    draggingRef.current = false;
    setDragging(false);
    document.body.classList.remove('dragging');
    dividerRef.current?.classList.remove('dragging');
    // 持久化当前像素宽
    const w = leftRef.current?.getBoundingClientRect().width;
    if (w) localStorage.setItem('leftPaneWidthPxV2', String(Math.round(w)));
    window.removeEventListener('pointermove', onDividerPointerMove);
  }

  // 挂载 xterm（同时初始化两个终端）
  useEffect(() => {
    if (!termHostRef.current || !termHostRef2.current) return;

    // 保存清理函数的引用
    let cleanupFns = [];

    // 动态导入xterm以避免SSR问题
    (async () => {
      const { Terminal } = await import('xterm');
      const { FitAddon } = await import('xterm-addon-fit');

      const term = new Terminal({
        fontSize: 13,
        cursorBlink: true,
        convertEol: true,
        scrollback: 2000,
        theme: {
          background: '#ffffff',
          foreground: '#333333',
          cursor: '#333333',
          cursorAccent: '#ffffff',
          selectionBackground: 'rgba(0, 120, 215, 0.3)',
          // ANSI 颜色（亮色主题）
          black: '#000000',
          red: '#cd3131',
          green: '#00bc00',
          yellow: '#949800',
          blue: '#0451a5',
          magenta: '#bc05bc',
          cyan: '#0598bc',
          white: '#555555',
          brightBlack: '#666666',
          brightRed: '#cd3131',
          brightGreen: '#14ce14',
          brightYellow: '#b5ba00',
          brightBlue: '#0451a5',
          brightMagenta: '#bc05bc',
          brightCyan: '#0598bc',
          brightWhite: '#a5a5a5'
        }
      });
      const fit = new FitAddon();
      term.loadAddon(fit);
      term.open(termHostRef.current);
      try { fit.fit(); } catch {}
      termRef.current = term;
      fitRef.current = fit;

      // 输入 -> WS
      term.onData((data) => sendInput(data));

      // 自适应尺寸 - 使用 rAF 节流
      const onResize = () => {
        if (fitRafRef.current) return;
        fitRafRef.current = requestAnimationFrame(() => {
          fitRafRef.current = null;
          try { fit.fit(); sendResize(); } catch {}
        });
      };
      const ro = new ResizeObserver(onResize);
      ro.observe(termHostRef.current);
      window.addEventListener('resize', onResize);
      cleanupFns.push(() => {
        ro.disconnect();
        window.removeEventListener('resize', onResize);
        if (fitRafRef.current) cancelAnimationFrame(fitRafRef.current);
      });

      // 监听终端粘贴事件（全局监听，检查焦点）
      const onPaste = async (e) => {
        // 检查焦点是否在终端区域
        const activeElement = document.activeElement;
        const terminalContainer = termHostRef.current;
        if (!terminalContainer || !terminalContainer.contains(activeElement)) {
          return; // 焦点不在终端，不处理
        }

        const items = e.clipboardData?.items;
        if (!items) return;

        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          if (item.type.startsWith('image/')) {
            e.preventDefault();
            e.stopPropagation();
            const blob = item.getAsFile();
            if (!blob) continue;

            try {
              // 上传图片到服务器
              const formData = new FormData();
              formData.append('image', blob, `screenshot-${Date.now()}.png`);
              const res = await fetch(`${BASE}/api/upload-image`, {
                method: 'POST',
                body: formData
              });

              if (res.ok) {
                const { path } = await res.json();
                // 只发送路径到终端，后面加一个空格方便继续输入
                sendInput(path + ' ');
              } else {
                const error = await res.text();
                sendInput(`echo "❌ Upload failed: ${error}"\n`);
              }
            } catch (err) {
              sendInput(`echo "❌ Upload error: ${err.message}"\n`);
            }
            break;
          }
        }
      };

      // 在 document 级别监听粘贴（捕获阶段）
      document.addEventListener('paste', onPaste, true);
      cleanupFns.push(() => {
        document.removeEventListener('paste', onPaste, true);
      });

      // 首次连接 Claude 终端
      reconnectWS();

      // ===== 初始化 Bash 终端 =====
      console.log('[Bash] Starting initialization, termHostRef2.current:', termHostRef2.current);
      if (!termHostRef2.current) {
        console.error('[Bash] termHostRef2.current is null, cannot initialize');
      } else {
        const term2 = new Terminal({
          fontSize: 13,
          cursorBlink: true,
          convertEol: true,
          scrollback: 2000,
          theme: {
            background: '#ffffff',
            foreground: '#333333',
            cursor: '#333333',
            cursorAccent: '#ffffff',
            selectionBackground: 'rgba(0, 120, 215, 0.3)',
            black: '#000000',
            red: '#cd3131',
            green: '#00bc00',
            yellow: '#949800',
            blue: '#0451a5',
            magenta: '#bc05bc',
            cyan: '#0598bc',
            white: '#555555',
            brightBlack: '#666666',
            brightRed: '#cd3131',
            brightGreen: '#14ce14',
            brightYellow: '#b5ba00',
            brightBlue: '#0451a5',
            brightMagenta: '#bc05bc',
            brightCyan: '#0598bc',
            brightWhite: '#a5a5a5'
          }
        });
        const fit2 = new FitAddon();
        term2.loadAddon(fit2);
        term2.open(termHostRef2.current);
        try { fit2.fit(); } catch {}
        termRef2.current = term2;
        fitRef2.current = fit2;
        console.log('[Bash] Terminal created and opened');

        term2.onData((data) => {
          const ws = wsRef2.current;
          if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'input', data }));
        });

        const onResize2 = () => {
          requestAnimationFrame(() => {
            try { fit2.fit(); sendResize2(); } catch {}
          });
        };
        const ro2 = new ResizeObserver(onResize2);
        ro2.observe(termHostRef2.current);
        cleanupFns.push(() => ro2.disconnect());

        // 连接 Bash 终端
        console.log('[Bash] About to call reconnectWS2()');
        reconnectWS2();
        console.log('[Bash] reconnectWS2() called');
      } // end if termHostRef2.current

      // ===== 初始化 Bash 2 终端 =====
      if (termHostRef3.current) {
        const term3 = new Terminal({
          fontSize: 13,
          cursorBlink: true,
          convertEol: true,
          scrollback: 2000,
          theme: {
            background: '#ffffff',
            foreground: '#333333',
            cursor: '#333333',
            cursorAccent: '#ffffff',
            selectionBackground: 'rgba(0, 120, 215, 0.3)',
            black: '#000000',
            red: '#cd3131',
            green: '#00bc00',
            yellow: '#949800',
            blue: '#0451a5',
            magenta: '#bc05bc',
            cyan: '#0598bc',
            white: '#555555',
            brightBlack: '#666666',
            brightRed: '#cd3131',
            brightGreen: '#14ce14',
            brightYellow: '#b5ba00',
            brightBlue: '#0451a5',
            brightMagenta: '#bc05bc',
            brightCyan: '#0598bc',
            brightWhite: '#a5a5a5'
          }
        });
        const fit3 = new FitAddon();
        term3.loadAddon(fit3);
        term3.open(termHostRef3.current);
        try { fit3.fit(); } catch {}
        termRef3.current = term3;
        fitRef3.current = fit3;

        term3.onData((data) => {
          const ws = wsRef3.current;
          if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'input', data }));
        });

        const onResize3 = () => {
          requestAnimationFrame(() => {
            try { fit3.fit(); sendResize3(); } catch {}
          });
        };
        const ro3 = new ResizeObserver(onResize3);
        ro3.observe(termHostRef3.current);
        cleanupFns.push(() => ro3.disconnect());

        reconnectWS3();
      } // end if termHostRef3.current
    })();

    return () => {
      try { wsRef.current?.close(); } catch {}
      try { wsRef2.current?.close(); } catch {}
      try { wsRef3.current?.close(); } catch {}
      cleanupFns.forEach(fn => fn());
      if (termRef.current) {
        termRef.current.dispose();
        termRef.current = null;
      }
      if (termRef2.current) {
        termRef2.current.dispose();
        termRef2.current = null;
      }
      if (termRef3.current) {
        termRef3.current.dispose();
        termRef3.current = null;
      }
      fitRef2.current = null;
      fitRef3.current = null;
      if (outputTimerRef.current) {
        clearTimeout(outputTimerRef.current);
      }
      if (resizeBusyTimerRef.current) {
        clearTimeout(resizeBusyTimerRef.current);
      }
      if (finalizeTimerRef.current) {
        clearTimeout(finalizeTimerRef.current);
      }
    };
  }, []);

  // 渲染划词翻译视图
  const selectedTextView = useMemo(() => {
    if (!selectedTextTranslation) return null;
    const { text, translation, file, line, lineRange } = selectedTextTranslation;

    // 构建标题
    let title = '🌐 划词翻译';
    if (file) {
      const fileName = file.split('/').pop();
      title += ` | 📄 ${fileName}`;
      if (lineRange) {
        title += ` | 📍 LINES: ${lineRange}`;
      } else if (line) {
        title += ` | 📍 LINE: ${line}`;
      }
    }

    return (
      <div className="edit-file-view">
        <div className="mirror-title">{title}</div>
        <div className="hunk-block">
          {translation && (
            <div className="hunk-section plus-section">
              <div className="hunk-label">+ 翻译</div>
              <pre className="hunk-content">{translation}</pre>
            </div>
          )}
          <div className="hunk-section minus-section">
            <div className="hunk-label">− 原文</div>
            <pre className="hunk-content">{text}</pre>
          </div>
        </div>
      </div>
    );
  }, [selectedTextTranslation]);

  // 使用 useMemo 渲染 Edit file 视图，避免拖拽时丢失
  // 优先使用state，但如果state被清空就用ref备份
  const editFileView = useMemo(() => {
    const data = editFileData || editFileDataRef.current;
    if (!data) return null;

    const { file, blocks, translations } = data;

    // 内联 renderMirrorView 逻辑，避免函数引用问题
    return (
      <div className="edit-file-view">
        <div className="mirror-title">✏️ Edit file {file}</div>
        {blocks.map(({ idx, oldTxt, newTxt }) => {
          const cnOld = translations?.[`${idx}:---`] || '';
          const cnNew = translations?.[`${idx}:+++`] || '';

          // 计算 diff
          let diff = null;
          if (cnOld && cnNew) {
            const oldWords = cnOld.split(/(\s+|[，。；！？、])/);
            const newWords = cnNew.split(/(\s+|[，。；！？、])/);
            const m = oldWords.length;
            const n = newWords.length;
            const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

            for (let i = 1; i <= m; i++) {
              for (let j = 1; j <= n; j++) {
                if (oldWords[i - 1] === newWords[j - 1]) {
                  dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                  dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
              }
            }

            const result = [];
            let i = m, j = n;
            while (i > 0 || j > 0) {
              if (i > 0 && j > 0 && oldWords[i - 1] === newWords[j - 1]) {
                result.unshift({ type: 'equal', text: oldWords[i - 1] });
                i--; j--;
              } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
                result.unshift({ type: 'add', text: newWords[j - 1] });
                j--;
              } else if (i > 0) {
                result.unshift({ type: 'del', text: oldWords[i - 1] });
                i--;
              }
            }
            diff = result;
          }

          return (
            <div key={idx} className="hunk-block">
              {diff && (
                <div className="hunk-section diff-section">
                  <div className="hunk-label">📝 详细对比 (hunk {idx+1})</div>
                  <div className="diff-content">
                    {diff.map((part, i) => {
                      if (part.type === 'equal') {
                        return <span key={i}>{part.text}</span>;
                      } else if (part.type === 'del') {
                        return <span key={i} className="diff-del">{part.text}</span>;
                      } else {
                        return <span key={i} className="diff-add">{part.text}</span>;
                      }
                    })}
                  </div>
                </div>
              )}
              {cnOld && (
                <div className="hunk-section minus-section">
                  <div className="hunk-label">− 原文 (hunk {idx+1})</div>
                  <pre className="hunk-content">{cnOld}</pre>
                </div>
              )}
              {cnNew && (
                <div className="hunk-section plus-section">
                  <div className="hunk-label">+ 修改后 (hunk {idx+1})</div>
                  <pre className="hunk-content">{cnNew}</pre>
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  }, [editFileData]);

  return (
    <div style={{height:'100vh', display:'flex', flexDirection:'column'}}>
      <header style={{display:'flex',gap:12,alignItems:'center',padding:'10px 14px',borderBottom:'1px solid #eee'}}>
        <button className="btn" onClick={()=>fetch(`${BASE}/api/compile`,{method:'POST'})}>⚡ 一键编译（⌘⇧R）</button>
        <button className="btn" onClick={()=>{
          const a = document.createElement('a');
          a.href = `${BASE}/api/download-tex`;
          a.download = 'tex.zip';
          a.click();
        }}>📦 下载 tex 目录</button>
        <div style={{fontSize:'13px', color:'#333'}}>
          💡 <strong>双击</strong>单词 或 <strong>选择文本</strong> 即可定位 LaTeX 源码
        </div>
        <div style={{marginLeft:'auto', color:'#666'}}>{status}</div>
      </header>
      <div className="split" ref={splitRef}>
        <iframe
          ref={iframeRef}
          id="viewer"
          className="pdf-pane"
          src={`${BASE}/pdfjs/web/viewer.html?file=${encodeURIComponent(`${BASE}/pdf`)}&zoom=page-fit#pagemode=none`}
        />

        {/* 自定义分割条 */}
        <div
          ref={dividerRef}
          className={`divider${dragging ? ' dragging' : ''}`}
          onPointerDown={onDividerPointerDown}
          title="拖拽调整左右宽度"
        />

        <aside ref={leftRef} className="left-pane" style={{ width: leftWidth ? `${leftWidth}px` : undefined }}>
          {/* ===== 左上：可交互终端（双终端标签页） ===== */}
          <div ref={bottomRef} className="left-bottom" style={{ height: terminalHeight ? `${terminalHeight}px` : undefined }}>
            <div className="terminal-bar">
              {/* 终端标签页 */}
              <button
                className={`btn btn-sm ${activeTerminal === 0 ? 'btn-active' : ''}`}
                onClick={() => setActiveTerminal(0)}
                style={{ background: activeTerminal === 0 ? '#e6f7ff' : undefined, borderColor: activeTerminal === 0 ? '#1890ff' : undefined }}
              >
                <span className={`dot ${cliStatus==='connected'?'ok':(cliStatus==='error'?'err':'')}`} style={{marginRight: 4}}></span>
                Claude
              </button>
              <button
                className={`btn btn-sm ${activeTerminal === 1 ? 'btn-active' : ''}`}
                onClick={() => setActiveTerminal(1)}
                style={{ background: activeTerminal === 1 ? '#e6f7ff' : undefined, borderColor: activeTerminal === 1 ? '#1890ff' : undefined }}
              >
                <span className={`dot ${cliStatus2==='connected'?'ok':(cliStatus2==='error'?'err':'')}`} style={{marginRight: 4}}></span>
                Bash
              </button>
              <button
                className={`btn btn-sm ${activeTerminal === 2 ? 'btn-active' : ''}`}
                onClick={() => setActiveTerminal(2)}
                style={{ background: activeTerminal === 2 ? '#e6f7ff' : undefined, borderColor: activeTerminal === 2 ? '#1890ff' : undefined }}
              >
                <span className={`dot ${cliStatus3==='connected'?'ok':(cliStatus3==='error'?'err':'')}`} style={{marginRight: 4}}></span>
                Bash 2
              </button>
              <span className="grow"></span>
              <button className="btn btn-sm" onClick={()=>fileInputRef.current?.click()}>📷 上传图片</button>
              <button className="btn btn-sm" onClick={()=>{ (activeTerminal === 0 ? termRef : activeTerminal === 1 ? termRef2 : termRef3).current?.clear(); }}>清屏</button>
              <button className="btn btn-sm" onClick={()=>sendInput('\x03')}>Ctrl+C</button>
              {activeTerminal === 0 && (
                <button className="btn btn-sm" onClick={()=>{
                  reconnectWS(); setTimeout(()=>sendInput(`claude${CLAUDE_MODEL ? ` --model ${CLAUDE_MODEL}` : ''} --resume\n`), 1000);
                }}>历史会话</button>
              )}
              <button className="btn btn-sm" onClick={()=> activeTerminal === 0 ? reconnectWS() : activeTerminal === 1 ? reconnectWS2() : reconnectWS3()}>重连</button>
            </div>
            {/* 终端容器 - 使用绝对定位切换显示 */}
            <div style={{ position: 'relative', flex: '1 1 auto', overflow: 'hidden' }}>
              {/* Claude 终端 */}
              <div ref={termHostRef} className="terminal-host" style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                visibility: activeTerminal === 0 ? 'visible' : 'hidden',
                zIndex: activeTerminal === 0 ? 1 : 0
              }} />
              {/* Bash 终端 */}
              <div ref={termHostRef2} className="terminal-host" style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                visibility: activeTerminal === 1 ? 'visible' : 'hidden',
                zIndex: activeTerminal === 1 ? 1 : 0
              }} />
              {/* Bash 2 终端 */}
              <div ref={termHostRef3} className="terminal-host" style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                visibility: activeTerminal === 2 ? 'visible' : 'hidden',
                zIndex: activeTerminal === 2 ? 1 : 0
              }} />
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={handleFileUpload}
            />
          </div>

          {/* 垂直分割条 */}
          <div
            ref={vDividerRef}
            className={`v-divider${vDragging ? ' dragging' : ''}`}
            onPointerDown={onVDividerPointerDown}
            title="拖拽调整终端高度"
          />

          {/* ===== 左下：翻译 ===== */}
          <div className="left-top">
            <div className="pane-header">
              🌏 翻译
              {isTranslating && <span className="translating-indicator"></span>}
            </div>
            <div className="result-box">
              {editFileView ? (
                editFileView
              ) : selectedTextView ? (
                selectedTextView
              ) : typeof result === 'string' ? (
                <pre className="mono">{result}</pre>
              ) : (
                result
              )}
            </div>
            {Array.isArray(candidates) && candidates.length>1 && (
              <div className="cand-list">
                <div className="cand-title">找到多个可能位置（点击选择）：</div>
                <ul className="cand-ul">
                  {candidates.map((c, i) => (
                    <li key={i} className="cand" onClick={()=>renderResult(c)}>
                      <div className="cand-head">
                        <b>{c.file}</b>:{c.line} <span style={{opacity:.6}}>({c.method}, score {c.score ?? '-'})</span>
                      </div>
                      <pre className="cand-snippet">{c.snippet}</pre>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </aside>
        {/* 拖拽遮罩（阻挡 iframe 捕获鼠标） */}
        <div className={`drag-overlay${dragging ? ' active' : ''}`} />
      </div>

      {/* 候选列表已集成到左侧面板 */}

      {/* Ask Ackleaf 浮动按钮 */}
      {askButton.show && (
        <button
          onClick={handleAskClaude}
          style={{
            position: 'fixed',
            left: askButton.x,
            top: askButton.y,
            padding: '6px 12px',
            backgroundColor: '#667eea',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '12px',
            fontWeight: '600',
            boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
            zIndex: 10000,
            whiteSpace: 'nowrap'
          }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          🤖 Ask Ackleaf
        </button>
      )}
    </div>
  );
}