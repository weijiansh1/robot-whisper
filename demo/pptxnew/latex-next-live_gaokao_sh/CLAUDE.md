# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LaTeX Next Live** is an integrated live LaTeX compilation and preview system with bidirectional synchronization via SyncTeX. It combines:
- **Frontend**: Next.js 14 + React 18 (client-side PDF viewer interaction)
- **Backend**: Express.js server with real-time communication (SSE + WebSocket)
- **LaTeX Engine**: macTeX/TeX Live (latexmk + xelatex)
- **SyncTeX Integration**: Reverse PDF-to-source lookup for precise source code navigation
- **AI Integration**: Claude Code terminal interface + translation API (Qwen-MT)

### Technology Stack
- **Next.js** 14.2.5 - Full-stack framework
- **Express** 4.19.2 - Backend server
- **PDF.js** - Client-side PDF rendering
- **xterm.js** 5.3.0 - Terminal emulation
- **node-pty** 1.0.0 - Pseudo-terminal for CLI interaction
- **WebSocket (ws)** 8.18.3 - Real-time terminal communication
- **Chokidar** 3.6.0 - File system watching for auto-compilation
- **Python** 3.x - Auxiliary reverse search script

---

## Build & Run

### Development
```bash
npm install
npm run dev
```
- Runs on `http://localhost:3000` (default PORT=3000)
- Auto-recompiles LaTeX on `.tex`, `.bib`, `.sty`, `.cls` file changes
- HMR (Hot Module Reload) for Next.js frontend
- SSE broadcasts compilation status to connected clients

### Production
```bash
npm run build
npm run start
```
- Builds Next.js app (`.next` directory)
- Launches server with production optimizations
- Disables React Strict Mode for subprocess stability

### Environment Variables (`.env`)
```bash
# Translation API (optional, for selected text translation)
DASHSCOPE_API_KEY=sk-xxxxx           # Qwen-MT API key
DASHSCOPE_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-mt-turbo

# Claude Code integration (optional)
CLAUDE_CODE_BIN=claude
CLAUDE_CODE_ARGS=code
PROJECT_ROOT=/path/to/tex/folder
# CLAUDE_CODE_TOKEN=your-secret-token   # Optional WebSocket token
```

---

## Architecture Overview

```
Client                          Server                          File System
┌─────────────────────┐        ┌──────────────────────┐         ┌─────────────┐
│  Browser (Next.js)  │        │   Express Server     │         │ LaTeX Files │
├─────────────────────┤        ├──────────────────────┤         ├─────────────┤
│ PDF.js Viewer       │◄──────►│ /api/revsearch       │◄────────│ tex/*.tex   │
│ xterm Terminal      │        │ /api/compile         │         │ tex/*.bib   │
│ React Components    │        │ /ws/terminal (WS)    │         │ build/*.pdf │
│ Event Handlers      │        │ /api/sse (SSE)       │         │ output/...  │
└─────────────────────┘        │ /api/qwen-translate  │         └─────────────┘
                               │ /pdf (static serve)  │
                               └──────────────────────┘
                                    │
                                    ▼
                               latexmk (child process)
                               SyncTeX query
                               Python revsearch.py
```

---

## File Structure

```
latex-next-live/
├── app/                          # Next.js App Router
│   ├── page.jsx                 # Main React component (1702 lines)
│   │   ├── PDF viewer iframe handling
│   │   ├── Dual terminal integration (Claude + Bash)
│   │   ├── File selection & revsearch
│   │   ├── Translation UI (Edit file + selected text)
│   │   └── Drag-to-resize panels
│   ├── layout.jsx               # Root layout wrapper
│   └── globals.css              # Global styles (xterm, layout, animations)
│
├── server.js                     # Express backend (649 lines)
│   ├── SSE setup for compilation events
│   ├── /api/compile - manual compilation trigger
│   ├── /api/revsearch - PDF→LaTeX reverse lookup
│   ├── /api/qwen-translate - AI translation proxy
│   ├── /api/source - code snippet reader
│   ├── /api/upload-image - screenshot upload
│   ├── /api/download-tex - directory export
│   ├── /pdf - static PDF serving
│   ├── /ws/terminal - Claude CLI WebSocket
│   ├── /ws/terminal-codex - Bash WebSocket
│   └── File watcher (chokidar) for auto-compile
│
├── scripts/
│   └── revsearch.py             # Python fuzzy reverse search helper
│                                 # Uses ripgrep + SyncTeX scoring
│
├── tex/                          # LaTeX source (Overleaf git clone)
│   ├── main.tex                 # Main document (ICLR 2026)
│   ├── macros.tex               # Custom macros
│   ├── references.bib           # Bibliography
│   ├── sections/                # Chapter files
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── method.tex
│   │   ├── dynamics.tex
│   │   ├── experiments.tex
│   │   ├── results.tex
│   │   ├── discussion.tex
│   │   ├── related_work.tex
│   │   ├── conclusion.tex
│   │   └── appendix.tex
│   ├── figs/                    # Figures (png, pdf)
│   ├── *.sty                    # Style files (icml2026, algorithm, etc.)
│   └── icml2026.bst             # Bibliography style
│
├── build/                        # Compilation output
│   ├── main.pdf                 # Final PDF
│   ├── main.synctex.gz          # SyncTeX database
│   └── .aux/                    # Auxiliary files (logs, aux, bbl, etc.)
│
├── public/
│   └── pdfjs/                   # PDF.js viewer (static)
│       └── web/viewer.html      # PDF viewer interface
│
├── .next/                        # Next.js build output (gitignore)
├── node_modules/                # Dependencies
├── tmp/                          # Temporary uploads
├── package.json                  # npm dependencies & scripts
├── next.config.js               # Next.js config
├── .env                         # Environment variables
├── README.md                    # User documentation
└── CLAUDE.md                    # This file
```

---

## Core Components

### 1. Frontend: app/page.jsx (1703 lines)

#### Main Functional Areas:

**A. PDF Viewer Integration**
- Embeds PDF.js viewer in iframe
- Loads PDF from `/pdf` endpoint with cache-busting timestamp
- Preserves scroll position & zoom on recompilation
- Handles document load events for event attachment

**B. Text Selection & Reverse Lookup**
```javascript
// Key coordinates transformation:
// Browser pixels → PDF coordinates (via viewport.convertToPdfPoint)
// PDF coordinates → SyncTeX coordinates (Y-axis flip for PDF bottom-left origin)
// Coordinates sent to /api/revsearch for server-side SyncTeX query
```

**C. Dual Terminal System**
- **Claude Terminal** (`/ws/terminal`): Runs `claude --model claude-opus-4-5-20251101`
  - Captures "Edit file" blocks from CLI output
  - Parses diff-style edits for AI-assisted modifications
  - Translates changes via Qwen-MT API
  - Displays side-by-side comparison (original vs. modified)

- **Bash Terminal** (`/ws/terminal-codex`): Interactive shell
  - Standard bash shell for manual commands
  - Supports file uploads via clipboard paste
  - Terminal resize detection with ResizeObserver

**D. Parser State Machine (Claude Terminal)**
```
Capturing = false
    ↓ [Match "Edit file <path>"]
Capturing = true
    ├ [Collect diff lines (-, +, line numbers)]
    └ [Match "Do you want..."] → Finalize & Translate
Capturing = false [Reset]
```

**E. Translation Pipeline**
- Dual modes: "Edit file" mode (Claude) vs. "Selected text" mode (PDF selection)
- Segments sent to `/api/qwen-translate` in parallel
- Results cached in state, displayed with diff highlighting

**F. Drag-to-Resize Panels**
- **Horizontal**: Left sidebar (code editor simulation) ↔ Right PDF viewer
- **Vertical**: Top panel (results) ↔ Bottom terminal
- Persistence via localStorage (`leftPaneWidthPxV2`, `terminalHeightPx`)
- Min/max constraints to prevent panel collapse

#### Key State Management:
```javascript
- status: Compilation & connection status
- result: Reverse lookup or translation result
- editFileData: Parsed Claude edits with translations
- selectedTextTranslation: Selected PDF text + translation
- candidates: Multiple SyncTeX match candidates
- leftWidth, terminalHeight: Pixel dimensions (persisted)
- cliStatus: WebSocket connection state (Claude)
- cliStatus2: WebSocket connection state (Bash)
- activeTerminal: Which tab is visible (0=Claude, 1=Bash)
```

#### Critical Hooks:
1. **SSE Connection** - Auto-reconnects on compile events
2. **Keyboard Shortcuts** - Cmd+Shift+R or Ctrl+Shift+R for manual compile
3. **File Upload** - Clipboard paste images → `/api/upload-image`
4. **WebSocket** - Terminal I/O with resize signaling
5. **ResizeObserver** - Terminal fitting with RAF throttling

---

### 2. Backend: server.js (872 lines)

#### Request Flow:

**A. SSE Event Broadcast**
```
chokidar detects .tex/.bib change
    ↓
runLatexmk() triggered (debounced)
    ↓
latexmk subprocess runs in TEX_ROOT
    ↓
If successful: broadcast 'compiled' event to all connected SSE clients
    ↓
Browser: /api/sse receives event → reload iframe (preserving page state)
```

**B. Reverse Search API: POST /api/revsearch**
```
Input:
{
  page: 1,
  h: 100,        // X coordinate (PDF space)
  v: 500,        // Y coordinate (PDF space, from bottom)
  selectedText: "example text",
  endPage, endH, endV: range for multi-line selection
}

Processing:
1. Call SyncTeX: synctex edit -o PAGE:H:V:PDF_PATH
2. Parse output for: Input: <file>, Line: <line>, Column: <col>
3. Read source file & extract context (±2 lines)
4. If selection text doesn't match context, fallback to Python revsearch
5. Return result with latexText, lineRange, snippet

Output:
{
  method: "synctex",
  file: "/path/to/file.tex",
  line: 42,
  lineRange: "42-45",
  latexText: "actual source code",
  snippet: "context with line numbers",
  column: 5,
  adjusted: false      // true if coordinate offset applied
}
```

**C. Coordinate Adjustment Loop**
- Initial SyncTeX query fails: try offset coordinates
- Offsets: [0,0], [0,-3], [0,3], [-3,0], [3,0], ... up to 16 offsets
- Stops on first match; handles edge cases (list indentation, ligatures)

**D. Translation API: POST /api/qwen-translate**
```
Input:
{
  segments: [
    { idx: 0, tag: '---', text: "old text" },
    { idx: 0, tag: '+++', text: "new text" }
  ],
  source_lang: "en",
  target_lang: "zh"
}

Processing:
- Calls Qwen-MT model via DASHSCOPE_BASE URL
- Uses chat/completions endpoint with translation_options
- Parallel requests (one per segment)
- Returns translations keyed by "idx:tag"

Output:
{
  ok: true,
  results: [
    { idx: 0, tag: '---', translation: "旧文本" },
    { idx: 0, tag: '+++', translation: "新文本" }
  ]
}
```

**E. File Watching & Auto-Compilation**
```javascript
chokidar.watch(TEX_ROOT, {
  ignored: /(^|[/\\])(\.git|\.DS_Store)$/i,
  ignoreInitial: true,
  awaitWriteFinish: { stabilityThreshold: 200, pollInterval: 50 }
})
```
- Watches for: `.tex`, `.bib`, `.sty`, `.cls`, `.png`, `.jpg`, `.pdf`, `.eps`, `.svg`
- Stability threshold: waits 200ms after file write finishes
- Max 1 compile at a time (flag: `compiling`)

**F. WebSocket Terminals**
```
/ws/terminal (Claude):
  - Spawns: /bin/bash -lc "claude --model claude-opus-4-5-20251101"
  - CWD: process.env.CLAUDE_CWD || ROOT
  - PTY columns: 80, rows: 24
  - Handles: input, output, resize messages

/ws/terminal-codex (Bash):
  - Spawns: /bin/bash -l (empty CMD → interactive shell)
  - Same PTY config
```

#### Configuration (Top of server.js):
```javascript
const ROOT = __dirname;
const TEX_ROOT = path.join(ROOT, 'tex');
const OUT_DIR = path.join(ROOT, 'build');
const TMP_DIR = path.join(ROOT, 'tmp');
const MAIN_TEX = 'main.tex';             // ICLR 2026 conference paper
const PDF_PATH = path.join(OUT_DIR, 'main.pdf');
```

**Compilation Command**:
```bash
latexmk -pdf -synctex=1 -interaction=nonstopmode -halt-on-error \
  -file-line-error -auxdir=<OUT_DIR>/.aux -outdir=<OUT_DIR> \
  main.tex
```
- Generates `.synctex.gz` for reverse lookup
- Auxiliary files isolated in `.aux/` directory
- PDF output directly to `OUT_DIR`

---

### 3. Helper: scripts/revsearch.py (109 lines)

**Purpose**: Fuzzy reverse search when SyncTeX doesn't find exact match

**Algorithm**:
1. Normalize input text (remove zero-width chars, handle ligatures, hyphenation)
2. Try SyncTeX query with provided coordinates
3. If SyncTeX succeeds: score the candidate
4. Search repository with ripgrep: `rg -F --hidden "g:*.tex" "<text>"`
5. Score each candidate by word overlap
6. Return single best result or top 10 candidates (sorted by score)

**Scoring**:
```
score = 0.6 (if synctex) + 0.3 * (overlapping_words / total_words)
```

---

## Communication Protocols

### 1. Server-Sent Events (SSE) - `/api/sse`
**Purpose**: Real-time compilation status broadcasts

**Events**:
```
Event: connected
Data: "ok"

Event: compiled
Data: {
  "success": true,
  "tag": "watch",           // "watch", "manual", "startup"
  "logTail": "last 6000 chars of stderr/stdout"
}
```

**Client Handling**:
- Auto-reconnect on error (3s delay)
- Reload iframe on successful compile (preserve page state)

### 2. WebSocket (WS) - `/ws/terminal` & `/ws/terminal-codex`
**Purpose**: Bidirectional terminal I/O

**Message Format**:
```json
// Client → Server (Input):
{ "type": "input", "data": "command text" }

// Client → Server (Resize):
{ "type": "resize", "cols": 80, "rows": 24 }

// Server → Client (Output):
{ "type": "output", "data": "terminal output bytes" }
{ "type": "status", "data": "connection info" }
```

**Connection Handling**:
- Uses `node-pty` for pseudo-terminal spawning
- Automatic termination on client disconnect
- Resize events trigger `fit.fit()` on client side

---

## Key Workflows

### Workflow 1: LaTeX Compilation (Auto-Triggered)
```
User edits tex/main.tex (or tex/sections/*.tex)
    ↓ [Chokidar detects change]
    ↓
Server: runLatexmk('watch')
    ↓
  [Check compiling flag to prevent overlaps]
    ↓
  exec(latexmk ...) with TEXINPUTS env var pointing to tex/
    ↓
  [300ms later]
    ↓
  Check if main.pdf exists
    ↓
  broadcast('compiled', { success: true, tag: 'watch' })
    ↓
All SSE clients receive event
    ↓
Browser: Reload iframe with new ?ts=timestamp
    ↓
Preserve page number, zoom level, scroll position
```

### Workflow 2: PDF Reverse Lookup (Double-Click)
```
User double-clicks word in PDF
    ↓
JavaScript captures selection range
    ↓
  [Get bounding box, find page element by data-pageNumber]
    ↓
  Calculate PDF coordinates via viewport.convertToPdfPoint()
    ↓
  Flip Y-axis: syncY = pageHeight - pdfY
    ↓
POST /api/revsearch with { page, h, v, selectedText }
    ↓
Server: synctex edit -o <page>:<h>:<v>:<PDF_PATH>
    ↓
Parse output: file, line, column
    ↓
renderResult(): display source code snippet + selected text
    ↓
Trigger translation if segment selected
```

### Workflow 3: Claude Code Edit (Terminal)
```
Claude outputs:
  Edit file /path/to/file.tex
  12 - old line content
  12 + new line content
  Do you want to make this edit?
    ↓
Frontend parser captures block
    ↓
  Extract blocks: { idx, oldTxt, newTxt }
    ↓
POST /api/qwen-translate with segments
    ↓
Get translations back
    ↓
Display in left panel with diff highlighting:
  −  Original English
  +  Translated result
  📝 Detailed comparison (with word-level diff)
```

### Workflow 4: Selected Text Translation (PDF)
```
User selects text in PDF
    ↓
Reverse lookup finds source location
    ↓
renderResult() extracts selectedText
    ↓
UI switches to "划词翻译" (word selection translation) view
    ↓
POST /api/qwen-translate with segment
    ↓
Display: "原文 (−)" and "翻译 (+)"
    ↓
Optionally send to Claude terminal via "Ask Ackleaf" button
```

---

## Data Flow: Coordinates

### PDF Space → SyncTeX Space

**Step 1: Browser Pixel Coordinates**
- User double-clicks at screen position (100, 200)
- Bounding box of clicked word

**Step 2: Page-Relative Coordinates**
```javascript
const pageRect = pageEl.getBoundingClientRect();  // Page's screen position
const x = clickX - pageRect.left;                 // Pixel offset on page
const y = clickY - pageRect.top;
```

**Step 3: PDF Coordinates**
```javascript
const [pdfX, pdfY] = pageView.viewport.convertToPdfPoint(x, y);
// Returns: PDF coordinate system (origin at lower-left)
```

**Step 4: SyncTeX Coordinates**
```javascript
const pageHeight = pageView.viewport.viewBox[3];  // PDF page height
const syncX = Math.round(pdfX);
const syncY = Math.round(pageHeight - pdfY);      // Flip Y-axis!
```

**SyncTeX Input**: `synctex edit -o 1:100:500:/path/to/output.pdf`
- Page 1, X=100pt, Y=500pt (from bottom)

**SyncTeX Output**:
```
Input:main.tex
Line:42
Column:5
```

---

## Performance Optimizations

### Frontend
1. **RAF Throttling**: Panel resizing uses `requestAnimationFrame()`
2. **Debounced Resize**: ResizeObserver for terminal fitted to RAF
3. **Parser Freeze**: During terminal resize, parser halts to prevent garbled state
4. **Incremental ANSI Stripping**: Process terminal output chunk-by-chunk
5. **useMemo**: Edit file view memoized to persist during drag
6. **localStorage**: Panel dimensions persisted across sessions

### Backend
1. **Compilation Lock**: Only one latexmk process at a time
2. **File Write Stability**: 200ms wait before recompiling (chokidar)
3. **SSE Broadcast**: Efficient Set iteration, no per-client buffering
4. **SyncTeX Caching**: No caching (synchronous queries are fast)
5. **Coordinate Offset Loop**: Stops on first match (max 16 attempts)

---

## Configuration & Customization

### Change Main LaTeX File
Edit `server.js` (lines 30-31):
```javascript
const MAIN_TEX = 'main.tex';             // Change this path
const PDF_PATH = path.join(OUT_DIR, 'main.pdf');
```

### Change Compilation Command
Edit `server.js` line 78:
```javascript
const cmd = `latexmk -pdf -synctex=1 ... "${MAIN_TEX}"`;
```

Example for pdflatex:
```bash
latexmk -pdf -pdflatex -synctex=1 ...
```

### Add Custom Environment Variables
Edit `.env`:
```bash
CLAUDE_CODE_BIN=claude-3.5-sonnet
PORT=4000
```

### Adjust Terminal Defaults
Edit `app/page.jsx` (search for "new Terminal"):
```javascript
const term = new Terminal({
  fontSize: 13,              // Font size
  scrollback: 2000,          // Buffer lines
  cursorBlink: true,         // Cursor animation
  theme: { ... }             // Color scheme
});
```

---

## Debugging & Troubleshooting

### Enable Verbose Logging
**Backend** (`server.js`):
- Console logs already present (search for `console.log`)
- Enable SyncTeX debug output: add `-v` flag to synctex command

**Frontend** (`app/page.jsx`):
- Open browser DevTools → Console
- Look for `[Parser]`, `[WS]`, `[Resize]` prefixed logs
- Network tab shows `/api/revsearch` requests/responses

### Common Issues

**Issue**: SyncTeX not finding source
- Check: Build directory has `.synctex.gz` file
- Check: Coordinates are reasonable (h > 0, v > 0)
- Check: PDF file path is correct
- Fallback: Coordinate offset loop will try nearby positions

**Issue**: Compilation infinite loop
- Check: File watchers on build output directory (may trigger recompile)
- Solution: Ensure `OUT_DIR` not in watch scope
- Check: `ignoreInitial: true` prevents startup recompile

**Issue**: Terminal not connecting
- Check: WebSocket port is accessible (not blocked by firewall)
- Check: Server is running (look for "LaTeX Live on http://...")
- Check: Browser console for WebSocket errors
- Try: Manual reconnect button in UI

**Issue**: Translation not working
- Check: `.env` has valid `DASHSCOPE_API_KEY`
- Check: `DASHSCOPE_BASE` URL is correct
- Check: Network tab shows POST to `/api/qwen-translate`
- Try: Increase request timeout if network is slow

---

## Deployment Notes

### Local Development
```bash
npm run dev
# Runs on localhost:3000
# Auto-restart on server.js changes
```

### Production Server (Linux/macOS)
```bash
npm run build
npm run start

# Or use pm2:
pm2 start server.js --name "latex-live"
pm2 save && pm2 startup
```

### Docker Considerations
Would need:
- Node.js 18+
- TeX Live (full install)
- SyncTeX (usually included)
- Python 3.x (for revsearch.py)

Example Dockerfile structure:
```dockerfile
FROM node:18
RUN apt-get install -y texlive-full
COPY . /app
WORKDIR /app
RUN npm ci
ENV NODE_ENV=production
CMD ["npm", "run", "start"]
```

---

## Security Notes

- **No authentication**: Current implementation has no user auth
- **File paths**: Exposed via `/api/source` (reads arbitrary files in tex/)
- **Code execution**: WebSocket terminals run as current user
- **Translation API**: API key exposed in `.env` (should use secrets manager in production)

Recommendations:
1. Add authentication middleware for production
2. Restrict `/api/source` to tex/ directory only
3. Use environment-based API keys (not committed)
4. Run server with least-privilege user account
5. Use HTTPS + WSS in production

---

## Key Dependencies & Versions

| Package | Version | Purpose |
|---------|---------|---------|
| next | 14.2.5 | Full-stack framework |
| react | 18.3.1 | UI components |
| express | 4.19.2 | Backend API |
| ws | 8.18.3 | WebSocket |
| node-pty | 1.0.0 | Terminal spawning |
| xterm | 5.3.0 | Terminal emulation |
| pdfjs-dist | 4.7.76 | PDF rendering |
| chokidar | 3.6.0 | File watching |
| multer | 2.0.2 | File uploads |
| body-parser | 1.20.3 | JSON parsing |
| dotenv | 17.2.3 | Environment variables |

---

## Future Enhancement Ideas

1. **Collaborative Editing**: Multiple users via CRDT
2. **Version Control Integration**: Auto-commit to git on compile
3. **Diff Visualization**: Side-by-side PDF comparison
4. **Custom Snippets**: User-defined LaTeX templates
5. **CI/CD Integration**: GitHub Actions for automated builds
6. **Database Logging**: Track compilation history
7. **Mobile Support**: Responsive design for tablets
8. **Dark Theme**: CSS variables for theme switching
9. **Plugin System**: Extensible backend architecture
10. **Overleaf Sync**: Full bidirectional Git sync with Overleaf projects

---

## File Checklist for Modifications

- `server.js`: API endpoints, compilation logic, WebSocket handling
- `app/page.jsx`: UI logic, event handlers, terminal integration
- `app/globals.css`: Layout, colors, animations
- `scripts/revsearch.py`: Fuzzy search algorithm
- `.env`: Configuration (API keys, paths)
- `package.json`: Dependencies and scripts
- `next.config.js`: Next.js build settings

---

**Last Updated**: 2026-01-28
**Status**: Production-ready
**Maintenance**: Active (Overleaf integration, Qwen-MT translation, Claude Code)
**Current Paper**: ICLR 2026 (Overleaf project: `<YOUR_PROJECT_ID>`)
