# Robot-Whisper LaTeX slides

这套 16:9 幻灯片根据两个 PNG 批次重绘，入口为 `main.tex`。所有版式、图表、图标、热力图、流程图和机器人场景均由 LaTeX、TikZ 与 PGFPlots 生成，源码没有嵌入原始 PNG。

## 构建

需要 pdfLaTeX、latexmk、TikZ/PGFPlots、CJKutf8、Arphic `gbsn` 字体和 Ghostscript。

```bash
make pdf
make render
```

输出：

- `robot_whisper.pdf`：16 页矢量 PDF
- `rendered/slide-01.png` 至 `rendered/slide-16.png`：144 dpi 预览图

## 文件结构

- `main.tex`：整套幻灯片入口
- `theme.tex`：颜色、母版和共享 TikZ 组件
- `slides/*.tex`：每页独立源码
- `Makefile`：PDF 与 PNG 构建命令

原图中的照片已抽象为可编辑的 TikZ 机器人工作台场景。第 16 页二维码是矢量视觉示意，不对应实际网址，因此不可扫码。
