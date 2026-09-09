# cckfdu-site — HiMoE 博客 / 短片 / trap 动画更新包

针对 `cckfdu-site`(Astro 静态站)的一组增量改动。**保持原目录结构**,
按相对路径覆盖到项目根目录即可。

```bash
# 在 cckfdu-site 项目根目录下
unzip -o cckfdu-himoe-updates.zip
rsync -av cckfdu-himoe-updates/src/ src/
rsync -av cckfdu-himoe-updates/public/ public/
npm run build          # 或 npm run deploy
```

没有新增依赖,`package.json` / `astro.config.mjs` / `src/layouts/*` 均未改动。
不修改任何既有文件,全部为新增 —— 因此本包没有 `patches/` 目录。

---

## 一、文件清单

| 文件 | 状态 | 说明 |
|---|---|---|
| `src/pages/blog/the-state-the-tokens-never-showed.md` | **新增** | 博客文章(英文) |
| `src/pages/blog/the-state-the-tokens-never-showed-zh.md` | **新增** | 同文中文版,二者页脚互链 |
| `public/assets/himoe_fig_rescue.png` | **新增** | 首图:报警→分叉→救回,三线全实测(69 KB) |
| `public/assets/himoe_fig_trap.png` | **新增** | 文内图:32 条同状态 rollout 的到目标距离,trap 的物理定义(105 KB) |
| `public/assets/himoe_fig_heat.png` | **新增** | 文内图:成功 vs trap 的路由热力图,深层水平条纹(70 KB) |
| `public/assets/himoe_fig_fan.png` | **新增** | 文内图:352 条分支扇面 + 各自报警点(904 KB) |
| `public/assets/himoe_fig_transfer.png` | **新增** | 文内图:一个阈值在两批语料上的误报对齐(54 KB) |
| `public/assets/himoe_fig_candidates.png` | **新增** | 文内图:报警步 8 个候选的路由变化(44 KB) |
| `public/assets/himoe_success.mp4` | **新增** | 开场视频:同状态成功(571 KB / 18.6 s) |
| `public/assets/himoe_failure.mp4` | **新增** | 开场视频:同状态失败(681 KB / 26.0 s) |
| `public/assets/himoe_rescue_trunk.mp4` | **新增** | 救援段视频:原噪声跑满失败(743 KB / 26.0 s) |
| `public/assets/himoe_rescue_ok.mp4` | **新增** | 救援段视频:报警步重采样 8 步完成(167 KB / 3.9 s) |
| `public/himoe-pattern-film.html` | **新增** | 50 秒视觉短片,自包含单文件(20 KB) |
| `public/himoe-trap-animation.html` | **新增** | trap 逐帧动画,自包含单文件、数据内嵌(168 KB) |

媒体合计约 3.3 MB;四段视频均已 `+faststart` 重封装,可在 HTTP 上拖动。

---

## 二、两处需要知道的约定

### ⚠️ 文内媒体用了根绝对路径 `/assets/…`

`Post.astro` 只给 **frontmatter 首图**加 `import.meta.env.BASE_URL` 前缀;
正文里的 `<video>` / `<img>` 走的是 markdown 内嵌 HTML,写的是根绝对路径。
在正式站(根部署)下没有问题;如果用 `PUBLIC_BASE` 做前缀化的预览构建,
这些文内媒体会 404 —— 与站内旧文在同一限制下,不是本包引入的新问题。
要在预览下也可用,需给 layout 加统一的资源前缀处理,那超出本包范围。

### 两个 `public/` 下的独立 HTML 不走站点管线

短片与 trap 动画是自包含单文件(无外部 JS/CSS/数据依赖,短片对
`prefers-reduced-motion` 停在结尾静帧),放进 `public/` 原样透传即可,
不参与 Astro 构建,也不会被站点样式影响。短片支持 `?lang=en` 切英文字幕;
两篇文章分别按语言链到对应版本。

---

## 三、各文件改了什么

### `src/pages/blog/the-state-the-tokens-never-showed{,-zh}.md`(新增)

面向普通读者的叙述文,约 1500 词 / 2600 字,结构固定为三部分加一个尾问:

1. 开场 —— 同一模拟器状态、只差一条采样噪声的两段视频,一成一败;
   点题「分开它们的那件事从未出现在任何 token 里」
2. `Part one` —— 物理世界的 trap 长什么样(两副面孔:悬停不动 / 原地打转),以及 token 侧的
   三条局限:动作流照常输出、长度两解、判决在终点
3. `Part two` —— 路由神经元的 pattern:动则换人、卡则冻结;一个自归一比值
   加一个冻结阈值;80.1% / 82.0% 与阈值跨语料;误报=慢成功 → 警报语义是
   「卡住」而非「必败」
4. `Part three` —— 老实话:成因平凡(卡住→观测不变→输入不变→专家不换,
   r=+0.93),平凡恰是可靠;身份是场景指纹、变化才是状态
5. `One more thinking` —— 报警步存状态换噪声,3/8 救回、8 步对 52 步;
   四条边界(0/8、随机时刻同 3/8、幅度选不出、无预警)如实列出

写作上沿用站内博文的做法:数字后面必跟"实际意味着什么";机制、局限与
肯定结论同等篇幅;一处 blockquote 留给最泼冷水的那句对照结果。
频繁出现的术语只有一个:**pattern**(路由器点亮的 top-4 名单),
首次出现即定义,不引入其他行话。

### `public/himoe-pattern-film.html`(新增)

50 秒五幕视觉短片,与站内其他 film 同一语汇(药丸标签的省略版:深底、
细线辉光、逐笔 draw-on、衬线斜体大字、一幕一个发光对象、底部 DOM 字幕带):

1. pattern 是什么 —— 32 点环,4 个亮点翻牌
2. 对比 —— 「手臂在动」持续换人 vs 「手臂卡住」冻结变橙,霜圈合拢
3. r(t) —— 示意曲线下潜,三个金点 1·2·3,报警环扩散
4. 两群 —— 蓝浮橙沉的示意扇面,再落两个实测大数 80.1% · 82.0%
5. 救援 —— 分叉:橙线沉底打 ✗(52 步),蓝线弹回提前打 ✓(8 步),
   收尾卡「报警 → 重采样 → 救回 · 26.0 s → 3.9 s」

**曲线为示意、数字为实测**,末幕字幕明写这一句。节奏参数集中在文件底部
assembly 段(`STORY` 每幕秒数、`FADE` 交叠),单独调即可。

### `public/himoe-trap-animation.html`(新增)

既有的 trap 逐帧动画原样收入(数据内嵌,32 条 rollout 与深层路由对照),
博文第一部分链向它。页内自带 2026-08-27 的成因更正说明,与博文第三部分
的表述一致。

### 四段视频与四张图(新增)

全部由实验产物直接导出:图 1–3 从 352/512 条语料与服务端路由日志重绘,
救援两段视频来自触发式干预实跑的留档(报警步 q32,检测器常数事先冻结)。
图内不含示意线;示意只出现在短片里并已声明。

---

## 四、验证

```bash
npm run build && npm run preview     # http://localhost:4321
```

建议看这几处:

- `/blog/the-state-the-tokens-never-showed/` —— 首图、两组并排视频能否播放与拖动
- `/blog/the-state-the-tokens-never-showed-zh/` —— 中文排版、页脚互链
- `/himoe-pattern-film.html` 与 `?lang=en` —— 两种字幕各看一遍,确认自动循环
- `/himoe-trap-animation.html` —— 载入后点「播放」

四段视频均为 mp4/H.264 + faststart;若站点由不支持 Range 请求的静态服务
托管,视频将不可拖动(可播放)—— 站内现有部署(Cloudflare/Netlify 类)不受影响。
