/* Render the page's chart JS under a minimal DOM shim and check the SVG geometry.
   Catches what the palette validator cannot: text outside the viewBox, marks
   outside the plot rect, and overlapping labels. */
import { readFileSync } from "node:fs";

const NS = "http://www.w3.org/2000/svg";

class Node {
  constructor(name) {
    this.nodeName = name; this.attrs = {}; this.children = []; this.style = {};
    this._text = ""; this.listeners = {};
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  hasAttribute(k) { return k in this.attrs; }
  removeAttribute(k) { delete this.attrs[k]; }
  appendChild(c) { this.children.push(c); c.parent = this; return c; }
  addEventListener(t, f) { (this.listeners[t] ||= []).push(f); }
  set textContent(v) { this._text = String(v); }
  get textContent() { return this._text; }
  set innerHTML(v) { this._html = String(v); }
  get innerHTML() { return this._html ?? ""; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 920, height: 400 }; }
  querySelector() { return null; }
  walk(fn) { fn(this); this.children.forEach(c => c.walk(fn)); }
}

const ids = {};
const doc = {
  createElementNS: (_ns, n) => new Node(n),
  createElement: n => new Node(n),
  getElementById: id => (ids[id] ||= new Node("div")),
  documentElement: new Node("html"),
  addEventListener() {},
};
doc.documentElement.setAttribute("data-theme", "light");

const VARS = {
  "--surface": "#fcfcfb", "--grid": "#e1e0d9", "--axis": "#c3c2b7",
  "--muted": "#898781", "--ink": "#0b0b0b",
  "--s1": "#2a78d6", "--s2": "#eb6834", "--s3": "#1baf7a",
  "--pos": "#2a78d6", "--neg": "#e34948", "--seq": "#2a78d6", "--good": "#006300",
};

globalThis.document = doc;
globalThis.getComputedStyle = () => ({ getPropertyValue: n => VARS[n] ?? "#000" });
globalThis.addEventListener = () => {};
globalThis.setTimeout = (f) => f;
globalThis.clearTimeout = () => {};

const html = readFileSync(
  new URL("../viz/himoe-moe-signal-explainer.html", import.meta.url), "utf8");
const script = html.slice(html.lastIndexOf("<script>") + 8, html.lastIndexOf("</script>"));
new Function(script)();

// ---- checks -------------------------------------------------------------
let fail = 0, warn = 0;
const CHART_IDS = ["c1", "c2", "c3", "c4", "c5"];

// approximate advance widths; CJK is full-width, digits/latin about half
const charW = (ch, size) => (/[　-鿿＀-￯]/.test(ch) ? 1.0 : 0.55) * size;
const textWidth = (s, size) => [...s].reduce((a, c) => a + charW(c, size), 0);
const SIZE = { tk: 11.5, al: 12, dl: 12, ann: 11.5 };

for (const id of CHART_IDS) {
  const host = ids[id];
  const svg = host.children.find(c => c.nodeName === "svg");
  if (!svg) { console.log(`[FAIL] ${id}: no svg produced`); fail++; continue; }
  const [, , W, H] = svg.getAttribute("viewBox").split(/\s+/).map(Number);
  const texts = [];
  let outside = 0, marksOut = 0;

  svg.walk(n => {
    if (n.nodeName === "text") {
      const x = +n.attrs.x, y = +n.attrs.y;
      const size = SIZE[n.attrs.class] ?? 12;
      const w = textWidth(n._text, size);
      const anchor = n.attrs["text-anchor"] || "start";
      const x0 = anchor === "middle" ? x - w / 2 : anchor === "end" ? x - w : x;
      const box = { x0, x1: x0 + w, y0: y - size * 0.8, y1: y + size * 0.25, t: n._text };
      texts.push(box);
      // allow a small bleed for axis labels that sit in the margin
      if (box.x0 < -8 || box.x1 > W + 8 || box.y0 < -26 || box.y1 > H + 8) {
        console.log(`[FAIL] ${id}: text "${n._text}" outside viewBox `
          + `(x ${box.x0.toFixed(0)}..${box.x1.toFixed(0)} / 0..${W}, `
          + `y ${box.y0.toFixed(0)}..${box.y1.toFixed(0)} / 0..${H})`);
        outside++;
      }
    }
    if (n.nodeName === "circle") {
      const cx = +n.attrs.cx, cy = +n.attrs.cy, r = +n.attrs.r;
      if (cx - r < -2 || cx + r > W + 2 || cy - r < -2 || cy + r > H + 2) marksOut++;
    }
  });

  // pairwise label overlap
  let overlaps = 0;
  for (let i = 0; i < texts.length; i++)
    for (let j = i + 1; j < texts.length; j++) {
      const a = texts[i], b = texts[j];
      if (a.x0 < b.x1 - 1 && b.x0 < a.x1 - 1 && a.y0 < b.y1 - 1 && b.y0 < a.y1 - 1) {
        console.log(`[WARN] ${id}: labels overlap — "${a.t}" x "${b.t}"`);
        overlaps++;
      }
    }
  fail += outside + marksOut; warn += overlaps;
  console.log(`${outside + marksOut + overlaps === 0 ? "[PASS]" : "[----]"} `
    + `${id}: ${texts.length} labels, viewBox ${W}x${H}, `
    + `outside=${outside} marksOutside=${marksOut} overlaps=${overlaps}`);
}

// table views must exist for every chart (relief rule + no color-only encoding)
for (const t of ["tb1", "tb2", "tb3", "tb4", "tb5"]) {
  const host = ids[t];
  const ok = host.children.some(c => c.nodeName === "table");
  console.log(`${ok ? "[PASS]" : "[FAIL]"} ${t}: table view ${ok ? "present" : "MISSING"}`);
  if (!ok) fail++;
}

console.log(`\n${fail === 0 ? "GEOMETRY OK" : "GEOMETRY FAILURES: " + fail}`
  + (warn ? ` (${warn} overlap warnings)` : ""));
process.exit(fail ? 1 : 0);
