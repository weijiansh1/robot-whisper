/* Walk the reversal animation under a recording canvas shim.
   Checks: no runtime error anywhere on the timeline, every draw inside the
   1280x720 frame, no NaN/undefined text, all 8 scenes paint, and no two
   sufficiently-opaque text labels collide within a frame. */
import { readFileSync } from "node:fs";

const W = 1280, H = 720;
let ops = [], bad = [], texts = [];
const track = (k, x, y) => {
  if (!Number.isFinite(x) || !Number.isFinite(y)) { bad.push(`${k} NaN`); return; }
  if (x < -60 || x > W + 60 || y < -40 || y > H + 40)
    bad.push(`${k} outside frame (${x.toFixed(0)},${y.toFixed(0)})`);
};
const charW = (ch, s) => (/[　-鿿＀-￯]/.test(ch) ? 1.0 : 0.55) * s;
const width = (s, size) => [...String(s)].reduce((a, ch) => a + charW(ch, size), 0);

function makeCtx() {
  return {
    canvas: { width: 2560, height: 1440 },
    globalAlpha: 1, fillStyle: "", strokeStyle: "", lineWidth: 1, lineCap: "", lineJoin: "",
    font: "400 26px sans", textAlign: "left", shadowColor: "", shadowBlur: 0,
    letterSpacing: "0px",
    save() {}, restore() {}, scale() {}, beginPath() {}, clip() {},
    moveTo(x, y) { track("moveTo", x, y); },
    lineTo(x, y) { track("lineTo", x, y); },
    rect(x, y) { track("rect", x, y); },
    roundRect(x, y, w, h) {
      track("roundRect", x, y);
      if (!(w >= 0) || !(h >= 0)) bad.push(`roundRect ${w}x${h}`);
      ops.push("r");
    },
    arc(x, y, r) { track("arc", x, y); if (!(r >= 0)) bad.push(`arc r=${r}`); ops.push("a"); },
    fill() { ops.push("f"); }, stroke() { ops.push("s"); },
    fillRect(x, y) { track("fillRect", x, y); ops.push("fr"); },
    fillText(s, x, y) {
      track("fillText", x, y);
      if (s == null || /undefined|NaN|\[object/.test(String(s))) bad.push(`text "${s}"`);
      const size = parseFloat((this.font.match(/(\d+(?:\.\d+)?)px/) || [0, 26])[1]);
      const w = width(s, size);
      const a = this.textAlign;
      const x0 = a === "center" ? x - w / 2 : a === "right" ? x - w : x;
      if (this.globalAlpha > 0.35 && String(s).trim())
        texts.push({ x0, x1: x0 + w, y0: y - size * 0.8, y1: y + size * 0.25, s: String(s) });
      ops.push("t");
    },
    measureText(s) { return { width: width(s, 26) }; },
  };
}

class E {
  constructor() {
    this.listeners = {}; this.dataset = {}; this._html = ""; this.value = "0";
    this.textContent = ""; this.style = {};
    this.classList = { toggle() {}, add() {}, remove() {} };
  }
  addEventListener(t, f) { (this.listeners[t] ||= []).push(f); }
  fire(t, ev) { (this.listeners[t] || []).forEach(f => f(ev)); }
  set innerHTML(v) { this._html = String(v); }
  get innerHTML() { return this._html; }
  getContext() { return (this._ctx ||= makeCtx()); }
}
const els = {};
const get = id => (els[id] ||= new E());
globalThis.document = {
  getElementById: get, querySelectorAll: () => [], createElement: () => new E(),
};
globalThis.requestAnimationFrame = () => 0;

const html = readFileSync(new URL("../viz/himoe-moe-detect-cloud.html", import.meta.url), "utf8");
const script = html.slice(html.lastIndexOf("<script>") + 8, html.lastIndexOf("</script>"));
try { new Function(script)(); }
catch (e) { console.log("[FAIL] threw on load:", e.message); process.exit(1); }
console.log(`[PASS] loaded; landing frame ${ops.length} ops`);

const scrub = get("sc"), clk = get("tm");
const BOUND = [0,5,10,15,20,26];
let errors = 0, collisions = 0, steps = 0, empty = 0;
const sceneOps = {};
for (let v = 0; v <= 1000; v += 3) {
  ops = []; bad = []; texts = [];
  try { scrub.fire("input", { target: { value: String(v) } }); }
  catch (e) { console.log(`[FAIL] threw at ${v}: ${e.message}`); errors++; continue; }
  steps++;
  const t = parseFloat(clk.textContent);
  let sc = 0;
  for (let k = 0; k < BOUND.length; k++) if (t >= BOUND[k]) sc = k;
  sceneOps[sc] = Math.max(sceneOps[sc] || 0, ops.length);
  if (ops.length < 6) empty++;
  if (bad.length) {
    console.log(`[FAIL] ${v}: ${[...new Set(bad)].slice(0, 2).join(" | ")}`);
    errors += bad.length;
  }
  for (let i = 0; i < texts.length; i++)
    for (let j = i + 1; j < texts.length; j++) {
      const a = texts[i], b = texts[j];
      if (a.x0 < b.x1 - 2 && b.x0 < a.x1 - 2 && a.y0 < b.y1 - 2 && b.y0 < a.y1 - 2) {
        console.log(`[WARN] scrub=${v}: "${a.s}" x "${b.s}"`);
        collisions++;
      }
    }
}
console.log(`[${errors ? "FAIL" : "PASS"}] walked ${steps} positions, ${errors} problems, ${empty} near-empty frames`);
console.log(`[${collisions ? "FAIL" : "PASS"}] ${collisions} label collisions`);
for (let s = 0; s < 6; s++)
  console.log(`  ${sceneOps[s] ? "[PASS]" : "[FAIL]"} segment ${s + 1}: peak ${sceneOps[s] || 0} ops`);
const allScenes = [...Array(6).keys()].every(s => sceneOps[s]);
const ok = !errors && !collisions && allScenes;
console.log("\n" + (ok ? "REVERSAL OK" : "REVERSAL HAS FAILURES"));
process.exit(ok ? 0 : 1);

