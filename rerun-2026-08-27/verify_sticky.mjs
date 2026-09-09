/* Drive the sticky-core scene across its whole timeline under a recording
   canvas shim: no runtime errors, nothing drawn outside the 1200x780 stage,
   no NaN/undefined text, captions advance, chrome renders. */
import { readFileSync } from "node:fs";

const W = 1200, H = 780;
let ops = [], bad = [];
const track = (k, x, y) => {
  if (!Number.isFinite(x) || !Number.isFinite(y)) { bad.push(`${k} NaN (${x},${y})`); return; }
  if (x < -90 || x > W + 90 || y < -40 || y > H + 40)
    bad.push(`${k} outside stage (${x.toFixed(0)},${y.toFixed(0)})`);
};

function makeCtx() {
  return {
    canvas: { width: 2400, height: 1560 },
    globalAlpha: 1, fillStyle: "", strokeStyle: "", lineWidth: 1, lineCap: "",
    lineJoin: "", font: "", textAlign: "left", shadowColor: "", shadowBlur: 0,
    save() {}, restore() {}, scale() {}, beginPath() {},
    moveTo(x, y) { track("moveTo", x, y); },
    lineTo(x, y) { track("lineTo", x, y); },
    rect(x, y, w, h) { track("rect", x, y); track("rect2", x + w, y + h); },
    roundRect(x, y, w, h) {
      track("roundRect", x, y); track("roundRect2", x + w, y + h);
      if (!Number.isFinite(w) || !Number.isFinite(h) || w < 0 || h < 0)
        bad.push(`roundRect bad size ${w}x${h}`);
      ops.push("roundRect");
    },
    arc(x, y, r) { track("arc", x, y); if (!(r >= 0)) bad.push(`arc r=${r}`); ops.push("arc"); },
    fill() { ops.push("fill"); }, stroke() { ops.push("stroke"); },
    fillRect(x, y) { track("fillRect", x, y); ops.push("fillRect"); },
    fillText(s, x, y) {
      track("fillText", x, y);
      if (s == null || /undefined|NaN|\[object/.test(String(s)))
        bad.push(`fillText suspicious "${s}"`);
      ops.push("text");
    },
    measureText: s => ({ width: String(s).length * 7 }),
  };
}

class E {
  constructor() {
    this.listeners = {}; this.dataset = {}; this._html = ""; this.value = "0";
    this.textContent = ""; this._hidden = false; this.style = {};
    this.classList = { toggle() {}, add() {}, remove() {} };
  }
  addEventListener(t, f) { (this.listeners[t] ||= []).push(f); }
  fire(t, ev) { (this.listeners[t] || []).forEach(f => f(ev)); }
  set innerHTML(v) { this._html = String(v); }
  get innerHTML() { return this._html; }
  setAttribute(k) { if (k === "hidden") this._hidden = true; }
  removeAttribute(k) { if (k === "hidden") this._hidden = false; }
  hasAttribute(k) { return k === "hidden" ? this._hidden : false; }
  getContext() { return (this._ctx ||= makeCtx()); }
}
const els = {};
const get = id => (els[id] ||= new E());
globalThis.document = { getElementById: get, querySelectorAll: () => [], createElement: () => new E() };
globalThis.requestAnimationFrame = () => 0;

const html = readFileSync(new URL("../viz/himoe-moe-sticky-core.html", import.meta.url), "utf8");
const script = html.slice(html.lastIndexOf("<script>") + 8, html.lastIndexOf("</script>"));
try { new Function(script)(); }
catch (e) { console.log("[FAIL] threw on load:", e.message); process.exit(1); }
console.log(`[PASS] loaded; landing frame drew ${ops.length} ops`);

const scrub = get("scrub"), cap = get("cap");
const seen = new Set();
let errors = 0, minOps = Infinity, steps = 0;
for (let v = 0; v <= 1000; v += 4) {
  ops = []; bad = [];
  try { scrub.fire("input", { target: { value: String(v) } }); }
  catch (e) { console.log(`[FAIL] threw at scrub=${v}: ${e.message}`); errors++; continue; }
  steps++; seen.add(cap.dataset.i); minOps = Math.min(minOps, ops.length);
  if (ops.length < 40) { console.log(`[FAIL] scrub=${v} drew only ${ops.length} ops`); errors++; }
  if (bad.length) {
    console.log(`[FAIL] scrub=${v}: ${[...new Set(bad)].slice(0, 3).join(" | ")}`);
    errors += bad.length;
  }
}
console.log(`[${errors ? "FAIL" : "PASS"}] walked ${steps} positions, min ${minOps} ops/frame, ${errors} problems`);
console.log(`[${seen.size === 6 ? "PASS" : "FAIL"}] captions advanced through ${seen.size}/6 beats`);
for (const [id, needle, label] of [["hero", "AUC", "hero figures"], ["lg", "粘滞", "legend"],
                                   ["tbl", "<table", "table view"], ["foot", "怎么算", "method note"]]) {
  const ok = get(id).innerHTML.includes(needle);
  console.log(`[${ok ? "PASS" : "FAIL"}] ${label}`);
  if (!ok) errors++;
}
const ok = !errors && seen.size === 6;
console.log("\n" + (ok ? "SCENE OK" : "SCENE HAS FAILURES"));
process.exit(ok ? 0 : 1);
