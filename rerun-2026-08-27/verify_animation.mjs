/* Drive the trap animation across its whole timeline under a recording canvas
   shim. Verifies: no runtime errors at any point on the clock, every draw stays
   inside the 1160x660 stage, all five acts actually paint, and the caption
   advances. */
import { readFileSync } from "node:fs";

const W = 1160, H = 660;
const ops = [];
let bad = [];

const track = (kind, x, y) => {
  if (!Number.isFinite(x) || !Number.isFinite(y)) { bad.push(`${kind} NaN (${x},${y})`); return; }
  if (x < -80 || x > W + 80 || y < -40 || y > H + 40) bad.push(`${kind} out of stage (${x.toFixed(0)},${y.toFixed(0)})`);
};

function makeCtx() {
  const c = {
    canvas: { width: 2320, height: 1320 },
    globalAlpha: 1, fillStyle: "", strokeStyle: "", lineWidth: 1,
    font: "", textAlign: "left", lineJoin: "", lineCap: "",
    _path: [],
    save() {}, restore() {}, scale() {}, beginPath() { this._path = []; },
    moveTo(x, y) { track("moveTo", x, y); this._path.push([x, y]); },
    lineTo(x, y) { track("lineTo", x, y); this._path.push([x, y]); },
    arc(x, y, r) { track("arc", x, y); if (!Number.isFinite(r) || r < 0) bad.push(`arc bad r=${r}`); ops.push("arc"); },
    fill() { ops.push("fill"); }, stroke() { ops.push("stroke"); },
    fillRect(x, y, w, h) { track("fillRect", x, y); track("fillRect2", x + w, y + h); ops.push("fillRect"); },
    strokeRect(x, y, w, h) { track("strokeRect", x, y); ops.push("strokeRect"); },
    fillText(s, x, y) {
      track("fillText", x, y);
      if (s === undefined || s === null || /undefined|NaN|\[object/.test(String(s)))
        bad.push(`fillText suspicious: "${s}"`);
      ops.push("text:" + s);
    },
    measureText: s => ({ width: String(s).length * 7 }),
  };
  return c;
}

class E {
  constructor(id) {
    this.id = id; this.listeners = {}; this.dataset = {}; this._html = "";
    this.textContent = ""; this.value = "0"; this._hidden = false;
    this.classList = { toggle() {}, add() {}, remove() {} };
    this.style = {};
  }
  addEventListener(t, f) { (this.listeners[t] ||= []).push(f); }
  fire(t, ev) { (this.listeners[t] || []).forEach(f => f(ev)); }
  set innerHTML(v) { this._html = String(v); this._kids = null; }
  get innerHTML() { return this._html; }
  setAttribute(k, v) { if (k === "hidden") this._hidden = true; }
  removeAttribute(k) { if (k === "hidden") this._hidden = false; }
  hasAttribute(k) { return k === "hidden" ? this._hidden : false; }
  getContext() { return (this._ctx ||= makeCtx()); }
}

const els = {};
const get = id => (els[id] ||= new E(id));
globalThis.document = {
  getElementById: get,
  querySelectorAll: () => [],
  createElement: () => new E("tmp"),
};
globalThis.requestAnimationFrame = () => 0;

const html = readFileSync(
  new URL("../viz/himoe-moe-trap-animation.html", import.meta.url), "utf8");
const script = html.slice(html.lastIndexOf("<script>") + 8, html.lastIndexOf("</script>"));

try { new Function(script)(); }
catch (e) { console.log("[FAIL] script threw on load:", e.message); process.exit(1); }
console.log("[PASS] script loaded, initial render produced", ops.length, "ops");

// walk the timeline via the scrubber listener
const scrub = get("scrub"), cap = get("cap"), clock = get("clock");
const seen = new Set(), actOps = {};
let steps = 0, errors = 0;
for (let v = 0; v <= 1000; v += 5) {
  ops.length = 0; bad = [];
  scrub.value = String(v);
  try { scrub.fire("input", { target: { value: String(v) } }); }
  catch (e) { console.log(`[FAIL] threw at scrub=${v}: ${e.message}`); errors++; continue; }
  steps++;
  seen.add(cap.dataset.i);
  const t = parseFloat(clock.textContent);
  const act = t < 6.5 ? 0 : t < 17.5 ? 1 : t < 32.5 ? 2 : t < 43.5 ? 3 : 4;
  actOps[act] = Math.max(actOps[act] || 0, ops.length);
  // t=0 is the first tick of a fade-in and is legitimately sparse
  if (v > 0 && ops.length < 10) { console.log(`[FAIL] scrub=${v} (t=${t}s) drew only ${ops.length} ops`); errors++; }
  if (bad.length) { console.log(`[FAIL] scrub=${v}: ${[...new Set(bad)].slice(0, 3).join(" | ")}`); errors += bad.length; }
}

console.log(`[${errors ? "FAIL" : "PASS"}] walked ${steps} timeline positions, ${errors} problems`);
for (const a of [0, 1, 2, 3, 4])
  console.log(`  ${actOps[a] ? "[PASS]" : "[FAIL]"} act ${a + 1}: peak ${actOps[a] || 0} draw ops`);
console.log(`[${seen.size === 6 ? "PASS" : "FAIL"}] captions advanced through ${seen.size}/6 beats`);
console.log(`[${get("tbl").innerHTML.includes("<table") ? "PASS" : "FAIL"}] data table rendered`);
console.log(`[${get("lg").innerHTML.includes("成功") ? "PASS" : "FAIL"}] legend rendered`);

const ok = !errors && seen.size === 6 && [0, 1, 2, 3, 4].every(a => actOps[a]);
console.log("\n" + (ok ? "ANIMATION OK" : "ANIMATION HAS FAILURES"));
process.exit(ok ? 0 : 1);
