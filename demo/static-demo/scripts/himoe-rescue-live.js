/* ===================================================================
   Rescue, live — the r(t) curve grows while the two futures play.
   Left: trunk in ink, alarm ring at q32, the original noise sinking
   red to 52 steps, the resampled branch climbing teal to done-in-8.
   Right: both futures as synced video frames. One clock, draggable.
   =================================================================== */
(function () {
  'use strict';
  var root = document.getElementById('rescue-live');
  if (!root) return;
  var cv = document.getElementById('rlive-canvas'), ctx = cv.getContext('2d');
  var vT = document.getElementById('rlive-trunk'), vO = document.getElementById('rlive-ok');
  var okwrap = document.getElementById('rlive-okwrap');
  var btn = document.getElementById('rlive-play');
  var bar = document.getElementById('rlive-bar'), fill = bar && bar.querySelector('i');
  var plot = cv.parentNode;
  var zh = document.documentElement.dataset.lang !== 'en';
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  var TR = [1.02, 1.0597, 1.0328, 1.0339, 0.9671, 1.0114, 1.0215, 1.0556, 1.0549, 1.0977, 1.0274, 1.0136, 1.0015, 0.975, 1.0492, 1.0676, 1.0539, 1.0725, 0.9955, 0.9788, 1.0188, 1.0092, 1.0748, 1.0854, 1.0228, 1.0129, 0.9961, 0.9759, 0.9591, 0.952, 0.941, 0.933, 0.8496, 0.9102, 0.9278, 0.9464, 0.9239, 0.8757, 0.8761, 0.8316, 0.9078, 0.9314, 0.8934, 0.9409, 0.899, 0.9502, 0.9389, 0.8565, 0.7673, 0.614, 0.5918, 0.5907];                    /* trunk r(t), q0..q51 (head nulls) */
  var RS = [0.8496, 0.9428, 0.9522, 0.9949, 0.9974, 0.9444, 0.941, 0.9077];                    /* rescued r(t), from q32 */
  var A = 32, TH = 0.95, STEP = 0.5, DUR = 26.0, OFF = 16.0, RDUR = 3.9, RATE = 1.5;

  var C = {};
  function readColors() {
    var cs = getComputedStyle(document.documentElement);
    var g = function (n, f) { var v = cs.getPropertyValue(n).trim(); return v || f; };
    C.ink = g('--ink', '#1D1A14'); C.gray = g('--gray', '#5B5448');
    C.teal = g('--teal', '#0C6E61'); C.gold = g('--gold', '#8C5E0E');
    C.red = g('--red', '#AC372B'); C.line = g('--line-strong', 'rgba(29,26,20,.32)');
  }
  readColors();

  var DPR = 1, W = 0, H = 0;
  function resize() {
    DPR = Math.min(2, window.devicePixelRatio || 1);
    var r = plot.getBoundingClientRect();
    W = Math.max(1, Math.round(r.width)); H = Math.max(1, Math.round(r.height));
    cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0); draw();
  }

  var ML = 46, MR = 14, MT = 12, MB = 28, Y0 = 0.30, Y1 = 1.30;
  function X(q) { return ML + q / 52 * (W - ML - MR); }
  function Y(v) { v = Math.max(Y0, Math.min(Y1, v)); return MT + (Y1 - v) / (Y1 - Y0) * (H - MT - MB); }

  function path(pts) {
    ctx.beginPath(); var started = false;
    for (var i = 0; i < pts.length; i++) {
      if (pts[i] == null) continue;
      var x = X(pts[i][0]), y = Y(pts[i][1]);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
  }
  function series(arr, from, upto, q0) {
    var out = [];
    for (var q = from; q <= upto && q - q0 < arr.length; q++) {
      var v = arr[q - q0]; if (v == null) continue; out.push([q, v]);
    }
    return out;
  }
  function lerpAt(arr, q0, qf) {                       /* value at fractional step */
    var i = Math.floor(qf) - q0, u = qf - Math.floor(qf);
    var a = arr[i], b = arr[i + 1];
    if (a == null) return null; if (b == null) return a;
    return a + (b - a) * u;
  }

  function draw() {
    if (!W) return;
    var t = vT.currentTime || 0, sf = t / STEP;        /* fractional step */
    ctx.clearRect(0, 0, W, H);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    /* axes */
    ctx.strokeStyle = C.line; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ML, H - MB); ctx.lineTo(W - MR, H - MB); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ML, MT); ctx.lineTo(ML, H - MB); ctx.stroke();
    ctx.fillStyle = C.gray; ctx.font = '11px "IBM Plex Mono",monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (var q = 0; q <= 50; q += 10) ctx.fillText(String(q), X(q), H - MB + 6);
    ctx.save(); ctx.translate(12, (MT + H - MB) / 2); ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = 'middle';
    ctx.font = '11px "Hanken Grotesk","Noto Sans SC",sans-serif';
    ctx.fillText(zh ? '神经元 pattern 变化率 r(t)' : 'neuron-pattern change r(t)', 0, 0);
    ctx.restore();
    /* theta */
    ctx.setLineDash([5, 5]); ctx.strokeStyle = C.gray; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ML, Y(TH)); ctx.lineTo(W - MR, Y(TH)); ctx.stroke(); ctx.setLineDash([]);
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('θ = 0.95', ML + 6, Y(TH) - 3);
    /* trunk: ink before the alarm, red after */
    var pre = series(TR, 0, Math.min(sf, A), 0);
    if (sf > 32) { } /* partial tip below */
    path(pre); ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.globalAlpha = .85; ctx.stroke(); ctx.globalAlpha = 1;
    if (sf > A) {
      var post = series(TR, A, Math.min(sf, 51), 0);
      path(post); ctx.strokeStyle = C.red; ctx.lineWidth = 2.2; ctx.stroke();
    }
    /* partial segment tip on the active curve */
    var tipV = lerpAt(TR, 0, Math.min(sf, 51));
    if (tipV != null && sf < 51.9) {
      ctx.fillStyle = sf >= A ? C.red : C.ink;
      ctx.beginPath(); ctx.arc(X(Math.min(sf, 51)), Y(tipV), 3.2, 0, 7); ctx.fill();
    }
    /* rescued branch from q32 */
    if (sf >= A) {
      var rb = series(RS, A, Math.min(sf, A + RS.length - 1), A);
      path(rb); ctx.strokeStyle = C.teal; ctx.lineWidth = 2.6; ctx.stroke();
      var rV = lerpAt(RS, A, Math.min(sf, A + RS.length - 1));
      if (rV != null && sf < A + RS.length - 1) {
        ctx.fillStyle = C.teal; ctx.beginPath(); ctx.arc(X(Math.min(sf, A + RS.length - 1)), Y(rV), 3.2, 0, 7); ctx.fill();
      }
      if (sf >= A + RS.length - 1) {                    /* done: check + label */
        var ex = X(A + RS.length - 1), ey = Y(RS[RS.length - 1]);
        ctx.strokeStyle = C.teal; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(ex + 10, ey - 2); ctx.lineTo(ex + 15, ey + 4); ctx.lineTo(ex + 24, ey - 8); ctx.stroke();
        ctx.fillStyle = C.teal; ctx.font = '600 12px "Hanken Grotesk","Noto Sans SC",sans-serif';
        ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(zh ? '8 步 · 完成' : '8 steps · done', ex + 30, ey + 4);
      }
      /* alarm ring */
      var av = TR[A] != null ? TR[A] : TH;
      var pulse = 1 + 0.15 * Math.sin((t - OFF) * 4);
      ctx.strokeStyle = C.gold; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.arc(X(A), Y(av), 7 * (sf < A + 2 ? pulse : 1), 0, 7); ctx.stroke();
      ctx.fillStyle = C.gold; ctx.font = '600 11px "Hanken Grotesk","Noto Sans SC",sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(zh ? '报警 q32' : 'alarm q32', X(A), Y(av) - 12);
    }
    if (sf >= 51.5) {                                   /* trunk end: cross + label */
      var fx = X(51), fy = Y(TR[51] != null ? TR[51] : 0.4);
      ctx.strokeStyle = C.red; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(fx - 16, fy - 14); ctx.lineTo(fx - 6, fy - 4);
      ctx.moveTo(fx - 6, fy - 14); ctx.lineTo(fx - 16, fy - 4); ctx.stroke();
      ctx.fillStyle = C.red; ctx.font = '600 12px "Hanken Grotesk","Noto Sans SC",sans-serif';
      ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      ctx.fillText(zh ? '52 步 · 失败' : '52 steps · failed', fx, fy + 6);
    }
    /* cursor */
    var cx2 = X(Math.min(sf, 52));
    ctx.strokeStyle = C.gold; ctx.globalAlpha = .55; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx2, MT); ctx.lineTo(cx2, H - MB); ctx.stroke(); ctx.globalAlpha = 1;
  }

  /* ---- clock + sync ---- */
  var playing = false, scrubbing = false, wasPlaying = false;
  vT.playbackRate = RATE; vO.playbackRate = RATE;
  function syncOk() {
    var t = vT.currentTime || 0;
    if (t / STEP >= A) {
      okwrap.classList.remove('is-wait');
      var want = Math.max(0, Math.min(t - OFF, RDUR - 0.05));
      if (Math.abs((vO.currentTime || 0) - want) > 0.35) vO.currentTime = want;
      if (playing && vO.paused && want < RDUR - 0.1) vO.play().catch(function () {});
      if (!playing && !vO.paused) vO.pause();
    } else {
      okwrap.classList.add('is-wait');
      if (!vO.paused) vO.pause();
      if ((vO.currentTime || 0) > 0.05) vO.currentTime = 0;
    }
  }
  function play() { playing = true; vT.play().catch(function () {}); setBtn(); }
  function pause() { playing = false; vT.pause(); vO.pause(); setBtn(); }
  function setBtn() { if (!btn) return; btn.textContent = playing ? '❚❚' : '▶'; btn.classList.toggle('is-dim', playing); }
  vT.addEventListener('ended', function () {
    if (!playing) return;
    vT.currentTime = 0; vO.currentTime = 0; vO.pause();
    vT.play().catch(function () {});
  });
  (function tick() {
    syncOk(); draw();
    if (fill) fill.style.width = ((vT.currentTime || 0) / DUR * 100) + '%';
    requestAnimationFrame(tick);
  })();

  function seekTo(clientX) {
    var r = bar.getBoundingClientRect();
    var u = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    vT.currentTime = u * (vT.duration || DUR) * 0.999;
    syncOk();
  }
  if (bar) {
    bar.addEventListener('pointerdown', function (e) {
      scrubbing = true; wasPlaying = playing; pause();
      try { bar.setPointerCapture(e.pointerId); } catch (err) {}
      seekTo(e.clientX); e.preventDefault();
    });
    bar.addEventListener('pointermove', function (e) { if (scrubbing) seekTo(e.clientX); });
    bar.addEventListener('pointerup', function () { scrubbing = false; if (wasPlaying) play(); });
    bar.addEventListener('pointercancel', function () { scrubbing = false; });
  }
  if (btn) btn.addEventListener('click', function (e) { e.stopPropagation(); playing ? pause() : play(); });
  root.addEventListener('click', function (e) {
    if (e.target === btn || (bar && bar.contains(e.target))) return;
    playing ? pause() : play();
  });
  document.addEventListener('visibilitychange', function () { if (document.hidden && playing) pause(); });
  if (!reduce && 'IntersectionObserver' in window) {
    new IntersectionObserver(function (es) {
      es.forEach(function (en) {
        if (en.isIntersecting) { if (!playing && !scrubbing) play(); }
        else if (playing) pause();
      });
    }, { threshold: 0.3 }).observe(root);
  } else if (reduce) { vT.currentTime = DUR * 0.999; }

  window.radRescue = { setTheme: function () { readColors(); draw(); } };
  var rz; window.addEventListener('resize', function () { clearTimeout(rz); rz = setTimeout(resize, 150); }, { passive: true });
  resize();
})();
