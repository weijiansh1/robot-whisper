/* ===================================================================
   The pair, live — two real r(t) curves grow while their rollouts
   play. Teal = success (finishes at 38); red = failure (three steps
   under θ, alarm at q34, sinks to timeout). One clock, draggable.
   =================================================================== */
(function () {
  'use strict';
  var root = document.getElementById('duo-live');
  if (!root) return;
  var cv = document.getElementById('dlive-canvas'), ctx = cv.getContext('2d');
  var vOK = document.getElementById('dlive-ok'), vBAD = document.getElementById('dlive-bad');
  var btn = document.getElementById('dlive-play');
  var bar = document.getElementById('dlive-bar'), fill = bar && bar.querySelector('i');
  var plot = cv.parentNode;
  var zh = document.documentElement.dataset.lang !== 'en';
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  var ROK = [1.087, 1.1195, 1.0955, 1.0763, 1.087, 1.024, 0.948, 0.933, 0.913, 0.958, 0.953, 1.055, 1.225, 1.227, 1.227, 1.171, 1.087, 1.097, 1.145, 1.263, 1.22, 1.202, 1.142, 1.012, 1.031, 1.091, 1.073, 1.058, 1.095, 1.119, 1.284, 1.347, 1.265, 1.19, 1.093, 0.994, 1.01, 1.054];
  var RBAD = [1.106, 1.1385, 1.1145, 1.0953, 1.106, 1.042, 0.967, 0.941, 0.894, 0.93, 0.958, 1.079, 1.224, 1.216, 1.182, 1.106, 1.054, 1.067, 1.116, 1.234, 1.218, 1.211, 1.164, 1.053, 1.002, 1.038, 1.031, 1.023, 1.062, 0.954, 0.935, 0.964, 0.806, 0.744, 0.605, 0.412, 0.392, 0.378, 0.409, 0.413, 0.428, 0.389, 0.361, 0.356, 0.351, 0.388, 0.39, 0.355, 0.332, 0.341, 0.357, 0.405];
  var TH = 0.925, AF = 34, STEP = 0.5, DUR = 26.0, RATE = 1.5;
  var EXTRA = [{"k": "ok", "r": [1.059, 1.0915, 1.0675, 1.0483, 1.059, 0.995, 0.978, 0.936, 0.941, 0.915, 0.988, 1.146, 1.24, 1.296, 1.209, 1.135, 1.064, 1.047, 1.095, 1.187, 1.18, 1.194, 1.156, 1.047, 1.053, 0.995, 0.994, 0.999, 1.014, 1.165, 1.297, 1.322, 1.305, 1.181, 1.111, 1.044, 1.058, 1.092], "a": null}, {"k": "ok", "r": [0.838, 0.8705, 0.8465, 0.8273, 0.838, 0.844, 0.902, 1.038, 1.163, 1.192, 1.082, 1.03, 0.991, 0.988, 1.065, 1.136, 1.103, 1.088, 1.013, 0.915, 0.941, 0.934, 0.952, 0.972, 0.986, 1.05, 1.175, 1.199, 1.133, 1.03, 0.939, 0.889, 0.907, 0.954], "a": null}, {"k": "ok", "r": [0.883, 0.9155, 0.8915, 0.8723, 0.883, 0.863, 0.899, 0.989, 1.117, 1.222, 1.209, 1.122, 1.069, 1.029, 1.042, 1.1, 1.17, 1.148, 1.103, 1.03, 0.911, 0.904, 0.928, 0.93, 0.96, 0.995, 0.957, 1.081, 1.192, 1.217, 1.222, 1.132, 1.035, 0.957, 0.946, 0.93], "a": null}, {"k": "ok", "r": [0.908, 0.9405, 0.9165, 0.8973, 0.908, 0.894, 0.893, 0.971, 1.092, 1.2, 1.227, 1.149, 1.073, 1.016, 1.0, 1.067, 1.135, 1.133, 1.151, 1.103, 0.994, 1.01, 0.997, 1.003, 1.062, 1.033, 1.075, 1.177, 1.184, 1.152, 1.071, 0.985, 0.922, 0.947, 0.959], "a": null}, {"k": "ok", "r": [1.028, 1.0605, 1.0365, 1.0173, 1.028, 1.028, 0.962, 0.96, 0.972, 1.049, 1.223, 1.353, 1.387, 1.323, 1.266, 1.203, 1.183, 1.252, 1.29, 1.263, 1.272, 1.199, 1.083, 1.086, 1.075, 1.075, 1.071, 1.086, 1.192, 1.353, 1.416, 1.382, 1.263, 1.127, 1.103, 1.086, 1.089], "a": null}, {"k": "ok", "r": [1.041, 1.0735, 1.0495, 1.0303, 1.041, 1.048, 0.956, 0.968, 0.959, 1.055, 1.217, 1.324, 1.365, 1.321, 1.252, 1.22, 1.204, 1.293, 1.261, 1.222, 1.218, 1.075, 1.056, 1.075, 1.066, 1.071, 1.083, 1.062, 1.111, 1.083, 0.947, 0.827, 0.667, 0.568, 0.618, 0.668, 0.692, 0.69, 0.797, 0.867, 0.977], "a": 33}, {"k": "ok", "r": [1.063, 1.0955, 1.0715, 1.0523, 1.063, 0.952, 0.871, 0.915, 0.937, 0.974, 1.014, 0.877, 0.913, 1.01, 1.055, 1.122, 1.108, 0.997, 1.038, 1.081, 1.17, 1.151, 1.098, 1.051, 0.944, 1.001, 0.98, 1.054, 1.047, 0.984, 1.07, 1.137, 1.177, 1.193, 1.087, 0.996, 0.997, 0.986, 0.948, 0.986], "a": null}, {"k": "bad", "r": [0.854, 0.8865, 0.8625, 0.8433, 0.854, 0.818, 0.855, 0.995, 1.146, 1.211, 1.156, 1.081, 1.046, 1.026, 1.043, 1.124, 1.135, 1.093, 1.095, 0.995, 0.935, 0.957, 0.94, 0.942, 0.976, 0.895, 0.891, 0.905, 0.951, 0.91, 0.788, 0.631, 0.664, 0.762, 0.876, 0.987, 0.833, 0.8, 0.79, 0.744, 0.774, 0.791, 0.779, 0.908, 0.943, 0.956, 0.952], "a": 27}, {"k": "bad", "r": [1.026, 1.0585, 1.0345, 1.0153, 1.026, 0.904, 0.926, 0.962, 0.974, 0.958, 0.864, 0.703, 0.725, 0.902, 1.054, 1.195, 1.131, 0.996, 0.933, 0.927, 1.024, 1.13, 1.099, 1.062, 0.955, 0.837, 0.895, 0.915, 0.936, 0.919, 0.8, 0.725, 0.621, 0.572, 0.527, 0.478, 0.44, 0.409, 0.366, 0.344, 0.332, 0.325, 0.475, 0.69, 1.0, 1.191, 1.184, 1.147, 1.01, 0.914, 0.899], "a": 27}, {"k": "bad", "r": [1.038, 1.0705, 1.0465, 1.0273, 1.038, 0.905, 0.941, 0.961, 0.962, 0.978, 0.883, 0.843, 0.949, 1.062, 1.173, 1.205, 1.071, 1.02, 0.983, 1.016, 1.119, 1.097, 1.02, 0.8689, 0.7298, 0.6302, 0.5941, 0.5262, 0.4412, 0.3875, 0.3733, 0.3815, 0.406, 0.4283, 0.4314, 0.413, 0.3871, 0.3732, 0.3819, 0.4065, 0.4286, 0.4312, 0.4126, 0.3867, 0.3732, 0.3822, 0.407, 0.4288, 0.4311, 0.4121, 0.3863], "a": 25}, {"k": "bad", "r": [1.067, 1.0995, 1.0755, 1.0563, 1.067, 0.993, 0.972, 0.944, 0.933, 0.942, 0.932, 0.871, 0.797, 0.715, 0.692, 0.828, 0.988, 1.083, 1.163, 1.106, 0.962, 0.967, 0.911, 0.947, 1.097, 1.109, 1.091, 0.956, 0.8061, 0.7359, 0.6505, 0.6108, 0.5634, 0.5013, 0.4574, 0.4318, 0.4202, 0.4313, 0.4567, 0.4772, 0.4773, 0.4569, 0.4314, 0.4202, 0.4317, 0.4572, 0.4774, 0.477, 0.4564, 0.431, 0.4202], "a": 30}, {"k": "bad", "r": [0.999, 1.0315, 1.0075, 0.9883, 0.999, 1.031, 1.039, 1.011, 1.001, 1.092, 1.19, 1.3, 1.289, 1.133, 1.055, 1.036, 1.116, 1.208, 1.139, 1.069, 0.932, 0.887, 0.922, 0.939, 1.03, 1.028, 0.9612, 0.9169, 0.8092, 0.6496, 0.5181, 0.4164, 0.3722, 0.3505, 0.3256, 0.3164, 0.3298, 0.3557, 0.3744, 0.3719, 0.35, 0.3253, 0.3165, 0.3302, 0.3561, 0.3746, 0.3716, 0.3495, 0.3249], "a": null}, {"k": "bad", "r": [1.04, 1.0725, 1.0485, 1.0293, 1.04, 1.019, 0.927, 0.934, 0.96, 0.99, 0.99, 0.929, 0.951, 1.014, 1.133, 1.203, 1.154, 1.022, 0.96, 0.958, 1.044, 1.118, 1.042, 0.962, 0.8021, 0.6978, 0.6438, 0.5569, 0.4867, 0.4433, 0.4329, 0.4283, 0.4053, 0.3813, 0.3746, 0.3901, 0.4162, 0.4331, 0.428, 0.4047, 0.381, 0.3747, 0.3906, 0.4167, 0.4332, 0.4277, 0.4042, 0.3807, 0.3748, 0.391, 0.4171, 0.4333], "a": 26}, {"k": "bad", "r": [0.931, 0.9635, 0.9395, 0.9203, 0.931, 0.957, 1.014, 1.038, 1.069, 1.008, 1.029, 1.126, 1.239, 1.274, 1.253, 1.134, 1.087, 1.071, 1.097, 1.22, 1.176, 1.112, 0.998, 0.88, 0.902, 0.986, 1.047, 1.023, 0.8522, 0.6683, 0.5442, 0.4589, 0.4435, 0.4301, 0.4384, 0.4645, 0.4793, 0.4717, 0.4474, 0.4248, 0.421, 0.4389, 0.4649, 0.4794, 0.4713, 0.4469, 0.4245, 0.4212, 0.4394, 0.4654], "a": 30}];

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
  var ML = 46, MR = 14, MT = 12, MB = 28, Y0 = 0.30, Y1 = 1.45;
  function X(q) { return ML + q / 52 * (W - ML - MR); }
  function Y(v) { v = Math.max(Y0, Math.min(Y1, v)); return MT + (Y1 - v) / (Y1 - Y0) * (H - MT - MB); }

  function drawCurve(arr, upto, col, lw) {
    ctx.beginPath(); var started = false;
    var n = Math.min(upto, arr.length - 1);
    for (var q = 0; q <= n; q++) {
      if (arr[q] == null) continue;
      var x = X(q), y = Y(arr[q]);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
    return n;
  }

  function draw() {
    if (!W) return;
    var t = vBAD.currentTime || 0, sf = t / STEP;
    ctx.clearRect(0, 0, W, H);
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
    ctx.fillText(zh ? '神经元 pattern' : 'neuron pattern', 0, 0);
    ctx.restore();
    /* theta */
    ctx.setLineDash([5, 5]); ctx.strokeStyle = C.gray; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ML, Y(TH)); ctx.lineTo(W - MR, Y(TH)); ctx.stroke(); ctx.setLineDash([]);
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('θ', ML + 6, Y(TH) - 3);
    /* background fan: more real curves, faint */
    for (var e = 0; e < EXTRA.length; e++) {
      var ex2 = EXTRA[e];
      ctx.globalAlpha = 0.16;
      drawCurve(ex2.r, Math.floor(sf), ex2.k === 'ok' ? C.teal : C.red, 1.1);
      ctx.globalAlpha = 1;
      if (ex2.a != null && sf >= ex2.a && ex2.r[ex2.a] != null) {
        ctx.fillStyle = C.gold; ctx.globalAlpha = 0.55;
        ctx.beginPath(); ctx.arc(X(ex2.a), Y(ex2.r[ex2.a]), 2.2, 0, 7); ctx.fill();
        ctx.globalAlpha = 1;
      }
    }
    /* the starring pair */
    var nOK = drawCurve(ROK, Math.floor(sf), C.teal, 2.2);
    var nBAD = drawCurve(RBAD, Math.floor(sf), C.red, 2.2);
    /* heads */
    if (sf < ROK.length - 1 && ROK[nOK] != null) { ctx.fillStyle = C.teal; ctx.beginPath(); ctx.arc(X(nOK), Y(ROK[nOK]), 3.2, 0, 7); ctx.fill(); }
    if (sf < RBAD.length - 1 && RBAD[nBAD] != null) { ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(X(nBAD), Y(RBAD[nBAD]), 3.2, 0, 7); ctx.fill(); }
    /* success end: check */
    if (sf >= ROK.length - 1) {
      var ex = X(ROK.length - 1), ey = Y(ROK[ROK.length - 1]);
      ctx.strokeStyle = C.teal; ctx.lineWidth = 2.4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(ex + 8, ey - 2); ctx.lineTo(ex + 13, ey + 4); ctx.lineTo(ex + 22, ey - 8); ctx.stroke();
      ctx.fillStyle = C.teal; ctx.font = '600 12px "Hanken Grotesk","Noto Sans SC",sans-serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(zh ? '38 步 · 完成' : '38 steps · done', ex + 28, ey);
    }
    /* failure end: cross */
    if (sf >= 51.5) {
      var fx = X(51), fy = Y(RBAD[51] != null ? RBAD[51] : 0.42);
      ctx.strokeStyle = C.red; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(fx - 14, fy - 12); ctx.lineTo(fx - 4, fy - 2);
      ctx.moveTo(fx - 4, fy - 12); ctx.lineTo(fx - 14, fy - 2); ctx.stroke();
      ctx.fillStyle = C.red; ctx.font = '600 12px "Hanken Grotesk","Noto Sans SC",sans-serif';
      ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      ctx.fillText(zh ? '52 步 · 超时' : '52 steps · timeout', fx, fy + 8);
    }
    /* alarm ring at q34 */
    if (sf >= AF) {
      var av = RBAD[AF] != null ? RBAD[AF] : TH;
      var pulse = 1 + 0.15 * Math.sin((t - AF * STEP) * 4);
      ctx.strokeStyle = C.gold; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.arc(X(AF), Y(av), 7 * (sf < AF + 2 ? pulse : 1), 0, 7); ctx.stroke();
      ctx.fillStyle = C.gold; ctx.font = '600 11px "Hanken Grotesk","Noto Sans SC",sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(zh ? '报警 q34' : 'alarm q34', X(AF), Y(av) - 12);
    }
    /* cursor */
    var cx2 = X(Math.min(sf, 52));
    ctx.strokeStyle = C.gold; ctx.globalAlpha = .5; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx2, MT); ctx.lineTo(cx2, H - MB); ctx.stroke(); ctx.globalAlpha = 1;
  }

  var playing = false, scrubbing = false, wasPlaying = false;
  vOK.playbackRate = RATE; vBAD.playbackRate = RATE;
  function syncOK() {
    if (!vOK.duration) return;
    var want = Math.min(vBAD.currentTime || 0, vOK.duration - 0.05);
    if (Math.abs((vOK.currentTime || 0) - want) > 0.4) vOK.currentTime = want;
    if (playing && vOK.paused && want < vOK.duration - 0.1) vOK.play().catch(function () {});
    if (!playing && !vOK.paused) vOK.pause();
  }
  function play() { playing = true; vBAD.play().catch(function () {}); setBtn(); }
  function pause() { playing = false; vBAD.pause(); vOK.pause(); setBtn(); }
  function setBtn() { if (!btn) return; btn.textContent = playing ? '❚❚' : '▶'; btn.classList.toggle('is-dim', playing); }
  vBAD.addEventListener('ended', function () {
    if (!playing) return;
    vBAD.currentTime = 0; vOK.currentTime = 0;
    vBAD.play().catch(function () {});
  });
  (function tick() {
    syncOK(); draw();
    if (fill) fill.style.width = ((vBAD.currentTime || 0) / DUR * 100) + '%';
    requestAnimationFrame(tick);
  })();

  function seekTo(clientX) {
    var r = bar.getBoundingClientRect();
    var u = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    vBAD.currentTime = u * (vBAD.duration || DUR) * 0.999;
    syncOK();
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
  } else if (reduce) { vBAD.currentTime = DUR * 0.999; }

  window.radDuo = { setTheme: function () { readColors(); draw(); } };
  var rz; window.addEventListener('resize', function () { clearTimeout(rz); rz = setTimeout(resize, 150); }, { passive: true });
  resize();
})();
