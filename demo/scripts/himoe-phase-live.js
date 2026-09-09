/* ===================================================================
   Phase space, live — schematic orbits over a real background.
   Six choreographed sample orbits trace themselves in (r, Δr):
   successes circle the cruising zone and land; loops run a clean
   closed cycle; stagnation dives left and dies on the fixed point,
   a gold ring flashing on the way down. The scatter of final states
   (all 352 rollouts) and the two zones are measured; the orbits are
   schematic. Click to pause; pauses off-screen; loops forever.
   =================================================================== */
(function () {
  'use strict';
  var box = document.getElementById('phase-live');
  if (!box) return;
  var cv = box.querySelector('canvas'), ctx = cv.getContext('2d');
  var zh = document.documentElement.dataset.lang !== 'en';
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  var CLOUDS = {"ok": [[0.9226, -0.0383], [0.8426, -0.0324], [0.9826, -0.0358], [0.9706, -0.0173], [0.743, -0.069], [0.987, -0.0655], [0.979, -0.047], [0.975, -0.0459], [0.9888, -0.0627], [0.9328, -0.059], [0.986, -0.0373], [1.0542, -0.0265], [0.9572, -0.0368], [0.913, -0.0438], [1.0806, -0.0321], [1.0186, -0.0453], [0.9904, -0.0569], [1.0368, -0.0276], [1.0824, -0.0437], [1.2306, -0.0315], [0.8756, -0.0243], [0.8362, -0.053], [0.9146, -0.0385], [1.125, -0.0386], [0.9964, -0.0371], [0.9278, -0.0469], [1.203, -0.0132], [1.1878, 0.0277], [1.1658, -0.0057], [1.139, -0.0575], [1.003, -0.0497], [1.0728, -0.0531], [1.1134, -0.0575], [1.0272, -0.0569], [0.8788, -0.0223], [0.9096, -0.0132], [1.0682, -0.0559], [1.0764, -0.0499], [1.0552, -0.0547], [1.0038, -0.0279], [1.0104, -0.0549], [1.0896, -0.0582], [0.9674, -0.057], [1.168, -0.0513], [1.0094, -0.0552], [1.0772, -0.0305], [0.9996, -0.0436], [1.012, -0.073], [0.943, -0.0412], [1.0504, -0.0292], [1.1, -0.0475], [0.8274, 0.0477], [1.0444, -0.0409], [1.0558, -0.0557], [1.0972, -0.0487], [1.065, -0.0347], [1.0552, -0.0312], [0.975, -0.0158], [1.0946, -0.0533], [1.1706, -0.0423], [0.8046, 0.0525], [1.0838, -0.0375], [1.0966, -0.061], [1.1012, -0.0686], [1.1596, -0.0169], [0.8724, -0.0468], [1.0448, -0.0413], [1.0652, -0.0534], [1.034, -0.0279], [0.9018, -0.0121], [1.1146, -0.0725], [1.056, -0.059], [1.1094, -0.0775], [1.0946, -0.0479], [1.1336, -0.0582], [0.96, -0.0451], [1.0476, -0.0469], [1.133, -0.0409], [0.9148, -0.0193], [1.0874, -0.0421], [1.0948, -0.0579], [1.0302, -0.0596], [1.0496, -0.0503], [1.0976, -0.0517], [1.075, -0.0521], [0.9024, -0.0278], [1.0584, -0.024], [1.0656, -0.0534], [1.1246, -0.0557], [1.1062, -0.0563], [0.9768, -0.0457], [0.9918, -0.0543], [1.0, -0.0532], [0.978, -0.0532], [1.0334, -0.0435], [1.0, -0.04], [1.1088, -0.0355], [0.9786, -0.0565], [1.0294, -0.0551], [1.03, -0.0331], [0.9632, -0.0366], [0.979, -0.0553], [0.7796, 0.0501], [0.986, -0.0547], [0.8896, -0.0458], [0.9122, -0.0691], [0.6034, 0.0439], [0.9238, -0.0259], [0.8836, 0.0037], [1.0128, -0.0371], [0.967, -0.053], [0.8294, -0.0336], [0.9548, -0.0262], [0.9598, -0.0429], [0.7986, -0.0198], [0.9718, -0.0292], [0.9438, -0.0505]], "lp": [[0.8258, -0.0585], [0.6354, -0.0411], [1.0838, 0.0167], [1.0308, -0.0039], [0.7248, -0.0459], [0.9788, 0.0079], [0.9466, 0.0267], [0.8426, -0.0443], [0.8916, 0.0219], [0.5304, -0.0683], [0.8034, -0.0589], [1.122, 0.0229], [0.9546, -0.0335], [0.9784, 0.0195], [1.1132, 0.0155], [1.1246, 0.0157], [0.9224, 0.0166], [1.0726, -0.0361], [1.0624, 0.0025], [0.8514, -0.0437], [1.1358, 0.0483], [0.724, -0.0693], [0.4004, 0.0039], [0.8066, -0.0317], [0.887, -0.0227], [1.1812, 0.0189], [1.1638, -0.0142], [0.8384, -0.0255], [1.1338, 0.0361], [0.938, -0.0428], [0.9746, 0.0107], [1.2138, 0.0667], [0.5724, -0.0092], [0.974, -0.0361], [0.947, -0.0401], [0.9376, -0.0219], [1.0144, 0.0459], [0.5128, 0.0014], [0.966, -0.0401], [0.9336, -0.0465], [0.9632, 0.0555], [1.0536, 0.02], [0.8624, -0.0497], [1.0134, 0.0355], [1.0004, -0.0241], [1.0282, -0.0162], [0.8726, -0.0439], [0.723, 0.0358], [1.1076, 0.0492], [1.0818, 0.0484], [0.877, -0.0355], [0.9602, -0.0343], [1.016, -0.0356], [1.1034, 0.0173], [1.0842, 0.0015], [1.0694, 0.0389], [0.892, -0.0517], [1.037, -0.0221], [0.9998, 0.0265], [1.0798, 0.0475], [1.0518, -0.0296], [1.0252, -0.0253], [0.9478, -0.0307], [0.9894, -0.0397], [0.985, 0.0393], [1.0302, 0.0221], [0.9286, -0.0289], [1.0544, -0.0195], [1.0036, -0.0151], [1.0928, 0.0029], [1.0636, -0.0175], [0.9936, -0.0425], [0.9332, -0.0109], [0.9986, 0.0479], [0.9814, 0.0135], [0.4626, 0.002], [1.1162, -0.0035], [0.5878, -0.0561], [0.9392, -0.0277], [1.0016, -0.0343], [0.955, -0.0274], [0.9446, -0.035], [1.0826, -0.0317], [0.9126, -0.0441], [0.938, -0.0486], [1.046, -0.0039], [0.9702, -0.0264], [1.0144, -0.0287], [0.9948, -0.0199], [1.0222, -0.0285], [1.0846, -0.0078], [0.9684, -0.0416], [1.0412, -0.0389], [1.0618, -0.0368], [1.0708, -0.0266], [1.0248, -0.0484], [1.0312, -0.0121], [1.041, -0.0628], [1.13, 0.0399], [1.213, 0.0249], [0.9352, -0.0605], [0.9918, -0.0435], [0.9172, 0.0316], [1.0356, -0.0355], [1.1604, -0.0279], [1.0492, 0.0011], [1.061, -0.023], [0.993, -0.0489], [1.1384, -0.0273], [1.066, 0.0486], [0.675, -0.0986], [1.026, 0.0219], [0.9098, -0.0439], [1.0524, 0.0275], [0.5944, -0.0353], [1.1008, 0.0492], [0.973, -0.0387], [0.8828, -0.0211], [0.8672, -0.0593], [0.775, -0.0681], [0.7486, -0.0444], [1.096, 0.0132], [1.037, 0.0049], [0.9928, 0.0001], [0.803, -0.0248], [1.1324, 0.0349], [1.0972, 0.0129], [0.8878, -0.0581], [1.0236, 0.0086], [1.1338, 0.0293], [1.0422, 0.0055], [0.9616, -0.0441], [0.7164, -0.0895], [0.8726, 0.0205], [0.9714, -0.0045], [0.9006, 0.0457], [0.908, -0.0608], [0.7032, -0.0454], [0.8366, -0.0383], [0.7722, 0.0141], [0.8828, 0.0205], [1.0552, 0.038], [0.7608, -0.0394], [0.8582, -0.0371], [1.0018, -0.016], [0.8826, -0.0465], [0.7204, -0.0193], [0.674, -0.0759], [0.8884, -0.056], [0.9076, 0.0361]], "st": [[0.448, 0.0041], [0.4912, 0.043], [0.4272, 0.0113], [0.3264, 0.0048], [0.4654, 0.0004], [0.454, 0.0172], [1.0184, 0.1259], [0.4742, -0.0421], [0.5018, -0.0023], [0.4404, -0.0139], [0.3754, -0.0012], [0.4674, 0.0249], [0.5206, 0.0296], [0.4568, 0.0263], [0.4402, -0.0159], [0.3352, -0.0166], [0.4246, -0.0015], [0.5216, 0.0054], [0.4086, -0.0255], [0.391, -0.0059], [0.4806, 0.0037], [0.367, -0.0122], [0.4656, 0.0157], [0.6514, 0.0609], [0.4292, -0.0011], [0.447, -0.0069], [0.464, -0.0039], [0.416, -0.0009], [0.5844, 0.0322], [0.401, 0.0161], [0.4924, 0.0261], [0.3568, -0.0204], [0.415, -0.0059], [0.5912, 0.0295], [0.4306, -0.0078], [0.4914, -0.021], [0.376, -0.007], [0.494, 0.0134], [0.4074, 0.0013], [0.8852, 0.1095], [0.4906, -0.0037], [0.4638, -0.0043], [0.4546, -0.0011], [0.392, -0.0259], [0.4276, -0.0123], [0.447, -0.0163], [0.5708, 0.0283], [0.3768, -0.0227], [0.4726, -0.0051]]};

  var C = {};
  function readColors() {
    var cs = getComputedStyle(document.documentElement);
    var g = function (n, f) { var v = cs.getPropertyValue(n).trim(); return v || f; };
    C.ink = g('--ink', '#1D1A14'); C.gray = g('--gray', '#5B5448');
    C.teal = g('--teal', '#0C6E61'); C.gold = g('--gold', '#8C5E0E');
    C.red = g('--red', '#AC372B'); C.line = g('--line-strong', 'rgba(29,26,20,.32)');
  }
  readColors();
  function withA(col, a) {
    if (col[0] === '#') {
      var r = parseInt(col.slice(1, 3), 16), g2 = parseInt(col.slice(3, 5), 16), b = parseInt(col.slice(5, 7), 16);
      return 'rgba(' + r + ',' + g2 + ',' + b + ',' + a + ')';
    }
    return col;
  }
  function ease(t) { t = t < 0 ? 0 : t > 1 ? 1 : t; return t * t * t * (t * (6 * t - 15) + 10); }

  /* ---- schematic orbits ---- */
  function mkSuccess(off, cx, laps) {
    /* healthy thought never repeats: an aperiodic wander inside the
       cruising zone (incommensurate sines), then it settles */
    var pts = [], N = 150, i;
    for (i = 0; i < N; i++) {
      var t = i / N * 10 + off;
      var x = cx + 0.075 * Math.sin(2.3 * t + off) + 0.045 * Math.sin(5.13 * t + 1.1)
                 + 0.020 * Math.sin(8.7 * t);
      var y = -0.018 + 0.034 * Math.sin(3.71 * t + 0.6) + 0.018 * Math.sin(1.63 * t + off)
                     + 0.010 * Math.sin(7.9 * t + 2.0);
      pts.push([x, y]);
    }
    var last = pts[N - 1];
    for (i = 1; i <= 22; i++) {
      var u = ease(i / 22);
      pts.push([last[0] + (cx - 0.01 - last[0]) * u, last[1] + (-0.035 - last[1]) * u]);
    }
    return { k: 'ok', pts: pts, a: null, loop: false };
  }
  function mkLoop(off, rx, ry, tilt) {
    var pts = [], N = 130, i, aIdx = 0, minx = 1e9;
    for (i = 0; i < N; i++) {
      var ang = 6.2832 * i / N + off;
      var x = rx * Math.cos(ang), y = ry * Math.sin(ang);
      var xr = 0.94 + x * Math.cos(tilt) - y * Math.sin(tilt);
      var yr = 0.004 + x * Math.sin(tilt) * 0.4 + y * Math.cos(tilt);
      pts.push([xr, yr]);
      if (xr < minx) { minx = xr; aIdx = i; }       // alarm at the cycle's lowest r
    }
    return { k: 'lp', pts: pts, a: aIdx, loop: true };
  }
  function mkStag(off) {
    var pts = [], i, aIdx = null;
    for (i = 0; i < 55; i++) {                     /* a healthy lap first */
      var t = i / 55, ang = 6.2832 * 0.9 * t + off;
      pts.push([1.01 + 0.10 * Math.cos(ang), -0.015 + 0.040 * Math.sin(ang)]);
    }
    var start = pts[pts.length - 1];
    for (i = 1; i <= 75; i++) {                    /* the dive */
      var u = i / 75, eu = ease(u);
      var r = start[0] + (0.452 - start[0]) * eu;
      var dr = -0.072 * Math.sin(Math.PI * Math.min(1, u * 1.05)) * (1 - 0.25 * u) + 0.006 * Math.sin(9 * u + off);
      pts.push([r, dr * (u > 0.92 ? (1 - u) / 0.08 * 0.3 + 0.0 : 1)]);
      if (aIdx === null && r < 0.85) aIdx = pts.length - 1;
    }
    pts.push([0.452, 0.0]);
    return { k: 'st', pts: pts, a: aIdx, loop: false };
  }
  var ORBITS = [
    mkSuccess(0.4, 1.02, 2.1), mkSuccess(2.6, 0.99, 1.7),
    mkLoop(0.9, 0.165, 0.058, 0.5), mkLoop(3.6, 0.145, 0.066, -0.4),
    mkStag(1.2), mkStag(4.4)
  ];
  var STARTS = [0, 14, 6, 26, 20, 58];             /* staggered entrances */
  var MAXPTS = 0;
  for (var i2 = 0; i2 < ORBITS.length; i2++) MAXPTS = Math.max(MAXPTS, STARTS[i2] + ORBITS[i2].pts.length);
  var SPEED = 16, HOLD = 2.2, T = MAXPTS / SPEED + HOLD;

  var DPR = 1, W = 0, H = 0;
  function resize() {
    DPR = Math.min(2, window.devicePixelRatio || 1);
    var r = box.getBoundingClientRect();
    W = Math.max(1, Math.round(r.width)); H = Math.max(1, Math.round(r.height));
    cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0); draw();
  }
  var ML = 52, MR = 16, MT = 14, MB = 36;
  var X0 = 0.28, X1 = 1.38, Y0 = -0.15, Y1 = 0.15;
  function X(r) { return ML + (r - X0) / (X1 - X0) * (W - ML - MR); }
  function Y(v) { return MT + (Y1 - v) / (Y1 - Y0) * (H - MT - MB); }
  function dot(x, y, r, col) { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); }
  function ringD(x, y, r, col) { ctx.strokeStyle = col; ctx.lineWidth = 1.1; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke(); }

  function zone(cx, cy, rx, ry, col) {
    ctx.save(); ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.ellipse(X(cx), Y(cy), rx / (X1 - X0) * (W - ML - MR), ry / (Y1 - Y0) * (H - MT - MB), 0, 0, 7);
    ctx.lineWidth = 1.3; ctx.strokeStyle = col; ctx.stroke();
    ctx.setLineDash([]); ctx.restore();
  }
  function label(x, y, txt, col, size, bold) {
    ctx.font = (bold ? '600 ' : '400 ') + (size || 12) + 'px "Hanken Grotesk","Noto Sans SC",sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = col; ctx.fillText(txt, x, y);
  }

  function bg() {
    ctx.strokeStyle = C.line; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ML, H - MB); ctx.lineTo(W - MR, H - MB); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ML, MT); ctx.lineTo(ML, H - MB); ctx.stroke();
    ctx.strokeStyle = withA(C.gray, 0.30);
    ctx.beginPath(); ctx.moveTo(ML, Y(0)); ctx.lineTo(W - MR, Y(0)); ctx.stroke();
    ctx.fillStyle = C.gray; ctx.font = '11px "IBM Plex Mono",monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (var r = 0.4; r <= 1.31; r += 0.2) ctx.fillText(r.toFixed(1), X(r), H - MB + 6);
    ctx.fillText('r(t)', (ML + W - MR) / 2, H - MB + 20);
    ctx.save(); ctx.translate(14, (MT + H - MB) / 2); ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    ctx.fillText('Δr', 0, 0); ctx.restore();
    var i, p;
    for (i = 0; i < CLOUDS.ok.length; i++) { p = CLOUDS.ok[i]; dot(X(p[0]), Y(p[1]), 2.3, withA(C.teal, 0.22)); }
    for (i = 0; i < CLOUDS.lp.length; i++) { p = CLOUDS.lp[i]; ringD(X(p[0]), Y(p[1]), 2.7, withA(C.red, 0.18)); }
    for (i = 0; i < CLOUDS.st.length; i++) { p = CLOUDS.st[i]; dot(X(p[0]), Y(p[1]), 2.5, withA(C.red, 0.26)); }
    zone(1.01, -0.045, 0.10, 0.038, withA(C.teal, 0.7));
    label(X(1.01), Y(0.012), zh ? '巡航(正常干活)' : 'cruising (healthy work)', withA(C.teal, 0.95), 12, true);
    zone(0.454, -0.001, 0.085, 0.031, withA(C.red, 0.75));
    label(X(0.454), Y(-0.052), zh ? '不动点(停滞)' : 'fixed point (stagnation)', withA(C.red, 0.95), 12, true);
    label(X(0.90), Y(0.118), zh ? '极限环(打转)' : 'limit cycles (looping)', withA(C.red, 0.75), 11, false);
  }

  function draw() {
    if (!W) return;
    ctx.clearRect(0, 0, W, H);
    bg();
    var s = t * SPEED;
    for (var o = 0; o < ORBITS.length; o++) {
      var ob = ORBITS[o], pts = ob.pts, n = pts.length;
      var col = ob.k === 'ok' ? C.teal : C.red;
      var prog = s - STARTS[o];
      if (prog < 1) continue;
      var hi = ob.loop ? prog : Math.min(prog, n - 1);
      var iHi = Math.floor(hi);
      function P(idx) { return pts[ob.loop ? ((idx % n) + n) % n : Math.max(0, Math.min(n - 1, idx))]; }
      /* faint history: full closed cycle for loops, path-so-far otherwise */
      ctx.beginPath();
      var lim = ob.loop ? Math.min(iHi, n) : iHi;
      for (var i = 0; i <= lim; i++) {
        var q = P(i);
        if (i === 0) ctx.moveTo(X(q[0]), Y(q[1])); else ctx.lineTo(X(q[0]), Y(q[1]));
      }
      if (ob.loop && iHi >= n) ctx.closePath();
      ctx.strokeStyle = withA(col, 0.16); ctx.lineWidth = 1.2; ctx.lineJoin = 'round'; ctx.stroke();
      /* comet tail */
      var tail = 16;
      for (i = iHi - tail; i < iHi; i++) {
        if (i < 1) continue;
        var a = 0.10 + 0.6 * (1 - (iHi - i) / tail);
        var q0 = P(i - 1), q1 = P(i);
        ctx.strokeStyle = withA(col, a); ctx.lineWidth = 1.8;
        ctx.beginPath(); ctx.moveTo(X(q0[0]), Y(q0[1])); ctx.lineTo(X(q1[0]), Y(q1[1])); ctx.stroke();
      }
      /* head */
      var hq = P(iHi);
      if (!ob.loop && prog >= n - 1) {
        if (ob.k === 'st') { dot(X(hq[0]), Y(hq[1]), 3.4, withA(col, 0.95)); ringD(X(hq[0]), Y(hq[1]), 6.5, withA(col, 0.4)); }
        else dot(X(hq[0]), Y(hq[1]), 3.2, withA(col, 0.95));
      } else {
        dot(X(hq[0]), Y(hq[1]), 3.4, withA(col, 0.95));
        if (ob.k === 'lp') ringD(X(hq[0]), Y(hq[1]), 5.6, withA(col, 0.35));
      }
      /* alarm: loops re-ring on every pass; dives ring once; gold dot stays */
      if (ob.a != null) {
        var aq = pts[ob.a];
        var seen = prog >= ob.a;
        var since = prog - ob.a;                 /* one alarm only — it latches */
        if (seen && since >= 0 && since < 14) {
          var u = since / 14;
          ctx.strokeStyle = withA(C.gold, (1 - u) * 0.95); ctx.lineWidth = 2.2;
          ctx.beginPath(); ctx.arc(X(aq[0]), Y(aq[1]), 5 + 26 * ease(u), 0, 7); ctx.stroke();
          ctx.strokeStyle = withA(C.gold, (1 - u) * 0.45); ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.arc(X(aq[0]), Y(aq[1]), 3 + 14 * ease(u), 0, 7); ctx.stroke();
          dot(X(aq[0]), Y(aq[1]), 3.0, withA(C.gold, 0.95));
        } else if (seen) dot(X(aq[0]), Y(aq[1]), 2.4, withA(C.gold, 0.8));
      }
    }
  }

  var t = 0, playing = false, raf = 0, last = 0;
  function tick(now) {
    if (!playing) return;
    if (!last) last = now;
    var dt = Math.min(0.05, (now - last) / 1000); last = now;
    t += dt; if (t >= T) t = 0;
    draw(); raf = requestAnimationFrame(tick);
  }
  function play() { if (playing || reduce) return; playing = true; last = 0; raf = requestAnimationFrame(tick); }
  function pause() { playing = false; cancelAnimationFrame(raf); }

  box.addEventListener('click', function () { playing ? pause() : play(); });
  document.addEventListener('visibilitychange', function () { if (document.hidden) pause(); });
  if (!reduce && 'IntersectionObserver' in window) {
    new IntersectionObserver(function (es) {
      es.forEach(function (en) { en.isIntersecting ? play() : pause(); });
    }, { threshold: 0.25 }).observe(box);
  }
  if (reduce) t = T - 0.01;

  window.radPhase = { setTheme: function () { readColors(); draw(); } };
  var rz; window.addEventListener('resize', function () { clearTimeout(rz); rz = setTimeout(resize, 150); }, { passive: true });
  resize();
})();
