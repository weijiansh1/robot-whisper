"use strict";

const EXPERT_COLORS = [
  "#2f6f9f", "#d45b45", "#27866f", "#d0922f", "#745ca3", "#3d8d9a", "#bb4f74", "#668b3d",
  "#8a6642", "#536fbe", "#d06f2d", "#4f9461", "#9960a4", "#268598", "#b34b43", "#758136",
  "#355f78", "#c27b54", "#3f7e75", "#aa8738", "#6b69a8", "#547d91", "#af557b", "#6f8d52",
  "#805b4e", "#4775a4", "#b86935", "#578869", "#855f91", "#387c88", "#a84f55", "#777647",
];

const state = {
  data: null,
  masks: null,
  episodesById: new Map(),
  scene: null,
  failureEpisode: null,
  successEpisode: null,
  layerIndex: 0,
  pcaMode: "soft",
  pcaCoords: {},
  pcaScreenPoints: [],
  step: 0,
  playing: false,
  loop: true,
  timer: null,
  timelineCache: new Map(),
  metricCache: new Map(),
  timelineGeometry: new Map(),
};

const els = {
  scene: document.querySelector("#scene-select"),
  failure: document.querySelector("#failure-select"),
  success: document.querySelector("#success-select"),
  layer: document.querySelector("#layer-select"),
  prev: document.querySelector("#step-prev"),
  play: document.querySelector("#play-toggle"),
  next: document.querySelector("#step-next"),
  loop: document.querySelector("#loop-toggle"),
  range: document.querySelector("#step-range"),
  output: document.querySelector("#step-output"),
  speed: document.querySelector("#speed-select"),
  alignment: document.querySelector("#alignment-status"),
  comparison: document.querySelector("#comparison-status"),
  load: document.querySelector("#load-state"),
  source: document.querySelector("#source-note"),
  expertKey: document.querySelector("#expert-key"),
  pcaButtons: [...document.querySelectorAll("[data-pca-mode]")],
  pcaExplained: document.querySelector("#pca-explained"),
  pcaDistance: document.querySelector("#pca-distance"),
  pcaProgress: document.querySelector("#pca-progress"),
  pcaRepresentation: document.querySelector("#pca-representation"),
  pcaCanvas: document.querySelector("#pca-canvas"),
  pcaTooltip: document.querySelector("#pca-tooltip"),
};

const panelEls = Object.fromEntries(["failure", "success"].map((side) => [side, {
  panel: document.querySelector(`.${side}-panel`),
  episode: document.querySelector(`#${side}-episode`),
  meta: document.querySelector(`#${side}-meta`),
  persistence: document.querySelector(`#${side}-persistence`),
  reference: document.querySelector(`#${side}-reference`),
  lag1: document.querySelector(`#${side}-lag1`),
  curvature: document.querySelector(`#${side}-curvature`),
  indicator: document.querySelector(`#${side}-indicator`),
  indicatorNote: document.querySelector(`#${side}-indicator-note`),
  timeline: document.querySelector(`#${side}-timeline`),
  trail: document.querySelector(`#${side}-trail`),
  layerLabel: document.querySelector(`#${side}-layer-label`),
  lagGrid: document.querySelector(`#${side}-lag-grid`),
}]));

function decodeUint32(base64) {
  const binary = window.atob(base64);
  if (binary.length % 4 !== 0) throw new Error("route mask payload is not uint32-aligned");
  const bytes = new Uint8Array(binary.length);
  const chunk = 1 << 20;
  for (let start = 0; start < binary.length; start += chunk) {
    const end = Math.min(binary.length, start + chunk);
    for (let index = start; index < end; index += 1) bytes[index] = binary.charCodeAt(index);
  }
  return new Uint32Array(bytes.buffer);
}

function decodePcaCoordinates(spec, expectedRows) {
  const coordinates = new Float32Array(expectedRows * 2);
  let decodedRows = 0;
  for (const group of Object.values(spec.groups)) {
    const binary = window.atob(group.coords_base64);
    const expectedBytes = group.projected_rows * 2 * Int16Array.BYTES_PER_ELEMENT;
    if (binary.length !== expectedBytes) {
      throw new Error(`PCA group ${group.scene} has ${binary.length} bytes; expected ${expectedBytes}`);
    }
    if (group.row_offset < 0 || group.row_offset + group.projected_rows > expectedRows) {
      throw new Error(`PCA group ${group.scene} is outside the control-row axis`);
    }
    const bytes = new Uint8Array(binary.length);
    const chunk = 1 << 20;
    for (let start = 0; start < binary.length; start += chunk) {
      const end = Math.min(binary.length, start + chunk);
      for (let index = start; index < end; index += 1) bytes[index] = binary.charCodeAt(index);
    }
    const quantized = new Int16Array(bytes.buffer);
    const target = group.row_offset * 2;
    for (let index = 0; index < quantized.length; index += 1) {
      coordinates[target + index] = quantized[index] * group.scale[index % 2];
    }
    decodedRows += group.projected_rows;
  }
  if (decodedRows !== expectedRows) throw new Error(`decoded ${decodedRows} PCA rows; expected ${expectedRows}`);
  return coordinates;
}

function popcount32(value) {
  let x = value >>> 0;
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function maskExperts(mask) {
  const experts = [];
  for (let expert = 0; expert < 32; expert += 1) {
    if (((mask >>> expert) & 1) !== 0) experts.push(expert);
  }
  return experts;
}

function overlapMasks(a, b) {
  return popcount32((a & b) >>> 0) / 4;
}

function episode(side) {
  return state.episodesById.get(side === "failure" ? state.failureEpisode : state.successEpisode);
}

function rowStride() {
  return state.data.axes.layers.length * state.data.axes.tokens.length;
}

function maskAt(item, step, layerIndex, tokenIndex) {
  const tokenCount = state.data.axes.tokens.length;
  const index = (item.offset + step) * rowStride() + layerIndex * tokenCount + tokenIndex;
  return state.masks[index];
}

function samePositionOverlap(item, aStep, bStep, layerIndex, tokenIndex) {
  return overlapMasks(
    maskAt(item, aStep, layerIndex, tokenIndex),
    maskAt(item, bStep, layerIndex, tokenIndex),
  );
}

function validateData(raw) {
  if (raw?.schema?.name !== "himoe_moe_control_trajectory_v1" || raw.schema.version !== 2) {
    throw new Error("unexpected trajectory schema");
  }
  if (raw.axes.layers.length !== 8 || raw.axes.tokens.length !== 10) {
    throw new Error("viewer expects 8 HB layers and 10 action tokens");
  }
  if (!Array.isArray(raw.episodes) || raw.episodes.length !== 512) {
    throw new Error("viewer expects 512 episodes");
  }
  if (!raw.pca?.soft?.groups || !raw.pca?.top4?.groups) {
    throw new Error("viewer requires grouped soft and Top-4 PCA data");
  }
  return raw;
}

function normalizeData(raw) {
  const data = validateData(raw);
  const masks = decodeUint32(data.masks_base64);
  const expected = data.source.rows * data.axes.layers.length * data.axes.tokens.length;
  if (masks.length !== expected) {
    throw new Error(`decoded ${masks.length} route masks; expected ${expected}`);
  }
  for (let index = 0; index < Math.min(4096, masks.length); index += 1) {
    if (popcount32(masks[index]) !== data.axes.top_k) {
      throw new Error("decoded route mask is not Top-4");
    }
  }
  const pcaCoords = {
    soft: decodePcaCoordinates(data.pca.soft, data.source.rows),
    top4: decodePcaCoordinates(data.pca.top4, data.source.rows),
  };
  return { data, masks, pcaCoords };
}

function option(value, label) {
  const node = document.createElement("option");
  node.value = String(value);
  node.textContent = label;
  return node;
}

function mixedScenes() {
  const outcomes = new Map();
  for (const item of state.data.episodes) {
    if (!outcomes.has(item.scene)) outcomes.set(item.scene, new Set());
    outcomes.get(item.scene).add(item.success);
  }
  return [...outcomes.entries()]
    .filter(([, values]) => values.size === 2)
    .map(([scene]) => Number(scene))
    .sort((a, b) => a - b);
}

function sceneEpisodes(scene = state.scene) {
  return state.data.episodes.filter((item) => item.scene === scene);
}

function episodesFor(scene, success) {
  return state.data.episodes
    .filter((item) => item.scene === scene && item.success === success)
    .sort((a, b) => a.noise - b.noise);
}

function populateControls() {
  els.scene.replaceChildren();
  for (const scene of mixedScenes()) {
    const failures = episodesFor(scene, false).length;
    const successes = episodesFor(scene, true).length;
    els.scene.append(option(scene, `state ${scene} · ${failures}F / ${successes}S`));
  }
  els.layer.replaceChildren();
  state.data.axes.layers.forEach((layer, index) => {
    const group = index < 4 ? "Early" : "Late";
    els.layer.append(option(index, `L${layer} · ${group}`));
  });
}

function populateEpisodeSelects(preferredFailure = null, preferredSuccess = null) {
  const failures = episodesFor(state.scene, false);
  const successes = episodesFor(state.scene, true);
  els.failure.replaceChildren(...failures.map((item) => option(
    item.episode,
    `ep ${item.episode} · noise ${item.noise} · ${item.length} controls`,
  )));
  els.success.replaceChildren(...successes.map((item) => option(
    item.episode,
    `ep ${item.episode} · noise ${item.noise} · ${item.length} controls`,
  )));

  const failureIds = new Set(failures.map((item) => item.episode));
  const successIds = new Set(successes.map((item) => item.episode));
  state.failureEpisode = failureIds.has(preferredFailure) ? preferredFailure : failures[0].episode;
  const longestSuccess = successes.reduce((best, item) => item.length > best.length ? item : best, successes[0]);
  state.successEpisode = successIds.has(preferredSuccess) ? preferredSuccess : longestSuccess.episode;
  els.failure.value = String(state.failureEpisode);
  els.success.value = String(state.successEpisode);
}

function pairMaxStep() {
  return Math.max(episode("failure").length, episode("success").length) - 1;
}

function setReady(ready) {
  [els.scene, els.failure, els.success, els.layer, els.prev, els.play, els.next, els.loop, els.range, els.speed]
    .forEach((element) => { element.disabled = !ready; });
  els.pcaButtons.forEach((element) => { element.disabled = !ready; });
}

function setStep(value) {
  const max = pairMaxStep();
  state.step = Math.max(0, Math.min(max, Number(value)));
  els.range.value = String(state.step);
  renderAll();
}

function stopPlayback() {
  if (state.timer !== null) window.clearInterval(state.timer);
  state.timer = null;
  state.playing = false;
  els.play.innerHTML = "&#9654;";
  els.play.title = "播放";
  els.play.setAttribute("aria-label", "播放");
}

function startPlayback() {
  stopPlayback();
  state.playing = true;
  els.play.innerHTML = "&#10074;&#10074;";
  els.play.title = "暂停";
  els.play.setAttribute("aria-label", "暂停");
  state.timer = window.setInterval(() => {
    const max = pairMaxStep();
    if (state.step >= max) {
      if (state.loop) setStep(0);
      else stopPlayback();
      return;
    }
    setStep(state.step + 1);
  }, Number(els.speed.value));
}

function togglePlayback() {
  if (state.playing) stopPlayback();
  else startPlayback();
}

function percent(value, digits = 0) {
  if (!Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function signedPoints(value) {
  if (!Number.isFinite(value)) return "—";
  const points = value * 100;
  return `${points >= 0 ? "+" : ""}${points.toFixed(1)} pp`;
}

function currentLag1(item, step) {
  if (step < 1 || step >= item.length) return null;
  let sum = 0;
  let count = 0;
  for (let layer = 0; layer < state.data.axes.layers.length; layer += 1) {
    for (let token = 0; token < state.data.axes.tokens.length; token += 1) {
      sum += samePositionOverlap(item, step, step - 1, layer, token);
      count += 1;
    }
  }
  return sum / count;
}

function rollingMetrics(item, end) {
  const windowSize = state.data.axes.window;
  if (end < windowSize - 1 || end >= item.length) return null;
  const key = `${item.episode}:${end}`;
  if (state.metricCache.has(key)) return state.metricCache.get(key);
  const start = end - windowSize + 1;
  const curve = [];
  for (let lag = 1; lag <= 5; lag += 1) {
    let sum = 0;
    let count = 0;
    for (let step = start + lag; step <= end; step += 1) {
      for (let layer = 0; layer < state.data.axes.layers.length; layer += 1) {
        for (let token = 0; token < state.data.axes.tokens.length; token += 1) {
          sum += samePositionOverlap(item, step, step - lag, layer, token);
          count += 1;
        }
      }
    }
    curve.push(sum / count);
  }
  const persistence = curve.reduce((total, value) => total + value, 0) / curve.length;
  const curvature = curve[1] - 0.5 * (curve[0] + curve[2]);
  const result = { curve, persistence, curvature };
  state.metricCache.set(key, result);
  return result;
}

function successReference(scene, step) {
  return state.data.success_reference[String(scene)]?.[String(step)] ?? null;
}

function indicatorFor(metrics, item, step) {
  if (!metrics) {
    return { label: "积累中", className: "neutral", note: "需要 16 个控制步" };
  }
  const ref = successReference(item.scene, step);
  if (!ref) return { label: "无参考", className: "neutral", note: "该绝对步没有成功样本" };
  const survivor = step >= 35 ? " · 幸存者参考" : "";
  if (metrics.persistence >= ref.p95) {
    return { label: "高持久", className: "high", note: `高于成功 P95 · n=${ref.n}${survivor}` };
  }
  if (metrics.persistence >= ref.p75) {
    return { label: "偏高", className: "elevated", note: `高于成功 P75 · n=${ref.n}${survivor}` };
  }
  return { label: "常规", className: "normal", note: `成功参考内 · n=${ref.n}${survivor}` };
}

function overlapColor(value, alpha = 1) {
  const stops = [
    [241, 243, 240],
    [207, 226, 217],
    [143, 196, 174],
    [72, 151, 124],
    [20, 103, 84],
  ];
  const scaled = Math.max(0, Math.min(1, value)) * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const mix = scaled - index;
  const rgb = stops[index].map((channel, channelIndex) => Math.round(
    channel + (stops[index + 1][channelIndex] - channel) * mix,
  ));
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

function timelineValues(item, layerIndex) {
  const key = `${item.episode}:${layerIndex}`;
  if (state.timelineCache.has(key)) return state.timelineCache.get(key);
  const tokenCount = state.data.axes.tokens.length;
  const values = new Float32Array(item.length * tokenCount);
  values.fill(Number.NaN);
  for (let step = 1; step < item.length; step += 1) {
    for (let token = 0; token < tokenCount; token += 1) {
      values[step * tokenCount + token] = samePositionOverlap(
        item, step, step - 1, layerIndex, token,
      );
    }
  }
  state.timelineCache.set(key, values);
  return values;
}

function prepCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function pcaPoint(item, step, mode = state.pcaMode) {
  const coordinates = state.pcaCoords[mode];
  const index = (item.offset + step) * 2;
  return [coordinates[index], coordinates[index + 1]];
}

function pcaCoordinateLabel(value, span) {
  if (span < 0.1) return value.toFixed(3);
  if (span < 2) return value.toFixed(2);
  return value.toFixed(1);
}

function renderPca() {
  const failure = episode("failure");
  const success = episode("success");
  const selectedIds = new Set([failure.episode, success.episode]);
  const paths = sceneEpisodes().map((item) => ({
    side: item.success ? "success" : "failure",
    item,
    color: item.success ? "#168168" : "#c95048",
    label: item.success ? "成功" : "失败",
    selected: selectedIds.has(item.episode),
  })).sort((a, b) => Number(a.selected) - Number(b.selected));
  const { ctx, width, height } = prepCanvas(els.pcaCanvas);
  const left = width < 520 ? 44 : 58;
  const right = width < 520 ? 18 : 28;
  const top = 18;
  const bottom = 38;
  const plotWidth = Math.max(1, width - left - right);
  const plotHeight = Math.max(1, height - top - bottom);

  const allPoints = paths.flatMap(({ item }) => Array.from(
    { length: item.length }, (_, step) => pcaPoint(item, step),
  ));
  let minX = Math.min(...allPoints.map((point) => point[0]));
  let maxX = Math.max(...allPoints.map((point) => point[0]));
  let minY = Math.min(...allPoints.map((point) => point[1]));
  let maxY = Math.max(...allPoints.map((point) => point[1]));
  const rawX = Math.max(maxX - minX, 1e-7);
  const rawY = Math.max(maxY - minY, 1e-7);
  minX -= rawX * 0.10;
  maxX += rawX * 0.10;
  minY -= rawY * 0.10;
  maxY += rawY * 0.10;

  const viewMinX = minX;
  const viewMaxX = maxX;
  const viewMinY = minY;
  const viewMaxY = maxY;
  const visibleWidth = viewMaxX - viewMinX;
  const visibleHeight = viewMaxY - viewMinY;
  const scaleX = plotWidth / visibleWidth;
  const scaleY = plotHeight / visibleHeight;
  const toScreen = ([x, y]) => [
    left + (x - viewMinX) * scaleX,
    top + (viewMaxY - y) * scaleY,
  ];

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfb";
  ctx.fillRect(left, top, plotWidth, plotHeight);
  ctx.strokeStyle = "#e3e8e4";
  ctx.lineWidth = 1;
  ctx.font = "600 9px system-ui, sans-serif";
  ctx.fillStyle = "#77827d";
  for (let tick = 0; tick <= 5; tick += 1) {
    const fraction = tick / 5;
    const x = left + fraction * plotWidth;
    const y = top + fraction * plotHeight;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + plotHeight);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + plotWidth, y);
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(
      pcaCoordinateLabel(viewMinX + fraction * visibleWidth, visibleWidth),
      x,
      top + plotHeight + 7,
    );
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(
      pcaCoordinateLabel(viewMaxY - fraction * visibleHeight, visibleHeight),
      left - 7,
      y,
    );
  }

  if (viewMinX <= 0 && viewMaxX >= 0) {
    const x = toScreen([0, 0])[0];
    ctx.strokeStyle = "#bdc7c1";
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + plotHeight);
    ctx.stroke();
  }
  if (viewMinY <= 0 && viewMaxY >= 0) {
    const y = toScreen([0, 0])[1];
    ctx.strokeStyle = "#bdc7c1";
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + plotWidth, y);
    ctx.stroke();
  }

  state.pcaScreenPoints = [];
  for (const path of paths) {
    const points = Array.from({ length: path.item.length }, (_, step) => ({
      world: pcaPoint(path.item, step),
      step,
    }));
    const screen = points.map((point) => toScreen(point.world));
    for (let step = 0; step < screen.length; step += 1) {
      state.pcaScreenPoints.push({
        x: screen[step][0],
        y: screen[step][1],
        world: points[step].world,
        step,
        side: path.side,
        item: path.item,
        label: path.label,
      });
    }

    const stroke = (end, alpha, lineWidth) => {
      ctx.strokeStyle = path.color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = lineWidth;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      for (let step = 0; step <= end; step += 1) {
        const [x, y] = screen[step];
        if (step === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    };
    stroke(screen.length - 1, path.selected ? 0.18 : 0.065, path.selected ? 1.3 : 0.8);
    const activeEnd = Math.min(state.step, path.item.length - 1);
    stroke(activeEnd, path.selected ? 0.96 : 0.28, path.selected ? 2.6 : 1.05);

    if (!path.selected) {
      if (state.step < path.item.length) {
        const [x, y] = screen[activeEnd];
        ctx.fillStyle = path.color;
        ctx.globalAlpha = 0.58;
        ctx.beginPath();
        ctx.arc(x, y, 2.3, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      continue;
    }

    ctx.fillStyle = path.color;
    ctx.globalAlpha = 0.35;
    for (let pointStep = 0; pointStep <= activeEnd; pointStep += 1) {
      const [x, y] = screen[pointStep];
      ctx.beginPath();
      ctx.arc(x, y, 1.7, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    for (const landmark of state.data.landmarks) {
      if (landmark.step >= path.item.length) continue;
      const [x, y] = screen[landmark.step];
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = path.color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(x, y, 3.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    const [currentX, currentY] = screen[activeEnd];
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = path.color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(currentX, currentY, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = path.color;
    ctx.font = "800 10px system-ui, sans-serif";
    ctx.textAlign = currentX > left + plotWidth * 0.78 ? "right" : "left";
    ctx.textBaseline = currentY < top + 18 ? "top" : "bottom";
    ctx.fillText(
      `${path.side === "failure" ? "F" : "S"} · t${activeEnd}`,
      currentX + (ctx.textAlign === "right" ? -9 : 9),
      currentY + (ctx.textBaseline === "top" ? 7 : -7),
    );
  }

  ctx.fillStyle = "#56635d";
  ctx.font = "700 10px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText("PC1", left + plotWidth, height - 2);
  ctx.save();
  ctx.translate(11, top);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText("PC2", 0, 0);
  ctx.restore();

  const spec = state.data.pca[state.pcaMode];
  const group = spec.groups[String(state.scene)];
  const explained = group.explained_variance_ratio.reduce((total, value) => total + value, 0);
  els.pcaExplained.textContent = percent(explained, 1);
  const bothActive = state.step < failure.length && state.step < success.length;
  if (bothActive) {
    const a = pcaPoint(failure, state.step);
    const b = pcaPoint(success, state.step);
    els.pcaDistance.textContent = Math.hypot(a[0] - b[0], a[1] - b[1]).toFixed(3);
  } else {
    els.pcaDistance.textContent = "—";
  }
  const activeFailure = paths.filter((path) => path.side === "failure" && state.step < path.item.length).length;
  const activeSuccess = paths.filter((path) => path.side === "success" && state.step < path.item.length).length;
  els.pcaProgress.textContent = `t${String(state.step).padStart(2, "0")} · 活跃 ${activeFailure}F / ${activeSuccess}S`;
  els.pcaRepresentation.textContent = `${spec.representation} · 初始状态 ${state.scene} · ${group.episode_count} trajectories · fit ${group.fit_rows.toLocaleString()} pre-split rows · project ${group.projected_rows.toLocaleString()} full rows`;
}

function nearestPcaPoint(event, maxDistance = 14) {
  const rect = els.pcaCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let nearest = null;
  let best = maxDistance * maxDistance;
  for (const point of state.pcaScreenPoints) {
    const distance = (point.x - x) ** 2 + (point.y - y) ** 2;
    if (distance < best) {
      nearest = point;
      best = distance;
    }
  }
  return nearest ? { point: nearest, x, y, width: rect.width, height: rect.height } : null;
}

function renderTimeline(side, item) {
  const canvas = panelEls[side].timeline;
  const { ctx, width, height } = prepCanvas(canvas);
  const tokenCount = state.data.axes.tokens.length;
  const maxSteps = pairMaxStep() + 1;
  const left = 31;
  const right = 8;
  const top = 16;
  const bottom = 24;
  const plotWidth = Math.max(1, width - left - right);
  const plotHeight = Math.max(1, height - top - bottom);
  const cellWidth = plotWidth / maxSteps;
  const cellHeight = plotHeight / tokenCount;
  const values = timelineValues(item, state.layerIndex);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f6f8f6";
  ctx.fillRect(left, top, plotWidth, plotHeight);

  for (let token = 0; token < tokenCount; token += 1) {
    ctx.fillStyle = token % 2 === 0 ? "rgba(22, 35, 32, 0.025)" : "rgba(255,255,255,0)";
    ctx.fillRect(left, top + token * cellHeight, plotWidth, cellHeight);
    ctx.fillStyle = "#68736f";
    ctx.font = "600 9px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`T${token + 1}`, left - 5, top + (token + 0.5) * cellHeight);
  }

  for (let step = 1; step < item.length; step += 1) {
    for (let token = 0; token < tokenCount; token += 1) {
      const value = values[step * tokenCount + token];
      ctx.fillStyle = overlapColor(value);
      ctx.fillRect(
        left + step * cellWidth + 0.2,
        top + token * cellHeight + 0.2,
        Math.max(0.4, cellWidth - 0.4),
        Math.max(0.4, cellHeight - 0.4),
      );
    }
  }

  if (item.length < maxSteps) {
    const endX = left + item.length * cellWidth;
    ctx.fillStyle = "rgba(80, 91, 86, 0.08)";
    ctx.fillRect(endX, top, left + plotWidth - endX, plotHeight);
    ctx.strokeStyle = "rgba(80, 91, 86, 0.25)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(endX, top);
    ctx.lineTo(endX, top + plotHeight);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const markerColors = ["#3978a8", "#765b9e", "#d79a35"];
  state.data.landmarks.forEach((mark, index) => {
    if (mark.step >= maxSteps) return;
    const x = left + (mark.step + 0.5) * cellWidth;
    ctx.strokeStyle = markerColors[index];
    ctx.globalAlpha = 0.72;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + plotHeight);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  });

  const cursorX = left + (state.step + 0.5) * cellWidth;
  ctx.strokeStyle = "#17231f";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cursorX, top - 4);
  ctx.lineTo(cursorX, top + plotHeight + 3);
  ctx.stroke();

  ctx.fillStyle = "#68736f";
  ctx.font = "600 9px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let tick = 0; tick < maxSteps; tick += 10) {
    ctx.fillText(String(tick), left + (tick + 0.5) * cellWidth, top + plotHeight + 6);
  }
  if ((maxSteps - 1) % 10 !== 0) {
    ctx.fillText(String(maxSteps - 1), left + (maxSteps - 0.5) * cellWidth, top + plotHeight + 6);
  }

  state.timelineGeometry.set(side, { left, top, plotWidth, plotHeight, maxSteps });
}

function routeCell(item, step, layerIndex, tokenIndex, current) {
  const cell = document.createElement("div");
  cell.className = `route-cell${current ? " is-current" : ""}`;
  cell.setAttribute("role", "cell");
  const mask = maskAt(item, step, layerIndex, tokenIndex);
  const experts = maskExperts(mask);
  cell.title = `t${String(step).padStart(2, "0")} · token${tokenIndex + 1} · L${state.data.axes.layers[layerIndex]} · experts ${experts.join(", ")}`;
  for (const expert of experts) {
    const chip = document.createElement("span");
    chip.className = "expert-chip";
    chip.style.backgroundColor = EXPERT_COLORS[expert];
    cell.append(chip);
  }
  return cell;
}

function renderTrail(side, item) {
  const target = panelEls[side].trail;
  const active = state.step < item.length;
  const effective = Math.min(state.step, item.length - 1);
  const rows = state.data.axes.window;
  const fragment = document.createDocumentFragment();
  const corner = document.createElement("div");
  corner.className = "trail-header";
  corner.textContent = "t";
  corner.setAttribute("role", "columnheader");
  fragment.append(corner);
  for (const token of state.data.axes.tokens) {
    const header = document.createElement("div");
    header.className = "trail-header";
    header.textContent = `T${token}`;
    header.setAttribute("role", "columnheader");
    fragment.append(header);
  }
  for (let row = 0; row < rows; row += 1) {
    const step = effective - rows + 1 + row;
    const current = active && step === state.step;
    const label = document.createElement("div");
    label.className = `trail-step${current ? " is-current" : ""}`;
    label.textContent = step >= 0 ? String(step).padStart(2, "0") : "—";
    label.setAttribute("role", "rowheader");
    fragment.append(label);
    for (let token = 0; token < state.data.axes.tokens.length; token += 1) {
      if (step < 0) {
        const empty = document.createElement("div");
        empty.className = "trail-empty";
        empty.textContent = "·";
        fragment.append(empty);
      } else {
        fragment.append(routeCell(item, step, state.layerIndex, token, current));
      }
    }
  }
  target.replaceChildren(fragment);
}

function renderLagGrid(side, item) {
  const target = panelEls[side].lagGrid;
  const active = state.step < item.length;
  const fragment = document.createDocumentFragment();
  const corner = document.createElement("div");
  corner.className = "lag-header";
  corner.textContent = "lag";
  fragment.append(corner);
  for (const token of state.data.axes.tokens) {
    const header = document.createElement("div");
    header.className = "lag-header";
    header.textContent = `T${token}`;
    fragment.append(header);
  }
  for (let lag = 1; lag <= 5; lag += 1) {
    const label = document.createElement("div");
    label.className = "lag-label";
    label.textContent = `−${lag}`;
    fragment.append(label);
    for (let token = 0; token < state.data.axes.tokens.length; token += 1) {
      if (!active || state.step < lag) {
        const empty = document.createElement("div");
        empty.className = "lag-empty";
        empty.textContent = "—";
        fragment.append(empty);
      } else {
        const value = samePositionOverlap(item, state.step, state.step - lag, state.layerIndex, token);
        const cell = document.createElement("div");
        cell.className = "lag-cell";
        cell.style.background = overlapColor(value);
        cell.textContent = `${Math.round(value * 100)}`;
        cell.title = `token${token + 1}: t${state.step} vs t${state.step - lag} = ${percent(value)}`;
        fragment.append(cell);
      }
    }
  }
  target.replaceChildren(fragment);
}

function renderPanel(side) {
  const item = episode(side);
  const ui = panelEls[side];
  const active = state.step < item.length;
  const metrics = rollingMetrics(item, state.step);
  const lag1 = currentLag1(item, state.step);
  const ref = successReference(item.scene, state.step);
  const indicator = indicatorFor(metrics, item, state.step);

  ui.panel.classList.toggle("is-ended", !active);
  ui.episode.textContent = `episode ${item.episode}`;
  ui.meta.textContent = `state ${item.scene} · noise ${item.noise} · ${item.length} controls`;
  ui.persistence.textContent = active && metrics ? percent(metrics.persistence, 1) : "—";
  ui.reference.textContent = ref ? `成功 P50 ${percent(ref.p50, 1)} · P95 ${percent(ref.p95, 1)}` : "成功参考 —";
  ui.lag1.textContent = active ? percent(lag1, 1) : "—";
  ui.curvature.textContent = active && metrics ? signedPoints(metrics.curvature) : "—";
  ui.indicator.textContent = active ? indicator.label : `已结束 t${item.length - 1}`;
  ui.indicator.className = `indicator ${active ? indicator.className : "normal"}`;
  ui.indicatorNote.textContent = active ? indicator.note : `${item.action_steps} environment actions`;
  ui.layerLabel.textContent = `L${state.data.axes.layers[state.layerIndex]} · d${state.data.source.denoise_step}`;

  renderTimeline(side, item);
  renderTrail(side, item);
  renderLagGrid(side, item);
  return metrics;
}

function renderTopline(failureMetrics, successMetrics) {
  const max = pairMaxStep();
  els.range.max = String(max);
  els.output.value = `t${String(state.step).padStart(2, "0")} / t${String(max).padStart(2, "0")}`;
  els.alignment.textContent = `绝对控制步 t${String(state.step).padStart(2, "0")}`;
  if (failureMetrics && successMetrics) {
    const delta = failureMetrics.persistence - successMetrics.persistence;
    els.comparison.textContent = `失败 − 成功持久度 ${signedPoints(delta)}`;
  } else if (state.step >= episode("success").length) {
    els.comparison.textContent = `成功轨迹已在 t${episode("success").length - 1} 结束`;
  } else {
    els.comparison.textContent = `窗口 ${Math.min(state.step + 1, 16)} / 16`;
  }
}

function renderExpertKey() {
  const fragment = document.createDocumentFragment();
  EXPERT_COLORS.forEach((color, expert) => {
    const item = document.createElement("span");
    item.className = "expert-key-item";
    item.style.backgroundColor = color;
    item.textContent = String(expert);
    item.title = `expert ${expert}`;
    fragment.append(item);
  });
  els.expertKey.replaceChildren(fragment);
}

function updateQuery() {
  if (!state.data || !["http:", "https:"].includes(window.location.protocol)) return;
  const query = new URLSearchParams({
    scene: String(state.scene),
    failure: String(state.failureEpisode),
    success: String(state.successEpisode),
    layer: String(state.data.axes.layers[state.layerIndex]),
    pca: state.pcaMode,
    step: String(state.step),
  });
  const url = new URL(window.location.href);
  url.search = query.toString();
  try {
    window.history.replaceState(null, "", url);
  } catch (error) {
    // Sandboxed previewers may expose an HTTP URL while denying history writes.
    if (!(error instanceof DOMException) || error.name !== "SecurityError") throw error;
  }
}

function renderAll() {
  if (!state.data) return;
  renderPca();
  const failureMetrics = renderPanel("failure");
  const successMetrics = renderPanel("success");
  renderTopline(failureMetrics, successMetrics);
  updateQuery();
}

function selectInitialPair() {
  const query = new URLSearchParams(window.location.search);
  const defaults = state.data.defaults;
  const requestedScene = query.has("scene") ? Number(query.get("scene")) : null;
  state.scene = Number.isFinite(requestedScene) && mixedScenes().includes(requestedScene)
    ? requestedScene
    : defaults.scene;
  els.scene.value = String(state.scene);
  const requestedFailure = query.has("failure") ? Number(query.get("failure")) : null;
  const requestedSuccess = query.has("success") ? Number(query.get("success")) : null;
  populateEpisodeSelects(
    Number.isFinite(requestedFailure) ? requestedFailure : defaults.failure_episode,
    Number.isFinite(requestedSuccess) ? requestedSuccess : defaults.success_episode,
  );
  const requestedLayer = query.has("layer") ? Number(query.get("layer")) : null;
  const layerIndex = state.data.axes.layers.indexOf(requestedLayer);
  state.layerIndex = layerIndex >= 0 ? layerIndex : 0;
  els.layer.value = String(state.layerIndex);
  const requestedPca = query.get("pca");
  state.pcaMode = requestedPca === "top4" ? "top4" : "soft";
  els.pcaButtons.forEach((button) => {
    const active = button.dataset.pcaMode === state.pcaMode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  const requestedStep = query.has("step") ? Number(query.get("step")) : null;
  state.step = Number.isFinite(requestedStep)
    ? Math.max(0, Math.min(pairMaxStep(), requestedStep))
    : 0;
  els.range.value = String(state.step);
}

function bindEvents() {
  els.scene.addEventListener("change", () => {
    stopPlayback();
    state.scene = Number(els.scene.value);
    populateEpisodeSelects();
    setStep(0);
  });
  els.failure.addEventListener("change", () => {
    stopPlayback();
    state.failureEpisode = Number(els.failure.value);
    setStep(Math.min(state.step, pairMaxStep()));
  });
  els.success.addEventListener("change", () => {
    stopPlayback();
    state.successEpisode = Number(els.success.value);
    setStep(Math.min(state.step, pairMaxStep()));
  });
  els.layer.addEventListener("change", () => {
    state.layerIndex = Number(els.layer.value);
    renderAll();
  });
  els.pcaButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.pcaMode = button.dataset.pcaMode;
      els.pcaButtons.forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle("is-active", active);
        candidate.setAttribute("aria-pressed", String(active));
      });
      els.pcaTooltip.hidden = true;
      renderAll();
    });
  });
  els.prev.addEventListener("click", () => {
    stopPlayback();
    setStep(state.step - 1);
  });
  els.next.addEventListener("click", () => {
    stopPlayback();
    setStep(state.step + 1);
  });
  els.play.addEventListener("click", togglePlayback);
  els.loop.addEventListener("click", () => {
    state.loop = !state.loop;
    els.loop.setAttribute("aria-pressed", String(state.loop));
  });
  els.range.addEventListener("input", () => {
    stopPlayback();
    setStep(Number(els.range.value));
  });
  els.speed.addEventListener("change", () => {
    if (state.playing) startPlayback();
  });

  for (const side of ["failure", "success"]) {
    panelEls[side].timeline.addEventListener("click", (event) => {
      const geometry = state.timelineGeometry.get(side);
      if (!geometry) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const relative = (x - geometry.left) / geometry.plotWidth;
      if (relative < 0 || relative > 1) return;
      stopPlayback();
      setStep(Math.floor(relative * geometry.maxSteps));
    });
  }

  els.pcaCanvas.addEventListener("mousemove", (event) => {
    const nearest = nearestPcaPoint(event);
    if (!nearest) {
      els.pcaTooltip.hidden = true;
      return;
    }
    const { point, x, y, width, height } = nearest;
    els.pcaTooltip.textContent = `${point.label} · ep ${point.item.episode} · t${point.step} · PC1 ${point.world[0].toFixed(3)} · PC2 ${point.world[1].toFixed(3)}`;
    if (x > width / 2) {
      els.pcaTooltip.style.left = "auto";
      els.pcaTooltip.style.right = `${Math.max(8, width - x + 10)}px`;
    } else {
      els.pcaTooltip.style.left = `${Math.max(8, x + 10)}px`;
      els.pcaTooltip.style.right = "auto";
    }
    els.pcaTooltip.style.top = `${Math.max(18, Math.min(height - 18, y))}px`;
    els.pcaTooltip.hidden = false;
  });
  els.pcaCanvas.addEventListener("mouseleave", () => { els.pcaTooltip.hidden = true; });
  els.pcaCanvas.addEventListener("click", (event) => {
    const nearest = nearestPcaPoint(event, 18);
    if (!nearest) return;
    stopPlayback();
    if (nearest.point.item.success) {
      state.successEpisode = nearest.point.item.episode;
      els.success.value = String(state.successEpisode);
    } else {
      state.failureEpisode = nearest.point.item.episode;
      els.failure.value = String(state.failureEpisode);
    }
    setStep(nearest.point.step);
  });

  window.addEventListener("keydown", (event) => {
    const tag = document.activeElement?.tagName;
    if (tag === "SELECT" || tag === "INPUT") return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      stopPlayback();
      setStep(state.step - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      stopPlayback();
      setStep(state.step + 1);
    } else if (event.key === " ") {
      event.preventDefault();
      togglePlayback();
    }
  });

  const resize = new ResizeObserver(() => {
    if (!state.data) return;
    window.requestAnimationFrame(() => {
      renderTimeline("failure", episode("failure"));
      renderTimeline("success", episode("success"));
      renderPca();
    });
  });
  resize.observe(panelEls.failure.timeline);
  resize.observe(panelEls.success.timeline);
  resize.observe(els.pcaCanvas);
}

function displayError(error) {
  stopPlayback();
  els.load.className = "load-state is-error";
  els.load.textContent = `无法读取轨迹数据：${error.message}`;
  setReady(false);
  console.error(error);
}

async function loadData() {
  try {
    const embedded = document.querySelector("#trajectory-data");
    let raw;
    if (embedded?.textContent.trim()) {
      raw = JSON.parse(embedded.textContent);
    } else {
      const response = await fetch("./data.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      raw = await response.json();
    }
    const normalized = normalizeData(raw);
    state.data = normalized.data;
    state.masks = normalized.masks;
    state.pcaCoords = normalized.pcaCoords;
    for (const item of state.data.episodes) state.episodesById.set(item.episode, item);
    populateControls();
    selectInitialPair();
    renderExpertKey();
    els.source.textContent = `数据源：${state.data.source.route_store} · d${state.data.source.denoise_step}`;
    setReady(true);
    els.load.className = "load-state is-ready";
    renderAll();
  } catch (error) {
    displayError(error instanceof Error ? error : new Error(String(error)));
  }
}

bindEvents();
loadData();
