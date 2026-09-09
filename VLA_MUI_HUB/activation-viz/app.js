"use strict";

const COLORS = {
  ink: "#172321",
  muted: "#64716c",
  grid: "#d7ddd8",
  success: "#168568",
  failure: "#d14f45",
  early: "#267c8a",
  late: "#b56d22",
  paper: "#fbfcfa",
};

const FALLBACK_NORM_CV = {
  early: 0.0005795878096250817,
  late: 0.000664150677039288,
};

const FALLBACK_EVIDENCE = [
  {
    name: "all_mixed",
    label: "All mixed · 40 pools",
    feature: "Late token drift",
    auc: 0.4782,
    rawP: 0.4295,
    maxP: 0.9608,
  },
  {
    name: "balanced_min4",
    label: "Balanced · 16 pools",
    feature: "Early centroid convergence",
    auc: 0.4637,
    rawP: 0.2342,
    maxP: 0.7986,
  },
  {
    name: "non_long_mixed",
    label: "Non-long · 27 pools",
    feature: "Early token drift",
    auc: 0.4378,
    rawP: 0.0604,
    maxP: 0.3499,
  },
  {
    name: "non_long_balanced_min4",
    label: "Non-long balanced · 6 pools",
    feature: "Late centroid convergence",
    auc: 0.5676,
    rawP: 0.2205,
    maxP: 0.7813,
  },
];

const FEATURE_LABELS = {
  early_rms: "Early hidden RMS",
  late_rms: "Late hidden RMS",
  early_token_adj_cosdist: "Early token drift",
  late_token_adj_cosdist: "Late token drift",
  early_flow_endpoint_cosdist: "Early flow endpoint drift",
  late_flow_endpoint_cosdist: "Late flow endpoint drift",
  early_pool_centroid_d9_rms: "Early d9 centroid distance",
  late_pool_centroid_d9_rms: "Late d9 centroid distance",
  early_pool_centroid_convergence: "Early centroid convergence",
  late_pool_centroid_convergence: "Late centroid convergence",
  early_pool_knn4_d9_rms: "Early d9 4-NN dispersion",
  late_pool_knn4_d9_rms: "Late d9 4-NN dispersion",
};

const SUBSET_LABELS = {
  all_mixed: "All mixed · 40 pools",
  balanced_min4: "Balanced · 16 pools",
  non_long_mixed: "Non-long · 27 pools",
  non_long_balanced_min4: "Non-long balanced · 6 pools",
};

const els = {
  task: document.querySelector("#task-select"),
  state: document.querySelector("#state-select"),
  stepRange: document.querySelector("#step-range"),
  stepOutput: document.querySelector("#step-output"),
  prev: document.querySelector("#step-prev"),
  next: document.querySelector("#step-next"),
  play: document.querySelector("#play-toggle"),
  segments: [...document.querySelectorAll(".segment")],
  load: document.querySelector("#load-state"),
  pca: document.querySelector("#pca-canvas"),
  pcaExplained: document.querySelector("#pca-explained"),
  pcaShell: document.querySelector("#scatter-shell"),
  curve: document.querySelector("#curve-canvas"),
  curveDelta: document.querySelector("#curve-delta"),
  tooltip: document.querySelector("#plot-tooltip"),
  layerTable: document.querySelector("#layer-table"),
  candidateStrip: document.querySelector("#candidate-strip"),
  selectedSummary: document.querySelector("#selected-summary"),
  evidenceBody: document.querySelector("#evidence-body"),
  driftNote: document.querySelector("#drift-note"),
  sourceNote: document.querySelector("#source-note"),
  stateTokenControl: document.querySelector("#state-token-control"),
  findingPools: document.querySelector("#finding-pools"),
  findingEarly: document.querySelector("#finding-early"),
  findingLate: document.querySelector("#finding-late"),
  metricOutcome: document.querySelector("#metric-outcome"),
  metricOutcomeNote: document.querySelector("#metric-outcome-note"),
  metricDispersion: document.querySelector("#metric-dispersion"),
  metricDispersionNote: document.querySelector("#metric-dispersion-note"),
  metricRetentionLabel: document.querySelector("#metric-retention-label"),
  metricRetention: document.querySelector("#metric-retention"),
  metricRetentionNote: document.querySelector("#metric-retention-note"),
  metricNormCv: document.querySelector("#metric-norm-cv"),
  metricNormCvNote: document.querySelector("#metric-norm-cv-note"),
  scatterRetention: document.querySelector("#scatter-retention"),
  trajectoryToggle: document.querySelector("#trajectory-toggle"),
};

const state = {
  data: null,
  taskId: "",
  poolId: "",
  block: "early",
  step: 0,
  candidate: 0,
  hoverCandidate: null,
  projectedPoints: [],
  showTrajectories: false,
  timer: null,
};

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function finiteNumber(value, fallback = null) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function booleanValue(value) {
  if (typeof value === "string") {
    return value.toLowerCase() === "true" || value === "1" || value.toLowerCase() === "success";
  }
  return Boolean(value);
}

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function normalizeSeries(value) {
  if (Array.isArray(value)) {
    return value.map((item) => finiteNumber(item)).filter((item) => item !== null);
  }
  if (value && typeof value === "object") {
    return normalizeSeries(firstDefined(value.values, value.steps, value.series));
  }
  return [];
}

function normalizePcaSteps(view) {
  let steps = firstDefined(
    view?.pca?.steps,
    view?.pca_steps,
    view?.steps,
    view?.coordinates,
  );
  if (!Array.isArray(steps)) return [];

  // Tolerate candidate-major [32,10,2] as well as the canonical [10,32,2].
  if (steps.length === 32 && Array.isArray(steps[0]) && steps[0].length === 10) {
    steps = Array.from({ length: 10 }, (_, d) => steps.map((candidate) => candidate[d]));
  }
  return steps.slice(0, 10).map((row) => asArray(row).slice(0, 32).map((point) => {
    if (Array.isArray(point)) return [finiteNumber(point[0], 0), finiteNumber(point[1], 0)];
    return [finiteNumber(point?.x, 0), finiteNumber(point?.y, 0)];
  }));
}

function pcaDispersion(steps) {
  return steps.map((points) => {
    if (!points.length) return 0;
    const cx = points.reduce((sum, point) => sum + point[0], 0) / points.length;
    const cy = points.reduce((sum, point) => sum + point[1], 0) / points.length;
    return Math.sqrt(points.reduce((sum, point) => {
      const dx = point[0] - cx;
      const dy = point[1] - cy;
      return sum + dx * dx + dy * dy;
    }, 0) / points.length);
  });
}

function normalizeView(view = {}) {
  const pcaSteps = normalizePcaSteps(view);
  let dispersion = normalizeSeries(firstDefined(
    view?.raw_cloud?.dispersion_rms,
    view?.raw_cloud?.dispersion,
    view.dispersion,
    view.cloud_dispersion,
    view.centroid_dispersion,
  ));
  if (!dispersion.length) dispersion = pcaDispersion(pcaSteps);

  let retention = normalizeSeries(firstDefined(view?.raw_cloud?.retention, view.retention));
  if (!retention.length && dispersion.length && dispersion[0] !== 0) {
    retention = dispersion.map((value) => value / dispersion[0]);
  }
  return {
    layers: asArray(view.layers).map(Number).filter(Number.isFinite),
    pcaSteps,
    explained: asArray(firstDefined(
      view?.pca?.explained_variance_ratio,
      view?.pca?.explained_variance,
    )).map(Number),
    candidateCentered: booleanValue(view?.pca?.candidate_centered_per_denoise),
    dispersion,
    retention,
  };
}

function normalizeLayer(layer, index) {
  const layerId = finiteNumber(firstDefined(layer?.layer, layer?.id, layer?.index), [2, 3, 4, 5, 12, 13, 14, 15][index]);
  const dispersion = normalizeSeries(firstDefined(layer?.dispersion_rms, layer?.dispersion, layer?.cloud_dispersion));
  let retention = normalizeSeries(layer?.retention);
  if (!retention.length && dispersion.length && dispersion[0] !== 0) {
    retention = dispersion.map((value) => value / dispersion[0]);
  }
  return { layer: layerId, dispersion, retention };
}

function taskLookup(rawTasks) {
  const result = new Map();
  const entries = Array.isArray(rawTasks)
    ? rawTasks.map((task, index) => [firstDefined(task?.id, task?.task_id, index), task])
    : Object.entries(rawTasks || {});
  entries.forEach(([key, task]) => {
    const id = String(firstDefined(task?.id, task?.task_id, key));
    const label = String(firstDefined(
      task?.label,
      task?.display_name,
      task?.task_name,
      task?.name,
      task?.prompt,
      id,
    ));
    result.set(id, { id, label, raw: task });
  });
  return result;
}

function normalizeData(raw) {
  const tasks = taskLookup(raw.tasks);
  const pools = asArray(raw.pools).map((pool, poolIndex) => {
    const taskId = String(firstDefined(pool.task_id, pool.task, pool.task_name, "task"));
    if (!tasks.has(taskId)) {
      tasks.set(taskId, {
        id: taskId,
        label: String(firstDefined(pool.task_label, pool.task_name, pool.task, taskId)),
        raw: {},
      });
    }
    const candidates = asArray(pool.candidates).map((candidate, candidateIndex) => ({
      index: candidateIndex,
      id: String(firstDefined(candidate.id, candidate.candidate_id, candidateIndex)),
      seed: firstDefined(candidate.seed, candidate.flow_noise_seed, candidateIndex),
      success: booleanValue(firstDefined(candidate.success, candidate.outcome, false)),
    }));
    const early = normalizeView(pool?.views?.early || pool.early || {});
    const late = normalizeView(pool?.views?.late || pool.late || {});
    const inferredK = Math.max(candidates.length, early.pcaSteps[0]?.length || 0, late.pcaSteps[0]?.length || 0);
    while (candidates.length < inferredK) {
      const index = candidates.length;
      candidates.push({ index, id: String(index), seed: index, success: false });
    }
    return {
      id: String(firstDefined(pool.id, pool.pool_id, `${taskId}:${pool.init_state_id ?? poolIndex}`)),
      taskId,
      initState: firstDefined(pool.init_state_id, pool.state_id, pool.state, poolIndex),
      candidates,
      views: { early, late },
      layers: asArray(firstDefined(pool.layers, raw?.global?.layers)).map(normalizeLayer),
      stateToken: pool.state_token_range || pool.state_token || {},
      raw: pool,
    };
  });

  if (!pools.length) throw new Error("data.json 中没有 pools[]");
  return { raw, tasks, pools };
}

function currentPool() {
  return state.data?.pools.find((pool) => pool.id === state.poolId) || null;
}

function currentView(block = state.block) {
  return currentPool()?.views?.[block] || null;
}

function setReady(ready) {
  [els.task, els.state, els.stepRange, els.prev, els.next, els.play, els.trajectoryToggle].forEach((control) => {
    control.disabled = !ready;
  });
}

function displayError(error) {
  stopPlayback();
  setReady(false);
  els.load.className = "load-state is-error";
  els.load.textContent = `无法读取 data.json：${error.message}`;
}

function successCounts(pool) {
  const successes = pool.candidates.filter((candidate) => candidate.success).length;
  return { successes, failures: pool.candidates.length - successes };
}

function mostBalancedMixedPool(pools) {
  return pools
    .map((pool) => ({ pool, counts: successCounts(pool) }))
    .filter(({ counts }) => counts.successes > 0 && counts.failures > 0)
    .sort((a, b) => (
      Math.abs(a.counts.successes - a.counts.failures) - Math.abs(b.counts.successes - b.counts.failures)
      || Number(a.pool.initState) - Number(b.pool.initState)
    ))[0]?.pool;
}

function taskLabel(taskId) {
  return state.data?.tasks.get(taskId)?.label || taskId;
}

function populateTaskOptions() {
  els.task.replaceChildren();
  const taskIds = [...new Set(state.data.pools.map((pool) => pool.taskId))];
  taskIds.forEach((taskId) => {
    const option = document.createElement("option");
    option.value = taskId;
    option.textContent = taskLabel(taskId).replaceAll("_", " ");
    option.title = option.textContent;
    els.task.append(option);
  });
  els.task.value = state.taskId;
}

function populateStateOptions(preferredPoolId = "") {
  const pools = state.data.pools
    .filter((pool) => pool.taskId === state.taskId)
    .sort((a, b) => Number(a.initState) - Number(b.initState));
  els.state.replaceChildren();
  pools.forEach((pool) => {
    const counts = successCounts(pool);
    const option = document.createElement("option");
    option.value = pool.id;
    option.textContent = `state ${pool.initState} · ${counts.successes}S / ${counts.failures}F`;
    els.state.append(option);
  });
  const preferred = pools.find((pool) => pool.id === preferredPoolId);
  const mixed = mostBalancedMixedPool(pools);
  const next = preferred || mixed || pools[0];
  state.poolId = next?.id || "";
  els.state.value = state.poolId;
}

function selectInitialPool() {
  const queryPool = new URLSearchParams(window.location.search).get("pool");
  const requested = state.data.pools.find((pool) => pool.id === queryPool);
  const mixed = mostBalancedMixedPool(state.data.pools);
  const initial = requested || mixed || state.data.pools[0];
  state.taskId = initial.taskId;
  state.poolId = initial.id;
  state.candidate = 0;
  populateTaskOptions();
  populateStateOptions(initial.id);
}

function updateQuery() {
  if (!state.poolId || !window.history?.replaceState) return;
  try {
    const url = new URL(window.location.href);
    url.searchParams.set("pool", state.poolId);
    window.history.replaceState({}, "", url);
  } catch (_) {
    // Some browsers disallow history mutation for local file:// documents.
  }
}

function prepCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function drawEmpty(ctx, width, height, text) {
  ctx.fillStyle = COLORS.muted;
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, width / 2, height / 2);
}

function pcaProjection(view, width, height) {
  const points = view?.pcaSteps.flat() || [];
  if (!points.length) return null;
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const pad = { left: 24, right: 20, top: 18, bottom: 24 };
  const innerWidth = Math.max(1, width - pad.left - pad.right);
  const innerHeight = Math.max(1, height - pad.top - pad.bottom);
  const dataMinX = Math.min(...xs);
  const dataMaxX = Math.max(...xs);
  const dataMinY = Math.min(...ys);
  const dataMaxY = Math.max(...ys);
  const centerX = (dataMinX + dataMaxX) / 2;
  const centerY = (dataMinY + dataMaxY) / 2;
  const spanX = Math.max(dataMaxX - dataMinX, 1e-9) * 1.16;
  const spanY = Math.max(dataMaxY - dataMinY, 1e-9) * 1.16;
  const scale = Math.min(innerWidth / spanX, innerHeight / spanY);
  const plotCenterX = pad.left + innerWidth / 2;
  const plotCenterY = pad.top + innerHeight / 2;
  const minX = centerX - innerWidth / (2 * scale);
  const maxX = centerX + innerWidth / (2 * scale);
  const minY = centerY - innerHeight / (2 * scale);
  const maxY = centerY + innerHeight / (2 * scale);
  return {
    x: (value) => plotCenterX + (value - centerX) * scale,
    y: (value) => plotCenterY - (value - centerY) * scale,
    raw: { minX, maxX, minY, maxY },
    plot: { left: pad.left, right: width - pad.right, top: pad.top, bottom: height - pad.bottom },
  };
}

function projectStep(points, count, projection) {
  return points.slice(0, count).map((point, candidateIndex) => ({
    candidateIndex,
    x: projection.x(point[0]),
    y: projection.y(point[1]),
    raw: point,
  }));
}

function convexHull(points) {
  if (points.length < 3) return points.slice();
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (origin, a, b) => (
    (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x)
  );
  const half = (ordered) => {
    const result = [];
    ordered.forEach((point) => {
      while (result.length >= 2 && cross(result.at(-2), result.at(-1), point) <= 0) result.pop();
      result.push(point);
    });
    return result;
  };
  return [...half(sorted).slice(0, -1), ...half([...sorted].reverse()).slice(0, -1)];
}

function drawHull(ctx, points, stroke, fill, lineWidth = 1.5) {
  const hull = convexHull(points);
  if (hull.length < 3) return;
  ctx.save();
  ctx.strokeStyle = stroke;
  ctx.fillStyle = fill;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.beginPath();
  hull.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y); else ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawOutcomePoint(ctx, x, y, success, radius, hollow = false) {
  ctx.save();
  const color = success ? COLORS.success : COLORS.failure;
  ctx.fillStyle = hollow ? COLORS.paper : color;
  ctx.strokeStyle = hollow ? "rgba(100, 113, 108, 0.52)" : COLORS.paper;
  ctx.lineWidth = hollow ? 1.15 : 1.5;
  if (success || hollow) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  } else {
    ctx.translate(x, y);
    ctx.rotate(Math.PI / 4);
    ctx.beginPath();
    ctx.rect(-radius, -radius, radius * 2, radius * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function renderScatter() {
  const { ctx, width, height } = prepCanvas(els.pca);
  const pool = currentPool();
  const view = currentView();
  if (!pool || !view?.pcaSteps.length) {
    state.projectedPoints = [];
    drawEmpty(ctx, width, height, "当前池缺少 PCA 轨迹");
    return;
  }
  const projection = pcaProjection(view, width, height);
  if (!projection) return;

  // Zero axes are meaningful when they fall inside the fitted PCA range.
  ctx.save();
  ctx.strokeStyle = "rgba(100, 113, 108, 0.35)";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 4]);
  if (projection.raw.minX <= 0 && projection.raw.maxX >= 0) {
    const x0 = projection.x(0);
    ctx.beginPath(); ctx.moveTo(x0, projection.plot.top); ctx.lineTo(x0, projection.plot.bottom); ctx.stroke();
  }
  if (projection.raw.minY <= 0 && projection.raw.maxY >= 0) {
    const y0 = projection.y(0);
    ctx.beginPath(); ctx.moveTo(projection.plot.left, y0); ctx.lineTo(projection.plot.right, y0); ctx.stroke();
  }
  ctx.restore();

  const count = Math.min(pool.candidates.length, ...view.pcaSteps.map((step) => step.length));
  const baselinePoints = projectStep(view.pcaSteps[0] || [], count, projection);
  const activePoints = projectStep(view.pcaSteps[state.step] || [], count, projection);
  const previousPoints = state.step > 0
    ? projectStep(view.pcaSteps[state.step - 1] || [], count, projection)
    : [];
  const blockColor = COLORS[state.block];

  drawHull(ctx, baselinePoints, "rgba(100, 113, 108, 0.62)", "rgba(100, 113, 108, 0.075)", 1.35);
  drawHull(ctx, activePoints, blockColor, state.block === "early"
    ? "rgba(38, 124, 138, 0.105)"
    : "rgba(181, 109, 34, 0.105)", 1.9);

  if (state.showTrajectories) {
    const order = Array.from({ length: count }, (_, index) => index)
      .sort((a, b) => Number(a === state.candidate) - Number(b === state.candidate));
    order.forEach((candidateIndex) => {
      const candidate = pool.candidates[candidateIndex];
      const selected = candidateIndex === state.candidate;
      const hovered = candidateIndex === state.hoverCandidate;
      ctx.save();
      ctx.strokeStyle = candidate.success ? COLORS.success : COLORS.failure;
      ctx.globalAlpha = selected ? 0.82 : hovered ? 0.58 : 0.13;
      ctx.lineWidth = selected ? 2.2 : hovered ? 1.5 : 0.8;
      ctx.beginPath();
      view.pcaSteps.forEach((stepPoints, stepIndex) => {
        const point = stepPoints[candidateIndex];
        if (!point) return;
        const x = projection.x(point[0]);
        const y = projection.y(point[1]);
        if (stepIndex === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    });
  }

  if (previousPoints.length) {
    ctx.save();
    ctx.strokeStyle = blockColor;
    ctx.globalAlpha = 0.24;
    ctx.lineWidth = 1;
    activePoints.forEach((point, index) => {
      const previous = previousPoints[index];
      if (!previous) return;
      ctx.beginPath();
      ctx.moveTo(previous.x, previous.y);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
    });
    ctx.restore();
    previousPoints.forEach((point) => drawOutcomePoint(ctx, point.x, point.y, true, 3.15, true));
  }

  state.projectedPoints = activePoints;
  state.projectedPoints.forEach(({ candidateIndex, x, y }) => {
    const candidate = pool.candidates[candidateIndex];
    const selected = candidateIndex === state.candidate;
    const hovered = candidateIndex === state.hoverCandidate;
    drawOutcomePoint(ctx, x, y, candidate.success, hovered ? 5.8 : 4.7);
    if (selected) {
      ctx.save();
      ctx.strokeStyle = COLORS.ink;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  });

  if (activePoints.length) {
    const centerX = activePoints.reduce((sum, point) => sum + point.x, 0) / activePoints.length;
    const centerY = activePoints.reduce((sum, point) => sum + point.y, 0) / activePoints.length;
    ctx.save();
    ctx.strokeStyle = COLORS.ink;
    ctx.fillStyle = COLORS.ink;
    ctx.lineWidth = 1.7;
    ctx.beginPath();
    ctx.moveTo(centerX - 6, centerY); ctx.lineTo(centerX + 6, centerY);
    ctx.moveTo(centerX, centerY - 6); ctx.lineTo(centerX, centerY + 6);
    ctx.stroke();
    ctx.font = "10px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText("中心", centerX + 8, centerY - 3);
    ctx.restore();
  }
}

function globalView(block) {
  return state.data?.raw?.global?.views?.[block] || {};
}

function curveFor(block) {
  const local = currentPool()?.views?.[block]?.dispersion || [];
  if (local.length) return local;
  return normalizeSeries(firstDefined(
    globalView(block).dispersion_median_by_pool,
    globalView(block).dispersion,
  ));
}

function renderCurve() {
  const { ctx, width, height } = prepCanvas(els.curve);
  const curves = { early: curveFor("early"), late: curveFor("late") };
  const values = [...curves.early, ...curves.late].filter(Number.isFinite);
  if (!values.length) {
    drawEmpty(ctx, width, height, "缺少 dispersion 序列");
    els.curveDelta.textContent = "—";
    return;
  }
  const pad = { left: 42, right: 14, top: 13, bottom: 25 };
  const minY = Math.min(0, ...values) * 0.92;
  const maxY = Math.max(...values) * 1.08 || 1;
  const x = (index) => pad.left + (index / 9) * (width - pad.left - pad.right);
  const y = (value) => height - pad.bottom - ((value - minY) / (maxY - minY)) * (height - pad.top - pad.bottom);

  ctx.save();
  ctx.font = "10px Inter, sans-serif";
  ctx.fillStyle = COLORS.muted;
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let tick = 0; tick <= 2; tick += 1) {
    const value = minY + ((maxY - minY) * tick) / 2;
    const py = y(value);
    ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(width - pad.right, py); ctx.stroke();
    ctx.fillText(value.toFixed(2), pad.left - 6, py);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("d0", x(0), height - pad.bottom + 7);
  ctx.fillText("d9", x(9), height - pad.bottom + 7);
  ctx.restore();

  ["early", "late"].forEach((block) => {
    const series = curves[block];
    if (!series.length) return;
    const active = block === state.block;
    ctx.save();
    ctx.strokeStyle = COLORS[block];
    ctx.fillStyle = COLORS[block];
    ctx.globalAlpha = active ? 1 : 0.42;
    ctx.lineWidth = active ? 2.5 : 1.5;
    ctx.beginPath();
    series.slice(0, 10).forEach((value, index) => {
      if (index === 0) ctx.moveTo(x(index), y(value)); else ctx.lineTo(x(index), y(value));
    });
    ctx.stroke();
    const current = Math.min(state.step, series.length - 1);
    ctx.beginPath();
    ctx.arc(x(current), y(series[current]), active ? 4.5 : 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  });

  const active = curves[state.block];
  if (active.length) {
    const current = active[Math.min(state.step, active.length - 1)];
    const ratio = active[0] ? current / active[0] : 0;
    els.curveDelta.textContent = `当前 d${state.step} / d0 · ${formatPercent(ratio)}`;
  }
}

function retentionValue(layer) {
  if (layer.retention.length) return layer.retention[layer.retention.length - 1];
  if (layer.dispersion.length && layer.dispersion[0] !== 0) {
    return layer.dispersion[layer.dispersion.length - 1] / layer.dispersion[0];
  }
  return null;
}

function isLayerActive(layer, view) {
  if (view?.layers.length) return view.layers.includes(layer);
  return state.block === "early" ? layer <= 5 : layer >= 12;
}

function renderLayers() {
  const pool = currentPool();
  const layers = pool?.layers.length
    ? pool.layers
    : asArray(state.data?.raw?.global?.layers).map(normalizeLayer);
  els.layerTable.replaceChildren();
  if (!layers.length) {
    const empty = document.createElement("p");
    empty.className = "panel-caption";
    empty.textContent = "当前数据没有 per-layer dispersion。";
    els.layerTable.append(empty);
    return;
  }
  const view = currentView();
  layers.slice(0, 8).forEach((layer) => {
    const ratioRaw = retentionValue(layer);
    const ratio = ratioRaw !== null && ratioRaw > 2 ? ratioRaw / 100 : ratioRaw;
    const active = isLayerActive(layer.layer, view);
    const block = layer.layer <= 5 ? "early" : "late";
    const row = document.createElement("div");
    row.className = `layer-row${active ? " is-active" : ""}`;
    row.setAttribute("role", "row");

    const label = document.createElement("span");
    label.className = "layer-label";
    label.setAttribute("role", "cell");
    label.textContent = `L${layer.layer}`;

    const track = document.createElement("div");
    track.className = "layer-track";
    track.setAttribute("role", "cell");
    const fill = document.createElement("div");
    fill.className = `layer-fill ${block}`;
    fill.style.width = ratio === null ? "0%" : `${Math.max(0, Math.min(100, ratio * 100))}%`;
    track.append(fill);

    const value = document.createElement("span");
    value.className = "layer-value";
    value.setAttribute("role", "cell");
    value.textContent = ratio === null ? "—" : formatPercent(ratio);
    if (layer.dispersion.length) {
      row.title = `L${layer.layer}: ${layer.dispersion[0].toFixed(3)} → ${layer.dispersion.at(-1).toFixed(3)}`;
    }
    row.append(label, track, value);
    els.layerTable.append(row);
  });
}

function candidateRadii(view) {
  const points = view?.pcaSteps[state.step] || [];
  if (!points.length) return [];
  const cx = points.reduce((sum, point) => sum + point[0], 0) / points.length;
  const cy = points.reduce((sum, point) => sum + point[1], 0) / points.length;
  return points.map((point) => Math.hypot(point[0] - cx, point[1] - cy));
}

function renderCandidates() {
  const pool = currentPool();
  if (!pool) return;
  const radii = candidateRadii(currentView());
  const maxRadius = Math.max(...radii, 1e-9);
  const scroll = els.candidateStrip.parentElement.scrollLeft;
  els.candidateStrip.replaceChildren();
  pool.candidates.slice(0, 32).forEach((candidate, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate-item";
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(index === state.candidate));
    button.setAttribute("aria-label", `候选 ${index + 1}，seed ${candidate.seed}，${candidate.success ? "成功" : "失败"}`);
    button.dataset.candidate = String(index);

    const status = document.createElement("span");
    status.className = `status-bar ${candidate.success ? "success" : "failure"}`;
    const level = document.createElement("span");
    level.className = "candidate-level";
    const relativeRadius = (radii[index] || 0) / maxRadius;
    level.style.height = `${Math.max(3, relativeRadius * 34)}px`;
    const number = document.createElement("span");
    number.className = "candidate-index";
    number.textContent = String(index + 1).padStart(2, "0");
    button.title = `seed ${candidate.seed} · ${candidate.success ? "成功" : "失败"} · 二维相对半径 ${relativeRadius.toFixed(2)}（本步最大=1，不可跨步比较）`;
    button.append(status, level, number);
    button.addEventListener("click", () => selectCandidate(index));
    button.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const delta = event.key === "ArrowLeft" ? -1 : 1;
      const next = Math.max(0, Math.min(pool.candidates.length - 1, index + delta));
      selectCandidate(next);
      requestAnimationFrame(() => els.candidateStrip.children[next]?.focus());
    });
    els.candidateStrip.append(button);
  });
  els.candidateStrip.parentElement.scrollLeft = scroll;

  const candidate = pool.candidates[state.candidate];
  const relativeRadius = maxRadius ? (radii[state.candidate] || 0) / maxRadius : 0;
  els.selectedSummary.textContent = candidate
    ? `#${String(state.candidate + 1).padStart(2, "0")} · seed ${candidate.seed} · ${candidate.success ? "成功" : "失败"} · 二维相对半径 ${relativeRadius.toFixed(2)}（本步最大=1，不可跨步比较）`
    : "选择候选查看位置";
}

function globalNormCv(block) {
  const view = globalView(block);
  const value = finiteNumber(firstDefined(
    view.matched_site_norm_cv,
    view?.raw_cloud?.matched_site_norm_cv,
    view.matched_site_hidden_norm_cv,
    view.norm_cv_matched_site,
    state.data?.raw?.global?.matched_site_norm_cv?.[block],
  ));
  if (value === null) return FALLBACK_NORM_CV[block];
  return value > 1 ? value / 100 : value;
}

function retentionAt(view, step) {
  const retained = view?.retention?.[Math.min(step, view.retention.length - 1)];
  if (Number.isFinite(retained)) return retained > 2 ? retained / 100 : retained;
  const dispersion = view?.dispersion || [];
  const value = dispersion[Math.min(step, dispersion.length - 1)];
  return Number.isFinite(value) && dispersion[0] ? value / dispersion[0] : null;
}

function globalEndpointRetention(block) {
  const view = globalView(block);
  const endpoint = finiteNumber(view?.endpoint?.d9_over_d0_pool_median);
  if (endpoint !== null) return endpoint > 2 ? endpoint / 100 : endpoint;
  const retention = normalizeSeries(view.retention_median_by_pool);
  if (retention.length) return retention.at(-1) > 2 ? retention.at(-1) / 100 : retention.at(-1);
  const dispersion = normalizeSeries(view.dispersion_median_by_pool);
  return dispersion.length && dispersion[0] ? dispersion.at(-1) / dispersion[0] : null;
}

function renderFinding() {
  els.findingPools.textContent = String(state.data?.pools.length || "—");
  els.findingEarly.textContent = formatPercent(globalEndpointRetention("early"), 0);
  els.findingLate.textContent = formatPercent(globalEndpointRetention("late"), 0);
}

function renderMetrics() {
  const pool = currentPool();
  const view = currentView();
  if (!pool) return;
  const counts = successCounts(pool);
  els.metricOutcome.textContent = `${counts.successes} S / ${counts.failures} F`;
  els.metricOutcomeNote.textContent = `state ${pool.initState} · K=${pool.candidates.length}`;

  const dispersion = view?.dispersion || [];
  const current = dispersion[Math.min(state.step, dispersion.length - 1)];
  els.metricDispersion.textContent = Number.isFinite(current) ? current.toFixed(4) : "—";
  els.metricDispersionNote.textContent = `${state.block === "early" ? "HB2–5" : "HB12–15"} · d${state.step}`;

  const retained = retentionAt(view, state.step);
  const finalRetained = retentionAt(view, 9);
  els.metricRetentionLabel.textContent = `当前 d${state.step} / d0 retention`;
  els.metricRetention.textContent = formatPercent(retained);
  els.metricRetentionNote.textContent = `最终 d9 = ${formatPercent(finalRetained)} · raw hidden RMS`;
  els.scatterRetention.textContent = formatPercent(retained);
  const explained = view?.explained?.reduce((sum, value) => sum + value, 0);
  els.pcaExplained.textContent = formatPercent(explained, 1);

  els.metricNormCv.textContent = formatPercent(globalNormCv(state.block), 3);
  els.metricNormCvNote.textContent = "all-pool median · 跨 K32 候选";

  const stateTokenMax = finiteNumber(firstDefined(
    pool?.stateToken?.max,
    pool?.stateToken?.max_range,
    state.data?.raw?.global?.state_token_range?.max,
    state.data?.raw?.global?.state_token_max_candidate_range,
  ));
  els.stateTokenControl.textContent = stateTokenMax === null
    ? "state token Δmax = —"
    : `state token Δmax = ${stateTokenMax.toFixed(stateTokenMax === 0 ? 0 : 6)}`;
}

function extractProbeRows(rawProbe) {
  if (!rawProbe) return [];
  let subsets = [];
  if (Array.isArray(rawProbe)) subsets = rawProbe;
  else if (Array.isArray(rawProbe.subsets)) subsets = rawProbe.subsets;
  else if (Array.isArray(rawProbe.rows)) subsets = rawProbe.rows;
  else {
    subsets = Object.entries(rawProbe)
      .filter(([, value]) => value && typeof value === "object" && !Array.isArray(value))
      .map(([name, value]) => ({ name, ...value }));
  }
  return subsets.map((subset) => {
    let strongest = firstDefined(subset.strongest, subset.best_feature, subset.feature, subset.features);
    if (Array.isArray(strongest)) {
      strongest = [...strongest].sort((a, b) => Math.abs(finiteNumber(b.auc, 0.5) - 0.5) - Math.abs(finiteNumber(a.auc, 0.5) - 0.5))[0];
    }
    if (!strongest || typeof strongest !== "object") strongest = subset;
    const name = String(firstDefined(subset.name, subset.id, ""));
    const featureName = String(firstDefined(strongest.name, strongest.id, strongest.feature, subset.feature_name, "—"));
    return {
      name,
      label: String(firstDefined(subset.label, SUBSET_LABELS[name], name)),
      feature: FEATURE_LABELS[featureName] || featureName.replaceAll("_", " "),
      auc: finiteNumber(firstDefined(strongest.auc, subset.auc)),
      rawP: finiteNumber(firstDefined(strongest.p_raw_two_sided, strongest.raw_p, strongest.p_raw, subset.raw_p)),
      maxP: finiteNumber(firstDefined(strongest.p_maxT, strongest.maxT_p, strongest.max_p, subset.maxT_p)),
    };
  }).filter((row) => row.name && row.auc !== null);
}

function renderEvidence() {
  const supplied = extractProbeRows(firstDefined(
    state.data?.raw?.success_probe,
    state.data?.raw?.success_screen,
  ));
  const suppliedByName = new Map(supplied.map((row) => [row.name, row]));
  const rows = FALLBACK_EVIDENCE.map((fallback) => ({
    ...fallback,
    ...(suppliedByName.get(fallback.name) || {}),
  }));
  supplied.forEach((row) => {
    if (!rows.some((existing) => existing.name === row.name)) rows.push(row);
  });

  els.evidenceBody.replaceChildren();
  rows.slice(0, 6).forEach((row) => {
    const tr = document.createElement("tr");
    [
      row.label,
      row.feature,
      formatNumber(row.auc, 4),
      formatP(row.rawP),
      formatP(row.maxP),
    ].forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.append(td);
    });
    els.evidenceBody.append(tr);
  });
  const note = firstDefined(
    state.data?.raw?.success_probe?.drift_note,
    state.data?.raw?.global?.drift_note,
  );
  if (note) els.driftNote.textContent = String(note);
}

function formatNumber(value, digits = 3) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

function formatPercent(ratio, digits = 1) {
  return Number.isFinite(ratio) ? `${(ratio * 100).toFixed(digits)}%` : "—";
}

function formatP(value) {
  if (!Number.isFinite(value)) return "—";
  if (value < 0.0001) return "<0.0001";
  return value.toFixed(4);
}

function renderAll() {
  if (!state.data) return;
  els.stepRange.value = String(state.step);
  els.stepOutput.textContent = `d${state.step} / d9`;
  els.segments.forEach((button) => {
    const active = button.dataset.block === state.block;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  els.trajectoryToggle.setAttribute("aria-pressed", String(state.showTrajectories));
  els.trajectoryToggle.title = state.showTrajectories ? "隐藏完整 d0–d9 轨迹" : "显示完整 d0–d9 轨迹";
  renderFinding();
  renderMetrics();
  renderScatter();
  renderCurve();
  renderLayers();
  renderCandidates();
}

function selectCandidate(index) {
  const pool = currentPool();
  if (!pool) return;
  state.candidate = Math.max(0, Math.min(pool.candidates.length - 1, index));
  renderScatter();
  renderCandidates();
}

function setStep(step) {
  state.step = Math.max(0, Math.min(9, Number(step)));
  renderAll();
}

function stopPlayback() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  els.play.innerHTML = "<span aria-hidden=\"true\">&#9654;</span>";
  els.play.setAttribute("aria-label", "播放");
  els.play.title = "播放";
}

function togglePlayback() {
  if (state.timer) {
    stopPlayback();
    return;
  }
  els.play.textContent = "Ⅱ";
  els.play.setAttribute("aria-label", "暂停");
  els.play.title = "暂停";
  state.timer = window.setInterval(() => setStep((state.step + 1) % 10), 650);
}

function showTooltip(event, projected) {
  const pool = currentPool();
  const candidate = pool?.candidates[projected.candidateIndex];
  if (!candidate) return;
  els.tooltip.replaceChildren();
  const heading = document.createElement("strong");
  heading.textContent = `#${String(projected.candidateIndex + 1).padStart(2, "0")} · ${candidate.success ? "成功" : "失败"}`;
  const details = document.createElement("span");
  details.textContent = `seed ${candidate.seed} · d${state.step} · (${projected.raw[0].toFixed(2)}, ${projected.raw[1].toFixed(2)})`;
  els.tooltip.append(heading, details);
  const shell = els.pcaShell.getBoundingClientRect();
  const left = Math.min(event.clientX - shell.left, shell.width - 205);
  const top = Math.min(event.clientY - shell.top, shell.height - 65);
  els.tooltip.style.left = `${Math.max(0, left)}px`;
  els.tooltip.style.top = `${Math.max(0, top)}px`;
  els.tooltip.hidden = false;
}

function nearestProjectedPoint(event) {
  const rect = els.pca.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let nearest = null;
  let distance = 13;
  state.projectedPoints.forEach((point) => {
    const candidateDistance = Math.hypot(point.x - x, point.y - y);
    if (candidateDistance < distance) {
      nearest = point;
      distance = candidateDistance;
    }
  });
  return nearest;
}

function bindEvents() {
  els.task.addEventListener("change", () => {
    stopPlayback();
    state.taskId = els.task.value;
    state.step = 0;
    state.candidate = 0;
    populateStateOptions();
    updateQuery();
    renderAll();
  });
  els.state.addEventListener("change", () => {
    stopPlayback();
    state.poolId = els.state.value;
    state.step = 0;
    state.candidate = 0;
    updateQuery();
    renderAll();
  });
  els.segments.forEach((button) => button.addEventListener("click", () => {
    state.block = button.dataset.block;
    renderAll();
  }));
  els.stepRange.addEventListener("input", () => {
    stopPlayback();
    setStep(els.stepRange.value);
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
  els.trajectoryToggle.addEventListener("click", () => {
    state.showTrajectories = !state.showTrajectories;
    els.trajectoryToggle.setAttribute("aria-pressed", String(state.showTrajectories));
    els.trajectoryToggle.title = state.showTrajectories ? "隐藏完整 d0–d9 轨迹" : "显示完整 d0–d9 轨迹";
    renderScatter();
  });

  els.pca.addEventListener("pointermove", (event) => {
    const nearest = nearestProjectedPoint(event);
    const nextHover = nearest?.candidateIndex ?? null;
    if (nextHover !== state.hoverCandidate) {
      state.hoverCandidate = nextHover;
      renderScatter();
    }
    if (nearest) showTooltip(event, nearest);
    else els.tooltip.hidden = true;
  });
  els.pca.addEventListener("pointerleave", () => {
    state.hoverCandidate = null;
    els.tooltip.hidden = true;
    renderScatter();
  });
  els.pca.addEventListener("click", (event) => {
    const nearest = nearestProjectedPoint(event);
    if (nearest) selectCandidate(nearest.candidateIndex);
  });
  window.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, button")) return;
    if (event.key === "ArrowLeft") { stopPlayback(); setStep(state.step - 1); }
    if (event.key === "ArrowRight") { stopPlayback(); setStep(state.step + 1); }
    if (event.key === " ") { event.preventDefault(); togglePlayback(); }
  });

  const resize = new ResizeObserver(() => {
    if (!state.data) return;
    window.requestAnimationFrame(() => {
      renderScatter();
      renderCurve();
    });
  });
  resize.observe(els.pca);
  resize.observe(els.curve);
}

async function loadData() {
  try {
    const embedded = document.querySelector("#activation-data");
    let raw;
    if (embedded?.textContent.trim()) {
      raw = JSON.parse(embedded.textContent);
    } else {
      const response = await fetch("./data.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      raw = await response.json();
    }
    state.data = normalizeData(raw);
    selectInitialPool();
    setReady(true);
    els.load.className = "load-state is-ready";
    renderEvidence();
    const source = raw.source;
    if (typeof source === "string") els.sourceNote.textContent = `数据源：${source}`;
    updateQuery();
    renderAll();
  } catch (error) {
    displayError(error instanceof Error ? error : new Error(String(error)));
  }
}

bindEvents();
loadData();
