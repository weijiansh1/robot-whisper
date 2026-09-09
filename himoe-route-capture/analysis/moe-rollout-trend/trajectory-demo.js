(() => {
  "use strict";

  const DATA = window.HIMOE_ROLLOUT_DEMO;
  const ATLAS = DATA && DATA.trajectory_routes;
  if (!ATLAS || ATLAS.version < 2) {
    document.getElementById("stage").style.display = "none";
    document.getElementById("error").style.display = "block";
    return;
  }

  const $ = (id) => document.getElementById(id);
  const TASK_NAMES = {
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside": "抽屉放碗",
    "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": "双摩卡壶上炉",
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": "ramekin 上黑碗",
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": "炉上黑碗"
  };
  const DEFAULT_TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove";
  const COLORS = {
    success: "#56d6a6",
    failure: "#ff6b60",
    gold: "#f2bf5b",
    blue: "#62a9ff"
  };

  const initialTask = ATLAS.tasks.find((task) => task.task === DEFAULT_TASK) || ATLAS.tasks[0];
  const state = {
    task: initialTask.task,
    scene: initialTask.default_scene,
    focus: "all",
    progress: 0,
    playing: !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    speed: 1,
    startHold: 1,
    endHold: 0,
    lastFrame: performance.now(),
    markers: []
  };

  function clamp(value, low = 0, high = 1) {
    return Math.max(low, Math.min(high, value));
  }

  function mix(a, b, amount) {
    return a + (b - a) * amount;
  }

  function smooth(amount) {
    const x = clamp(amount);
    return x * x * (3 - 2 * x);
  }

  function fmt(value, digits = 3, signed = false) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "n/a";
    const number = Number(value);
    return `${signed && number > 0 ? "+" : ""}${number.toFixed(digits)}`;
  }

  function currentTask() {
    return ATLAS.tasks.find((task) => task.task === state.task);
  }

  function currentScene() {
    return currentTask().scenes.find((scene) => scene.scene === Number(state.scene));
  }

  function selectedRoute() {
    if (state.focus === "all") return null;
    return currentScene().routes.find((route) => route.episode === Number(state.focus)) || null;
  }

  function maxIndex() {
    const route = selectedRoute();
    return Math.max(0, (route ? route.chunks : currentScene().max_chunks) - 1);
  }

  function focusEpisodes(scene) {
    if (state.focus !== "all") return new Set([Number(state.focus)]);
    return new Set(Object.values(scene.highlighted));
  }

  function routePoint(route, progress) {
    const last = route.points.length - 1;
    if (progress >= last) return route.points[last];
    const low = Math.max(0, Math.floor(progress));
    const amount = smooth(progress - low);
    return [
      mix(route.points[low][0], route.points[low + 1][0], amount),
      mix(route.points[low][1], route.points[low + 1][1], amount)
    ];
  }

  function routeScore(route, progress) {
    const values = route.routing_percentile;
    if (!values.length) return null;
    const last = values.length - 1;
    const low = Math.min(last, Math.max(0, Math.floor(progress)));
    const high = Math.min(last, low + 1);
    const a = values[low];
    const b = values[high];
    if (a === null && b === null) return null;
    if (a === null) return b;
    if (b === null || low === high) return a;
    return mix(a, b, smooth(progress - low));
  }

  function canvasContext() {
    const canvas = $("route-canvas");
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.max(1, Math.round(rect.width * dpr));
    const pixelHeight = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { canvas, ctx, width: rect.width, height: rect.height };
  }

  function stageTransform(scene, width, height) {
    const points = scene.routes.flatMap((route) => route.points);
    let minX = Math.min(...points.map((point) => point[0]));
    let maxX = Math.max(...points.map((point) => point[0]));
    let minY = Math.min(...points.map((point) => point[1]));
    let maxY = Math.max(...points.map((point) => point[1]));
    const dx = Math.max(1e-6, maxX - minX);
    const dy = Math.max(1e-6, maxY - minY);
    minX -= dx * 0.12;
    maxX += dx * 0.12;
    minY -= dy * 0.18;
    maxY += dy * 0.18;
    const mobile = width < 620;
    const rect = mobile
      ? { left: 25, top: 150, width: width - 50, height: Math.max(220, height - 400) }
      : { left: 58, top: 168, width: Math.max(380, width - 420), height: Math.max(300, height - 298) };
    const scale = Math.min(rect.width / (maxX - minX), rect.height / (maxY - minY));
    const dataWidth = (maxX - minX) * scale;
    const dataHeight = (maxY - minY) * scale;
    const originX = rect.left + (rect.width - dataWidth) / 2 - minX * scale;
    const originY = rect.top + (rect.height - dataHeight) / 2 + maxY * scale;
    return {
      rect,
      point: (point) => [originX + point[0] * scale, originY - point[1] * scale]
    };
  }

  function drawGrid(ctx, transform) {
    const rect = transform.rect;
    ctx.save();
    ctx.strokeStyle = "#222a28";
    ctx.lineWidth = 1;
    for (let axis = 0; axis <= 8; axis += 1) {
      const x = rect.left + rect.width * axis / 8;
      const y = rect.top + rect.height * axis / 8;
      ctx.beginPath();
      ctx.moveTo(x, rect.top);
      ctx.lineTo(x, rect.top + rect.height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(rect.left, y);
      ctx.lineTo(rect.left + rect.width, y);
      ctx.stroke();
    }
    ctx.strokeStyle = "#36413e";
    ctx.strokeRect(rect.left, rect.top, rect.width, rect.height);
    ctx.fillStyle = "#63706b";
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("PC1 (EEF xyz)", rect.left + rect.width, rect.top + rect.height + 18);
    ctx.save();
    ctx.translate(rect.left - 17, rect.top + 4);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "right";
    ctx.fillText("PC2", 0, 0);
    ctx.restore();
    ctx.restore();
  }

  function traceRoute(ctx, route, progress, transform, options) {
    const last = route.points.length - 1;
    const end = Math.min(last, progress);
    const whole = Math.floor(end);
    const fraction = end - whole;
    ctx.beginPath();
    const start = transform.point(route.points[0]);
    ctx.moveTo(start[0], start[1]);
    for (let chunk = 1; chunk <= whole; chunk += 1) {
      const point = transform.point(route.points[chunk]);
      ctx.lineTo(point[0], point[1]);
    }
    if (whole < last && fraction > 0) {
      const point = transform.point(routePoint(route, end));
      ctx.lineTo(point[0], point[1]);
    }
    ctx.strokeStyle = options.color;
    ctx.globalAlpha = options.alpha;
    ctx.lineWidth = options.width;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function meanRoute(scene, success) {
    scene._meanRoutes ||= {};
    const key = success ? "success" : "failure";
    if (scene._meanRoutes[key]) return scene._meanRoutes[key];
    const routes = scene.routes.filter((route) => route.success === success);
    if (!routes.length) return null;
    const points = Array.from({ length: scene.max_chunks }, (_, chunk) => {
      const held = routes.map((route) => route.points[Math.min(chunk, route.points.length - 1)]);
      return [
        held.reduce((sum, point) => sum + point[0], 0) / held.length,
        held.reduce((sum, point) => sum + point[1], 0) / held.length
      ];
    });
    scene._meanRoutes[key] = { points };
    return scene._meanRoutes[key];
  }

  function drawArrow(ctx, from, to, color, label) {
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    const length = Math.hypot(dx, dy);
    if (length < 8) return;
    const ux = dx / length;
    const uy = dy / length;
    const tip = [to[0] - ux * 10, to[1] - uy * 10];
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.7;
    ctx.beginPath();
    ctx.moveTo(from[0], from[1]);
    ctx.lineTo(tip[0], tip[1]);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(tip[0], tip[1]);
    ctx.lineTo(tip[0] - ux * 8 + uy * 4, tip[1] - uy * 8 - ux * 4);
    ctx.lineTo(tip[0] - ux * 8 - uy * 4, tip[1] - uy * 8 + ux * 4);
    ctx.closePath();
    ctx.fill();
    ctx.font = "italic 13px Cambria Math, serif";
    ctx.textAlign = "center";
    ctx.fillText(label, (from[0] + tip[0]) / 2 - uy * 10, (from[1] + tip[1]) / 2 + ux * 10);
    ctx.restore();
  }

  function covarianceEllipse(points) {
    const meanX = points.reduce((sum, point) => sum + point[0], 0) / points.length;
    const meanY = points.reduce((sum, point) => sum + point[1], 0) / points.length;
    let xx = 0;
    let yy = 0;
    let xy = 0;
    points.forEach((point) => {
      const dx = point[0] - meanX;
      const dy = point[1] - meanY;
      xx += dx * dx;
      yy += dy * dy;
      xy += dx * dy;
    });
    xx /= points.length;
    yy /= points.length;
    xy /= points.length;
    const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
    const trace = xx + yy;
    const root = Math.sqrt(Math.max(0, (xx - yy) ** 2 + 4 * xy * xy));
    return {
      x: meanX,
      y: meanY,
      rx: 2 * Math.sqrt(Math.max(4, (trace + root) / 2)) + 5,
      ry: 2 * Math.sqrt(Math.max(4, (trace - root) / 2)) + 5,
      angle
    };
  }

  function drawCluster(ctx, points, color, label, bounds) {
    if (points.length < 2) return;
    const ellipse = covarianceEllipse(points);
    ellipse.rx = Math.min(ellipse.rx, bounds.width * 0.24);
    ellipse.ry = Math.min(ellipse.ry, bounds.height * 0.24);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.58;
    ctx.lineWidth = 1.3;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.ellipse(ellipse.x, ellipse.y, ellipse.rx, ellipse.ry, ellipse.angle, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "11px Inter, system-ui, sans-serif";
    const labelOnLeft = ellipse.x + ellipse.rx + 88 > bounds.left + bounds.width;
    ctx.textAlign = labelOnLeft ? "right" : "left";
    ctx.fillText(label, labelOnLeft ? ellipse.x - ellipse.rx - 7 : ellipse.x + ellipse.rx + 7, ellipse.y - 3);
    ctx.restore();
  }

  function drawCallout(ctx, scene, transform, chunk, direction) {
    if (state.progress < chunk - 0.05 || state.progress > 18) return;
    const episode = scene.highlighted.failure;
    const route = scene.routes.find((item) => item.episode === episode);
    if (!route || route.points.length <= chunk) return;
    const taskGap = DATA.by_task.find((row) =>
      row.task === state.task &&
      row.signal === "per_query_success_manifold_percentile" &&
      row.chunk === chunk
    );
    const aggregate = DATA.gaps.find((row) =>
      row.signal === "per_query_success_manifold_percentile" && row.chunk === chunk
    );
    if (!taskGap || !aggregate) return;
    const point = transform.point(route.points[chunk]);
    const offsetX = chunk === 5 ? 52 : -58;
    const offsetY = direction * (chunk === 5 ? 46 : 62);
    const end = [point[0] + offsetX, point[1] + offsetY];
    const fade = state.progress <= 14 ? 1 : clamp((18 - state.progress) / 4);
    ctx.save();
    ctx.strokeStyle = COLORS.gold;
    ctx.fillStyle = COLORS.gold;
    ctx.globalAlpha = clamp((state.progress - chunk + 0.3) / 0.5) * fade;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(point[0], point[1]);
    ctx.lineTo(end[0], end[1]);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.textAlign = offsetX > 0 ? "left" : "right";
    ctx.font = "700 12px Inter, system-ui, sans-serif";
    const x = end[0] + (offsetX > 0 ? 5 : -5);
    ctx.fillText(`k${chunk}: contribution gap ${fmt(taskGap.failure_minus_success, 3, true)}`, x, end[1] - 4);
    ctx.fillStyle = "#9eaaa4";
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.fillText(`4-task ${fmt(aggregate.failure_minus_success, 3, true)}`, x, end[1] + 12);
    ctx.restore();
  }

  function drawHistoricalHalos(ctx, route, progress, transform) {
    const last = Math.min(Math.floor(progress), route.points.length - 1);
    const stride = route.points.length > 30 ? 2 : 1;
    for (let chunk = 0; chunk <= last; chunk += stride) {
      const score = route.routing_percentile[chunk];
      if (score === null) continue;
      const point = transform.point(route.points[chunk]);
      ctx.beginPath();
      ctx.arc(point[0], point[1], 3 + score * 5, 0, Math.PI * 2);
      ctx.strokeStyle = COLORS.gold;
      ctx.globalAlpha = 0.1 + score * 0.2;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function drawScene() {
    const { canvas, ctx, width, height } = canvasContext();
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#101315";
    ctx.fillRect(0, 0, width, height);
    const scene = currentScene();
    const transform = stageTransform(scene, width, height);
    const focus = focusEpisodes(scene);
    drawGrid(ctx, transform);

    [true, false].forEach((success) => {
      const route = meanRoute(scene, success);
      if (!route) return;
      traceRoute(ctx, route, state.progress, transform, {
        color: success ? COLORS.success : COLORS.failure,
        alpha: 0.07,
        width: 22
      });
    });

    const ordered = [...scene.routes].sort((a, b) => Number(focus.has(a.episode)) - Number(focus.has(b.episode)));
    state.markers = [];
    ordered.forEach((route) => {
      const highlighted = focus.has(route.episode);
      const color = route.success ? COLORS.success : COLORS.failure;
      const score = routeScore(route, state.progress);
      if (highlighted) {
        traceRoute(ctx, route, state.progress, transform, { color, alpha: 0.14, width: 12 });
      }
      traceRoute(ctx, route, state.progress, transform, {
        color,
        alpha: highlighted ? 0.98 : 0.2,
        width: highlighted ? 3.2 : 1.05
      });
      if (highlighted) drawHistoricalHalos(ctx, route, state.progress, transform);

      const point = transform.point(routePoint(route, state.progress));
      const ended = state.progress >= route.points.length - 1;
      ctx.beginPath();
      if (ended) {
        const radius = highlighted ? 5.2 : 2.6;
        ctx.rect(point[0] - radius, point[1] - radius, radius * 2, radius * 2);
      } else {
        ctx.arc(point[0], point[1], highlighted ? 5.5 : 2.4, 0, Math.PI * 2);
      }
      ctx.fillStyle = color;
      ctx.globalAlpha = highlighted ? 1 : ended ? 0.3 : 0.48;
      ctx.fill();
      ctx.globalAlpha = 1;
      if (highlighted && score !== null) {
        ctx.beginPath();
        ctx.arc(point[0], point[1], 8 + score * 7, 0, Math.PI * 2);
        ctx.strokeStyle = COLORS.gold;
        ctx.globalAlpha = 0.42 + score * 0.4;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      state.markers.push({ x: point[0], y: point[1], route, score, highlighted, ended });
    });

    const start = transform.point(scene.routes[0].points[0]);
    ctx.beginPath();
    ctx.arc(start[0], start[1], 6, 0, Math.PI * 2);
    ctx.fillStyle = "#f0f4f2";
    ctx.fill();
    ctx.font = "italic 17px Cambria Math, serif";
    ctx.fillStyle = "#edf2ef";
    ctx.textAlign = "left";
    ctx.fillText("s0", start[0] + 10, start[1] - 10);

    const arrowRoute = selectedRoute()
      || scene.routes.find((route) => route.episode === scene.highlighted.failure)
      || scene.routes.find((route) => route.episode === scene.highlighted.success);
    if (arrowRoute && state.progress > 0.15 && state.progress < arrowRoute.points.length - 1) {
      const low = Math.floor(state.progress);
      const from = transform.point(arrowRoute.points[low]);
      const to = transform.point(routePoint(arrowRoute, state.progress));
      drawArrow(ctx, from, to, COLORS.gold, `A${low}`);
    }

    const mobile = width < 620;
    if (state.progress >= 6.6 && scene.success > 1 && scene.failure > 1) {
      const successPoints = scene.routes
        .filter((route) => route.success)
        .map((route) => transform.point(routePoint(route, state.progress)));
      const failurePoints = scene.routes
        .filter((route) => !route.success)
        .map((route) => transform.point(routePoint(route, state.progress)));
      drawCluster(ctx, successPoints, COLORS.success, mobile ? "success" : "success / terminal held", transform.rect);
      drawCluster(ctx, failurePoints, COLORS.failure, mobile ? "failure" : "failure / terminal held", transform.rect);
    }

    drawCallout(ctx, scene, transform, 5, -1);
    drawCallout(ctx, scene, transform, 8, mobile ? 1 : 1);
    canvas._transform = transform;
    updateNarrative();
  }

  function chapterFor(k, active) {
    if (k === 0) return {
      number: "01",
      label: "SAME INITIAL STATE",
      title: "32 条路线，从同一个物理状态出发",
      copy: "观察相同，flow-noise seed 不同。颜色是事后 outcome；路线与 MoE 状态尚未接收任何执行后的反馈。",
      equation: "s<sub>0</sub> <span class=\"accent\">&#8594;</span> A<sub>0</sub>"
    };
    if (k <= 4) return {
      number: "02",
      label: "THE WORLD REMEMBERS",
      title: "每个 action chunk 都改写下一次起点",
      copy: "模型会重新推理，但环境不会重置。当前 chunk 的 MoE 只看当前观察，而当前观察已经携带前序动作造成的物理历史。",
      equation: "s<sub>k+1</sub> = F(s<sub>k</sub>, A<sub>k</sub>)"
    };
    if (k <= 8) return {
      number: "03",
      label: "FIXED COHORT WINDOW",
      title: "k0-k8：所有 2048 条 rollout 都仍在运行",
      copy: "金色 halo 是本 initial state 内的 HB routing-state 描述性距离；routed-contribution 对照只在这一无终止偏差窗口内报告。",
      equation: "&#916;<sub>MoE</sub>(k) <span class=\"accent\">&gt; 0</span>"
    };
    if (active[0] === 0 || active[1] === 0) return {
      number: "05",
      label: "RISK SET LIMIT",
      title: "同期对照消失：不能再计算成功-失败 MoE gap",
      copy: "路线仍可继续显示，但这一时刻只剩单一 outcome 的 rollout 在推理。页面保留物理终点，不把已终止轨迹伪装成新的 MoE 样本。",
      equation: "n<sub>S</sub>(k) = 0 <span class=\"accent\">&#8658;</span> &#916;<sub>routing</sub>(k) undefined"
    };
    return {
      number: "04",
      label: "FULL CLOSED LOOP",
      title: "几十个 chunk 中，路线与内部状态继续共同演化",
      copy: "这里是变长 risk set：只有尚未终止的成功和失败 rollout 参与当前 routing gap。它适合观察形成过程，但不能与固定 k0-k8 的效应量直接等同。",
      equation: "o<sub>k</sub> &#8594; z<sub>k</sub> &#8594; A<sub>k</sub> &#8594; o<sub>k+1</sub>"
    };
  }

  function updateNarrative() {
    const task = currentTask();
    const scene = currentScene();
    const k = Math.max(0, Math.min(maxIndex(), Math.floor(state.progress + 0.35)));
    const active = scene.active_counts[Math.min(k, scene.active_counts.length - 1)] || [0, 0];
    const chapter = chapterFor(k, active);
    $("chapter-number").textContent = chapter.number;
    $("chapter-label").textContent = chapter.label;
    $("chapter-title").textContent = chapter.title;
    $("chapter-copy").textContent = chapter.copy;
    $("equation").innerHTML = chapter.equation;
    $("readout-k").textContent = `k${k}`;

    const physical = scene.physical_centroid_gap_m[k];
    const routingGap = scene.routing_gap[k];
    const contributionGap = k < ATLAS.fixed_contribution_chunks ? scene.contribution_gap[k] : null;
    $("physical-gap").textContent = physical === null ? "n/a" : `${fmt(physical * 100, 2)} cm`;
    $("task-gap").textContent = fmt(routingGap, 3, true);
    $("cohort-gap").textContent = k < ATLAS.fixed_contribution_chunks
      ? fmt(contributionGap, 3, true)
      : "not estimated";
    $("reading").textContent = `${chapter.copy} 当前 active: ${active[0]}S / ${active[1]}F。`;

    const focus = selectedRoute();
    const focusText = focus
      ? `${focus.success ? "成功" : "失败"} ep ${focus.episode} · ${focus.chunks} chunks / ${focus.action_steps} actions`
      : `全部 32 条 · ${scene.min_chunks}-${scene.max_chunks} chunks`;
    $("scene-meta").innerHTML = `<strong>${TASK_NAMES[task.task]} · initial state ${scene.scene}</strong>${scene.success} success / ${scene.failure} failure · 当前 active ${active[0]}S/${active[1]}F<br>${focusText}<br>EEF PCA-2D retains ${(scene.pca_explained_fraction * 100).toFixed(1)}% variance`;
    $("route-range").value = state.progress;
    $("time-label").textContent = `k${state.progress.toFixed(2)} / k${maxIndex()}`;
    document.querySelectorAll("[data-k]").forEach((button) => {
      const nodeK = Number(button.dataset.k);
      button.classList.toggle("active", nodeK === k);
      button.classList.toggle("past", nodeK < state.progress);
    });
  }

  function timelineTicks(last) {
    if (last <= 12) return Array.from({ length: last + 1 }, (_, k) => k);
    if (window.innerWidth < 620) {
      const ticks = new Set([0, Math.min(8, last), last]);
      const step = last <= 24 ? 5 : 10;
      for (let k = step; k < last; k += step) {
        if (Math.abs(k - 8) >= 4 && last - k >= 4) ticks.add(k);
      }
      return [...ticks].sort((a, b) => a - b);
    }
    const step = last <= 24 ? 2 : 5;
    const ticks = new Set([0, Math.min(8, last), last]);
    for (let k = step; k < last; k += step) {
      if (last - k >= Math.max(2, step * 0.8)) ticks.add(k);
    }
    return [...ticks].sort((a, b) => a - b);
  }

  function rebuildTimeline() {
    const last = maxIndex();
    $("route-range").max = last;
    $("nodes").innerHTML = timelineTicks(last).map((k) => {
      const edge = k === 0 ? " edge-start" : k === last ? " edge-end" : "";
      const left = last ? (k / last) * 100 : 0;
      return `<button class="node${edge}" style="left:${left}%" type="button" data-k="${k}">k${k}</button>`;
    }).join("");
  }

  function rebuildRouteSelect() {
    const scene = currentScene();
    const select = $("route-select");
    select.replaceChildren();
    const all = document.createElement("option");
    all.value = "all";
    all.textContent = `全部 32 条 · 最长 ${scene.max_chunks} chunks`;
    select.appendChild(all);
    const routes = [...scene.routes].sort((a, b) => b.chunks - a.chunks || Number(a.success) - Number(b.success) || a.episode - b.episode);
    routes.forEach((route) => {
      const option = document.createElement("option");
      option.value = String(route.episode);
      option.textContent = `${route.success ? "成功" : "失败"} · ep ${route.episode} · ${route.chunks} chunks · seed ${route.seed}`;
      select.appendChild(option);
    });
    state.focus = "all";
    select.value = state.focus;
  }

  function rebuildSceneSelect() {
    const task = currentTask();
    const select = $("scene-select");
    select.replaceChildren();
    task.scenes.forEach((scene) => {
      const option = document.createElement("option");
      option.value = String(scene.scene);
      option.textContent = `state ${scene.scene} · S${scene.success}/F${scene.failure}`;
      select.appendChild(option);
    });
    select.value = String(state.scene);
    rebuildRouteSelect();
    rebuildTimeline();
  }

  function setPlaying(playing) {
    state.playing = playing;
    $("play").innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
    $("play").title = playing ? "暂停" : "播放";
    $("play").setAttribute("aria-label", playing ? "暂停" : "播放");
  }

  function replay() {
    state.progress = 0;
    state.startHold = 0.8;
    state.endHold = 0;
    setPlaying(true);
  }

  function seek(progress) {
    state.progress = clamp(progress, 0, maxIndex());
    state.startHold = 0;
    state.endHold = 0;
    setPlaying(false);
  }

  function animate(now) {
    const elapsed = Math.min(0.08, Math.max(0, (now - state.lastFrame) / 1000));
    state.lastFrame = now;
    if (state.playing) {
      if (state.startHold > 0) {
        state.startHold -= elapsed * state.speed;
      } else if (state.progress < maxIndex()) {
        const chunksPerSecond = Math.max(0.65, maxIndex() / 24);
        state.progress = Math.min(maxIndex(), state.progress + elapsed * state.speed * chunksPerSecond);
      } else {
        state.endHold += elapsed * state.speed;
        if (state.endHold >= 2.2) replay();
      }
    }
    drawScene();
    window.requestAnimationFrame(animate);
  }

  function nearestMarker(clientX, clientY) {
    const rect = $("route-canvas").getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let best = null;
    let distance = 14;
    state.markers.forEach((marker) => {
      const current = Math.hypot(marker.x - x, marker.y - y);
      if (current < distance || (current === distance && marker.highlighted)) {
        best = marker;
        distance = current;
      }
    });
    return best;
  }

  function showTooltip(event) {
    const marker = nearestMarker(event.clientX, event.clientY);
    const tooltip = $("route-tooltip");
    if (!marker) {
      tooltip.classList.remove("show");
      return;
    }
    const outcome = marker.route.success ? "成功" : "失败";
    const score = marker.score === null ? "当前无成功 risk-set 对照" : `HB routing percentile ${marker.score.toFixed(2)}`;
    tooltip.innerHTML = `<strong class="${marker.route.success ? "ok" : "bad"}">${outcome} rollout${marker.ended ? " · 已到记录终点" : ""}</strong>episode ${marker.route.episode} · seed ${marker.route.seed}<br>${marker.route.chunks} chunks · ${marker.route.action_steps} actions<br>${score}`;
    const stage = $("stage").getBoundingClientRect();
    tooltip.style.left = `${event.clientX - stage.left}px`;
    tooltip.style.top = `${event.clientY - stage.top}px`;
    tooltip.classList.add("show");
  }

  function initialize() {
    const taskSelect = $("task-select");
    ATLAS.tasks.forEach((task) => {
      const option = document.createElement("option");
      option.value = task.task;
      option.textContent = `${TASK_NAMES[task.task]} · ${task.min_chunks}-${task.max_chunks} chunks`;
      taskSelect.appendChild(option);
    });
    taskSelect.value = state.task;
    rebuildSceneSelect();

    taskSelect.addEventListener("change", () => {
      state.task = taskSelect.value;
      state.scene = currentTask().default_scene;
      rebuildSceneSelect();
      replay();
    });
    $("scene-select").addEventListener("change", (event) => {
      state.scene = Number(event.target.value);
      rebuildRouteSelect();
      rebuildTimeline();
      replay();
    });
    $("route-select").addEventListener("change", (event) => {
      state.focus = event.target.value;
      rebuildTimeline();
      replay();
    });
    $("play").addEventListener("click", () => setPlaying(!state.playing));
    $("replay").addEventListener("click", replay);
    $("speed-control").addEventListener("click", (event) => {
      const button = event.target.closest("[data-speed]");
      if (!button) return;
      state.speed = Number(button.dataset.speed);
      document.querySelectorAll("[data-speed]").forEach((item) => item.classList.toggle("active", item === button));
    });
    $("route-range").addEventListener("input", (event) => seek(Number(event.target.value)));
    $("nodes").addEventListener("click", (event) => {
      const button = event.target.closest("[data-k]");
      if (button) seek(Number(button.dataset.k));
    });
    $("route-canvas").addEventListener("pointermove", showTooltip);
    $("route-canvas").addEventListener("pointerleave", () => $("route-tooltip").classList.remove("show"));
    window.addEventListener("keydown", (event) => {
      const blocked = ["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName);
      if (event.code === "Space" && !blocked) {
        event.preventDefault();
        setPlaying(!state.playing);
      } else if (event.code === "ArrowLeft" && !blocked) {
        seek(Math.floor(state.progress) - 1);
      } else if (event.code === "ArrowRight" && !blocked) {
        seek(Math.floor(state.progress) + 1);
      }
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) setPlaying(false);
    });
    setPlaying(state.playing);
    updateNarrative();
    window.requestAnimationFrame(animate);
  }

  initialize();
})();
