(() => {
  "use strict";

  const DATA = window.HIMOE_ROLLOUT_DEMO;
  const ATLAS = DATA && DATA.trajectory_routes;
  if (!ATLAS || ATLAS.version < 3) {
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
    blue: "#62a9ff",
    grid: "#27302d"
  };
  const METRICS = {
    churn: {
      field: "routing_churn_percentile",
      label: "within-chunk routing churn percentile",
      short: "churn gap",
      symbol: "Cdenoise"
    },
    revision: {
      field: "routing_revision_percentile",
      label: "d0-to-d9 routing revision percentile",
      short: "d0→d9 gap",
      symbol: "V0→9"
    }
  };

  const initialTask = ATLAS.tasks.find((task) => task.task === DEFAULT_TASK) || ATLAS.tasks[0];
  const state = {
    task: initialTask.task,
    scene: initialTask.default_scene,
    focus: "all",
    metric: "churn",
    progress: 0,
    playing: !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
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
    const selected = selectedRoute();
    return Math.max(0, (selected ? selected.chunks : currentScene().max_chunks) - 1);
  }

  function focusEpisodes(scene) {
    if (state.focus !== "all") return new Set([Number(state.focus)]);
    return new Set(Object.values(scene.highlighted));
  }

  function metricValue(route, chunk, metric = state.metric) {
    const values = route[METRICS[metric].field];
    return chunk >= 0 && chunk < values.length ? values[chunk] : null;
  }

  function computationPoint(route, progress) {
    const last = route.chunks - 1;
    if (progress > last + 1e-6) return null;
    const low = Math.min(last, Math.max(0, Math.floor(progress)));
    const high = Math.min(last, low + 1);
    const amount = smooth(progress - low);
    const x0 = route.routing_percentile[low];
    const x1 = route.routing_percentile[high];
    const y0 = metricValue(route, low);
    const y1 = metricValue(route, high);
    const x = x0 === null || x1 === null ? null : mix(x0, x1, amount);
    const y = y0 === null || y1 === null ? null : mix(y0, y1, amount);
    return y === null ? null : { x, y };
  }

  function activeRoutes(scene, chunk) {
    return scene.routes.filter((route) => route.chunks > chunk);
  }

  function mean(values) {
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  }

  function quantile(values, q) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const position = (sorted.length - 1) * q;
    const low = Math.floor(position);
    const high = Math.ceil(position);
    return mix(sorted[low], sorted[high], position - low);
  }

  function groupGap(scene, chunk, accessor) {
    const active = activeRoutes(scene, chunk);
    const success = active.filter((route) => route.success).map(accessor).filter(Number.isFinite);
    const failure = active.filter((route) => !route.success).map(accessor).filter(Number.isFinite);
    if (!success.length || !failure.length) return null;
    return mean(failure) - mean(success);
  }

  function canvasContext(id) {
    const canvas = $(id);
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

  function plotTransform(width, height) {
    const mobile = width < 620;
    const rect = mobile
      ? { left: 34, top: 145, width: width - 88, height: Math.max(220, height - 400) }
      : { left: 62, top: 164, width: Math.max(380, width - 430), height: Math.max(300, height - 320) };
    return {
      rect,
      mobile,
      railX: mobile ? width - 25 : rect.left + rect.width + 34,
      point: (x, y) => [
        rect.left + clamp(x) * rect.width,
        rect.top + (1 - clamp(y)) * rect.height
      ]
    };
  }

  function drawAxes(ctx, transform, scene, chunk) {
    const { rect, mobile } = transform;
    ctx.save();
    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = "#65716c";
    ctx.lineWidth = 1;
    ctx.font = mobile ? "8px Inter, system-ui, sans-serif" : "9px Inter, system-ui, sans-serif";
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = tick / 4;
      const x = rect.left + value * rect.width;
      const y = rect.top + (1 - value) * rect.height;
      ctx.beginPath();
      ctx.moveTo(x, rect.top);
      ctx.lineTo(x, rect.top + rect.height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(rect.left, y);
      ctx.lineTo(rect.left + rect.width, y);
      ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(value.toFixed(2), x, rect.top + rect.height + 16);
      ctx.textAlign = "right";
      ctx.fillText(value.toFixed(2), rect.left - 8, y + 3);
    }
    ctx.strokeStyle = "#45514d";
    ctx.strokeRect(rect.left, rect.top, rect.width, rect.height);
    ctx.fillStyle = "#84908b";
    ctx.textAlign = "right";
    ctx.font = mobile ? "8px Inter, system-ui, sans-serif" : "10px Inter, system-ui, sans-serif";
    ctx.fillText("distance from active-success routing centroid · percentile", rect.left + rect.width, rect.top + rect.height + 31);
    ctx.save();
    ctx.translate(rect.left - (mobile ? 26 : 38), rect.top + rect.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText(METRICS[state.metric].label, 0, 0);
    ctx.restore();

    const successX = activeRoutes(scene, chunk)
      .filter((route) => route.success)
      .map((route) => route.routing_percentile[chunk])
      .filter(Number.isFinite);
    if (successX.length >= 2) {
      const q1 = quantile(successX, 0.25);
      const q3 = quantile(successX, 0.75);
      const left = rect.left + q1 * rect.width;
      const right = rect.left + q3 * rect.width;
      ctx.fillStyle = "rgba(86,214,166,0.08)";
      ctx.fillRect(left, rect.top, Math.max(1, right - left), rect.height);
      ctx.strokeStyle = "rgba(86,214,166,0.35)";
      ctx.setLineDash([4, 5]);
      ctx.beginPath();
      ctx.moveTo(left, rect.top);
      ctx.lineTo(left, rect.top + rect.height);
      ctx.moveTo(right, rect.top);
      ctx.lineTo(right, rect.top + rect.height);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(86,214,166,0.65)";
      ctx.textAlign = "left";
      ctx.fillText("success IQR", left + 4, rect.top + 14);
    }

    ctx.strokeStyle = "#4a5652";
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(transform.railX, rect.top);
    ctx.lineTo(transform.railX, rect.top + rect.height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#6d7974";
    ctx.textAlign = "center";
    ctx.font = mobile ? "7px Inter, system-ui, sans-serif" : "8px Inter, system-ui, sans-serif";
    if (mobile) {
      ctx.save();
      ctx.translate(transform.railX + 5, rect.top + rect.height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("no success reference", 0, 0);
      ctx.restore();
    } else {
      ctx.save();
      ctx.translate(transform.railX + 10, rect.top + rect.height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("safe reference unavailable", 0, 0);
      ctx.restore();
    }
    ctx.restore();
  }

  function pixelPoint(value, transform) {
    if (value.x === null) {
      return [transform.railX, transform.rect.top + (1 - clamp(value.y)) * transform.rect.height];
    }
    return transform.point(value.x, value.y);
  }

  function covarianceEllipse(points) {
    const x = mean(points.map((point) => point[0]));
    const y = mean(points.map((point) => point[1]));
    let xx = 0;
    let yy = 0;
    let xy = 0;
    points.forEach((point) => {
      const dx = point[0] - x;
      const dy = point[1] - y;
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
      x,
      y,
      rx: 1.7 * Math.sqrt(Math.max(9, (trace + root) / 2)) + 4,
      ry: 1.7 * Math.sqrt(Math.max(9, (trace - root) / 2)) + 4,
      angle
    };
  }

  function drawGroupEllipse(ctx, points, color, label, rect) {
    if (points.length < 3) return;
    const ellipse = covarianceEllipse(points);
    ellipse.rx = Math.min(ellipse.rx, rect.width * 0.24);
    ellipse.ry = Math.min(ellipse.ry, rect.height * 0.24);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.52;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.ellipse(ellipse.x, ellipse.y, ellipse.rx, ellipse.ry, ellipse.angle, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "9px Inter, system-ui, sans-serif";
    ctx.textAlign = ellipse.x > rect.left + rect.width * 0.72 ? "right" : "left";
    ctx.fillText(label, ellipse.x + (ellipse.x > rect.left + rect.width * 0.72 ? -ellipse.rx - 5 : ellipse.rx + 5), ellipse.y - 2);
    ctx.restore();
  }

  function drawMeanArrow(ctx, success, failure) {
    if (!success.length || !failure.length) return;
    const from = [mean(success.map((point) => point[0])), mean(success.map((point) => point[1]))];
    const to = [mean(failure.map((point) => point[0])), mean(failure.map((point) => point[1]))];
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    const length = Math.hypot(dx, dy);
    if (length < 12) return;
    const ux = dx / length;
    const uy = dy / length;
    ctx.save();
    ctx.strokeStyle = COLORS.gold;
    ctx.fillStyle = COLORS.gold;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(from[0], from[1]);
    ctx.lineTo(to[0] - ux * 8, to[1] - uy * 8);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(to[0], to[1]);
    ctx.lineTo(to[0] - ux * 9 + uy * 4, to[1] - uy * 9 - ux * 4);
    ctx.lineTo(to[0] - ux * 9 - uy * 4, to[1] - uy * 9 + ux * 4);
    ctx.closePath();
    ctx.fill();
    ctx.font = "700 9px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("failure - success", (from[0] + to[0]) / 2 - uy * 11, (from[1] + to[1]) / 2 + ux * 11);
    ctx.restore();
  }

  function drawTrail(ctx, route, transform) {
    const last = Math.min(route.chunks - 1, Math.floor(state.progress));
    if (last < 0) return;
    const color = route.success ? COLORS.success : COLORS.failure;
    const points = [];
    for (let chunk = 0; chunk <= last; chunk += 1) {
      const value = {
        x: route.routing_percentile[chunk],
        y: metricValue(route, chunk)
      };
      if (value.y === null) continue;
      points.push({ chunk, point: pixelPoint(value, transform), missing: value.x === null });
    }
    if (!points.length) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.17;
    ctx.lineWidth = 1.1;
    ctx.lineJoin = "round";
    ctx.beginPath();
    points.forEach(({ point }, axis) => {
      if (axis === 0) ctx.moveTo(point[0], point[1]);
      else ctx.lineTo(point[0], point[1]);
    });
    ctx.stroke();
    const recent = points.slice(-9);
    ctx.globalAlpha = 0.84;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    recent.forEach(({ point }, axis) => {
      if (axis === 0) ctx.moveTo(point[0], point[1]);
      else ctx.lineTo(point[0], point[1]);
    });
    ctx.stroke();
    points.forEach(({ chunk, point, missing }) => {
      const recentPoint = chunk >= Math.max(0, last - 8);
      const major = chunk === 0 || chunk === 8 || chunk === last || (recentPoint && chunk % 5 === 0);
      ctx.beginPath();
      ctx.arc(point[0], point[1], major ? 3.3 : 1.6, 0, Math.PI * 2);
      ctx.fillStyle = missing ? "#6f7b76" : color;
      ctx.globalAlpha = major ? 0.95 : recentPoint ? 0.42 : 0.12;
      ctx.fill();
      if (major) {
        ctx.fillStyle = "#9da9a3";
        ctx.globalAlpha = recentPoint || chunk === 0 || chunk === 8 ? 0.8 : 0.32;
        ctx.font = "8px Inter, system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`k${chunk}`, point[0] + 5, point[1] - 5);
      }
    });
    ctx.restore();
  }

  function drawComputationScene() {
    const { canvas, ctx, width, height } = canvasContext("state-canvas");
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#101315";
    ctx.fillRect(0, 0, width, height);
    const scene = currentScene();
    const transform = plotTransform(width, height);
    const chunk = Math.max(0, Math.min(maxIndex(), Math.floor(state.progress + 0.35)));
    const focus = focusEpisodes(scene);
    drawAxes(ctx, transform, scene, chunk);

    scene.routes.filter((route) => focus.has(route.episode)).forEach((route) => drawTrail(ctx, route, transform));

    const markers = [];
    scene.routes.forEach((route) => {
      const value = computationPoint(route, state.progress);
      if (!value) return;
      const point = pixelPoint(value, transform);
      markers.push({ route, value, point, highlighted: focus.has(route.episode) });
    });
    const validSuccess = markers.filter((marker) => marker.route.success && marker.value.x !== null).map((marker) => marker.point);
    const validFailure = markers.filter((marker) => !marker.route.success && marker.value.x !== null).map((marker) => marker.point);
    drawGroupEllipse(ctx, validSuccess, COLORS.success, "active success", transform.rect);
    drawGroupEllipse(ctx, validFailure, COLORS.failure, "active failure", transform.rect);
    drawMeanArrow(ctx, validSuccess, validFailure);

    markers.sort((a, b) => Number(a.highlighted) - Number(b.highlighted));
    state.markers = [];
    markers.forEach((marker) => {
      const color = marker.route.success ? COLORS.success : COLORS.failure;
      if (marker.highlighted) {
        ctx.beginPath();
        ctx.arc(marker.point[0], marker.point[1], 12, 0, Math.PI * 2);
        ctx.strokeStyle = COLORS.gold;
        ctx.globalAlpha = 0.72;
        ctx.lineWidth = 1.7;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(marker.point[0], marker.point[1], marker.highlighted ? 5.5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = marker.highlighted ? 1 : 0.52;
      ctx.fill();
      ctx.globalAlpha = 1;
      state.markers.push({ ...marker, x: marker.point[0], y: marker.point[1] });
    });

    canvas._transform = transform;
    drawDenoiseMicroscope(chunk);
    updateNarrative(chunk);
  }

  function heatColor(value, maximum) {
    if (!Number.isFinite(value) || maximum <= 0) return "#202725";
    const strength = clamp(Math.abs(value) / maximum);
    const target = value >= 0 ? [255, 107, 96] : [86, 214, 166];
    const base = [28, 35, 33];
    const amount = 0.18 + strength * 0.72;
    const rgb = base.map((channel, axis) => Math.round(mix(channel, target[axis], amount)));
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }

  function drawDenoiseMicroscope(chunk) {
    const { ctx, width, height } = canvasContext("denoise-canvas");
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#151a1a";
    ctx.fillRect(0, 0, width, height);
    $("micro-label").textContent = `k${chunk} · failure - success`;
    if (chunk >= ATLAS.fixed_contribution_chunks) {
      ctx.strokeStyle = "#34413c";
      ctx.setLineDash([4, 5]);
      ctx.strokeRect(1, 1, width - 2, height - 2);
      ctx.setLineDash([]);
      ctx.fillStyle = "#7b8882";
      ctx.font = height < 80 ? "9px Inter, system-ui, sans-serif" : "10px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("fixed routed-contribution detail ends at k8", width / 2, height / 2 + 3);
      return;
    }
    const scope = DATA.dynamics.by_scope[state.task];
    const matrix = scope.routed_rms.gap[chunk];
    const all = scope.routed_rms.gap.flat(2).map(Number).filter(Number.isFinite);
    const maximum = Math.max(...all.map(Math.abs), 1e-6);
    const compact = height < 80;
    const margin = { left: compact ? 20 : 30, right: 4, top: compact ? 3 : 16, bottom: compact ? 4 : 15 };
    const cellWidth = (width - margin.left - margin.right) / 10;
    const cellHeight = (height - margin.top - margin.bottom) / 4;
    const layers = DATA.dynamics.layers;
    for (let layer = 0; layer < 4; layer += 1) {
      for (let denoise = 0; denoise < 10; denoise += 1) {
        const value = Number(matrix[denoise][layer]);
        const x = margin.left + denoise * cellWidth;
        const y = margin.top + layer * cellHeight;
        ctx.fillStyle = heatColor(value, maximum);
        ctx.fillRect(x + 1, y + 1, Math.max(1, cellWidth - 2), Math.max(1, cellHeight - 2));
      }
      ctx.fillStyle = "#7d8a84";
      ctx.font = compact ? "7px Inter, system-ui, sans-serif" : "8px Inter, system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(`L${layers[layer]}`, margin.left - 4, margin.top + (layer + 0.62) * cellHeight);
    }
    if (!compact) {
      ctx.fillStyle = "#718079";
      ctx.textAlign = "center";
      ctx.font = "8px Inter, system-ui, sans-serif";
      for (let denoise = 0; denoise < 10; denoise += 1) {
        ctx.fillText(`d${denoise}`, margin.left + (denoise + 0.5) * cellWidth, height - 3);
      }
    }
  }

  function chapterFor(chunk, active) {
    if (chunk === 0) return {
      number: "01",
      label: "SAME OBSERVATION, DIFFERENT COMPUTATION",
      title: "相同起点，MoE 已产生不同内部路线",
      copy: "每个点只由当前第一个 action chunk 的 10 次去噪 HB routing 构成；此时尚未接收执行后的环境反馈。",
      equation: "R<sub>0,0:9</sub> <span class=\"accent\">&#8594;</span> (D<sub>safe</sub>, C<sub>denoise</sub>)"
    };
    if (chunk <= 4) return {
      number: "02",
      label: "FEEDBACK-CONDITIONED COMPUTATION",
      title: "环境改变输入，当前 MoE 状态随闭环迁移",
      copy: "轨迹线连接不同 chunk 的诊断坐标，但每个节点本身仍只使用该 chunk 内部的 routing；前序历史只通过当前 observation 进入。",
      equation: "o<sub>k</sub> &#8594; R<sub>k,0:9</sub> &#8594; A<sub>k</sub>"
    };
    if (chunk <= 8) return {
      number: "03",
      label: "FIXED COHORT TEST",
      title: "固定 cohort：检验失败群体是否离开成功 corridor",
      copy: "横向 gap 为正才表示失败群体距同期成功 routing 质心更远；不同 state、不同 k 可以不分离，纵轴也不预设 churn 越大越危险。",
      equation: "&#916;D<sub>safe</sub>(k), &#916;C<sub>denoise</sub>(k)"
    };
    if (active[0] < 2) return {
      number: "05",
      label: "SAFE REFERENCE LOST",
      title: "成功参考消失，横轴不再伪造风险距离",
      copy: "灰色虚线轨道只保留当前 chunk 的内部 churn/revision；没有至少两条 active success 时，safe-distance 被定义为不可估计。",
      equation: "n<sub>S</sub>(k) &lt; 2 <span class=\"accent\">&#8658;</span> D<sub>safe</sub>(k) undefined"
    };
    return {
      number: "04",
      label: "VARIABLE RISK SET",
      title: "长 rollout 在 MoE 状态空间中继续演化",
      copy: "当前坐标只比较仍在运行的 sibling rollouts。它适合观察 failure state 的形成，但后期坐标不能与固定 k0-k8 效应量直接等同。",
      equation: "h<sub>k</sub> = f(D<sub>safe</sub>, C<sub>denoise</sub>)"
    };
  }

  function updateNarrative(chunk) {
    const scene = currentScene();
    const active = scene.active_counts[Math.min(chunk, scene.active_counts.length - 1)] || [0, 0];
    const chapter = chapterFor(chunk, active);
    $("chapter-number").textContent = chapter.number;
    $("chapter-label").textContent = chapter.label;
    $("chapter-title").textContent = chapter.title;
    $("chapter-copy").textContent = chapter.copy;
    $("equation").innerHTML = chapter.equation;
    $("readout-k").textContent = `k${chunk}`;
    $("active-count").textContent = `${active[0]}S / ${active[1]}F`;
    $("distance-gap").textContent = fmt(scene.routing_gap[chunk], 3, true);
    const dynamicGap = groupGap(scene, chunk, (route) => metricValue(route, chunk));
    $("dynamic-gap").textContent = fmt(dynamicGap, 3, true);
    $("dynamic-gap-label").textContent = METRICS[state.metric].short;
    $("contribution-gap").textContent = chunk < ATLAS.fixed_contribution_chunks
      ? fmt(scene.contribution_gap[chunk], 3, true)
      : "not estimated";

    const focus = selectedRoute();
    let focusText = "两条金色轨迹分别是最长的成功和失败 sibling。";
    if (focus) {
      const x = focus.routing_percentile[chunk];
      const y = metricValue(focus, chunk);
      focusText = focus.chunks > chunk
        ? `焦点 ep ${focus.episode}: Dsafe=${fmt(x, 2)}, ${METRICS[state.metric].symbol}=${fmt(y, 2)}。`
        : `焦点 ep ${focus.episode} 已在 k${focus.chunks - 1} 后终止。`;
    }
    $("reading").textContent = `${chapter.copy} ${focusText} success centroid 使用事后 outcome 标签，当前是诊断坐标。`;
    $("chunk-range").value = state.progress;
    $("time-label").textContent = `k${state.progress.toFixed(2)} / k${maxIndex()}`;
    document.querySelectorAll("[data-k]").forEach((button) => {
      const node = Number(button.dataset.k);
      button.classList.toggle("active", node === chunk);
      button.classList.toggle("past", node < state.progress);
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
    $("chunk-range").max = last;
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
    all.textContent = `全部 active routes · 最长 ${scene.max_chunks} chunks`;
    select.appendChild(all);
    [...scene.routes]
      .sort((a, b) => b.chunks - a.chunks || Number(a.success) - Number(b.success) || a.episode - b.episode)
      .forEach((route) => {
        const option = document.createElement("option");
        option.value = String(route.episode);
        option.textContent = `${route.success ? "成功" : "失败"} · ep ${route.episode} · ${route.chunks} chunks · seed ${route.seed}`;
        select.appendChild(option);
      });
    state.focus = "all";
    select.value = state.focus;
  }

  function rebuildSceneSelect() {
    const select = $("scene-select");
    select.replaceChildren();
    currentTask().scenes.forEach((scene) => {
      const option = document.createElement("option");
      option.value = String(scene.scene);
      option.textContent = `s${scene.scene} · ${scene.success}S/${scene.failure}F`;
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

  function seek(value) {
    state.progress = clamp(value, 0, maxIndex());
    state.startHold = 0;
    state.endHold = 0;
    setPlaying(false);
  }

  function animate(now) {
    const elapsed = Math.min(0.08, Math.max(0, (now - state.lastFrame) / 1000));
    state.lastFrame = now;
    if (state.playing) {
      if (state.startHold > 0) {
        state.startHold -= elapsed;
      } else if (state.progress < maxIndex()) {
        const chunksPerSecond = Math.max(0.65, maxIndex() / 24);
        state.progress = Math.min(maxIndex(), state.progress + elapsed * chunksPerSecond);
      } else {
        state.endHold += elapsed;
        if (state.endHold >= 2.2) replay();
      }
    }
    drawComputationScene();
    window.requestAnimationFrame(animate);
  }

  function nearestMarker(clientX, clientY) {
    const rect = $("state-canvas").getBoundingClientRect();
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
    const tooltip = $("state-tooltip");
    if (!marker) {
      tooltip.classList.remove("show");
      return;
    }
    const outcome = marker.route.success ? "成功" : "失败";
    const x = marker.value.x === null ? "no active-success reference" : `safe-distance percentile ${marker.value.x.toFixed(2)}`;
    tooltip.innerHTML = `<strong class="${marker.route.success ? "ok" : "bad"}">${outcome} rollout</strong>episode ${marker.route.episode} · seed ${marker.route.seed}<br>${x}<br>${METRICS[state.metric].label} ${marker.value.y.toFixed(2)}`;
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
    $("metric-control").addEventListener("click", (event) => {
      const button = event.target.closest("[data-metric]");
      if (!button) return;
      state.metric = button.dataset.metric;
      document.querySelectorAll("[data-metric]").forEach((item) => item.classList.toggle("active", item === button));
    });
    $("play").addEventListener("click", () => setPlaying(!state.playing));
    $("replay").addEventListener("click", replay);
    $("chunk-range").addEventListener("input", (event) => seek(Number(event.target.value)));
    $("nodes").addEventListener("click", (event) => {
      const button = event.target.closest("[data-k]");
      if (button) seek(Number(button.dataset.k));
    });
    $("state-canvas").addEventListener("pointermove", showTooltip);
    $("state-canvas").addEventListener("pointerleave", () => $("state-tooltip").classList.remove("show"));
    window.addEventListener("resize", rebuildTimeline);
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
    drawComputationScene();
    window.requestAnimationFrame(animate);
  }

  initialize();
})();
