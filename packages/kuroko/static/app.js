let sessionId = null;
let portNames = [];
let lastGradientPayloads = [];

function getPolarizationMode() {
  return document.getElementById("polarizationMode").value || "cross";
}

function getTerminalPose() {
  const mode = document.getElementById("terminalPoseMode").value || "horizontal_scan";
  const angleInput = document.getElementById("terminalPoseAngle");
  return { mode, angleDeg: angleInput.value };
}

function validateTerminalPose(terminalPose) {
  if (terminalPose.mode === "fixed_angle" && terminalPose.angleDeg.trim() === "") {
    alert("请输入终端固定角度");
    return false;
  }
  return true;
}

function updateTerminalPoseAngleState() {
  const isFixedAngle = document.getElementById("terminalPoseMode").value === "fixed_angle";
  document.getElementById("terminalPoseAngle").disabled = !isFixedAngle;
}

function gridValue(grid, row, col) {
  const value = grid?.[row]?.[col];
  return Number.isFinite(value) ? value : NaN;
}

function finiteRange(grid) {
  const values = [];
  for (const row of grid || []) {
    for (const value of row || []) {
      if (Number.isFinite(value)) values.push(value);
    }
  }
  if (!values.length) return { min: 0, max: 1 };
  return { min: Math.min(...values), max: Math.max(...values) };
}

function radiusFromGainDb(gainDb, range) {
  if (!Number.isFinite(gainDb)) return 0;
  const floorDb = range.max - 40;
  const clamped = Math.max(gainDb, floorDb);
  return 0.04 + 0.96 * Math.pow(10, (clamped - range.max) / 20);
}

function buildSurface(payload, colorField) {
  const theta = payload.theta || [];
  const phi = payload.phi || [];
  const colorGrid = payload[colorField] || payload.z;
  const gainGrid = payload.gain_db || payload.z;
  const gainRange = finiteRange(gainGrid);
  const closeSeam = phi.length > 1 && Math.abs((phi[phi.length - 1] - phi[0]) % 360) > 1e-9;
  const columnCount = phi.length + (closeSeam ? 1 : 0);
  const x = [];
  const y = [];
  const z = [];
  const surfacecolor = [];
  const customdata = [];

  for (let ti = 0; ti < theta.length; ti += 1) {
    const xRow = [];
    const yRow = [];
    const zRow = [];
    const colorRow = [];
    const dataRow = [];
    const thetaRad = theta[ti] * Math.PI / 180;

    for (let pi = 0; pi < columnCount; pi += 1) {
      const sourcePi = pi === phi.length ? 0 : pi;
      const phiDeg = pi === phi.length ? phi[0] + 360 : phi[sourcePi];
      const phiRad = phiDeg * Math.PI / 180;
      const gainDb = gridValue(gainGrid, ti, sourcePi);
      const radius = radiusFromGainDb(gainDb, gainRange);
      const sinTheta = Math.sin(thetaRad);
      const color = gridValue(colorGrid, ti, sourcePi);

      xRow.push(radius * sinTheta * Math.cos(phiRad));
      yRow.push(radius * sinTheta * Math.sin(phiRad));
      zRow.push(radius * Math.cos(thetaRad));
      colorRow.push(color);
      dataRow.push([phiDeg % 360, theta[ti], gainDb, color]);
    }

    x.push(xRow);
    y.push(yRow);
    z.push(zRow);
    surfacecolor.push(colorRow);
    customdata.push(dataRow);
  }

  return { x, y, z, surfacecolor, customdata };
}

function plotElementId(portIndex, view) {
  return `plot-port-${portIndex}-${view}`;
}

function clearPatternGrid(message = "方向图尚未上传") {
  document.querySelectorAll(".patternPlot").forEach((plot) => Plotly.purge(plot));
  document.getElementById("patternGrid").innerHTML = `<div class="panel emptyPlot">${message}</div>`;
}

function ensurePatternPanels(payloads) {
  document.querySelectorAll(".patternPlot").forEach((plot) => Plotly.purge(plot));
  const grid = document.getElementById("patternGrid");
  grid.innerHTML = "";
  payloads.forEach((payload, idx) => {
    const portIndex = payload.port_index ?? idx;
    const pair = document.createElement("div");
    pair.className = "patternPair";

    const plot3d = document.createElement("div");
    plot3d.id = plotElementId(portIndex, "3d");
    plot3d.className = "panel patternPlot patternPlot3d";
    pair.appendChild(plot3d);

    const plot2d = document.createElement("div");
    plot2d.id = plotElementId(portIndex, "2d");
    plot2d.className = "panel patternPlot patternPlot2d";
    pair.appendChild(plot2d);

    grid.appendChild(pair);
  });
}

function renderPatternSet(payloads, title, colorField = "z") {
  const visiblePayloads = payloads || [];
  if (!visiblePayloads.length) {
    clearPatternGrid("方向图尚未上传");
    return;
  }
  ensurePatternPanels(visiblePayloads);
  visiblePayloads.forEach((payload, idx) => {
    const portIndex = payload.port_index ?? idx;
    const portName = portNames[portIndex] || `port ${portIndex}`;
    const baseTitle = `${portIndex}: ${portName} - ${title}`;
    renderPattern3d(payload, `${baseTitle} (3D)`, colorField, plotElementId(portIndex, "3d"), title);
    renderPattern2d(payload, `${baseTitle} (2D 展开)`, colorField, plotElementId(portIndex, "2d"), title);
  });
}

function renderPattern3d(payload, title, colorField = "z", targetId = "plot", colorTitle = title) {
  const surface = buildSurface(payload, colorField);
  const colorRange = finiteRange(surface.surfacecolor);
  const rowCount = surface.x.length;
  const columnCount = surface.x[0]?.length || 0;
  const x = [];
  const y = [];
  const z = [];
  const intensity = [];
  const customdata = [];
  const hovertext = [];
  const i = [];
  const j = [];
  const k = [];

  for (let ti = 0; ti < rowCount; ti += 1) {
    for (let pi = 0; pi < columnCount; pi += 1) {
      const [phiDeg, thetaDeg, gainDb, color] = surface.customdata[ti][pi];
      const gainText = Number.isFinite(gainDb) ? Number.parseFloat(gainDb.toPrecision(3)).toString() : "NaN";
      const colorText = Number.isFinite(color) ? Number.parseFloat(color.toPrecision(4)).toString() : "NaN";

      x.push(surface.x[ti][pi]);
      y.push(surface.y[ti][pi]);
      z.push(surface.z[ti][pi]);
      intensity.push(Number.isFinite(color) ? color : colorRange.min);
      customdata.push(surface.customdata[ti][pi]);
      hovertext.push(
        `phi=${phiDeg.toFixed(1)}°<br>` +
        `theta=${thetaDeg.toFixed(1)}°<br>` +
        `gain=${gainText} dB<br>` +
        `${colorTitle}=${colorText}`
      );
    }
  }

  for (let ti = 0; ti < rowCount - 1; ti += 1) {
    for (let pi = 0; pi < columnCount - 1; pi += 1) {
      const a = ti * columnCount + pi;
      const b = a + 1;
      const c = (ti + 1) * columnCount + pi;
      const d = c + 1;
      i.push(a, b);
      j.push(c, c);
      k.push(b, d);
    }
  }

  const trace = {
    type: "mesh3d",
    x,
    y,
    z,
    i,
    j,
    k,
    intensity,
    intensitymode: "vertex",
    customdata,
    text: hovertext,
    colorscale: "Turbo",
    showscale: true,
    cmin: colorRange.min,
    cmax: colorRange.max,
    colorbar: { title: colorTitle },
    hovertemplate: "%{text}<extra></extra>",
  };
  const layout = {
    title,
    margin: { t: 40, r: 10, b: 10, l: 10 },
    scene: {
      aspectmode: "data",
      xaxis: { title: "X", showspikes: false },
      yaxis: { title: "Y", showspikes: false },
      zaxis: { title: "Z", showspikes: false },
      camera: { eye: { x: 1.45, y: 1.45, z: 0.9 } },
    },
  };
  Plotly.newPlot(targetId, [trace], layout, { responsive: true, displaylogo: false });
}

function renderPattern2d(payload, title, colorField = "z", targetId = "plot2d", colorTitle = title) {
  const theta = payload.theta || [];
  const phi = payload.phi || [];
  const colorGrid = payload[colorField] || payload.z;
  const gainGrid = payload.gain_db || payload.z;
  const colorRange = finiteRange(colorGrid);
  const hovertext = (colorGrid || []).map((row, ti) => row.map((_, pi) => {
    const gainDb = gridValue(gainGrid, ti, pi);
    const color = gridValue(colorGrid, ti, pi);
    const gainText = Number.isFinite(gainDb) ? Number.parseFloat(gainDb.toPrecision(3)).toString() : "NaN";
    const colorText = Number.isFinite(color) ? Number.parseFloat(color.toPrecision(4)).toString() : "NaN";
    return (
      `phi=${Number(phi[pi] ?? NaN).toFixed(1)}°<br>` +
      `theta=${Number(theta[ti] ?? NaN).toFixed(1)}°<br>` +
      `gain=${gainText} dB<br>` +
      `${colorTitle}=${colorText}`
    );
  }));
  const trace = {
    type: "heatmap",
    x: phi,
    y: theta,
    z: colorGrid,
    text: hovertext,
    colorscale: "Turbo",
    zmin: colorRange.min,
    zmax: colorRange.max,
    colorbar: { title: colorTitle },
    hovertemplate: "%{text}<extra></extra>",
  };
  const traces = [trace];
  const clusters = (payload.channel_clusters || []).filter((point) => (
    Number.isFinite(point?.local_phi) && Number.isFinite(point?.theta)
  ));
  if (clusters.length) {
    const energies = clusters.map((point) => Number.isFinite(point.energy) ? Math.max(point.energy, 0) : 0);
    const energyRange = {
      min: Math.min(...energies),
      max: Math.max(...energies),
    };
    const markerSizes = energies.map((energy) => {
      const normalized = energyRange.max > energyRange.min
        ? (energy - energyRange.min) / (energyRange.max - energyRange.min)
        : 0.5;
      const radius = 5 + 13 * Math.sqrt(Math.max(normalized, 0));
      return 2 * radius;
    });
    traces.push({
      type: "scatter",
      mode: "markers",
      name: "Channel clusters",
      x: clusters.map((point) => point.local_phi),
      y: clusters.map((point) => point.theta),
      text: clusters.map((point) => {
        const phiText = Number.isFinite(point.phi) ? point.phi.toFixed(1) : "NaN";
        return (
          `${point.label || "cluster"}<br>` +
          `channel phi=${phiText}°<br>` +
          `local phi=${point.local_phi.toFixed(1)}°<br>` +
          `theta=${point.theta.toFixed(1)}°<br>` +
          `energy=${Number.isFinite(point.energy) ? point.energy.toPrecision(4) : "NaN"}`
        );
      }),
      marker: {
        color: "rgba(17, 24, 39, 0.45)",
        line: { color: "rgba(255, 255, 255, 0.9)", width: 1 },
        size: markerSizes,
        symbol: "circle",
      },
      hovertemplate: "%{text}<extra></extra>",
    });
  }
  const layout = {
    title,
    margin: { t: 40, r: 10, b: 52, l: 60 },
    xaxis: { title: "Phi (deg)", range: [0, 360], zeroline: false },
    yaxis: { title: "Theta (deg)", autorange: "reversed", zeroline: false },
  };
  Plotly.newPlot(targetId, traces, layout, { responsive: true, displaylogo: false });
}

function renderMiHistogram(miValues) {
  const histogram = document.getElementById("miHistogram");
  const values = (miValues || []).filter((value) => Number.isFinite(value));
  if (!values.length) {
    Plotly.purge("miHistogram");
    histogram.textContent = "MI 直方图尚未计算";
    return;
  }
  histogram.textContent = "";
  const trace = {
    type: "histogram",
    x: values,
    marker: { color: "#2563eb", line: { color: "#1e3a8a", width: 1 } },
    hovertemplate: "MI=%{x:.4g}<br>数量=%{y}<extra></extra>",
  };
  const layout = {
    title: "MI 直方图",
    margin: { t: 44, r: 16, b: 52, l: 52 },
    bargap: 0.06,
    xaxis: { title: "MI", zeroline: false },
    yaxis: { title: "数量", rangemode: "tozero" },
  };
  Plotly.newPlot("miHistogram", [trace], layout, { responsive: true, displaylogo: false });
}

async function uploadFiles() {
  const ffs = document.getElementById("ffsFiles").files;
  const channel = document.getElementById("channelFile").files[0];
  if (!ffs.length || !channel) {
    alert("请先选择 FFS 文件和 channel JSON");
    return;
  }
  const terminalPose = getTerminalPose();
  if (!validateTerminalPose(terminalPose)) return;

  const form = new FormData();
  for (const f of ffs) form.append("ffs_files", f);
  form.append("channel_file", channel);
  form.append("polarization_mode", getPolarizationMode());
  form.append("terminal_pose_mode", terminalPose.mode);
  if (terminalPose.mode === "fixed_angle") {
    form.append("terminal_pose_angle_deg", terminalPose.angleDeg);
  }

  const res = await fetch("/api/upload", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "上传失败");
    return;
  }

  sessionId = data.session_id;
  portNames = data.ports || [];
  lastGradientPayloads = [];
  renderMiHistogram([]);
  renderPatternSet(data.initial_heatmaps || [data.initial_heatmap], "方向图增益 (dB)");
  document.getElementById("statsBox").textContent =
    `channel: ${data.channel_name}\ntype: ${data.channel_type}\nport count: ${data.n_ports}`;
}

async function loadGainPatterns() {
  if (!sessionId) {
    alert("请先上传文件");
    return;
  }
  const payloads = [];
  const terminalPose = getTerminalPose();
  if (!validateTerminalPose(terminalPose)) return;
  for (let portIndex = 0; portIndex < portNames.length; portIndex += 1) {
    const params = new URLSearchParams({
      port_index: portIndex,
      polarization_mode: getPolarizationMode(),
      terminal_pose_mode: terminalPose.mode,
    });
    if (terminalPose.mode === "fixed_angle") {
      params.set("terminal_pose_angle_deg", terminalPose.angleDeg);
    }
    const res = await fetch(`/api/heatmap/${sessionId}?${params}`);
    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || "获取方向图失败");
      return;
    }
    payloads.push(data);
  }
  renderPatternSet(payloads, "方向图增益 (dB)");
}

async function computeGradient() {
  if (!sessionId) {
    alert("请先上传文件");
    return;
  }
  const terminalPose = getTerminalPose();
  if (!validateTerminalPose(terminalPose)) return;
  const payloads = [];
  document.getElementById("statsBox").textContent = "正在计算所有端口的 MI 梯度...";
  for (let portIndex = 0; portIndex < portNames.length; portIndex += 1) {
    const form = new FormData();
    form.append("port_index", portIndex);
    form.append("snr_db", document.getElementById("snrDb").value);
    form.append("num_snapshots", document.getElementById("numSnapshots").value);
    form.append("polarization_mode", getPolarizationMode());
    form.append("terminal_pose_mode", terminalPose.mode);
    if (terminalPose.mode === "fixed_angle") {
      form.append("terminal_pose_angle_deg", terminalPose.angleDeg);
    }

    const res = await fetch(`/api/gradient/${sessionId}`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || "计算梯度失败");
      return;
    }
    payloads.push(data);
  }

  lastGradientPayloads = payloads;
  document.getElementById("viewMode").value = "gradient";
  applyGradientView("gradient");
  renderMiHistogram(payloads[0]?.mi_values);
  document.getElementById("statsBox").textContent = JSON.stringify({
    ports: payloads.map((data) => ({
      port: data.port_index,
      name: portNames[data.port_index],
      rotations: data.rotation_count,
      terminal_pose: data.terminal_pose_mode === "fixed_angle" ? data.terminal_pose_angle_deg : "horizontal_scan",
      ...data.stats,
    })),
  }, null, 2);
}

function applyGradientView(viewMode) {
  if (viewMode === "gain") {
    loadGainPatterns();
    return;
  }
  if (!lastGradientPayloads.length) {
    alert("请先点击“计算 MI 梯度并着色”");
    return;
  }
  if (viewMode === "gradient") {
    renderPatternSet(lastGradientPayloads, "|dMI/dgain|", "z");
  } else if (viewMode === "gradlog") {
    renderPatternSet(lastGradientPayloads, "|dMI/dlog(gain)|", "grad_log_abs");
  } else if (viewMode === "mi_max") {
    renderPatternSet(lastGradientPayloads, "MI 最大值", "mi_max");
  } else if (viewMode === "mi_min") {
    renderPatternSet(lastGradientPayloads, "MI 最小值", "mi_min");
  }
}

async function withDisabledButton(buttonId, action) {
  const button = document.getElementById(buttonId);
  button.disabled = true;
  try {
    await action();
  } finally {
    button.disabled = false;
  }
}

document.getElementById("uploadBtn").addEventListener("click", () => withDisabledButton("uploadBtn", uploadFiles));
document.getElementById("refreshBtn").addEventListener("click", () => {
  const mode = document.getElementById("viewMode").value;
  if (mode === "gain") loadGainPatterns();
  else applyGradientView(mode);
});
document.getElementById("gradBtn").addEventListener("click", () => withDisabledButton("gradBtn", computeGradient));
document.getElementById("viewMode").addEventListener("change", (e) => applyGradientView(e.target.value));
document.getElementById("terminalPoseMode").addEventListener("change", updateTerminalPoseAngleState);
document.getElementById("polarizationMode").addEventListener("change", () => {
  lastGradientPayloads = [];
  renderMiHistogram([]);
  const mode = document.getElementById("viewMode").value;
  if (mode === "gain") loadGainPatterns();
});
updateTerminalPoseAngleState();
