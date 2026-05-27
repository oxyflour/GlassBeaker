let sessionId = null;
let currentPort = 0;
let lastGradientPayload = null;

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

function renderPattern3d(payload, title, colorField = "z") {
  const surface = buildSurface(payload, colorField);
  const colorRange = finiteRange(surface.surfacecolor);
  const trace = {
    type: "surface",
    x: surface.x,
    y: surface.y,
    z: surface.z,
    surfacecolor: surface.surfacecolor,
    customdata: surface.customdata,
    colorscale: "Turbo",
    cmin: colorRange.min,
    cmax: colorRange.max,
    colorbar: { title },
    contours: {
      x: { show: false },
      y: { show: false },
      z: { show: false },
    },
    hovertemplate:
      "phi=%{customdata[0]:.1f}°<br>" +
      "theta=%{customdata[1]:.1f}°<br>" +
      "gain=%{customdata[2]:.3g} dB<br>" +
      `${title}=%{customdata[3]:.4g}<extra></extra>`,
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
  Plotly.newPlot("plot", [trace], layout, { responsive: true, displaylogo: false });
}

async function uploadFiles() {
  const ffs = document.getElementById("ffsFiles").files;
  const channel = document.getElementById("channelFile").files[0];
  if (!ffs.length || !channel) {
    alert("请先选择 FFS 文件和 channel JSON");
    return;
  }

  const form = new FormData();
  for (const f of ffs) form.append("ffs_files", f);
  form.append("channel_file", channel);

  const res = await fetch("/api/upload", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "上传失败");
    return;
  }

  sessionId = data.session_id;
  currentPort = 0;
  lastGradientPayload = null;
  const portSelect = document.getElementById("portSelect");
  portSelect.innerHTML = "";
  data.ports.forEach((name, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `${idx}: ${name}`;
    portSelect.appendChild(opt);
  });
  renderPattern3d(data.initial_heatmap, "方向图增益 (dB)");
  document.getElementById("statsBox").textContent =
    `channel: ${data.channel_name}\ntype: ${data.channel_type}\nport count: ${data.n_ports}`;
}

async function loadGainPattern() {
  if (!sessionId) {
    alert("请先上传文件");
    return;
  }
  currentPort = parseInt(document.getElementById("portSelect").value || "0", 10);
  const res = await fetch(`/api/heatmap/${sessionId}?port_index=${currentPort}`);
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "获取方向图失败");
    return;
  }
  renderPattern3d(data, "方向图增益 (dB)");
}

async function computeGradient() {
  if (!sessionId) {
    alert("请先上传文件");
    return;
  }
  currentPort = parseInt(document.getElementById("portSelect").value || "0", 10);
  const form = new FormData();
  form.append("port_index", currentPort);
  form.append("snr_db", document.getElementById("snrDb").value);
  form.append("num_snapshots", document.getElementById("numSnapshots").value);

  const res = await fetch(`/api/gradient/${sessionId}`, { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "计算梯度失败");
    return;
  }

  lastGradientPayload = data;
  document.getElementById("viewMode").value = "gradient";
  applyGradientView("gradient");
  document.getElementById("statsBox").textContent = JSON.stringify({
    ...data.stats,
    rotations: data.rotation_count,
  }, null, 2);
}

function applyGradientView(viewMode) {
  if (viewMode === "gain") {
    loadGainPattern();
    return;
  }
  if (!lastGradientPayload) {
    alert("请先点击“计算 MI 梯度并着色”");
    return;
  }
  if (viewMode === "gradient") {
    renderPattern3d(lastGradientPayload, "|dMI/dgain|", "z");
  } else if (viewMode === "gradlog") {
    renderPattern3d(lastGradientPayload, "|dMI/dlog(gain)|", "grad_log_abs");
  }
}

document.getElementById("uploadBtn").addEventListener("click", uploadFiles);
document.getElementById("refreshBtn").addEventListener("click", () => {
  const mode = document.getElementById("viewMode").value;
  if (mode === "gain") loadGainPattern();
  else applyGradientView(mode);
});
document.getElementById("gradBtn").addEventListener("click", computeGradient);
document.getElementById("viewMode").addEventListener("change", (e) => applyGradientView(e.target.value));
document.getElementById("portSelect").addEventListener("change", () => {
  lastGradientPayload = null;
  const mode = document.getElementById("viewMode").value;
  if (mode === "gain") loadGainPattern();
});
