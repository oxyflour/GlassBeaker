from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MI Gradient Viewer")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

SESSIONS: dict[str, dict[str, Any]] = {}
POLARIZATION_MODES = {"cross", "vertical", "horizontal"}
TERMINAL_POSE_MODES = {"horizontal_scan", "fixed_angle"}


@dataclass
class FarFieldPattern:
    name: str
    frequency_hz: float | None
    radiated_power: float | None
    accepted_power: float | None
    stimulated_power: float | None
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    etheta: np.ndarray
    ephi: np.ndarray
    theta_unique: np.ndarray
    phi_unique: np.ndarray
    sample_to_grid: np.ndarray  # shape (n_points, 2) -> theta_idx, phi_idx
    grid_to_sample: np.ndarray | None = None

    @property
    def n_points(self) -> int:
        return int(self.theta_deg.size)


@dataclass
class ChannelComponent:
    theta_deg: float
    phi_deg: float
    alpha: complex
    pol: np.ndarray
    at: np.ndarray


def _is_float_token(s: str) -> bool:
    return bool(re.match(r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eEdD][+-]?\d+)?$", s))


def _to_float(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))


def parse_cst_ffs_text(text: str, name: str) -> FarFieldPattern:
    lines = text.splitlines()

    frequency_hz = None
    radiated_power = None
    accepted_power = None
    stimulated_power = None

    # Parse powers and frequency.
    for i, line in enumerate(lines):
        if "Radiated/Accepted/Stimulated Power" in line:
            vals: list[float] = []
            j = i + 1
            while j < len(lines) and len(vals) < 4:
                s = lines[j].strip()
                if s and not s.startswith("//"):
                    parts = s.split()
                    if len(parts) == 1 and _is_float_token(parts[0]):
                        vals.append(_to_float(parts[0]))
                j += 1
            if len(vals) >= 4:
                radiated_power, accepted_power, stimulated_power, frequency_hz = vals[:4]
            break

    start = None
    for i, line in enumerate(lines):
        if "Phi" in line and "Theta" in line and "Re(E_Theta)" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"{name}: cannot find CST FFS data header")

    rows: list[list[float]] = []
    splitter = re.compile(r"[,\s;]+")
    for line in lines[start:]:
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        parts = [p for p in splitter.split(s) if p]
        if len(parts) >= 6 and all(_is_float_token(p) for p in parts[:6]):
            rows.append([_to_float(p) for p in parts[:6]])

    if not rows:
        raise ValueError(f"{name}: no numeric field rows found")

    arr = np.asarray(rows, dtype=float)
    phi = np.mod(arr[:, 0], 360.0)
    theta = arr[:, 1]
    etheta = arr[:, 2] + 1j * arr[:, 3]
    ephi = arr[:, 4] + 1j * arr[:, 5]

    theta_unique = np.unique(theta)
    phi_unique = np.unique(phi)
    theta_index = {v: i for i, v in enumerate(theta_unique.tolist())}
    phi_index = {v: i for i, v in enumerate(phi_unique.tolist())}
    sample_to_grid = np.zeros((len(theta), 2), dtype=int)
    grid_to_sample = np.full((len(theta_unique), len(phi_unique)), -1, dtype=int)
    for i in range(len(theta)):
        ti = theta_index[theta[i]]
        pi = phi_index[phi[i]]
        sample_to_grid[i, 0] = ti
        sample_to_grid[i, 1] = pi
        grid_to_sample[ti, pi] = i

    return FarFieldPattern(
        name=name,
        frequency_hz=frequency_hz,
        radiated_power=radiated_power,
        accepted_power=accepted_power,
        stimulated_power=stimulated_power,
        theta_deg=theta,
        phi_deg=phi,
        etheta=etheta,
        ephi=ephi,
        theta_unique=theta_unique,
        phi_unique=phi_unique,
        sample_to_grid=sample_to_grid,
        grid_to_sample=grid_to_sample,
    )


def angular_distance2(theta_grid: np.ndarray, phi_grid: np.ndarray, theta: float, phi: float) -> np.ndarray:
    phi = phi % 360.0
    dtheta = theta_grid - theta
    dphi = np.abs(phi_grid - phi)
    dphi = np.minimum(dphi, 360.0 - dphi)
    scale = max(math.sin(math.radians(theta)), 1e-3)
    return dtheta * dtheta + (scale * dphi) * (scale * dphi)


def nearest_sample_index(pattern: FarFieldPattern, theta_deg: float, phi_deg: float) -> int:
    if pattern.grid_to_sample is not None:
        theta_idx = int(np.argmin(np.abs(pattern.theta_unique - theta_deg)))
        phi = phi_deg % 360.0
        dphi = np.abs(pattern.phi_unique - phi)
        phi_idx = int(np.argmin(np.minimum(dphi, 360.0 - dphi)))
        idx = int(pattern.grid_to_sample[theta_idx, phi_idx])
        if idx >= 0:
            return idx
    return int(np.argmin(angular_distance2(pattern.theta_deg, pattern.phi_deg, theta_deg, phi_deg % 360.0)))


def validate_polarization_mode(polarization_mode: str) -> str:
    if polarization_mode not in POLARIZATION_MODES:
        raise HTTPException(400, "Invalid polarization mode")
    return polarization_mode


def local_gain_linear(pattern: FarFieldPattern, polarization_mode: str = "cross") -> np.ndarray:
    if polarization_mode == "vertical":
        return np.abs(pattern.etheta)
    if polarization_mode == "horizontal":
        return np.abs(pattern.ephi)
    return np.sqrt(np.abs(pattern.etheta) ** 2 + np.abs(pattern.ephi) ** 2)


def gain_heatmap_payload(pattern: FarFieldPattern, port_index: int, polarization_mode: str) -> dict[str, Any]:
    gain_db = 20.0 * np.log10(np.maximum(local_gain_linear(pattern, polarization_mode), 1e-12))
    return {
        "mode": "gain_db",
        "port_index": port_index,
        "polarization_mode": polarization_mode,
        "theta": pattern.theta_unique.tolist(),
        "phi": pattern.phi_unique.tolist(),
        "z": heatmap_from_samples(pattern, gain_db),
    }


def heatmap_from_samples(pattern: FarFieldPattern, values: np.ndarray) -> list[list[float | None]]:
    z = np.full((len(pattern.theta_unique), len(pattern.phi_unique)), np.nan, dtype=float)
    for i, val in enumerate(values):
        ti, pi = pattern.sample_to_grid[i]
        z[ti, pi] = float(val)
    return [
        [float(value) if math.isfinite(float(value)) else None for value in row]
        for row in z.tolist()
    ]


def cplx(x: Any) -> complex:
    if isinstance(x, (int, float)):
        return complex(float(x), 0.0)
    if isinstance(x, str):
        return complex(x.replace("i", "j"))
    if isinstance(x, list) and len(x) == 2:
        return complex(float(x[0]), float(x[1]))
    if isinstance(x, dict):
        return complex(float(x.get("re", 0.0)), float(x.get("im", 0.0)))
    raise TypeError(f"Cannot parse complex value from {x!r}")


def path_gain(path: dict[str, Any]) -> complex:
    if "gain" in path:
        return cplx(path["gain"])
    if "gain_db" in path:
        mag = 10.0 ** (float(path["gain_db"]) / 20.0)
        phase = math.radians(float(path.get("phase_deg", 0.0)))
        return mag * np.exp(1j * phase)
    return 1.0 + 0.0j


def path_pol(path: dict[str, Any]) -> np.ndarray:
    pol = path.get("pol", [[1.0, 0.0], [0.0, 0.0]])
    p = np.asarray([cplx(pol[0]), cplx(pol[1])], dtype=np.complex128)
    n = np.linalg.norm(p)
    if n > 0:
        p = p / n
    return p


def path_tx_vector(path: dict[str, Any], nt_default: int = 1) -> np.ndarray:
    if "tx_vector" not in path:
        return np.ones((nt_default,), dtype=np.complex128)
    return np.asarray([cplx(v) for v in path["tx_vector"]], dtype=np.complex128)


def parse_channel_model(channel: dict[str, Any]) -> str:
    if isinstance(channel, dict) and ("snapshots" in channel or "paths" in channel):
        return "snapshots"
    if isinstance(channel, dict) and {"delays", "powers", "aoa", "zoa"}.issubset(channel.keys()):
        return "sionna_cdl"
    raise ValueError("Unsupported channel JSON format")


def build_simple_snapshot_data(patterns: list[FarFieldPattern], snapshot: dict[str, Any], nt_default: int = 1) -> dict[str, Any]:
    nr = len(patterns)
    paths = snapshot.get("paths", [])
    nt = nt_default
    for p in paths:
        if "tx_vector" in p:
            nt = len(p["tx_vector"])
            break

    components: list[ChannelComponent] = []

    for p in paths:
        theta = float(p.get("aoa_theta_deg", p.get("theta_deg")))
        phi = float(p.get("aoa_phi_deg", p.get("phi_deg")))
        alpha = path_gain(p)
        pol = path_pol(p)
        delay_s = float(p.get("delay_s", 0.0))
        freq_hz = float(p.get("freq_hz", snapshot.get("freq_hz", 0.0)))
        if delay_s and freq_hz:
            alpha *= np.exp(-1j * 2.0 * np.pi * freq_hz * delay_s)
        at = path_tx_vector(p, nt_default=nt)
        if at.size != nt:
            raise ValueError("Inconsistent tx_vector length across paths")
        components.append(ChannelComponent(theta, phi, alpha, pol, at))

    return {
        "nr": nr,
        "nt": nt,
        "components": components,
    }


def _db_to_linear_power(x: np.ndarray) -> np.ndarray:
    return 10.0 ** (x / 10.0)


def build_cdl_snapshot_data(patterns: list[FarFieldPattern], model: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    nr = len(patterns)
    aoa = np.asarray(model["aoa"], dtype=float)
    zoa = np.asarray(model["zoa"], dtype=float)
    powers = np.asarray(model["powers"], dtype=float)
    if np.min(powers) < 0 or np.max(powers) > 2.0:
        p_lin = _db_to_linear_power(powers)
    else:
        p_lin = powers.copy()
    p_lin = p_lin / max(np.sum(p_lin), 1e-15)

    xpr_db = float(model.get("xpr", 10.0))
    xpr_amp = 10.0 ** (-xpr_db / 20.0)

    components: list[ChannelComponent] = []

    for i in range(len(aoa)):
        theta = float(zoa[i])
        phi = float(aoa[i])
        alpha = math.sqrt(float(p_lin[i])) * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))

        # Simple polarization model from XPR.
        psi = rng.uniform(0.0, 2.0 * np.pi)
        pol = np.asarray([1.0 + 0.0j, xpr_amp * np.exp(1j * psi)], dtype=np.complex128)
        pol = pol / max(np.linalg.norm(pol), 1e-15)
        at = np.ones((1,), dtype=np.complex128)
        components.append(ChannelComponent(theta, phi, alpha, pol, at))

    return {
        "nr": nr,
        "nt": 1,
        "components": components,
    }


def build_realizations(patterns: list[FarFieldPattern], channel_data: dict[str, Any], num_snapshots: int) -> list[dict[str, Any]]:
    mode = parse_channel_model(channel_data)
    if mode == "snapshots":
        if "snapshots" in channel_data:
            snapshots = channel_data["snapshots"]
        else:
            snapshots = [channel_data]
        return [build_simple_snapshot_data(patterns, s) for s in snapshots]

    rng = np.random.default_rng(12345)
    return [build_cdl_snapshot_data(patterns, channel_data, rng) for _ in range(num_snapshots)]


def horizontal_rotation_angles(pattern: FarFieldPattern) -> np.ndarray:
    angles = np.unique(np.mod(pattern.phi_unique, 360.0))
    if angles.size == 0:
        return np.asarray([0.0], dtype=float)
    return angles.astype(float)


def build_rotated_channel_data(
    patterns: list[FarFieldPattern],
    realization: dict[str, Any],
    yaw_deg: float,
    polarization_mode: str = "cross",
) -> dict[str, Any]:
    nr = int(realization["nr"])
    nt = int(realization["nt"])
    H = np.zeros((nr, nt), dtype=np.complex128)
    port_indices: list[np.ndarray] = []
    port_responses: list[np.ndarray] = []
    alphas: list[complex] = []
    ats: list[np.ndarray] = []

    for comp in realization["components"]:
        ar = np.zeros((nr,), dtype=np.complex128)
        idxs = np.zeros((nr,), dtype=int)
        local_phi = comp.phi_deg - yaw_deg
        for m, pat in enumerate(patterns):
            idx = nearest_sample_index(pat, comp.theta_deg, local_phi)
            idxs[m] = idx
            eth = pat.etheta[idx]
            eph = pat.ephi[idx]
            if polarization_mode == "vertical":
                eph = 0.0 + 0.0j
            elif polarization_mode == "horizontal":
                eth = 0.0 + 0.0j
            ar[m] = np.conj(eth) * comp.pol[0] + np.conj(eph) * comp.pol[1]

        H += comp.alpha * np.outer(ar, np.conj(comp.at))
        port_indices.append(idxs)
        port_responses.append(ar)
        alphas.append(comp.alpha)
        ats.append(comp.at)

    return {
        "H": H,
        "path_indices": port_indices,
        "path_responses": port_responses,
        "alphas": alphas,
        "ats": ats,
    }


def mutual_information(H: np.ndarray, snr_linear: float) -> float:
    nr, nt = H.shape
    c = snr_linear / nt
    A = np.eye(nr, dtype=np.complex128) + c * (H @ H.conj().T)
    sign, logdet = np.linalg.slogdet(A)
    if sign.real <= 0:
        eig = np.linalg.eigvalsh(A)
        eig = np.maximum(eig.real, 1e-300)
        return float(np.sum(np.log2(eig)))
    return float(logdet.real / np.log(2.0))


def compute_mi_distribution_and_gradient(
    patterns: list[FarFieldPattern],
    realizations: list[dict[str, Any]],
    port_index: int,
    snr_db: float,
    polarization_mode: str = "cross",
    terminal_yaw_deg: float | None = None,
) -> dict[str, Any]:
    pat = patterns[port_index]
    g_lin = local_gain_linear(pat, polarization_mode)
    if terminal_yaw_deg is None:
        rotation_angles = horizontal_rotation_angles(pat)
    else:
        rotation_angles = np.asarray([terminal_yaw_deg % 360.0], dtype=float)
    grad_scale = np.zeros((pat.n_points,), dtype=float)
    mi_max = np.full((pat.n_points,), -np.inf, dtype=float)
    mi_min = np.full((pat.n_points,), np.inf, dtype=float)
    mi_values: list[float] = []

    snr_linear = 10.0 ** (snr_db / 10.0)

    for rel in realizations:
        mi_sum = 0.0
        for yaw_deg in rotation_angles:
            state = build_rotated_channel_data(patterns, rel, float(yaw_deg), polarization_mode)
            H = state["H"]
            nr, nt = H.shape
            c = snr_linear / nt
            A = np.eye(nr, dtype=np.complex128) + c * (H @ H.conj().T)
            B = np.linalg.solve(A, H)  # A^{-1} H
            mi_value = float(np.linalg.slogdet(A)[1].real / np.log(2.0))
            mi_sum += mi_value

            grouped_row_delta: dict[int, np.ndarray] = {}
            for idxs, ar, alpha, at in zip(
                state["path_indices"],
                state["path_responses"],
                state["alphas"],
                state["ats"],
            ):
                idx = int(idxs[port_index])
                r = ar[port_index]
                row_delta = alpha * r * np.conj(at)
                if idx in grouped_row_delta:
                    grouped_row_delta[idx] = grouped_row_delta[idx] + row_delta
                else:
                    grouped_row_delta[idx] = row_delta.copy()

            b_row = B[port_index, :]
            pref = 2.0 * c / np.log(2.0)
            for idx, drow in grouped_row_delta.items():
                mi_max[idx] = max(mi_max[idx], mi_value)
                mi_min[idx] = min(mi_min[idx], mi_value)
                dI_ds = pref * float(np.real(np.vdot(b_row, drow)))
                grad_scale[idx] += dI_ds
        mi_values.append(mi_sum)

    nreal = max(len(realizations), 1)
    grad_scale /= nreal
    mi_arr = np.asarray(mi_values, dtype=float)

    # Convert from dI/ds (local scaling) to dI/dgain.
    grad_gain = grad_scale / np.maximum(g_lin, 1e-12)
    grad_abs = np.abs(grad_gain)
    grad_log_abs = np.abs(grad_scale)  # dI/d log-gain approx, more stable for plotting if needed.

    return {
        "mi_values": mi_arr,
        "grad_gain": grad_gain,
        "grad_abs": grad_abs,
        "grad_log_abs": grad_log_abs,
        "mi_max": np.where(np.isfinite(mi_max), mi_max, np.nan),
        "mi_min": np.where(np.isfinite(mi_min), mi_min, np.nan),
        "gain_linear": g_lin,
        "gain_db": 20.0 * np.log10(np.maximum(g_lin, 1e-12)),
        "rotation_count": int(rotation_angles.size),
    }


def summary_stats(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/api/upload")
async def upload(
    ffs_files: list[UploadFile] = File(...),
    channel_file: UploadFile = File(...),
    polarization_mode: str = Form("cross"),
):
    polarization_mode = validate_polarization_mode(polarization_mode)
    if not ffs_files:
        raise HTTPException(400, "Please upload at least one .ffs file")
    if not channel_file.filename.lower().endswith(".json"):
        raise HTTPException(400, "Channel file must be JSON")

    patterns: list[FarFieldPattern] = []
    saved_files: list[str] = []
    for f in ffs_files:
        content = await f.read()
        try:
            text = content.decode("utf-8", errors="ignore")
            pat = parse_cst_ffs_text(text, f.filename)
        except Exception as e:
            raise HTTPException(400, f"Failed to parse {f.filename}: {e}")
        patterns.append(pat)
        out = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(f.filename).name}"
        out.write_bytes(content)
        saved_files.append(str(out))

    channel_content = await channel_file.read()
    try:
        channel_json = json.loads(channel_content.decode("utf-8"))
        channel_type = parse_channel_model(channel_json)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse channel JSON: {e}")

    sid = uuid.uuid4().hex
    SESSIONS[sid] = {
        "patterns": patterns,
        "channel": channel_json,
        "channel_type": channel_type,
        "saved_ffs": saved_files,
        "channel_name": channel_file.filename,
    }

    initial_heatmaps = [
        gain_heatmap_payload(pat, port_index, polarization_mode)
        for port_index, pat in enumerate(patterns)
    ]
    return {
        "session_id": sid,
        "n_ports": len(patterns),
        "ports": [p.name for p in patterns],
        "channel_type": channel_type,
        "channel_name": channel_file.filename,
        "initial_heatmap": initial_heatmaps[0],
        "initial_heatmaps": initial_heatmaps,
    }


@app.get("/api/heatmap/{session_id}")
def get_gain_heatmap(session_id: str, port_index: int = 0, polarization_mode: str = "cross"):
    polarization_mode = validate_polarization_mode(polarization_mode)
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    sess = SESSIONS[session_id]
    patterns: list[FarFieldPattern] = sess["patterns"]
    if port_index < 0 or port_index >= len(patterns):
        raise HTTPException(400, "Invalid port index")
    return gain_heatmap_payload(patterns[port_index], port_index, polarization_mode)


@app.post("/api/gradient/{session_id}")
def gradient(
    session_id: str,
    port_index: int = Form(0),
    snr_db: float = Form(10.0),
    num_snapshots: int = Form(200),
    polarization_mode: str = Form("cross"),
    terminal_pose_mode: str = Form("horizontal_scan"),
    terminal_pose_angle_deg: float | None = Form(None),
):
    polarization_mode = validate_polarization_mode(polarization_mode)
    if terminal_pose_mode not in TERMINAL_POSE_MODES:
        raise HTTPException(400, "Invalid terminal pose mode")
    terminal_yaw_deg = None
    if terminal_pose_mode == "fixed_angle":
        if terminal_pose_angle_deg is None or not math.isfinite(terminal_pose_angle_deg):
            raise HTTPException(400, "Terminal pose angle is required")
        terminal_yaw_deg = terminal_pose_angle_deg

    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    sess = SESSIONS[session_id]
    patterns: list[FarFieldPattern] = sess["patterns"]
    if port_index < 0 or port_index >= len(patterns):
        raise HTTPException(400, "Invalid port index")

    if sess["channel_type"] == "snapshots":
        realizations = build_realizations(patterns, sess["channel"], num_snapshots=1)
    else:
        realizations = build_realizations(patterns, sess["channel"], num_snapshots=max(1, num_snapshots))

    result = compute_mi_distribution_and_gradient(
        patterns,
        realizations,
        port_index,
        snr_db,
        polarization_mode,
        terminal_yaw_deg=terminal_yaw_deg,
    )
    pat = patterns[port_index]

    return {
        "mode": "gradient_abs_dmi_dgain",
        "port_index": port_index,
        "polarization_mode": polarization_mode,
        "terminal_pose_mode": terminal_pose_mode,
        "terminal_pose_angle_deg": terminal_yaw_deg,
        "snr_db": snr_db,
        "num_realizations": len(realizations),
        "rotation_count": result["rotation_count"],
        "theta": pat.theta_unique.tolist(),
        "phi": pat.phi_unique.tolist(),
        "z": heatmap_from_samples(pat, result["grad_abs"]),
        "gain_db": heatmap_from_samples(pat, result["gain_db"]),
        "grad_log_abs": heatmap_from_samples(pat, result["grad_log_abs"]),
        "mi_max": heatmap_from_samples(pat, result["mi_max"]),
        "mi_min": heatmap_from_samples(pat, result["mi_min"]),
        "mi_values": result["mi_values"].tolist(),
        "stats": summary_stats(result["mi_values"]),
    }
