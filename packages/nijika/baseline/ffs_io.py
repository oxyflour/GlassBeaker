from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np

POWER_PREFIX = "// Radiated/Accepted/Stimulated Power , Frequency"
FIELD_PREFIX = "// >> Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi):"
DIM_PREFIX = "// >> Total #phi samples, total #theta samples"


@dataclass(frozen=True, slots=True)
class FfsMetadata:
    frequencies_hz: np.ndarray
    angles_deg: np.ndarray
    radiated_power_w: np.ndarray
    accepted_power_w: np.ndarray
    stimulated_power_w: np.ndarray
    position_m: np.ndarray
    z_axis: np.ndarray
    x_axis: np.ndarray
    phi_count: int
    theta_count: int


def _read_sections(path: Path) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_name = ""
    current_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("//"):
            if current_name:
                sections.append((current_name, current_lines))
            current_name = line
            current_lines = []
            continue
        current_lines.append(line)
    if current_name:
        sections.append((current_name, current_lines))
    return sections


def _section_array(sections: list[tuple[str, list[str]]], prefix: str) -> np.ndarray:
    for name, lines in sections:
        if name == prefix:
            return np.loadtxt(StringIO("\n".join(lines)), ndmin=1)
    raise ValueError(f"missing section: {prefix}")


def _section_arrays(sections: list[tuple[str, list[str]]], prefix: str) -> list[np.ndarray]:
    arrays = [
        np.loadtxt(StringIO("\n".join(lines)), ndmin=2)
        for name, lines in sections
        if name == prefix
    ]
    if not arrays:
        raise ValueError(f"missing section: {prefix}")
    return arrays


def _metadata_from_sections(sections: list[tuple[str, list[str]]]) -> tuple[FfsMetadata, np.ndarray]:
    power = _section_array(sections, POWER_PREFIX).reshape(-1, 4)
    dims = _section_array(sections, DIM_PREFIX).astype(int).reshape(-1)
    field_sections = _section_arrays(sections, FIELD_PREFIX)
    angles = field_sections[0][:, :2].astype(np.float64, copy=False)
    fields = np.stack([section[:, 2:6] for section in field_sections]).astype(np.float64, copy=False)

    if len(field_sections) != power.shape[0]:
        raise ValueError("field section count does not match frequency count")
    if fields.shape[1] != int(dims[0]) * int(dims[1]):
        raise ValueError("angle grid size does not match declared dimensions")
    for section in field_sections[1:]:
        if not np.allclose(section[:, :2], angles, rtol=0.0, atol=1e-9):
            raise ValueError("angle grids differ across frequencies")

    metadata = FfsMetadata(
        frequencies_hz=power[:, 3].astype(np.float64, copy=False),
        angles_deg=angles,
        radiated_power_w=power[:, 0].astype(np.float64, copy=False),
        accepted_power_w=power[:, 1].astype(np.float64, copy=False),
        stimulated_power_w=power[:, 2].astype(np.float64, copy=False),
        position_m=_section_array(sections, "// Position").astype(np.float64, copy=False).reshape(3),
        z_axis=_section_array(sections, "// zAxis").astype(np.float64, copy=False).reshape(3),
        x_axis=_section_array(sections, "// xAxis").astype(np.float64, copy=False).reshape(3),
        phi_count=int(dims[0]),
        theta_count=int(dims[1]),
    )
    return metadata, fields


def load_ffs_sample(path: Path) -> tuple[FfsMetadata, np.ndarray]:
    return _metadata_from_sections(_read_sections(path))


def load_ffs_group(paths: list[Path]) -> tuple[FfsMetadata, np.ndarray]:
    if not paths:
        raise ValueError("paths must not be empty")
    base_metadata, base_field = load_ffs_sample(paths[0])
    metadata_parts = [base_metadata]
    field_parts = [base_field]
    for path in paths[1:]:
        metadata, field = load_ffs_sample(path)
        if (
            metadata.phi_count != base_metadata.phi_count
            or metadata.theta_count != base_metadata.theta_count
            or not np.allclose(metadata.angles_deg, base_metadata.angles_deg, rtol=0.0, atol=1e-9)
            or not np.allclose(metadata.position_m, base_metadata.position_m, rtol=0.0, atol=1e-12)
            or not np.allclose(metadata.z_axis, base_metadata.z_axis, rtol=0.0, atol=1e-12)
            or not np.allclose(metadata.x_axis, base_metadata.x_axis, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("FFS group members must share geometry and angle metadata")
        metadata_parts.append(metadata)
        field_parts.append(field)

    metadata = FfsMetadata(
        frequencies_hz=np.concatenate([item.frequencies_hz for item in metadata_parts]),
        angles_deg=base_metadata.angles_deg.copy(),
        radiated_power_w=np.concatenate([item.radiated_power_w for item in metadata_parts]),
        accepted_power_w=np.concatenate([item.accepted_power_w for item in metadata_parts]),
        stimulated_power_w=np.concatenate([item.stimulated_power_w for item in metadata_parts]),
        position_m=base_metadata.position_m.copy(),
        z_axis=base_metadata.z_axis.copy(),
        x_axis=base_metadata.x_axis.copy(),
        phi_count=base_metadata.phi_count,
        theta_count=base_metadata.theta_count,
    )
    return metadata, np.concatenate(field_parts, axis=0)


def write_ffs_sample(path: Path, metadata: FfsMetadata, field: np.ndarray) -> None:
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3 or field.shape[2] != 4:
        raise ValueError("field must have shape (freq_count, angle_count, 4)")
    if field.shape[0] != len(metadata.frequencies_hz):
        raise ValueError("field frequency count does not match metadata")
    if field.shape[1] != metadata.angles_deg.shape[0]:
        raise ValueError("field angle count does not match metadata")

    path.parent.mkdir(parents=True, exist_ok=True)
    freq_count = field.shape[0]
    angle_rows = metadata.angles_deg
    lines = [
        "// CST Farfield Source File",
        "",
        "// Version:",
        "3.0",
        "",
        "// Data Type",
        "Farfield",
        "",
        "// #Frequencies",
        str(freq_count),
        "",
        "// Position",
        " ".join(f"{value:.6e}" for value in metadata.position_m),
        "",
        "// zAxis",
        " ".join(f"{value:.6e}" for value in metadata.z_axis),
        "",
        "// xAxis",
        " ".join(f"{value:.6e}" for value in metadata.x_axis),
        "",
        POWER_PREFIX,
    ]
    for idx in range(freq_count):
        lines.extend(
            [
                f"{metadata.radiated_power_w[idx]:.6e}",
                f"{metadata.accepted_power_w[idx]:.6e}",
                f"{metadata.stimulated_power_w[idx]:.6e}",
                f"{metadata.frequencies_hz[idx]:.6e}",
                "",
                DIM_PREFIX,
                f"{metadata.phi_count} {metadata.theta_count}",
                "",
                FIELD_PREFIX,
            ]
        )
        rows = np.column_stack([angle_rows, field[idx]])
        lines.extend(" ".join(f"{value:.8e}" for value in row) for row in rows)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf8")
