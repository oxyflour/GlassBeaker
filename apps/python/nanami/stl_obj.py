from __future__ import annotations

import struct
from pathlib import Path


def is_ascii_stl(data: bytes) -> bool:
    head = data[:256].decode("ascii", errors="ignore").lower()
    return head.lstrip().startswith("solid") and b"\0" not in data[:256]


def parse_ascii_stl(text: str):
    normal = (0.0, 0.0, 1.0)
    vertices: list[tuple[float, float, float, tuple[float, float, float]]] = []
    for raw in text.splitlines():
        line = raw.strip().split()
        if not line:
            continue
        if line[0] == "facet" and len(line) >= 5:
            normal = tuple(float(value) for value in line[2:5])
        if line[0] == "vertex" and len(line) >= 4:
            point = tuple(float(value) for value in line[1:4])
            vertices.append((*point, normal))
    return vertices


def parse_binary_stl(data: bytes):
    count = struct.unpack_from("<I", data, 80)[0]
    offset = 84
    vertices: list[tuple[float, float, float, tuple[float, float, float]]] = []
    for _ in range(count):
        nx, ny, nz = struct.unpack_from("<3f", data, offset)
        offset += 12
        normal = (nx, ny, nz)
        for _ in range(3):
            x, y, z = struct.unpack_from("<3f", data, offset)
            offset += 12
            vertices.append((x, y, z, normal))
        offset += 2
    return vertices


def stl_to_obj(src: Path, dest: Path) -> None:
    data = src.read_bytes()
    rows = parse_ascii_stl(data.decode("utf-8", errors="ignore")) if is_ascii_stl(data) else parse_binary_stl(data)
    dest.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"# generated from {src.name}"]
    normals: list[tuple[float, float, float]] = []
    for x, y, z, normal in rows:
        lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
        normals.append(normal)
    for nx, ny, nz in normals:
        lines.append(f"vn {nx:.9g} {ny:.9g} {nz:.9g}")
    for face in range(0, len(rows), 3):
        a = face + 1
        b = face + 2
        c = face + 3
        lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
