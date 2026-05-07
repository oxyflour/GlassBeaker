from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


CACHE_VERSION = "v1"
CHUNK_SIZE = 1024 * 1024


def cache_root() -> Path:
    override = os.environ.get("USD_TO_MJCF_CACHE_DIR")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2] / "tmp" / "usd_to_mjcf_cache"


def mesh_cache_path(signature: str) -> Path:
    return cache_root() / CACHE_VERSION / "meshes" / f"{signature}.obj"


def mesh_source_cache_path(signature: str) -> Path:
    return cache_root() / CACHE_VERSION / "mesh_sources" / f"{signature}.txt"


def texture_cache_path(signature: str) -> Path:
    return cache_root() / CACHE_VERSION / "textures" / f"{signature}.png"


def file_digest(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def materialize_cached_file(cache_file: Path, output_file: Path) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        try:
            if output_file.samefile(cache_file):
                return
        except OSError:
            pass
        output_file.unlink()
    try:
        output_file.hardlink_to(cache_file)
    except OSError:
        shutil.copy2(cache_file, output_file)
