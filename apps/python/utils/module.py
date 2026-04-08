import importlib.util
import sys
import hashlib
from pathlib import Path

def module_name(abs_path: Path) -> str:
    digest = hashlib.sha1(str(abs_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"glassbeaker_dynamic_{abs_path.stem}_{digest}"

def load_module(abs_path: Path):
    name = module_name(abs_path)
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, abs_path)
    module = spec and importlib.util.module_from_spec(spec)
    if module and spec and spec.loader:
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module
    print(f"WARN: load from {abs_path} failed")
    return None
