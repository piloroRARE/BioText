#!/usr/bin/env python3
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def _load_legacy_start():
    candidates = [
        HERE / "start.py",
        HERE / "start.py (corrigé v2).py",
        HERE / "start (corrigé v2).py",
    ]
    for path in candidates:
        if path.exists():
            spec = spec_from_file_location("start_legacy", path)
            if spec is None or spec.loader is None:
                continue
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Unable to find the legacy start script.")


legacy = _load_legacy_start()


if __name__ == "__main__":
    # Reuse the original entry point behaviour while keeping the standard script path.
    legacy.main()
