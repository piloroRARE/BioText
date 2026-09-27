from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_legacy_module(name: str):
    candidates = [
        HERE / f"{name}.py (corrigé v2).py",
        HERE / f"{name} (corrigé v2).py",
        HERE / f"{name}.py",
    ]
    for path in candidates:
        if path.exists():
            spec = spec_from_file_location(f"{name}_legacy", path)
            if spec is None or spec.loader is None:
                continue
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"Unable to find the legacy implementation for {name!r}")


legacy = _load_legacy_module("biotext_light")

for key, value in legacy.__dict__.items():
    if key.startswith("__") and key.endswith("__"):
        continue
    globals()[key] = value


if __name__ == "__main__":
    legacy.train()
