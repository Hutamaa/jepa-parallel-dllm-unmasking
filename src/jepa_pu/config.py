from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_environment(value: Any) -> Any:
    """Expand ${NAME} and ${NAME:-default} strings in loaded YAML values."""
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        configured = os.environ.get(name)
        if configured:
            return configured
        if default is not None:
            return default
        raise ValueError(f"Required environment variable {name} is not set")

    return _ENV_PATTERN.sub(replace, value)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return _expand_environment(cfg)


def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    for key in ("data_dir", "trajectory_dir", "checkpoint_dir", "output_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
