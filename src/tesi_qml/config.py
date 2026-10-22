"""Caricamento rigoroso delle configurazioni e risoluzione dei path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Carica un file YAML la cui radice deve essere una mapping."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"La radice della configurazione deve essere una mapping: {config_path}")
    return value


def resolve_project_path(project_root: Path | str, value: Path | str) -> Path:
    """Risolve un path di configurazione rispetto alla radice senza usare cwd."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(project_root).resolve() / candidate).resolve()
