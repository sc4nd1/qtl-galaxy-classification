"""Utilità condivise dai comandi della pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_yaml


def base_parser(description: str, *, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument(
        "--dry-run", action="store_true", help="Mostra le azioni senza scrivere file"
    )
    return parser


def load_cli_config(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    return root, load_yaml(config_path)


def dry_run(args: argparse.Namespace, action: str) -> bool:
    if args.dry_run:
        print(f"DRY-RUN: {action}")
        return True
    return False


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
