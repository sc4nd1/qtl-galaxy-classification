#!/usr/bin/env python3
"""Orchestratore minimo per sessioni Colab persistenti e riprendibili."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from colab import create_state_archive, restore_state_archive


def _matrix_command(root: Path, *arguments: str) -> list[str]:
    return [
        sys.executable,
        str(root / "scripts" / "run_matrix.py"),
        "--project-root",
        str(root),
        *arguments,
    ]


def _run_session(root: Path, archive: Path, *, max_runs: int, device: str) -> int:
    if max_runs < 1:
        raise ValueError("--max-runs deve essere almeno uno")
    restored = restore_state_archive(root, archive)
    print(json.dumps({"state_restored": restored, "archive": str(archive)}), flush=True)
    for attempt in range(1, max_runs + 1):
        completed = subprocess.run(
            _matrix_command(root, "--max-runs", "1", "--device", device),
            check=False,
        )
        create_state_archive(root, archive)
        print(
            json.dumps(
                {
                    "session_attempt": attempt,
                    "exit_code": completed.returncode,
                    "state_saved": str(archive),
                }
            ),
            flush=True,
        )
        if completed.returncode != 0:
            return completed.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ripristina, esegue e salva una sessione della matrice Colab"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state-archive", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="esegue run singoli salvando dopo ciascuno")
    run_parser.add_argument("--max-runs", type=int, default=1)
    run_parser.add_argument("--device", default="cuda")
    subparsers.add_parser("restore", help="ripristina risultati e feature bank")
    subparsers.add_parser("save", help="salva risultati e feature bank")
    subparsers.add_parser("status", help="mostra lo stato corrente della matrice")
    args = parser.parse_args()
    root = args.project_root.resolve()
    archive = args.state_archive.resolve()

    if args.command == "run":
        return _run_session(root, archive, max_runs=args.max_runs, device=args.device)
    if args.command == "restore":
        restored = restore_state_archive(root, archive)
        print(json.dumps({"state_restored": restored, "archive": str(archive)}))
        return 0
    if args.command == "save":
        create_state_archive(root, archive)
        print(json.dumps({"state_saved": str(archive)}))
        return 0
    return subprocess.run(_matrix_command(root, "--report-only"), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
