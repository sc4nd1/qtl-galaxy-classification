#!/usr/bin/env python3
"""Materializza ed esegue in modo riprendibile la matrice primaria da 160 run."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from tesi_qml.cli import atomic_json, base_parser, load_cli_config
from tesi_qml.config import load_yaml
from tesi_qml.experiments import RunSpec
from tesi_qml.matrix import (
    build_feature_command,
    build_train_command,
    feature_bank_complete,
    find_run_directory,
    is_run_complete,
    materialize_primary_matrix,
)


def _spec_dict(spec: RunSpec) -> dict[str, object]:
    return {
        "run_id": spec.run_id,
        "task": spec.task,
        "model": spec.model,
        "qubits": spec.qubits,
        "repeat_seed": spec.repeat_seed,
        "iteration": spec.iteration,
        "n_layers": spec.n_layers,
        "subset_size": spec.subset_size,
    }


def _spec_from_row(row: dict[str, object]) -> RunSpec:
    return RunSpec(
        task=str(row["task"]),
        model=str(row["model"]),
        qubits=int(row["qubits"]) if row.get("qubits") is not None else None,
        repeat_seed=int(row["repeat_seed"]),
        iteration=int(row["iteration"]),
        n_layers=int(row["n_layers"]),
        subset_size=str(row["subset_size"]),
    )


def _load_state(path: Path, specs: list[RunSpec]) -> dict[str, object]:
    existing: dict[str, dict[str, object]] = {}
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        existing = {str(row["run_id"]): row for row in value.get("runs", [])}
    rows = []
    for spec in specs:
        row = existing.get(spec.run_id, {})
        rows.append(
            {
                **_spec_dict(spec),
                "status": row.get("status", "pending"),
                "run_dir": row.get("run_dir"),
                "error": row.get("error"),
            }
        )
    return {"version": 1, "expected_runs": len(specs), "runs": rows}


def _refresh_state(project_root: Path, state: dict[str, object]) -> None:
    for row in state["runs"]:
        spec = _spec_from_row(row)
        run_dir = find_run_directory(project_root, spec)
        row["run_dir"] = str(run_dir) if run_dir else None
        if run_dir and is_run_complete(run_dir):
            row["status"] = "complete"
            row["error"] = None
        elif row["status"] in {"running", "complete"}:
            row["status"] = "pending"


def _report(state: dict[str, object]) -> dict[str, int]:
    counts = Counter(str(row["status"]) for row in state["runs"])
    return {
        "total": int(state["expected_runs"]),
        "pending": counts["pending"],
        "running": counts["running"],
        "complete": counts["complete"],
        "failed": counts["failed"],
    }


def main() -> None:
    parser = base_parser(
        "Esegue la matrice primaria binaria/triclasse", default_config="configs/training.yaml"
    )
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument(
        "--matrix-path", type=Path, default=Path("results/matrix/primary.json")
    )
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--next", action="store_true", help="Esegue al massimo il prossimo run")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root, training_config = load_cli_config(args)
    dataset_path = (
        args.dataset_config if args.dataset_config.is_absolute() else root / args.dataset_config
    )
    if args.dry_run and not dataset_path.is_file():
        print("DRY-RUN: matrice primaria; dataset config non presente nella root temporanea")
        return
    dataset_config = load_yaml(dataset_path)
    specs = materialize_primary_matrix(dataset_config, training_config)
    training_path = args.config if args.config.is_absolute() else root / args.config

    if args.dry_run:
        print(json.dumps({"dry_run": True, "total": len(specs), "unique": len({s.run_id for s in specs})}))
        for spec in specs:
            command = build_train_command(
                sys.executable,
                root,
                spec,
                device=args.device,
                training_config=training_path,
                dataset_config=dataset_path,
            )
            print(" ".join(command))
        return

    if args.max_runs < 1:
        raise ValueError("--max-runs deve essere almeno uno")
    matrix_path = args.matrix_path if args.matrix_path.is_absolute() else root / args.matrix_path
    state = _load_state(matrix_path, specs)
    _refresh_state(root, state)
    atomic_json(matrix_path, state)
    if args.report_only:
        print(json.dumps(_report(state), indent=2, sort_keys=True))
        return

    limit = 1 if args.next else args.max_runs
    selected = [row for row in state["runs"] if row["status"] in {"pending", "failed"}][
        :limit
    ]
    for row in selected:
        spec = _spec_from_row(row)
        row["status"] = "running"
        row["error"] = None
        atomic_json(matrix_path, state)
        if not feature_bank_complete(root, spec):
            feature_command = build_feature_command(
                sys.executable,
                root,
                spec,
                device=args.device,
                dataset_config=dataset_path,
                training_config=training_path,
            )
            feature_result = subprocess.run(feature_command, check=False)
            if feature_result.returncode != 0 or not feature_bank_complete(root, spec):
                row["status"] = "failed"
                row["error"] = f"cache_features.py exit={feature_result.returncode}"
                atomic_json(matrix_path, state)
                continue
        command = build_train_command(
            sys.executable,
            root,
            spec,
            device=args.device,
            training_config=training_path,
            dataset_config=dataset_path,
        )
        completed = subprocess.run(command, check=False)
        run_dir = find_run_directory(root, spec)
        row["run_dir"] = str(run_dir) if run_dir else None
        if completed.returncode == 0 and run_dir and is_run_complete(run_dir):
            row["status"] = "complete"
        else:
            row["status"] = "failed"
            row["error"] = f"train.py exit={completed.returncode} o artefatti incompleti"
        atomic_json(matrix_path, state)
    print(json.dumps(_report(state), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
