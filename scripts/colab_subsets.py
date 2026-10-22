#!/usr/bin/env python3
"""Gestione e orchestrazione dell'analisi secondaria sui subset per Google Colab."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from tesi_qml.cli import atomic_json, atomic_parquet
from tesi_qml.config import load_yaml
from tesi_qml.data.features import FeatureBank
from tesi_qml.evaluation.plots import save_figure
from tesi_qml.evaluation.statistics import (
    aggregate_runs,
    holm_adjust,
    paired_model_comparison,
)
from tesi_qml.experiments import RunSpec, build_run_identity
from tesi_qml.matrix import materialize_primary_matrix
from tesi_qml.tasks import filter_feature_bank, get_task

try:
    from colab import create_state_archive, restore_state_archive
except ImportError:
    from scripts.colab import create_state_archive, restore_state_archive

try:
    from train import select_training_subset
except ImportError:
    from scripts.train import select_training_subset

SUBSET_SIZES = ("100", "300", "900")
EXPECTED_RUNS_PER_SUBSET = 160
EXPECTED_SUBSET_RUNS = 480
SUBSET_SAVE_INTERVAL = 10


def subset_matrix_path(project_root: Path | str, subset_size: str) -> Path:
    return Path(project_root) / "results" / "matrix" / f"subset_{subset_size}.json"


def subset_config_path(project_root: Path | str, subset_size: str) -> Path:
    return Path(project_root) / "configs" / f"training_subset_{subset_size}.yaml"


def write_subset_training_configs(
    project_root: Path | str, subset_sizes: Sequence[str] = SUBSET_SIZES
) -> list[Path]:
    root = Path(project_root)
    base = load_yaml(root / "configs" / "training.yaml")
    paths = []
    for subset_size in subset_sizes:
        config = deepcopy(base)
        config["primary_matrix"]["subset_size"] = str(subset_size)
        path = subset_config_path(root, subset_size)
        path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def matrix_report(path: Path | str, expected: int = EXPECTED_RUNS_PER_SUBSET) -> dict[str, int]:
    path = Path(path)
    if not path.is_file():
        return {
            "complete": 0,
            "failed": 0,
            "pending": expected,
            "running": 0,
            "total": expected,
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(str(row.get("status", "pending")) for row in state.get("runs", []))
    total = int(state.get("expected_runs", expected))
    return {
        "complete": counts["complete"],
        "failed": counts["failed"],
        "pending": counts["pending"],
        "running": counts["running"],
        "total": total,
    }


def subset_reports(project_root: Path | str) -> dict[str, dict[str, int]]:
    root = Path(project_root)
    return {size: matrix_report(subset_matrix_path(root, size)) for size in SUBSET_SIZES}


def _subset_state_files(project_root: Path | str) -> list[Path]:
    root = Path(project_root).resolve()
    results = root / "results"
    selected: set[Path] = set()
    for path in (results / "matrix").glob("subset_*.json"):
        if path.is_file():
            selected.add(path.resolve())
    runs_root = results / "runs"
    if runs_root.is_dir():
        for config_path in runs_root.rglob("run_config.json"):
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if str(config.get("subset_size")) in SUBSET_SIZES:
                selected.update(
                    path.resolve()
                    for path in config_path.parent.rglob("*")
                    if path.is_file() and not path.name.endswith(".part")
                )
    for relative in ("aggregate", "thesis_figures"):
        directory = results / relative
        if directory.is_dir():
            selected.update(path.resolve() for path in directory.rglob("*") if path.is_file())
    log_path = results / "subset_runner.log"
    if log_path.is_file():
        selected.add(log_path.resolve())
    return sorted(selected)


def save_subset_state(project_root: Path | str, archive_path: Path | str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    archive = Path(archive_path).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".part")
    files = _subset_state_files(root)
    with tarfile.open(temporary, "w:gz") as bundle:
        for path in files:
            bundle.add(path, arcname=path.relative_to(root).as_posix())
    temporary.replace(archive)
    return {
        "archive": str(archive),
        "files": len(files),
        "size_mb": round(archive.stat().st_size / 2**20, 2),
    }


def restore_subset_state(project_root: Path | str, archive_path: Path | str) -> bool:
    archive = Path(archive_path)
    if not archive.is_file():
        return False
    return restore_state_archive(project_root, archive)


def audit_nested_subsets(project_root: Path | str) -> list[dict[str, Any]]:
    root = Path(project_root)
    dataset_config = load_yaml(root / "configs" / "dataset.yaml")
    full_config = load_yaml(root / "configs" / "training.yaml")
    nested_audit = []
    for repeat_seed in dataset_config["splits"]["repeat_seeds"]:
        feature_root = root / "data" / "features" / f"seed_{repeat_seed}" / "iteration_1" / "train"
        if not (feature_root / "features.pt").is_file():
            continue
        full_bank = FeatureBank.load(feature_root)
        for task_name in ("binary", "three_class"):
            task = get_task(task_name)
            task_bank = filter_feature_bank(full_bank, task)
            identity = build_run_identity(
                RunSpec(task_name, "B0", None, repeat_seed),
                dataset_config=dataset_config,
                training_config=full_config,
            )
            subset_seed = identity["sub_seeds"]["dataloader"]
            selected_sets = {
                size: set(
                    select_training_subset(task_bank, size, subset_seed)
                    .metadata["objid"]
                    .astype(str)
                )
                for size in SUBSET_SIZES
            }
            full_set = set(task_bank.metadata["objid"].astype(str))
            assert selected_sets["100"] < selected_sets["300"] < selected_sets["900"] < full_set
            nested_audit.append(
                {
                    "seed": repeat_seed,
                    "task": task_name,
                    "sizes": {key: len(value) for key, value in selected_sets.items()},
                    "full": len(full_set),
                }
            )
    return nested_audit


def prepare_subsets(
    project_root: Path | str, archive_path: Path | str | None = None
) -> dict[str, Any]:
    root = Path(project_root)
    configs = write_subset_training_configs(root)
    restored = False
    if archive_path:
        restored = restore_subset_state(root, archive_path)

    dataset_config = load_yaml(root / "configs" / "dataset.yaml")
    full_config = load_yaml(root / "configs" / "training.yaml")
    full_ids = {spec.run_id for spec in materialize_primary_matrix(dataset_config, full_config)}
    subset_specs = [
        spec
        for config_path in configs
        for spec in materialize_primary_matrix(dataset_config, load_yaml(config_path))
    ]
    if len(subset_specs) != EXPECTED_SUBSET_RUNS:
        raise ValueError(f"Attesi {EXPECTED_SUBSET_RUNS} run, trovati {len(subset_specs)}")
    if len({spec.run_id for spec in subset_specs}) != EXPECTED_SUBSET_RUNS:
        raise ValueError("Trovati run duplicati nei subset")
    if not full_ids.isdisjoint({spec.run_id for spec in subset_specs}):
        raise ValueError("I run ID dei subset si sovrappongono alla matrice full")

    nested = audit_nested_subsets(root)
    reports = subset_reports(root)
    summary = {
        "configs_written": len(configs),
        "subset_state_restored": restored,
        "matrix_runs": len(subset_specs),
        "nested_audits": len(nested),
        "status": reports,
    }
    return summary


def run_subset_session(
    project_root: Path | str,
    archive_path: Path | str,
    device: str = "cuda",
    max_runs: int = 160,
    subset_sizes: Sequence[str] = SUBSET_SIZES,
) -> dict[str, dict[str, int]]:
    root = Path(project_root)
    if max_runs < 1:
        raise ValueError("max_runs deve essere positivo")
    subset_sizes = tuple(str(size) for size in subset_sizes)
    invalid_sizes = set(subset_sizes) - set(SUBSET_SIZES)
    if not subset_sizes or invalid_sizes:
        raise ValueError(f"Subset non validi: {sorted(invalid_sizes)}")
    reports = subset_reports(root)
    initial_complete = sum(report["complete"] for report in reports.values())
    next_save = initial_complete + SUBSET_SAVE_INTERVAL
    attempts = 0
    log_path = root / "results" / "subset_runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for subset_size in subset_sizes:
            while attempts < max_runs:
                before = matrix_report(subset_matrix_path(root, subset_size))
                if before["complete"] == before["total"]:
                    break
                command = [
                    sys.executable,
                    str(root / "scripts" / "run_matrix.py"),
                    "--project-root",
                    str(root),
                    "--config",
                    str(subset_config_path(root, subset_size)),
                    "--matrix-path",
                    str(subset_matrix_path(root, subset_size)),
                    "--max-runs",
                    "1",
                    "--device",
                    str(device),
                ]
                completed = subprocess.run(
                    command, cwd=root, text=True, capture_output=True, check=False
                )
                with log_path.open("a", encoding="utf-8") as log:
                    log.write("$ " + " ".join(command) + "\n")
                    log.write(completed.stdout or "")
                    log.write(completed.stderr or "")
                attempts += 1
                after = matrix_report(subset_matrix_path(root, subset_size))
                reports = subset_reports(root)
                overall_complete = sum(report["complete"] for report in reports.values())
                if completed.returncode != 0 or after["failed"]:
                    print(save_subset_state(root, archive_path), flush=True)
                    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                    print("\n".join(tail), flush=True)
                    raise RuntimeError(f"Run subset {subset_size} fallita; stato salvato")
                if overall_complete >= next_save:
                    saved = save_subset_state(root, archive_path)
                    print(
                        {
                            "complete": overall_complete,
                            "total": EXPECTED_SUBSET_RUNS,
                            "by_subset": reports,
                            "saved": saved,
                        },
                        flush=True,
                    )
                    while next_save <= overall_complete:
                        next_save += SUBSET_SAVE_INTERVAL
            if attempts >= max_runs:
                break
    except KeyboardInterrupt:
        print(save_subset_state(root, archive_path), flush=True)
        raise
    saved = save_subset_state(root, archive_path)
    reports = subset_reports(root)
    print({"attempts": attempts, "by_subset": reports, "saved": saved}, flush=True)
    return reports


def finalize_subsets(
    project_root: Path | str,
    subset_archive: Path | str | None = None,
    full_archive: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    reports = subset_reports(root)
    completed_subset_sizes = [
        size
        for size, report in reports.items()
        if report["complete"] == report["total"] and report["failed"] == 0
    ]

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "aggregate_results.py"),
            "--project-root",
            str(root),
            "--allow-incomplete",
        ],
        check=True,
    )

    dataset_config = load_yaml(root / "configs" / "dataset.yaml")
    aggregate_dir = root / "results" / "aggregate"
    runs_frame = pd.read_parquet(aggregate_dir / "runs.parquet")
    splits_frame = pd.read_parquet(root / "data" / "splits" / "all.parquet")

    full_training_counts = {}
    for repeat_seed in dataset_config["splits"]["repeat_seeds"]:
        training_rows = splits_frame.loc[
            (splits_frame["repeat_seed"] == repeat_seed)
            & (splits_frame["iteration"] == 1)
            & (splits_frame["split"] == "train")
        ]
        full_training_counts[("three_class", repeat_seed)] = len(training_rows)
        full_training_counts[("binary", repeat_seed)] = int(
            training_rows["label"].isin(["elliptical", "spiral"]).sum()
        )

    runs_frame["training_samples"] = [
        full_training_counts[(str(row.task), int(row.repeat_seed))]
        if str(row.subset_size) == "full"
        else int(row.subset_size)
        for row in runs_frame.itertuples(index=False)
    ]
    atomic_parquet(aggregate_dir / "runs_with_training_size.parquet", runs_frame)

    analysis_subset_sizes = {"full", *completed_subset_sizes}
    analysis_runs = runs_frame.loc[
        runs_frame["subset_size"].astype(str).isin(analysis_subset_sizes)
    ].copy()
    learning_summary = aggregate_runs(
        analysis_runs,
        metrics=("accuracy", "balanced_accuracy", "f1_macro", "training_seconds"),
        group_columns=("task", "model", "n_qubits", "subset_size", "training_samples"),
    )
    atomic_parquet(aggregate_dir / "subset_learning_curve.parquet", learning_summary)
    atomic_json(aggregate_dir / "subset_learning_curve.json", learning_summary.to_dict("records"))

    secondary_records = []
    for subset_size in completed_subset_sizes:
        subset_index = SUBSET_SIZES.index(subset_size)
        for task_index, task_name in enumerate(("binary", "three_class")):
            for qubit_index, qubits in enumerate((4, 8)):
                selected = analysis_runs.loc[
                    (analysis_runs["subset_size"].astype(str) == subset_size)
                    & (analysis_runs["task"] == task_name)
                    & (analysis_runs["n_qubits"] == qubits)
                ]
                comparison = paired_model_comparison(
                    selected,
                    "Q",
                    "B2",
                    "f1_macro",
                    seed=dataset_config["project_seed"]
                    + subset_index * 4
                    + task_index * 2
                    + qubit_index,
                )
                secondary_records.append(
                    {
                        "subset_size": subset_size,
                        "task": task_name,
                        "n_qubits": qubits,
                        **comparison,
                        "raw_p_value": comparison["permutation_p_value"],
                    }
                )
    if secondary_records:
        secondary_adjusted = holm_adjust([record["raw_p_value"] for record in secondary_records])
        for record, adjusted in zip(secondary_records, secondary_adjusted):
            record["holm_p_value"] = float(adjusted)
    secondary_frame = pd.DataFrame(secondary_records)
    if not secondary_frame.empty:
        atomic_parquet(aggregate_dir / "secondary_subset_hypotheses.parquet", secondary_frame)
    atomic_json(
        aggregate_dir / "secondary_subset_hypotheses.json", secondary_frame.to_dict("records")
    )

    plot_frame = learning_summary.copy()
    plot_frame["model_label"] = [
        str(row.model) if pd.isna(row.n_qubits) else f"{row.model}-{int(row.n_qubits)}"
        for row in plot_frame.itertuples(index=False)
    ]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "generate_figures.py"),
            "--project-root",
            str(root),
        ],
        check=False,
    )

    if not plot_frame.empty and "f1_macro_mean" in plot_frame.columns:
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for axis, task_name in zip(axes, ("binary", "three_class")):
            task_rows = plot_frame.loc[plot_frame["task"] == task_name]
            for label, group in task_rows.groupby("model_label", sort=True):
                group = group.sort_values("training_samples")
                means = group["f1_macro_mean"].to_numpy(float)
                lower = means - group["f1_macro_ci95_low"].to_numpy(float)
                upper = group["f1_macro_ci95_high"].to_numpy(float) - means
                axis.errorbar(
                    group["training_samples"],
                    means,
                    yerr=np.vstack([lower, upper]),
                    marker="o",
                    capsize=3,
                    label=label,
                )
            axis.set(xlabel="Campioni di training", ylabel="Macro-F1", title=task_name, ylim=(0, 1))
            axis.legend(fontsize="small", ncol=2)
        save_figure(figure, root / "results" / "thesis_figures" / "macro_f1_vs_training_samples")

    incremental = None
    if subset_archive:
        incremental = save_subset_state(root, subset_archive)
    if full_archive:
        create_state_archive(root, full_archive)

    summary = {
        "completed_subsets": completed_subset_sizes,
        "runs": len(runs_frame),
        "secondary_tests": len(secondary_frame),
        "subset_archive_saved": incremental,
        "full_archive_saved": str(full_archive) if full_archive else None,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gestione dell'analisi secondaria sui subset per Google Colab"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state-archive", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="prepara le configurazioni e ripristina lo stato")

    run_parser = subparsers.add_parser("run", help="esegue un batch di run subset")
    run_parser.add_argument(
        "--subset", choices=("100", "300", "900", "all"), default="all", help="dimensione subset"
    )
    run_parser.add_argument("--max-runs", type=int, default=160)
    run_parser.add_argument("--device", default="cuda")

    subparsers.add_parser("status", help="mostra lo stato dei subset")

    finalize_parser = subparsers.add_parser(
        "finalize", help="aggrega risultati, genera curve di apprendimento e salva lo stato"
    )
    finalize_parser.add_argument("--full-state-archive", type=Path, default=None)

    args = parser.parse_args()
    root = args.project_root.resolve()

    if args.command == "prepare":
        summary = prepare_subsets(root, args.state_archive)
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.command == "run":
        if not args.state_archive:
            parser.error("Il comando 'run' richiede --state-archive")
        sizes = SUBSET_SIZES if args.subset == "all" else (args.subset,)
        reports = run_subset_session(
            root,
            args.state_archive.resolve(),
            device=args.device,
            max_runs=args.max_runs,
            subset_sizes=sizes,
        )
        print(json.dumps(reports, indent=2))
        return 0

    if args.command == "status":
        reports = subset_reports(root)
        print(json.dumps(reports, indent=2))
        return 0

    if args.command == "finalize":
        full_archive = args.full_state_archive.resolve() if args.full_state_archive else None
        subset_archive = args.state_archive.resolve() if args.state_archive else None
        summary = finalize_subsets(root, subset_archive=subset_archive, full_archive=full_archive)
        print(json.dumps(summary, indent=2, default=str))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
