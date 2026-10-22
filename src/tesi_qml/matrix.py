"""Matrice primaria e controlli minimi sugli artefatti dei run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .experiments import RunSpec


def materialize_primary_matrix(
    dataset_config: dict[str, Any], training_config: dict[str, Any]
) -> list[RunSpec]:
    """Espande la configurazione bloccata nelle 160 identità preregistrate."""
    matrix = training_config["primary_matrix"]
    runs: list[RunSpec] = []
    for task in matrix["tasks"]:
        for repeat_seed in dataset_config["splits"]["repeat_seeds"]:
            for iteration in dataset_config["splits"]["primary_iterations"]:
                for model in matrix["models"]:
                    spec = RunSpec(
                        task=str(task),
                        model=str(model["name"]),
                        qubits=int(model["qubits"]) if "qubits" in model else None,
                        repeat_seed=int(repeat_seed),
                        iteration=int(iteration),
                        n_layers=int(matrix["depth"]),
                        subset_size=str(matrix["subset_size"]),
                    )
                    spec.validate_primary()
                    runs.append(spec)
    if len({run.run_id for run in runs}) != len(runs):
        raise ValueError("La matrice primaria contiene run duplicati")
    if len(runs) != 160:
        raise ValueError(f"La matrice primaria deve contenere 160 run, trovati {len(runs)}")
    return runs


def required_run_artifacts(run_dir: Path | str) -> tuple[Path, ...]:
    root = Path(run_dir)
    return (
        root / "run_config.json",
        root / "checkpoints" / "best.pt",
        root / "checkpoints" / "last.pt",
        root / "history.jsonl",
        root / "test_metrics.json",
        root / "test_predictions.parquet",
    )


def is_run_complete(
    run_dir: Path | str, *, expected_config_fingerprint: str | None = None
) -> bool:
    artifacts = required_run_artifacts(run_dir)
    if not all(path.is_file() for path in artifacts):
        return False
    try:
        config = json.loads(artifacts[0].read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    fingerprint = config.get("config_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        return False
    return expected_config_fingerprint is None or fingerprint == expected_config_fingerprint


def _matches_spec(config: dict[str, Any], spec: RunSpec) -> bool:
    return (
        config.get("task") == spec.task
        and config.get("model") == spec.model
        and config.get("n_qubits") == spec.qubits
        and config.get("repeat_seed") == spec.repeat_seed
        and config.get("iteration", config.get("fold")) == spec.iteration
        and config.get("n_layers") == spec.n_layers
        and str(config.get("subset_size", "full")) == spec.subset_size
    )


def find_run_directory(project_root: Path | str, spec: RunSpec) -> Path | None:
    base = (
        Path(project_root)
        / "results"
        / "runs"
        / spec.task
        / spec.model_tag
        / f"seed_{spec.repeat_seed}"
        / f"iteration_{spec.iteration}"
    )
    matches: list[Path] = []
    for path in sorted(base.glob("config_*/run_config.json")):
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if _matches_spec(config, spec):
            matches.append(path.parent)
    if len(matches) > 1:
        raise ValueError(f"Più directory corrispondono al run {spec.run_id}")
    return matches[0] if matches else None


def build_train_command(
    python_executable: str,
    project_root: Path | str,
    spec: RunSpec,
    *,
    device: str,
    training_config: Path | str,
    dataset_config: Path | str,
) -> list[str]:
    command = [
        python_executable,
        str(Path(project_root) / "scripts" / "train.py"),
        "--project-root",
        str(project_root),
        "--config",
        str(training_config),
        "--dataset-config",
        str(dataset_config),
        "--task",
        spec.task,
        "--model",
        spec.model,
        "--repeat-seed",
        str(spec.repeat_seed),
        "--iteration",
        str(spec.iteration),
        "--layers",
        str(spec.n_layers),
        "--subset-size",
        spec.subset_size,
        "--device",
        device,
        "--resume",
    ]
    if spec.qubits is not None:
        command.extend(("--qubits", str(spec.qubits)))
    return command


def feature_bank_complete(project_root: Path | str, spec: RunSpec) -> bool:
    base = (
        Path(project_root)
        / "data"
        / "features"
        / f"seed_{spec.repeat_seed}"
        / f"iteration_{spec.iteration}"
    )
    return all(
        (base / split / filename).is_file()
        for split in ("train", "validation", "test")
        for filename in ("features.pt", "metadata.parquet", "provenance.json")
    )


def build_feature_command(
    python_executable: str,
    project_root: Path | str,
    spec: RunSpec,
    *,
    device: str,
    dataset_config: Path | str,
    training_config: Path | str,
) -> list[str]:
    """Costruisce una sola cache condivisa dai due task per seed/iterazione."""
    return [
        python_executable,
        str(Path(project_root) / "scripts" / "cache_features.py"),
        "--project-root",
        str(project_root),
        "--config",
        str(dataset_config),
        "--training-config",
        str(training_config),
        "--repeat-seed",
        str(spec.repeat_seed),
        "--iteration",
        str(spec.iteration),
        "--device",
        device,
    ]
