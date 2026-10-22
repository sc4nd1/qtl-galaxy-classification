import json
import sys
from pathlib import Path

import pytest

from tesi_qml.config import load_yaml
from tesi_qml.experiments import RunSpec
from tesi_qml.matrix import (
    build_feature_command,
    build_train_command,
    find_run_directory,
    is_run_complete,
    materialize_primary_matrix,
    required_run_artifacts,
)


def _configs():
    return load_yaml("configs/dataset.yaml"), load_yaml("configs/training.yaml")


def test_primary_matrix_has_exactly_160_unique_iteration_one_runs() -> None:
    """Catches subsets, extra folds or missing task/model cells in the primary matrix."""
    dataset, training = _configs()
    runs = materialize_primary_matrix(dataset, training)

    assert len(runs) == 160
    assert len({run.run_id for run in runs}) == 160
    assert {run.iteration for run in runs} == {1}
    assert {run.task for run in runs} == {"binary", "three_class"}
    assert {run.repeat_seed for run in runs} == set(dataset["splits"]["repeat_seeds"])
    assert {run.model_tag for run in runs} == {
        "b0",
        "b1",
        "b2_q4",
        "b2_q8",
        "qf_q4",
        "qf_q8",
        "q_q4",
        "q_q8",
    }


def test_complete_run_requires_every_artifact_and_matching_config(tmp_path: Path) -> None:
    """Catches partial runs being marked complete after metrics alone."""
    run_dir = tmp_path / "run"
    fingerprint = "a" * 64
    for path in required_run_artifacts(run_dir):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "run_config.json":
            path.write_text(json.dumps({"config_fingerprint": fingerprint}), encoding="utf-8")
        else:
            path.write_bytes(b"artifact")

    assert is_run_complete(run_dir, expected_config_fingerprint=fingerprint)
    (run_dir / "test_predictions.parquet").unlink()
    assert not is_run_complete(run_dir, expected_config_fingerprint=fingerprint)
    assert not is_run_complete(run_dir, expected_config_fingerprint="b" * 64)


def test_find_run_directory_rejects_multiple_matching_configurations(tmp_path: Path) -> None:
    """Catches ambiguous resume when two run directories claim one matrix identity."""
    spec = RunSpec("binary", "QF", 4, 42)
    for suffix in ("one", "two"):
        run_dir = (
            tmp_path
            / "results"
            / "runs"
            / "binary"
            / "qf_q4"
            / "seed_42"
            / "iteration_1"
            / f"config_{suffix}"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "task": "binary",
                    "model": "QF",
                    "n_qubits": 4,
                    "repeat_seed": 42,
                    "iteration": 1,
                    "n_layers": 2,
                    "subset_size": "full",
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Più directory"):
        find_run_directory(tmp_path, spec)


def test_train_command_is_resumable_and_contains_only_run_identity(tmp_path: Path) -> None:
    spec = RunSpec("three_class", "Q", 8, 73)
    command = build_train_command(
        sys.executable,
        tmp_path,
        spec,
        device="cpu",
        training_config=Path("configs/training.yaml"),
        dataset_config=Path("configs/dataset.yaml"),
    )

    assert command[:2] == [sys.executable, str(tmp_path / "scripts" / "train.py")]
    assert command[command.index("--task") + 1] == "three_class"
    assert command[command.index("--model") + 1] == "Q"
    assert command[command.index("--qubits") + 1] == "8"
    assert "--resume" in command


def test_feature_command_is_task_agnostic_and_repeat_scoped(tmp_path: Path) -> None:
    """Catches binary and three-class tasks duplicating the same physical feature bank."""
    spec = RunSpec("binary", "Q", 8, 73)
    command = build_feature_command(
        sys.executable,
        tmp_path,
        spec,
        device="cpu",
        dataset_config=Path("configs/dataset.yaml"),
        training_config=Path("configs/training.yaml"),
    )

    assert command[command.index("--repeat-seed") + 1] == "73"
    assert command[command.index("--iteration") + 1] == "1"
    assert "--task" not in command
