from __future__ import annotations

import json
from pathlib import Path

from scripts.colab_subsets import (
    EXPECTED_RUNS_PER_SUBSET,
    EXPECTED_SUBSET_RUNS,
    SUBSET_SIZES,
    matrix_report,
    prepare_subsets,
    restore_subset_state,
    save_subset_state,
    write_subset_training_configs,
)
from tesi_qml.config import load_yaml


def test_write_subset_training_configs(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)
    base_config = {
        "primary_matrix": {
            "tasks": ["binary", "three_class"],
            "models": [{"name": "B0"}, {"name": "B1"}],
            "depth": 2,
            "subset_size": "full",
        }
    }
    import yaml

    (configs_dir / "training.yaml").write_text(yaml.safe_dump(base_config), encoding="utf-8")

    paths = write_subset_training_configs(tmp_path, subset_sizes=("100", "300"))
    assert len(paths) == 2
    assert (configs_dir / "training_subset_100.yaml").is_file()
    assert (configs_dir / "training_subset_300.yaml").is_file()

    c100 = load_yaml(configs_dir / "training_subset_100.yaml")
    assert c100["primary_matrix"]["subset_size"] == "100"


def test_matrix_report_returns_expected_counts_on_missing_or_populated_state(
    tmp_path: Path,
) -> None:
    missing_report = matrix_report(tmp_path / "nonexistent.json", expected=160)
    assert missing_report == {
        "complete": 0,
        "failed": 0,
        "pending": 160,
        "running": 0,
        "total": 160,
    }

    state_path = tmp_path / "matrix.json"
    state_path.write_text(
        json.dumps(
            {
                "expected_runs": 160,
                "runs": [
                    {"status": "complete"},
                    {"status": "complete"},
                    {"status": "failed"},
                    {"status": "running"},
                    {"status": "pending"},
                ],
            }
        ),
        encoding="utf-8",
    )
    report = matrix_report(state_path, expected=160)
    assert report["complete"] == 2
    assert report["failed"] == 1
    assert report["running"] == 1
    assert report["pending"] == 1
    assert report["total"] == 160


def test_save_and_restore_subset_state_roundtrip(tmp_path: Path) -> None:
    project = tmp_path / "project"
    archive = tmp_path / "drive" / "subset_state.tar.gz"

    (project / "results" / "matrix").mkdir(parents=True)
    (project / "results" / "runs" / "subset_run_1").mkdir(parents=True)
    (project / "results" / "matrix" / "subset_100.json").write_text(
        '{"complete": 1}\n', encoding="utf-8"
    )
    (project / "results" / "runs" / "subset_run_1" / "run_config.json").write_text(
        '{"subset_size": "100"}\n', encoding="utf-8"
    )
    (project / "results" / "runs" / "subset_run_1" / "metrics.json").write_text(
        '{"f1": 0.8}\n', encoding="utf-8"
    )
    (project / "results" / "subset_runner.log").write_text("log line\n", encoding="utf-8")

    info = save_subset_state(project, archive)
    assert info["files"] >= 3
    assert archive.is_file()

    restored = tmp_path / "restored"
    assert restore_subset_state(restored, archive)
    assert (restored / "results" / "matrix" / "subset_100.json").is_file()
    assert (restored / "results" / "runs" / "subset_run_1" / "metrics.json").is_file()
    assert (restored / "results" / "subset_runner.log").read_text(encoding="utf-8") == "log line\n"


def test_prepare_subsets_against_actual_repository_configs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary = prepare_subsets(repo_root)

    assert summary["configs_written"] == len(SUBSET_SIZES)
    assert summary["matrix_runs"] == EXPECTED_SUBSET_RUNS
    for size in SUBSET_SIZES:
        assert size in summary["status"]
        assert summary["status"][size]["total"] == EXPECTED_RUNS_PER_SUBSET
