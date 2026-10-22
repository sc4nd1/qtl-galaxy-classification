import json
from pathlib import Path

from scripts.aggregate_results import collect_run_records
from tesi_qml.matrix import required_run_artifacts


def _write_run(run_dir: Path, *, complete: bool) -> None:
    fingerprint = "a" * 64
    config = {
        "task": "three_class",
        "class_names": ["elliptical", "merger", "spiral"],
        "class_to_index": {"elliptical": 0, "merger": 1, "spiral": 2},
        "model": "Q",
        "n_qubits": 4,
        "n_layers": 2,
        "subset_size": "full",
        "repeat_seed": 42,
        "fold": 1,
        "experiment_fingerprint": "b" * 64,
        "config_fingerprint": fingerprint,
    }
    metrics = {
        "accuracy": 0.8,
        "balanced_accuracy": 0.79,
        "precision": {"macro": 0.78},
        "recall": {"macro": 0.77},
        "f1": {"macro": 0.76},
        "per_class": {"merger": {"precision": 0.6, "recall": 0.5, "f1": 0.55}},
        "roc_auc_ovr": {"macro": 0.85},
        "pr_auc": {"macro": 0.82},
        "log_loss": 0.5,
        "brier_score": 0.2,
        "ece": 0.04,
        "runtime": {
            "training_seconds": 10.0,
            "inference_seconds": 1.0,
            "total_parameters": 100,
            "trainable_parameters": 84,
            "quantum_parameters": 16,
        },
    }
    for path in required_run_artifacts(run_dir):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "run_config.json":
            path.write_text(json.dumps(config), encoding="utf-8")
        elif path.name == "test_metrics.json":
            path.write_text(json.dumps(metrics), encoding="utf-8")
        else:
            path.write_bytes(b"artifact")
    if not complete:
        (run_dir / "test_predictions.parquet").unlink()


def test_collect_run_records_uses_only_complete_runs_and_exports_merger_metrics(
    tmp_path: Path,
) -> None:
    """Catches partial artifacts leaking into inference and merger tables."""
    _write_run(tmp_path / "results" / "runs" / "complete", complete=True)
    _write_run(tmp_path / "results" / "runs" / "partial", complete=False)

    records = collect_run_records(tmp_path)

    assert len(records) == 1
    assert records[0]["task"] == "three_class"
    assert records[0]["n_qubits"] == 4
    assert records[0]["merger_precision"] == 0.6
    assert records[0]["merger_recall"] == 0.5
    assert records[0]["merger_f1"] == 0.55
