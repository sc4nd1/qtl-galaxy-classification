import io
import json
import os
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_colab_run_anomaly_audit.ipynb"
CONFIGURATIONS = (
    ("B0", None),
    ("B1", None),
    ("B2", 4),
    ("QF", 4),
    ("Q", 4),
    ("B2", 8),
    ("QF", 8),
    ("Q", 8),
)


def _run_payload(task: str, model: str, qubits: int | None, subset: str, seed: int):
    resource = {"100": 0.00, "300": 0.04, "900": 0.07, "full": 0.09}[subset]
    base = 0.84 if task == "binary" else 0.62
    model_effect = {"B0": -0.01, "B1": 0.0, "B2": 0.01, "QF": -0.005, "Q": 0.0}[model]
    f1 = base + resource + model_effect + seed * 0.001
    planted_performance = task == "three_class" and model == "Q" and qubits == 8 and subset == "100" and seed == 9
    planted_zero_gradient = task == "three_class" and model == "Q" and qubits == 4 and subset == "300" and seed == 7
    planted_slow_epoch = task == "binary" and model == "B2" and qubits == 4 and subset == "300" and seed == 6
    planted_non_monotonic = task == "binary" and model == "Q" and qubits == 4 and seed == 8
    if planted_non_monotonic and subset == "full":
        f1 -= 0.10

    validation_f1 = f1
    test_f1 = 0.20 if planted_performance else f1 - 0.004
    history = []
    for epoch in range(1, 9):
        elapsed = 100.0 if planted_slow_epoch and epoch == 5 else 0.5 + qubits_or_zero(qubits) / 100
        quantum_gradient = 0.0
        if model == "Q" and not planted_zero_gradient:
            quantum_gradient = 0.08 / epoch
        history.append(
            {
                "epoch": epoch,
                "elapsed_seconds": elapsed,
                "learning_rate": 0.001,
                "is_best": epoch == 8,
                "early_stopping_bad_epochs": 0,
                "gradient_norm": 1.2 / epoch,
                "quantum_gradient_norm": quantum_gradient,
                "quantum_gradient_variance": quantum_gradient**2 / 4,
                "train_loss": 0.8 / epoch,
                "validation_loss": 0.9 / epoch,
                "train_f1_macro": min(0.99, validation_f1 + 0.03 - 0.02 / epoch),
                "validation_f1_macro": validation_f1 - 0.04 / epoch,
                "train_accuracy": min(0.99, validation_f1 + 0.03 - 0.02 / epoch),
                "validation_accuracy": validation_f1 - 0.04 / epoch,
            }
        )

    runtime = {
        "best_epoch": 8,
        "last_epoch": 8,
        "stopped_early": False,
        "training_seconds": float(sum(row["elapsed_seconds"] for row in history)),
        "inference_seconds": 0.2 + qubits_or_zero(qubits) / 20,
        "total_parameters": 1000 + qubits_or_zero(qubits),
        "trainable_parameters": 1000 + qubits_or_zero(qubits),
    }
    per_class = {
        "elliptical": {"f1": test_f1 + 0.04, "precision": test_f1 + 0.03, "recall": test_f1 + 0.05, "support": 30},
        "spiral": {"f1": test_f1 + 0.02, "precision": test_f1 + 0.01, "recall": test_f1 + 0.03, "support": 30},
    }
    if task == "three_class":
        per_class["merger"] = {
            "f1": max(0.0, test_f1 - 0.12),
            "precision": max(0.0, test_f1 - 0.10),
            "recall": max(0.0, test_f1 - 0.14),
            "support": 30,
        }

    def metric_payload(metric_f1: float):
        return {
            "accuracy": metric_f1 + 0.002,
            "balanced_accuracy": metric_f1 + 0.001,
            "f1": {"macro": metric_f1, "micro": metric_f1 + 0.001, "weighted": metric_f1},
            "precision": {"macro": metric_f1 + 0.004},
            "recall": {"macro": metric_f1 + 0.001},
            "log_loss": 1.0 - metric_f1,
            "brier_score": (1.0 - metric_f1) / 2,
            "ece": 0.03 + seed / 1000,
            "per_class": per_class,
            "runtime": runtime,
            "n_samples": 90 if task == "three_class" else 60,
        }

    config = {
        "task": task,
        "model": model,
        "n_qubits": qubits,
        "n_layers": 2,
        "quantum_frozen": model == "QF",
        "repeat_seed": seed,
        "fold": 1,
        "iteration": 1,
        "subset_size": subset,
        "config_fingerprint": f"{task}-{model}-{qubits}-{subset}-{seed}",
        "runtime": {"requested_device": "cuda", "effective_device": "cuda", "pennylane_backend": "default.qubit" if model in {"Q", "QF"} else None},
        "environment": {"python": "3.13.15", "packages": {"numpy": "2.2.6"}},
        "training_config": {"max_epochs": 8, "patience": 3, "gradient_clip_norm": 1.0},
    }
    return config, metric_payload(validation_f1), metric_payload(test_f1), history


def qubits_or_zero(qubits: int | None) -> int:
    return 0 if qubits is None else qubits


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _write_run(archive: tarfile.TarFile, task: str, model: str, qubits: int | None, subset: str, seed: int) -> None:
    model_dir = model.lower() if qubits is None else f"{model.lower()}_q{qubits}"
    root = f"results/runs/{task}/{model_dir}/seed_{seed}/iteration_1/config_{seed:02d}{subset}"
    config, metrics, test_metrics, history = _run_payload(task, model, qubits, subset, seed)
    _add_bytes(archive, f"{root}/run_config.json", json.dumps(config).encode())
    _add_bytes(archive, f"{root}/metrics.json", json.dumps(metrics).encode())
    _add_bytes(archive, f"{root}/test_metrics.json", json.dumps(test_metrics).encode())
    history_payload = "\n".join(json.dumps(row) for row in history).encode()
    _add_bytes(archive, f"{root}/history.jsonl", history_payload)


def _make_archives(tmp_path: Path) -> tuple[Path, Path]:
    full = tmp_path / "colab_state.tar.gz"
    with tarfile.open(full, "w:gz") as archive:
        for task in ("binary", "three_class"):
            for model, qubits in CONFIGURATIONS:
                for seed in range(10):
                    _write_run(archive, task, model, qubits, "full", seed)

    subsets = tmp_path / "colab_subset_state.tar.gz"
    with tarfile.open(subsets, "w:gz") as archive:
        for subset in ("100", "300", "900"):
            for task in ("binary", "three_class"):
                for model, qubits in CONFIGURATIONS:
                    for seed in range(10):
                        _write_run(archive, task, model, qubits, subset, seed)
    return full, subsets


def test_audit_finds_planted_anomalies_without_flagging_expected_qf_zero_gradients(tmp_path: Path, monkeypatch) -> None:
    """Catches anomaly rules that miss real cases or confuse frozen gradients with failures."""
    full_archive, subset_archive = _make_archives(tmp_path)
    output = tmp_path / "run_anomaly_audit"
    monkeypatch.setenv("TESI_FULL_ARCHIVE", str(full_archive))
    monkeypatch.setenv("TESI_SUBSET_ARCHIVE", str(subset_archive))
    monkeypatch.setenv("TESI_ANOMALY_OUTPUT", str(output))
    monkeypatch.setenv("TESI_ANALYSIS_FAST", "1")
    monkeypatch.setenv("TESI_ANALYSIS_ZIP", "0")

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    original_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"cell-{index}", "exec"), namespace)
    finally:
        os.chdir(original_cwd)

    run_summary = pd.read_csv(output / "tables" / "run_summary.csv")
    flags = pd.read_csv(output / "tables" / "flags.csv")
    assert len(run_summary) == 640
    assert ((flags["category"] == "performance") & flags["run_path"].str.contains("three_class/q_q8/seed_9") & flags["run_path"].str.contains("config_09100")).any()
    assert ((flags["category"] == "quantum_gradient") & flags["run_path"].str.contains("three_class/q_q4/seed_7") & flags["run_path"].str.contains("config_07300")).any()
    assert ((flags["category"] == "epoch_time") & flags["run_path"].str.contains("binary/b2_q4/seed_6") & flags["run_path"].str.contains("config_06300")).any()
    assert not flags.loc[flags["category"].eq("quantum_gradient"), "model"].isin(["QF", "B0", "B1", "B2"]).any()

    non_monotonic = pd.read_csv(output / "tables" / "non_monotonic_resources.csv")
    assert (non_monotonic["run_path_larger"].str.contains("binary/q_q4/seed_8") & non_monotonic["larger_regime"].eq("full")).any()

    for name in (
        "01_flag_overview",
        "02_performance_outliers",
        "03_training_dynamics",
        "04_matched_contrasts",
        "05_resource_trajectories",
        "06_case_histories",
        "07_cost_outliers",
    ):
        for suffix in (".png", ".pdf"):
            path = output / "figures" / f"{name}{suffix}"
            assert path.is_file() and path.stat().st_size > 500, path

    report = (output / "interesting_cases.md").read_text(encoding="utf-8")
    assert "Non costituiscono criteri automatici di esclusione" in report
    assert (output / "tables" / "matched_contrasts.csv").is_file()
    assert (output / "tables" / "ranked_runs.csv").is_file()
    assert (output / "manifest.json").is_file()


def test_audit_notebook_does_not_import_training_dependencies() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    forbidden = ("import torch", "import jax", "import pennylane", "scripts/train.py")
    assert not any(token in source for token in forbidden)
