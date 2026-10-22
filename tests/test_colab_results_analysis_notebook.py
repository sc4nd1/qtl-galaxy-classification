import io
import json
import os
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_colab_results_analysis.ipynb"


CONFIGURATIONS = (
    ("B0", np.nan),
    ("B1", np.nan),
    ("B2", 4),
    ("QF", 4),
    ("Q", 4),
    ("B2", 8),
    ("QF", 8),
    ("Q", 8),
)


def _synthetic_runs(subset_sizes: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    training_samples = {"100": 100, "300": 300, "900": 900, "full": 2814}
    for subset_index, subset_size in enumerate(subset_sizes):
        for task_index, task in enumerate(("binary", "three_class")):
            for model_index, (model, qubits) in enumerate(CONFIGURATIONS):
                for seed in range(10):
                    baseline = 0.93 if task == "binary" else 0.73
                    resource_gain = 0.012 * subset_index
                    model_effect = {"B0": -0.006, "B1": -0.002, "B2": 0.0, "QF": -0.008, "Q": -0.005}[model]
                    qubit_effect = 0.001 if qubits == 8 else 0.0
                    noise = (seed - 4.5) * 0.001
                    f1 = baseline + resource_gain + model_effect + qubit_effect + noise
                    merger_f1 = f1 - 0.12 if task == "three_class" else np.nan
                    rows.append(
                        {
                            "task": task,
                            "model": model,
                            "repeat_seed": seed,
                            "fold": 1,
                            "n_qubits": qubits,
                            "subset_size": subset_size,
                            "f1_macro": f1,
                            "accuracy": f1 + 0.002,
                            "balanced_accuracy": f1 + 0.001,
                            "log_loss": 1.0 - f1,
                            "brier_score": (1.0 - f1) / 2,
                            "ece": 0.02 + 0.002 * model_index + 0.0005 * seed,
                            "merger_precision": merger_f1 + 0.015 if task == "three_class" else np.nan,
                            "merger_recall": merger_f1 - 0.015 if task == "three_class" else np.nan,
                            "merger_f1": merger_f1,
                            "training_seconds": (model_index + 1) * (subset_index + 1) * 3.0 + seed / 10,
                            "inference_seconds": (model_index + 1) * 0.2 + seed / 100,
                            "training_samples": (
                                1876 if subset_size == "full" and task == "binary"
                                else training_samples[subset_size]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _hypotheses(subset_sizes: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for subset_size in subset_sizes:
        for task in ("binary", "three_class"):
            for qubits in (4, 8):
                rows.append(
                    {
                        "subset_size": subset_size,
                        "task": task,
                        "n_qubits": qubits,
                        "model_a": "Q",
                        "model_b": "B2",
                        "metric": "f1_macro",
                        "difference_definition": "Q - B2",
                        "n_pairs": 10,
                        "mean_difference": -0.005,
                        "confidence_interval_95": "[-0.01, 0.001]",
                        "raw_p_value": 0.04,
                        "holm_p_value": 0.08,
                    }
                )
    return pd.DataFrame(rows)


def _write_parquet_member(archive: tarfile.TarFile, name: str, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    payload = buffer.getvalue()
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _make_archives(tmp_path: Path) -> tuple[Path, Path]:
    full_archive = tmp_path / "colab_state.tar.gz"
    with tarfile.open(full_archive, "w:gz") as archive:
        _write_parquet_member(
            archive,
            "results/aggregate/runs.parquet",
            _synthetic_runs(("full",)).drop(columns="training_samples"),
        )
        _write_parquet_member(
            archive,
            "results/aggregate/primary_hypotheses.parquet",
            _hypotheses(("full",)).drop(columns="subset_size"),
        )

    subset_archive = tmp_path / "colab_subset_state.tar.gz"
    with tarfile.open(subset_archive, "w:gz") as archive:
        _write_parquet_member(
            archive,
            "results/aggregate/runs_with_training_size.parquet",
            _synthetic_runs(("100", "300", "900")),
        )
        _write_parquet_member(
            archive,
            "results/aggregate/secondary_subset_hypotheses.parquet",
            _hypotheses(("100", "300", "900")),
        )
    return full_archive, subset_archive


def test_notebook_reads_archives_and_exports_every_analysis_block(tmp_path: Path, monkeypatch) -> None:
    """Catches archive loading or a requested analysis block disappearing."""
    full_archive, subset_archive = _make_archives(tmp_path)
    output_dir = tmp_path / "final_analysis"
    monkeypatch.setenv("TESI_FULL_ARCHIVE", str(full_archive))
    monkeypatch.setenv("TESI_SUBSET_ARCHIVE", str(subset_archive))
    monkeypatch.setenv("TESI_ANALYSIS_OUTPUT", str(output_dir))
    monkeypatch.setenv("TESI_ANALYSIS_FAST", "1")
    monkeypatch.setenv("TESI_ANALYSIS_ZIP", "0")

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    original_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            exec(compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(original_cwd)

    expected_figures = {
        "01a_absolute_100",
        "01b_absolute_300",
        "01c_absolute_900",
        "01d_absolute_full",
        "01e_resource_gap_vs_full",
        "02_q_minus_b2_paired",
        "03_learning_curves",
        "04a_binary_to_three_class",
        "04b_relative_quantum_penalty",
        "04c_merger_detailed",
        "04d_q_minus_qf",
        "04e_calibration",
        "05_cost_ratios",
        "appendix_a_qubit_sensitivity",
    }
    for figure_name in expected_figures:
        for suffix in (".png", ".pdf"):
            path = output_dir / "figures" / f"{figure_name}{suffix}"
            assert path.is_file() and path.stat().st_size > 500, path

    assert (output_dir / "tables" / "absolute_performance.csv").is_file()
    assert (output_dir / "tables" / "q_minus_b2_paired.csv").is_file()
    assert (output_dir / "tables" / "learning_curves.csv").is_file()
    assert (output_dir / "tables" / "binary_to_three_class.csv").is_file()
    assert (output_dir / "tables" / "cost_ratios.csv").is_file()
    assert (output_dir / "captions.md").is_file()
    assert (output_dir / "manifest.json").is_file()


def test_notebook_uses_only_data_analysis_dependencies() -> None:
    """Catches the analysis notebook importing the training/quantum stack."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    forbidden = ("import torch", "import jax", "import pennylane", "scripts/train.py")
    assert not any(token in source for token in forbidden)
