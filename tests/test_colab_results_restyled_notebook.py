import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "06_colab_result_analysis.ipynb"
REGIMES = ("100", "300", "900", "full")


def _write_input_tables(table_dir: Path) -> None:
    table_dir.mkdir(parents=True)
    paired_b2 = []
    paired_qf = []
    costs = []
    transitions = []
    mergers = []
    qubits = []
    tests = []
    samples = {"100": 100, "300": 300, "900": 900, "full": 2814}
    for task in ("binary", "three_class"):
        for regime_index, regime in enumerate(REGIMES):
            for n_qubits in (4, 8):
                tests.append({
                    "task": task, "subset_size": regime, "n_qubits": n_qubits,
                    "holm_p_value": 0.01 if task == "three_class" else 0.20,
                    "inference_family": "primario: full" if regime == "full" else "secondario: subset",
                })
                for seed in range(10):
                    base = (0.91 if task == "binary" else 0.70) + 0.012 * regime_index + seed * 0.0005
                    quantum_penalty = -0.004 if task == "binary" else -0.025
                    q = base + quantum_penalty - 0.001 * (n_qubits == 8)
                    qf = q - 0.006
                    common = {
                        "task": task, "subset_size": regime, "n_qubits": n_qubits,
                        "repeat_seed": seed, "fold": 1,
                    }
                    paired_b2.append({**common, "Q": q, "B2": base, "difference": q - base,
                                      "difference_pp": 100 * (q - base)})
                    paired_qf.append({**common, "Q": q, "QF": qf, "difference": q - qf,
                                      "difference_pp": 100 * (q - qf)})
                    for model, train_time in (("Q", 80 + 12 * regime_index),
                                              ("QF", 65 + 10 * regime_index)):
                        costs.append({
                            **common, "numerator_model": model, "cost_metric": "training_seconds",
                            "B2": 5 + regime_index, model: train_time + seed,
                            "ratio_to_b2": (train_time + seed) / (5 + regime_index),
                        })
                    if task == "three_class":
                        for model, value in (("B2", base - 0.10), ("Q", q - 0.14), ("QF", qf - 0.15)):
                            for metric in ("merger_precision", "merger_recall", "merger_f1"):
                                mergers.append({
                                    "n_qubits": n_qubits, "model": model, "subset_size": regime,
                                    "training_samples": samples[regime], "value_column": metric,
                                    "estimate": value, "ci95_low": value - 0.01, "ci95_high": value + 0.01,
                                })
            for seed in range(10):
                for model, difference in (("B2", -0.20), ("Q", -0.23), ("QF", -0.24)):
                    for n_qubits in (4, 8):
                        transitions.append({
                            "subset_size": regime, "model": model, "n_qubits": n_qubits,
                            "repeat_seed": seed, "fold": 1, "difference_pp": 100 * difference,
                        })
        for model, value in (("B2", 0.001), ("Q", -0.003), ("QF", -0.002)):
            for regime in REGIMES:
                for seed in range(10):
                    qubits.append({
                        "task": task, "subset_size": regime, "model": model,
                        "repeat_seed": seed, "fold": 1, "difference_pp": 100 * value,
                    })

    pd.DataFrame(paired_b2).to_csv(table_dir / "q_minus_b2_paired.csv", index=False)
    pd.DataFrame(paired_qf).to_csv(table_dir / "q_minus_qf.csv", index=False)
    pd.DataFrame(costs).to_csv(table_dir / "cost_ratios.csv", index=False)
    pd.DataFrame(transitions).to_csv(table_dir / "binary_to_three_class.csv", index=False)
    pd.DataFrame(mergers).to_csv(table_dir / "merger_detailed.csv", index=False)
    pd.DataFrame(qubits).to_csv(table_dir / "qubit_8_minus_4.csv", index=False)
    pd.DataFrame(tests).to_csv(table_dir / "q_minus_b2_holm_tests.csv", index=False)


def test_restyled_notebook_reads_csvs_and_exports_consolidated_figures(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "final_analysis"
    output_dir = tmp_path / "final_analysis_restyled"
    _write_input_tables(input_dir / "tables")
    monkeypatch.setenv("TESI_ANALYSIS_INPUT", str(input_dir))
    monkeypatch.setenv("TESI_RESTYLED_OUTPUT", str(output_dir))
    monkeypatch.setenv("TESI_ANALYSIS_FAST", "1")

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    original_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                exec(compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(original_cwd)

    expected = {
        "01_hero_learning_curves",
        "02_delta_q_minus_b2",
        "03_binary_to_three_class",
        "04_merger_detailed",
        "05_q_minus_qf",
        "06_qubit_sensitivity",
        "07_pareto_performance_cost",
    }
    for name in expected:
        for suffix in (".png", ".pdf"):
            path = output_dir / "figures" / f"{name}{suffix}"
            assert path.is_file() and path.stat().st_size > 500, path
    assert (output_dir / "captions.md").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_notebook"] == "04_colab_results_analysis.ipynb"
    assert len(manifest["figures"]) == 7


def test_restyled_notebook_is_csv_only_and_uses_editorial_palette() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for forbidden in ("tarfile", "read_parquet", "colab_state.tar.gz", "import torch", "import jax"):
        assert forbidden not in source
    for color in ("#1E3A8A", "#7C3AED", "#D97706", "#DC2626"):
        assert color in source
    assert "Parità classico–quantistico" in source
    assert "set_xticks(range(4)" in source
    assert "Differenza-in-differenze" in source
    assert "Direzione desiderabile" in source
    figure_cells = {
        "hero": next("".join(cell["source"]) for cell in notebook["cells"]
                     if "01_hero_learning_curves" in "".join(cell["source"])),
        "delta": next("".join(cell["source"]) for cell in notebook["cells"]
                      if "02_delta_q_minus_b2" in "".join(cell["source"])),
        "pareto": next("".join(cell["source"]) for cell in notebook["cells"]
                       if "07_pareto_performance_cost" in "".join(cell["source"])),
    }
    assert "set_xscale" not in figure_cells["hero"]
    assert "scatter(" not in figure_cells["hero"]
    assert "scatter(" not in figure_cells["delta"]
    assert "scatter(" not in figure_cells["pareto"]
    assert "adjust_text" in figure_cells["pareto"]
