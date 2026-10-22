from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib.use("Agg")

from tesi_qml.evaluation.plots import (
    plot_accuracy_cost,
    plot_architecture_schematic,
    plot_calibration,
    plot_confusion_matrix,
    plot_error_by_metadata,
    plot_gradient_norms,
    plot_learning_curves,
    plot_merger_metrics,
    plot_model_comparison,
    plot_paired_differences,
    plot_prediction_examples,
    plot_roc_pr,
    plot_subset_performance,
    plot_task_model_macro_f1,
    save_figure,
)


@pytest.mark.parametrize(
    "name,builder",
    [
        (
            "confusion",
            lambda: plot_confusion_matrix(
                np.array([[4, 1, 0], [0, 5, 0], [1, 0, 4]]), ("elliptical", "spiral", "merger")
            ),
        ),
        (
            "learning",
            lambda: plot_learning_curves(
                pd.DataFrame(
                    {
                        "epoch": [1, 2, 3],
                        "train_loss": [1.0, 0.7, 0.5],
                        "validation_loss": [1.1, 0.8, 0.6],
                        "train_accuracy": [0.4, 0.7, 0.8],
                        "validation_accuracy": [0.3, 0.6, 0.75],
                    }
                )
            ),
        ),
        (
            "subset",
            lambda: plot_subset_performance(
                pd.DataFrame(
                    {
                        "model": ["B1", "B1", "Q4", "Q4"],
                        "subset_size": [100, 200, 100, 200],
                        "f1_macro": [0.6, 0.7, 0.62, 0.73],
                    }
                )
            ),
        ),
        (
            "roc_pr",
            lambda: plot_roc_pr(
                np.array([0, 1, 2, 0, 1, 2]),
                np.array(
                    [
                        [0.8, 0.1, 0.1],
                        [0.1, 0.8, 0.1],
                        [0.1, 0.1, 0.8],
                        [0.7, 0.2, 0.1],
                        [0.2, 0.7, 0.1],
                        [0.1, 0.2, 0.7],
                    ]
                ),
                ("elliptical", "spiral", "merger"),
            ),
        ),
        (
            "calibration",
            lambda: plot_calibration(
                np.array([0, 1, 2, 0, 1, 2]),
                np.array(
                    [
                        [0.8, 0.1, 0.1],
                        [0.1, 0.8, 0.1],
                        [0.1, 0.1, 0.8],
                        [0.7, 0.2, 0.1],
                        [0.2, 0.7, 0.1],
                        [0.1, 0.2, 0.7],
                    ]
                ),
            ),
        ),
        (
            "gradients",
            lambda: plot_gradient_norms(
                pd.DataFrame(
                    {
                        "epoch": [1, 2, 3],
                        "gradient_norm": [1.0, 0.7, 0.5],
                        "quantum_gradient_norm": [0.2, 0.15, 0.1],
                    }
                )
            ),
        ),
        (
            "redshift_errors",
            lambda: plot_error_by_metadata(
                pd.DataFrame(
                    {
                        "redshift": np.linspace(0.01, 0.09, 12),
                        "correct": [True, False] * 6,
                    }
                ),
                "redshift",
            ),
        ),
        (
            "model_comparison",
            lambda: plot_model_comparison(
                pd.DataFrame(
                    {
                        "model": ["PL", "PL", "QISKIT", "QISKIT"],
                        "n_qubits": [4, 8, 4, 8],
                        "f1_macro": [0.65, 0.68, 0.63, 0.67],
                    }
                )
            ),
        ),
        (
            "accuracy_cost",
            lambda: plot_accuracy_cost(
                pd.DataFrame(
                    {
                        "model": ["B1", "PL", "QISKIT"],
                        "accuracy": [0.7, 0.72, 0.71],
                        "training_seconds": [10.0, 120.0, 180.0],
                        "trainable_parameters": [1000, 80, 80],
                    }
                )
            ),
        ),
        ("architecture", plot_architecture_schematic),
        (
            "task_model_macro_f1",
            lambda: plot_task_model_macro_f1(
                pd.DataFrame(
                    {
                        "task": ["binary", "binary", "three_class", "three_class"],
                        "model": ["B2", "Q", "B2", "Q"],
                        "n_qubits": [4, 4, 4, 4],
                        "repeat_seed": [42, 42, 42, 42],
                        "f1_macro": [0.70, 0.72, 0.60, 0.63],
                    }
                )
            ),
        ),
        (
            "paired_differences",
            lambda: plot_paired_differences(
                pd.DataFrame(
                    [
                        {
                            "task": "binary",
                            "model": model,
                            "n_qubits": 4,
                            "repeat_seed": seed,
                            "f1_macro": value,
                        }
                        for seed, values in ((42, (0.70, 0.69, 0.73)), (314, (0.71, 0.70, 0.74)))
                        for model, value in zip(("B2", "QF", "Q"), values)
                    ]
                )
            ),
        ),
        (
            "merger_metrics",
            lambda: plot_merger_metrics(
                pd.DataFrame(
                    {
                        "task": ["three_class"] * 4,
                        "model": ["B2", "B2", "Q", "Q"],
                        "n_qubits": [4, 4, 4, 4],
                        "merger_precision": [0.55, 0.58, 0.60, 0.62],
                        "merger_recall": [0.50, 0.52, 0.56, 0.57],
                        "merger_f1": [0.52, 0.55, 0.58, 0.59],
                    }
                )
            ),
        ),
    ],
)
def test_thesis_plots_export_nonempty_png_and_pdf(tmp_path: Path, name, builder):
    figure = builder()
    paths = save_figure(figure, tmp_path / name)
    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.is_file() and path.stat().st_size > 100 for path in paths)


def test_prediction_examples_export_correct_and_error_images(tmp_path: Path):
    paths = []
    for index, color in enumerate(("red", "blue")):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (32, 32), color=color).save(path)
        paths.append(str(path))
    predictions = pd.DataFrame(
        {
            "path": paths,
            "correct": [True, False],
            "true_class": ["elliptical", "merger"],
            "predicted_class": ["elliptical", "spiral"],
            "confidence": [0.9, 0.8],
        }
    )
    figure = plot_prediction_examples(predictions, project_root=tmp_path, n_each=1)
    exported = save_figure(figure, tmp_path / "examples")
    assert all(path.stat().st_size > 100 for path in exported)


def test_save_figure_releases_the_matplotlib_figure(tmp_path: Path):
    """Catches batch figure generation retaining every run in memory."""
    figure = plot_architecture_schematic()
    figure_number = figure.number
    assert plt.fignum_exists(figure_number)

    save_figure(figure, tmp_path / "architecture")

    assert not plt.fignum_exists(figure_number)
