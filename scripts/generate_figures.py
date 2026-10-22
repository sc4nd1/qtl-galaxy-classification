#!/usr/bin/env python3
import json

import pandas as pd

from tesi_qml.cli import base_parser, dry_run, load_cli_config
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


def main() -> None:
    parser = base_parser(
        "Genera figure PNG/PDF dagli artefatti salvati", default_config="configs/training.yaml"
    )
    args = parser.parse_args()
    root, _ = load_cli_config(args)
    if dry_run(args, "generazione figure per ogni run"):
        return
    count = 0
    for metrics_path in (root / "results").rglob("test_metrics.json"):
        run_dir = metrics_path.parent
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config_path = run_dir / "run_config.json"
        run_config = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        )
        class_names = tuple(
            run_config.get("class_names", ("elliptical", "merger", "spiral"))
        )
        figure_dir = run_dir / "figures"
        confusion = plot_confusion_matrix(
            metrics["confusion_matrix"]["raw"], class_names
        )
        save_figure(confusion, figure_dir / "confusion_matrix")
        save_figure(
            plot_confusion_matrix(
                metrics["confusion_matrix"]["normalized"],
                class_names,
                normalized=True,
                title="Confusion matrix normalizzata",
            ),
            figure_dir / "confusion_matrix_normalized",
        )
        predictions_path = run_dir / "test_predictions.parquet"
        if predictions_path.is_file():
            predictions = pd.read_parquet(predictions_path)
            probability_columns = [f"prob_{name}_model" for name in class_names]
            if set(probability_columns) <= set(predictions.columns):
                probabilities = predictions[probability_columns].to_numpy()
                truth = predictions["class_index"].to_numpy()
                save_figure(
                    plot_roc_pr(truth, probabilities, class_names),
                    figure_dir / "roc_pr",
                )
                save_figure(plot_calibration(truth, probabilities), figure_dir / "calibration")
            for column in ("redshift", "prob_elliptical", "prob_merger", "prob_spiral"):
                if column in predictions:
                    save_figure(
                        plot_error_by_metadata(predictions, column),
                        figure_dir / f"error_vs_{column}",
                    )
            if {"path", "correct", "true_class", "predicted_class", "confidence"} <= set(
                predictions.columns
            ):
                save_figure(
                    plot_prediction_examples(predictions, project_root=root),
                    figure_dir / "prediction_examples",
                )
        history_path = run_dir / "history.jsonl"
        if history_path.is_file():
            history = pd.read_json(history_path, lines=True)
            save_figure(plot_learning_curves(history), figure_dir / "learning_curves")
            if {"gradient_norm", "quantum_gradient_norm"} <= set(history.columns):
                save_figure(plot_gradient_norms(history), figure_dir / "gradient_norms")
        count += 1
    if not count:
        raise RuntimeError("Nessun risultato da visualizzare")
    aggregate_path = root / "results" / "aggregate" / "runs.parquet"
    if aggregate_path.is_file():
        runs = pd.read_parquet(aggregate_path)
        thesis_dir = root / "results" / "thesis_figures"
        save_figure(plot_architecture_schematic(), thesis_dir / "architecture_b2_qf_q")
        if {"task", "model", "n_qubits", "repeat_seed", "f1_macro"} <= set(runs.columns):
            save_figure(
                plot_task_model_macro_f1(runs), thesis_dir / "macro_f1_by_model_and_task"
            )
            if {"B2", "QF", "Q"} <= set(runs["model"]):
                save_figure(
                    plot_paired_differences(runs), thesis_dir / "paired_q_minus_controls"
                )
        if {"model", "subset_size", "f1_macro"} <= set(runs.columns):
            numeric_subset = runs.loc[runs["subset_size"].astype(str) != "full"].copy()
            if not numeric_subset.empty:
                numeric_subset["subset_size"] = numeric_subset["subset_size"].astype(int)
                save_figure(
                    plot_subset_performance(numeric_subset),
                    thesis_dir / "performance_vs_subset",
                )
        if {"model", "n_qubits", "f1_macro"} <= set(runs.columns):
            save_figure(plot_model_comparison(runs), thesis_dir / "comparison_4_vs_8_qubits")
        merger_columns = {
            "task",
            "model",
            "n_qubits",
            "merger_precision",
            "merger_recall",
            "merger_f1",
        }
        if merger_columns <= set(runs.columns):
            save_figure(plot_merger_metrics(runs), thesis_dir / "merger_metrics")
        if {"model", "accuracy", "training_seconds", "trainable_parameters"} <= set(runs.columns):
            save_figure(plot_accuracy_cost(runs), thesis_dir / "accuracy_vs_cost")
    print({"runs": count})


if __name__ == "__main__":
    main()
