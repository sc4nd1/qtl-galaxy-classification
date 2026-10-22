#!/usr/bin/env python3
import json

import pandas as pd

from tesi_qml.cli import atomic_json, atomic_parquet, base_parser, dry_run, load_cli_config
from tesi_qml.config import load_yaml
from tesi_qml.evaluation.statistics import (
    aggregate_runs,
    primary_hypothesis_tests,
    validate_primary_result_matrix,
)
from tesi_qml.matrix import is_run_complete


def collect_run_records(root) -> list[dict[str, object]]:
    """Legge soltanto run completi e compatibili con il proprio run_config."""
    records: list[dict[str, object]] = []
    for path in (root / "results" / "runs").rglob("test_metrics.json"):
        if not is_run_complete(path.parent):
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metadata = json.loads((path.parent / "run_config.json").read_text(encoding="utf-8"))
        runtime = metrics.get("runtime", {})
        merger = metrics.get("per_class", {}).get("merger", {})
        records.append(
            {
                "task": metadata.get("task"),
                "model": metadata.get("model"),
                "repeat_seed": metadata.get("repeat_seed"),
                "training_seed": metadata.get("training_seed"),
                "fold": metadata.get("fold", metadata.get("iteration")),
                "n_qubits": metadata.get("n_qubits"),
                "n_layers": metadata.get("n_layers"),
                "quantum_frozen": metadata.get("quantum_frozen", False),
                "subset_size": metadata.get("subset_size", "full"),
                "experiment_fingerprint": metadata.get("experiment_fingerprint"),
                "config_fingerprint": metadata.get("config_fingerprint"),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision_macro": metrics["precision"]["macro"],
                "recall_macro": metrics["recall"]["macro"],
                "f1_macro": metrics["f1"]["macro"],
                "roc_auc_macro": metrics["roc_auc_ovr"]["macro"],
                "pr_auc_macro": metrics["pr_auc"]["macro"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "ece": metrics["ece"],
                "merger_precision": merger.get("precision"),
                "merger_recall": merger.get("recall"),
                "merger_f1": merger.get("f1"),
                "training_seconds": runtime.get("training_seconds"),
                "inference_seconds": runtime.get("inference_seconds"),
                "total_parameters": runtime.get("total_parameters"),
                "trainable_parameters": runtime.get("trainable_parameters"),
                "quantum_parameters": runtime.get("quantum_parameters"),
                "run_dir": str(path.parent.relative_to(root)),
            }
        )
    return records


def main() -> None:
    parser = base_parser(
        "Aggrega i test set di tutti i run", default_config="configs/training.yaml"
    )
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root, _ = load_cli_config(args)
    if dry_run(args, "aggregazione dei test_metrics.json"):
        return
    records = collect_run_records(root)
    if not records:
        raise RuntimeError("Nessun test_metrics.json trovato")
    frame = pd.DataFrame(records)
    if frame["experiment_fingerprint"].isna().any():
        raise ValueError("Ogni run deve avere experiment_fingerprint in run_config.json")
    dataset_path = root / args.dataset_config
    dataset_config = load_yaml(dataset_path)
    primary_frame = frame.loc[frame["subset_size"].astype(str) == "full"].copy()
    if not args.allow_incomplete:
        validate_primary_result_matrix(
            primary_frame, repeat_seeds=dataset_config["splits"]["repeat_seeds"]
        )
    group_columns = (
        "task",
        "model",
        "n_qubits",
        "n_layers",
        "subset_size",
        "training_seed",
        "experiment_fingerprint",
    )
    summary = aggregate_runs(
        frame,
        metrics=(
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "roc_auc_macro",
            "pr_auc_macro",
            "log_loss",
            "brier_score",
            "ece",
        ),
        group_columns=group_columns,
    )
    output = root / "results" / "aggregate"
    atomic_parquet(output / "runs.parquet", frame)
    atomic_parquet(output / "summary.parquet", summary)
    atomic_json(output / "summary.json", summary.to_dict("records"))
    atomic_parquet(output / "binary.parquet", frame.loc[frame["task"] == "binary"])
    atomic_parquet(output / "three_class.parquet", frame.loc[frame["task"] == "three_class"])
    merger_columns = [
        "model",
        "n_qubits",
        "repeat_seed",
        "fold",
        "merger_precision",
        "merger_recall",
        "merger_f1",
    ]
    atomic_parquet(
        output / "merger.parquet",
        frame.loc[frame["task"] == "three_class", merger_columns],
    )
    if not args.allow_incomplete:
        primary = primary_hypothesis_tests(primary_frame, seed=dataset_config["project_seed"])
        atomic_parquet(output / "primary_hypotheses.parquet", primary)
        atomic_json(output / "primary_hypotheses.json", primary.to_dict("records"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
