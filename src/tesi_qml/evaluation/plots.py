"""Figure coerenti, vettoriali e pronte per la tesi."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve, roc_curve

COLORS = ("#3366CC", "#DC3912", "#109618", "#990099", "#FF9900")


def _axes(rows: int = 1, columns: int = 1, *, width: float = 6.4, height: float = 4.2):
    return plt.subplots(
        rows, columns, figsize=(width * columns, height * rows), constrained_layout=True
    )


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    *,
    normalized: bool = False,
    title: str | None = None,
) -> Figure:
    values = np.asarray(matrix)
    if values.shape != (len(class_names), len(class_names)):
        raise ValueError("Dimensione della confusion matrix incompatibile con le classi")
    if normalized:
        sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, sums, out=np.zeros_like(values, dtype=float), where=sums != 0)
    figure, axis = _axes()
    image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    threshold = float(values.max()) / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            text = f"{values[row, column]:.2f}" if normalized else f"{int(values[row, column])}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
            )
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=30, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Classe predetta")
    axis.set_ylabel("Classe reale")
    axis.set_title(title or ("Confusion matrix normalizzata" if normalized else "Confusion matrix"))
    figure.colorbar(image, ax=axis, shrink=0.85)
    return figure


def plot_learning_curves(history: pd.DataFrame) -> Figure:
    required = {"epoch", "train_loss", "validation_loss"}
    if missing := required - set(history.columns):
        raise ValueError(f"Colonne history mancanti: {sorted(missing)}")
    figure, axes = _axes(columns=2, width=5.2)
    axes[0].plot(history["epoch"], history["train_loss"], label="train", color=COLORS[0])
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation", color=COLORS[1])
    axes[0].set(xlabel="Epoca", ylabel="Loss", title="Loss")
    axes[0].legend()
    if {"train_accuracy", "validation_accuracy"} <= set(history.columns):
        axes[1].plot(history["epoch"], history["train_accuracy"], label="train", color=COLORS[0])
        axes[1].plot(
            history["epoch"], history["validation_accuracy"], label="validation", color=COLORS[1]
        )
        axes[1].set(xlabel="Epoca", ylabel="Accuracy", title="Accuracy", ylim=(0, 1))
        axes[1].legend()
    else:
        axes[1].set_visible(False)
    return figure


def plot_subset_performance(
    frame: pd.DataFrame,
    *,
    metric: str = "f1_macro",
) -> Figure:
    required = {"model", "subset_size", metric}
    if missing := required - set(frame.columns):
        raise ValueError(f"Colonne subset mancanti: {sorted(missing)}")
    figure, axis = _axes()
    for index, (model, group) in enumerate(frame.groupby("model", sort=True)):
        summary = group.groupby("subset_size")[metric].agg(["mean", "std"]).reset_index()
        axis.plot(
            summary["subset_size"],
            summary["mean"],
            marker="o",
            label=model,
            color=COLORS[index % len(COLORS)],
        )
        if summary["std"].notna().any():
            std = summary["std"].fillna(0)
            axis.fill_between(
                summary["subset_size"],
                summary["mean"] - std,
                summary["mean"] + std,
                alpha=0.15,
                color=COLORS[index % len(COLORS)],
            )
    axis.set(
        xlabel="Campioni di training",
        ylabel=metric.replace("_", " ").title(),
        title="Prestazioni per dimensione del training set",
    )
    axis.legend(title="Modello")
    return figure


def plot_roc_pr(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> Figure:
    truth = np.asarray(y_true)
    probs = np.asarray(probabilities)
    if probs.shape != (len(truth), len(class_names)):
        raise ValueError("Shape delle probabilità incompatibile")
    figure, axes = _axes(columns=2, width=5.2)
    for index, name in enumerate(class_names):
        binary = (truth == index).astype(int)
        if np.unique(binary).size < 2:
            continue
        false_positive, true_positive, _ = roc_curve(binary, probs[:, index])
        precision, recall, _ = precision_recall_curve(binary, probs[:, index])
        axes[0].plot(
            false_positive,
            true_positive,
            label=f"{name} (AUC={auc(false_positive, true_positive):.3f})",
            color=COLORS[index % len(COLORS)],
        )
        axes[1].plot(
            recall,
            precision,
            label=f"{name} (AUC={auc(recall, precision):.3f})",
            color=COLORS[index % len(COLORS)],
        )
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC one-vs-rest")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall one-vs-rest")
    for axis in axes:
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.legend(fontsize="small")
    return figure


def plot_calibration(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> Figure:
    truth = np.asarray(y_true)
    probs = np.asarray(probabilities)
    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predicted == truth
    edges = np.linspace(0, 1, n_bins + 1)
    centers: list[float] = []
    accuracies: list[float] = []
    counts: list[int] = []
    for left, right in pairwise(edges):
        mask = (
            (confidence >= left) & (confidence <= right)
            if left == 0
            else (confidence > left) & (confidence <= right)
        )
        if mask.any():
            centers.append(float(confidence[mask].mean()))
            accuracies.append(float(correct[mask].mean()))
            counts.append(int(mask.sum()))
    figure, axes = _axes(columns=2, width=5.2)
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", label="calibrazione perfetta")
    axes[0].plot(centers, accuracies, marker="o", color=COLORS[0], label="modello")
    axes[0].set(
        xlabel="Confidenza media",
        ylabel="Frazione corretta",
        title="Diagramma di affidabilità",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend()
    axes[1].bar(centers, counts, width=0.8 / n_bins, color=COLORS[0], alpha=0.8)
    axes[1].set(
        xlabel="Confidenza", ylabel="Campioni", title="Distribuzione della confidenza", xlim=(0, 1)
    )
    return figure


def plot_gradient_norms(history: pd.DataFrame) -> Figure:
    required = {"epoch", "gradient_norm", "quantum_gradient_norm"}
    if missing := required - set(history.columns):
        raise ValueError(f"Colonne gradienti mancanti: {sorted(missing)}")
    figure, axis = _axes()
    axis.plot(history["epoch"], history["gradient_norm"], label="totale", color=COLORS[0])
    axis.plot(
        history["epoch"],
        history["quantum_gradient_norm"],
        label="quantistico",
        color=COLORS[1],
    )
    axis.set(xlabel="Epoca", ylabel="Norma L2", title="Stabilità dei gradienti")
    if (history[["gradient_norm", "quantum_gradient_norm"]] > 0).all().all():
        axis.set_yscale("log")
    axis.legend()
    return figure


def plot_error_by_metadata(
    predictions: pd.DataFrame,
    column: str,
    *,
    n_bins: int = 6,
) -> Figure:
    if column not in predictions or "correct" not in predictions:
        raise ValueError(f"Servono le colonne {column} e correct")
    values = pd.to_numeric(predictions[column], errors="coerce")
    valid = predictions.loc[values.notna(), ["correct"]].copy()
    valid[column] = values.loc[values.notna()].astype(float)
    if valid.empty:
        raise ValueError(f"Nessun valore numerico disponibile per {column}")
    bins = pd.qcut(valid[column], q=min(n_bins, valid[column].nunique()), duplicates="drop")
    summary = (
        valid.assign(bin=bins, error=~valid["correct"].astype(bool))
        .groupby("bin", observed=True)
        .agg(center=(column, "mean"), error_rate=("error", "mean"), n=("error", "size"))
    )
    figure, axis = _axes()
    axis.plot(summary["center"], summary["error_rate"], marker="o", color=COLORS[1])
    for _, row in summary.iterrows():
        axis.annotate(
            f"n={int(row['n'])}",
            (row["center"], row["error_rate"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize="small",
        )
    axis.set(
        xlabel=column.replace("_", " ").title(),
        ylabel="Tasso di errore",
        title=f"Errori rispetto a {column.replace('_', ' ')}",
        ylim=(0, 1),
    )
    return figure


def plot_model_comparison(
    frame: pd.DataFrame,
    *,
    metric: str = "f1_macro",
) -> Figure:
    required = {"model", "n_qubits", metric}
    if missing := required - set(frame.columns):
        raise ValueError(f"Colonne confronto mancanti: {sorted(missing)}")
    figure, axis = _axes()
    valid = frame.dropna(subset=["n_qubits", metric])
    for index, (model, group) in enumerate(valid.groupby("model", sort=True)):
        summary = group.groupby("n_qubits")[metric].agg(["mean", "std"]).reset_index()
        axis.errorbar(
            summary["n_qubits"],
            summary["mean"],
            yerr=summary["std"].fillna(0),
            marker="o",
            capsize=3,
            label=model,
            color=COLORS[index % len(COLORS)],
        )
    axis.set(
        xlabel="Qubit / componenti",
        ylabel=metric.replace("_", " ").title(),
        title="Confronto 4 vs 8 qubit",
        xticks=sorted(valid["n_qubits"].unique()),
    )
    axis.legend(title="Modello")
    return figure


def plot_accuracy_cost(frame: pd.DataFrame) -> Figure:
    required = {"model", "accuracy", "training_seconds", "trainable_parameters"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Colonne costo mancanti: {sorted(missing)}")
    valid = frame.dropna(subset=["accuracy", "training_seconds", "trainable_parameters"])
    figure, axes = _axes(columns=2, width=5.2)
    for index, (model, group) in enumerate(valid.groupby("model", sort=True)):
        color = COLORS[index % len(COLORS)]
        axes[0].scatter(group["training_seconds"], group["accuracy"], label=model, color=color)
        axes[1].scatter(group["trainable_parameters"], group["accuracy"], label=model, color=color)
    axes[0].set(xlabel="Tempo di training [s]", ylabel="Accuracy", title="Accuratezza vs tempo")
    axes[1].set(xlabel="Parametri addestrabili", ylabel="Accuracy", title="Accuratezza vs capacità")
    if (valid["training_seconds"] > 0).all():
        axes[0].set_xscale("log")
    if (valid["trainable_parameters"] > 0).all():
        axes[1].set_xscale("log")
    for axis in axes:
        axis.set_ylim(0, 1)
        axis.legend(title="Modello", fontsize="small")
    return figure


def plot_architecture_schematic() -> Figure:
    """Schema compatto del confronto ResNet18 → B2/QF/Q."""
    figure, axis = _axes(width=8.0, height=3.4)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 4)
    axis.axis("off")
    boxes = (
        (1.2, 2.0, "ResNet18 congelata\n512 feature"),
        (4.0, 2.0, "Linear 512→q\ntanh(x)·π"),
        (6.8, 3.1, "B2\nLinear q→C"),
        (6.8, 2.0, "QF\nCircuito congelato"),
        (6.8, 0.9, "Q\nCircuito addestrabile"),
        (9.1, 2.0, "C logits"),
    )
    for x, y, label in boxes:
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": COLORS[0]},
        )
    axis.annotate("", xy=(3.0, 2.0), xytext=(2.2, 2.0), arrowprops={"arrowstyle": "->"})
    for y in (3.1, 2.0, 0.9):
        axis.annotate("", xy=(5.9, y), xytext=(5.0, 2.0), arrowprops={"arrowstyle": "->"})
        axis.annotate("", xy=(8.25, 2.0), xytext=(7.75, y), arrowprops={"arrowstyle": "->"})
    axis.set_title("Architetture del confronto a bottleneck appaiato")
    return figure


def _model_label(model: object, qubits: object) -> str:
    return str(model) if pd.isna(qubits) else f"{model}-{int(qubits)}"


def plot_task_model_macro_f1(frame: pd.DataFrame) -> Figure:
    required = {"task", "model", "n_qubits", "repeat_seed", "f1_macro"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Colonne macro-F1 mancanti: {sorted(missing)}")
    values = frame.dropna(subset=["task", "model", "f1_macro"]).copy()
    values["model_label"] = [
        _model_label(model, qubits)
        for model, qubits in zip(values["model"], values["n_qubits"])
    ]
    summary = (
        values.groupby(["model_label", "task"], sort=True)["f1_macro"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    models = sorted(summary["model_label"].unique())
    tasks = [task for task in ("binary", "three_class") if task in set(summary["task"])]
    figure, axis = _axes(width=8.0)
    width = 0.8 / max(len(tasks), 1)
    x = np.arange(len(models), dtype=float)
    for index, task in enumerate(tasks):
        task_rows = summary.loc[summary["task"] == task].set_index("model_label").reindex(models)
        means = task_rows["mean"].to_numpy(dtype=float)
        errors = (
            1.96
            * task_rows["std"].fillna(0).to_numpy(dtype=float)
            / np.sqrt(task_rows["count"].clip(lower=1).to_numpy(dtype=float))
        )
        axis.bar(
            x + (index - (len(tasks) - 1) / 2) * width,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            label=task,
            color=COLORS[index],
        )
    axis.set(
        xticks=x,
        xticklabels=models,
        ylabel="Macro-F1",
        title="Macro-F1 per modello e task (CI 95%)",
        ylim=(0, 1),
    )
    axis.tick_params(axis="x", rotation=35)
    axis.legend(title="Task")
    return figure


def plot_paired_differences(frame: pd.DataFrame) -> Figure:
    required = {"task", "model", "n_qubits", "repeat_seed", "f1_macro"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Colonne differenze appaiate mancanti: {sorted(missing)}")
    selected = frame.loc[frame["model"].isin(["B2", "QF", "Q"])]
    if selected.duplicated(["task", "n_qubits", "repeat_seed", "model"]).any():
        raise ValueError("Run duplicati nelle differenze appaiate")
    pivot = selected.pivot(
        index=["task", "n_qubits", "repeat_seed"], columns="model", values="f1_macro"
    ).dropna(subset=["B2", "QF", "Q"])
    if pivot.empty:
        raise ValueError("Nessuna terna B2/QF/Q completa")
    differences = pd.DataFrame(
        {"Q − B2": pivot["Q"] - pivot["B2"], "Q − QF": pivot["Q"] - pivot["QF"]}
    ).reset_index()
    figure, axis = _axes(width=8.0)
    positions = np.arange(len(differences), dtype=float)
    for index, column in enumerate(("Q − B2", "Q − QF")):
        axis.scatter(
            positions + (index - 0.5) * 0.18,
            differences[column],
            label=column,
            color=COLORS[index],
        )
    labels = [
        f"{row.task}/q{int(row.n_qubits)}/s{row.repeat_seed}"
        for row in differences.itertuples(index=False)
    ]
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        xticks=positions,
        xticklabels=labels,
        ylabel="Differenza macro-F1",
        title="Differenze appaiate per repeat seed",
    )
    axis.tick_params(axis="x", rotation=55, labelsize="small")
    axis.legend()
    return figure


def plot_merger_metrics(frame: pd.DataFrame) -> Figure:
    metrics = ("merger_precision", "merger_recall", "merger_f1")
    required = {"task", "model", "n_qubits", *metrics}
    if missing := required - set(frame.columns):
        raise ValueError(f"Colonne merger mancanti: {sorted(missing)}")
    values = frame.loc[frame["task"] == "three_class"].copy()
    values["model_label"] = [
        _model_label(model, qubits)
        for model, qubits in zip(values["model"], values["n_qubits"])
    ]
    summary = values.groupby("model_label", sort=True)[list(metrics)].mean()
    if summary.empty:
        raise ValueError("Nessun risultato three_class per merger")
    figure, axis = _axes(width=8.0)
    x = np.arange(len(summary), dtype=float)
    width = 0.25
    for index, metric in enumerate(metrics):
        axis.bar(
            x + (index - 1) * width,
            summary[metric].to_numpy(),
            width=width,
            label=metric.removeprefix("merger_").title(),
            color=COLORS[index],
        )
    axis.set(
        xticks=x,
        xticklabels=summary.index,
        ylabel="Score",
        title="Prestazioni specifiche della classe merger",
        ylim=(0, 1),
    )
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    return figure


def plot_prediction_examples(
    predictions: pd.DataFrame,
    *,
    project_root: Path | str,
    n_each: int = 4,
) -> Figure:
    required = {"path", "correct", "true_class", "predicted_class", "confidence"}
    if missing := required - set(predictions.columns):
        raise ValueError(f"Colonne esempi mancanti: {sorted(missing)}")
    if n_each < 1:
        raise ValueError("n_each deve essere positivo")
    root = Path(project_root)
    groups = (
        predictions.loc[predictions["correct"].astype(bool)].head(n_each),
        predictions.loc[~predictions["correct"].astype(bool)].head(n_each),
    )
    figure, axes = plt.subplots(
        2, n_each, figsize=(3.0 * n_each, 6.0), squeeze=False, constrained_layout=True
    )
    for row_index, (group, row_title) in enumerate(zip(groups, ("Corrette", "Errate"))):
        for column_index, axis in enumerate(axes[row_index]):
            axis.axis("off")
            if column_index >= len(group):
                continue
            record = group.iloc[column_index]
            image_path = Path(str(record["path"]))
            if not image_path.is_absolute():
                image_path = root / image_path
            with Image.open(image_path) as source:
                axis.imshow(source.convert("RGB"))
            axis.set_title(
                f"{row_title}: {record['true_class']} → {record['predicted_class']}\n"
                f"conf={float(record['confidence']):.2f}",
                fontsize="small",
            )
    return figure


def save_figure(figure: Figure, base_path: Path | str, *, dpi: int = 300) -> tuple[Path, Path]:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    figure.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf
