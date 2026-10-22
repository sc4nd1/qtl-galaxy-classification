"""Metriche multiclasse complete e JSON-serializzabili."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _optional_binary_score(y_binary: np.ndarray, score: np.ndarray, metric) -> float | None:
    if np.unique(y_binary).size < 2:
        return None
    return float(metric(y_binary, score))


def _expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for index in range(n_bins):
        if index == 0:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return result


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    n_bins: int = 15,
) -> dict[str, object]:
    """Calcola metriche discriminative, probabilistiche e di calibrazione."""
    truth = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    n_classes = len(class_names)
    if probs.shape != (len(truth), n_classes):
        raise ValueError("Shape delle probabilità incompatibile con campioni e classi")
    if len(truth) == 0:
        raise ValueError("Non è possibile valutare un insieme vuoto")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Le probabilità devono essere finite e comprese tra zero e uno")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("La somma delle probabilità per riga deve essere uno")
    # Softmax calcolate in float32 possono conservare, dopo il cast a float64,
    # un errore di somma maggiore della tolleranza interna di scikit-learn.
    probs = probs / probs.sum(axis=1, keepdims=True)
    if n_bins < 2:
        raise ValueError("n_bins deve essere almeno due")

    labels = np.arange(n_classes)
    predicted = probs.argmax(axis=1)
    raw_cm = confusion_matrix(truth, predicted, labels=labels)
    row_sum = raw_cm.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(
        raw_cm,
        row_sum,
        out=np.zeros_like(raw_cm, dtype=np.float64),
        where=row_sum != 0,
    )

    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, average=None, zero_division=0
    )
    per_class = {
        str(name): {
            "precision": float(per_precision[index]),
            "recall": float(per_recall[index]),
            "f1": float(per_f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }

    aggregates: dict[str, dict[str, float]] = {key: {} for key in ("precision", "recall", "f1")}
    for average in ("micro", "macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average=average, zero_division=0
        )
        aggregates["precision"][average] = float(precision)
        aggregates["recall"][average] = float(recall)
        aggregates["f1"][average] = float(f1)

    roc_per_class: dict[str, float | None] = {}
    pr_per_class: dict[str, float | None] = {}
    for index, name in enumerate(class_names):
        binary = (truth == index).astype(np.int64)
        roc_per_class[str(name)] = _optional_binary_score(binary, probs[:, index], roc_auc_score)
        pr_per_class[str(name)] = _optional_binary_score(
            binary, probs[:, index], average_precision_score
        )
    available_roc = [value for value in roc_per_class.values() if value is not None]
    available_pr = [value for value in pr_per_class.values() if value is not None]

    one_hot = np.eye(n_classes, dtype=np.float64)[truth]
    clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        **aggregates,
        "per_class": per_class,
        "confusion_matrix": {
            "raw": raw_cm.astype(int).tolist(),
            "normalized": normalized_cm.tolist(),
        },
        "roc_auc_ovr": {
            "macro": float(np.mean(available_roc)) if available_roc else None,
            "per_class": roc_per_class,
        },
        "pr_auc": {
            "macro": float(np.mean(available_pr)) if available_pr else None,
            "per_class": pr_per_class,
        },
        "log_loss": float(log_loss(truth, clipped, labels=labels)),
        "brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece": _expected_calibration_error(truth, probs, n_bins),
        "n_samples": len(truth),
    }
