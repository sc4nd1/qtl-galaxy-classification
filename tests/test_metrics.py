import math
import warnings

import numpy as np

from tesi_qml.training.metrics import compute_classification_metrics


def test_float32_softmax_roundoff_does_not_emit_probability_warning():
    probabilities = np.array(
        [
            [0.19160286, 0.24508366, 0.56331342],
            [0.15203156, 0.62482518, 0.22314322],
            [0.18565808, 0.68617254, 0.12816945],
        ],
        dtype=np.float32,
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = compute_classification_metrics(
            np.array([0, 1, 2]), probabilities, ("elliptical", "spiral", "merger")
        )
    assert np.isfinite(metrics["log_loss"])


CLASSES = ["elliptical", "merger", "spiral"]


def test_perfect_multiclass_predictions_produce_complete_metrics() -> None:
    """Catches missing per-class, ranking or calibration metrics in final artifacts."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.array(
        [
            [0.999, 0.0005, 0.0005],
            [0.0005, 0.999, 0.0005],
            [0.0005, 0.0005, 0.999],
            [0.999, 0.0005, 0.0005],
            [0.0005, 0.999, 0.0005],
            [0.0005, 0.0005, 0.999],
        ]
    )

    metrics = compute_classification_metrics(y_true, probabilities, CLASSES, n_bins=10)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["f1"]["macro"] == 1.0
    assert metrics["roc_auc_ovr"]["macro"] == 1.0
    assert metrics["pr_auc"]["macro"] == 1.0
    assert metrics["log_loss"] < 0.002
    assert metrics["brier_score"] < 0.00001
    assert math.isfinite(metrics["ece"])
    assert np.asarray(metrics["confusion_matrix"]["raw"]).shape == (3, 3)
    assert np.asarray(metrics["confusion_matrix"]["normalized"]).shape == (3, 3)
    assert set(metrics["per_class"]) == set(CLASSES)


def test_metrics_remain_serializable_when_a_class_is_absent() -> None:
    """Catches crashes on tiny stratified subsets with an absent evaluation class."""
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1]])

    metrics = compute_classification_metrics(y_true, probabilities, CLASSES)

    assert metrics["per_class"]["spiral"]["support"] == 0
    assert metrics["roc_auc_ovr"]["per_class"]["spiral"] is None
    assert metrics["pr_auc"]["per_class"]["spiral"] is None


def test_metrics_reject_invalid_probability_rows() -> None:
    """Catches logits or unnormalised scores being mislabeled as probabilities."""
    y_true = np.array([0, 1])
    invalid = np.array([[0.9, 0.9, 0.0], [0.1, 0.8, 0.1]])
    try:
        compute_classification_metrics(y_true, invalid, CLASSES)
    except ValueError as exc:
        assert "somma" in str(exc)
    else:
        raise AssertionError("invalid probabilities accepted")
