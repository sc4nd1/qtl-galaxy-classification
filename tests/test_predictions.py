import numpy as np
import pandas as pd

from tesi_qml.evaluation.predictions import prediction_frame


def test_prediction_frame_contains_traceable_sample_outputs() -> None:
    """Catches aggregate-only evaluation that cannot support thesis plots or error audit."""
    samples = pd.DataFrame(
        [
            {
                "objid": "1237654605318717697",
                "class_index": 0,
                "redshift": 0.04,
                "prob_elliptical": 0.91,
                "prob_merger": 0.0,
                "prob_spiral": 0.04,
                "sha256": "abc",
                "path": "data/images/elliptical/x.jpg",
            },
            {
                "objid": "1237654605318717698",
                "class_index": 2,
                "redshift": 0.06,
                "prob_elliptical": 0.02,
                "prob_merger": 0.0,
                "prob_spiral": 0.95,
                "sha256": "def",
                "path": "data/images/spiral/y.jpg",
            },
        ]
    )
    logits = np.array([[4.0, 0.0, -1.0], [0.0, 2.0, 1.0]])

    result = prediction_frame(
        samples,
        logits,
        ["elliptical", "merger", "spiral"],
        {"model": "b2_4", "fold": 1, "repeat_seed": 42, "subset_size": 50},
    )

    expected = {
        "objid",
        "true_class",
        "predicted_class",
        "correct",
        "confidence",
        "entropy",
        "logit_elliptical",
        "prob_elliptical_model",
        "prob_merger_model",
        "prob_spiral_model",
        "model",
        "fold",
        "repeat_seed",
        "subset_size",
    }
    assert expected.issubset(result.columns)
    assert result.loc[0, "objid"] == "1237654605318717697"
    assert bool(result.loc[0, "correct"])
    assert not bool(result.loc[1, "correct"])
    assert np.allclose(
        result[["prob_elliptical_model", "prob_merger_model", "prob_spiral_model"]].sum(1),
        1.0,
    )


def test_prediction_frame_rejects_metadata_logit_length_mismatch() -> None:
    """Catches silent sample/logit misalignment after batched inference."""
    samples = pd.DataFrame([{"objid": "1", "class_index": 0}])
    try:
        prediction_frame(samples, np.zeros((2, 3)), ["a", "b", "c"], {})
    except ValueError as exc:
        assert "lunghezza" in str(exc)
    else:
        raise AssertionError("mismatched logits accepted")
