import numpy as np
import pandas as pd

from tesi_qml.evaluation.statistics import (
    aggregate_runs,
    holm_adjust,
    paired_model_comparison,
    primary_hypothesis_tests,
    validate_experiment_matrix,
    validate_primary_result_matrix,
)


def _matched_results():
    rows = []
    for seed in (11, 29):
        for fold in range(4):
            baseline = 0.70 + fold * 0.01
            rows.extend(
                [
                    {"model": "B1", "seed": seed, "fold": fold, "f1_macro": baseline},
                    {"model": "Q4", "seed": seed, "fold": fold, "f1_macro": baseline + 0.05},
                ]
            )
    return pd.DataFrame(rows)


def test_paired_model_comparison_reports_ci_test_and_effect_size():
    comparison = paired_model_comparison(
        _matched_results(), "Q4", "B1", "f1_macro", seed=123, n_permutations=2000
    )
    assert comparison["pairing_keys"] == ["seed", "fold"]
    assert comparison["n_pairs"] == 8
    assert comparison["n_inference_blocks"] == 2
    assert comparison["inference_unit"] == "seed"
    np.testing.assert_allclose(comparison["mean_difference"], 0.05, atol=1e-12)
    assert len(comparison["confidence_interval_95"]) == 2
    assert 0.0 <= comparison["permutation_p_value"] <= 1.0
    assert "effect_size_cohen_dz" in comparison
    assert "wilcoxon_p_value" in comparison


def test_aggregate_runs_keeps_model_level_uncertainty():
    summary = aggregate_runs(_matched_results(), metrics=("f1_macro",))
    assert set(summary["model"]) == {"B1", "Q4"}
    assert {"f1_macro_mean", "f1_macro_std", "f1_macro_ci95_low", "f1_macro_ci95_high"} <= set(
        summary.columns
    )


def test_aggregate_runs_uses_repeat_seed_as_inference_unit_for_ci():
    """Catches optimistic CIs computed from correlated folds instead of repeat means."""
    frame = pd.DataFrame(
        [
            {"model": "PL", "repeat_seed": 11, "fold": 1, "f1_macro": 0.4},
            {"model": "PL", "repeat_seed": 11, "fold": 2, "f1_macro": 0.6},
            {"model": "PL", "repeat_seed": 11, "fold": 3, "f1_macro": 0.4},
            {"model": "PL", "repeat_seed": 11, "fold": 4, "f1_macro": 0.6},
            {"model": "PL", "repeat_seed": 29, "fold": 1, "f1_macro": 0.6},
            {"model": "PL", "repeat_seed": 29, "fold": 2, "f1_macro": 0.8},
            {"model": "PL", "repeat_seed": 29, "fold": 3, "f1_macro": 0.6},
            {"model": "PL", "repeat_seed": 29, "fold": 4, "f1_macro": 0.8},
        ]
    )

    summary = aggregate_runs(frame, metrics=("f1_macro",))
    row = summary.iloc[0]

    assert row["f1_macro_n"] == 2
    assert row["f1_macro_observation_n"] == 8
    assert row["f1_macro_inference_unit"] == "repeat_seed"
    np.testing.assert_allclose(row["f1_macro_mean"], 0.6)
    np.testing.assert_allclose(row["f1_macro_std"], np.sqrt(0.02))


def test_experiment_matrix_rejects_missing_or_duplicate_fold_runs():
    complete = pd.DataFrame(
        [
            {
                "experiment_fingerprint": "cfg-a",
                "repeat_seed": seed,
                "fold": fold,
                "accuracy": 0.7,
            }
            for seed in (11, 29)
            for fold in range(1, 5)
        ]
    )
    report = validate_experiment_matrix(complete, repeat_seeds=(11, 29), folds=(1, 2, 3, 4))
    assert report == {"cfg-a": 8}

    with np.testing.assert_raises_regex(ValueError, "incompleta"):
        validate_experiment_matrix(complete.iloc[:-1], repeat_seeds=(11, 29), folds=(1, 2, 3, 4))
    duplicated = pd.concat([complete, complete.iloc[[0]]], ignore_index=True)
    with np.testing.assert_raises_regex(ValueError, "duplicati"):
        validate_experiment_matrix(duplicated, repeat_seeds=(11, 29), folds=(1, 2, 3, 4))


def test_aggregate_keeps_qubits_layers_and_subset_separate():
    frame = pd.DataFrame(
        [
            {"model": "PL", "n_qubits": 4, "n_layers": 1, "subset_size": "50", "accuracy": 0.5},
            {"model": "PL", "n_qubits": 8, "n_layers": 4, "subset_size": "full", "accuracy": 0.9},
        ]
    )
    summary = aggregate_runs(
        frame,
        metrics=("accuracy",),
        group_columns=("model", "n_qubits", "n_layers", "subset_size"),
    )
    assert len(summary) == 2
    assert set(summary["accuracy_mean"]) == {0.5, 0.9}


def test_holm_adjust_matches_hand_computed_monotone_values() -> None:
    """Catches applying Holm multipliers without the cumulative monotonicity step."""
    np.testing.assert_allclose(
        holm_adjust([0.01, 0.04, 0.03, 0.20]),
        [0.04, 0.09, 0.09, 0.20],
    )


def test_primary_hypothesis_family_contains_exactly_four_corrected_q_vs_b2_tests() -> None:
    """Catches changing the preregistered family after observing results."""
    rows = []
    for task in ("binary", "three_class"):
        for qubits in (4, 8):
            for seed in range(10):
                baseline = 0.60 + seed * 0.005
                rows.extend(
                    [
                        {
                            "task": task,
                            "model": "B2",
                            "n_qubits": qubits,
                            "repeat_seed": seed,
                            "fold": 1,
                            "f1_macro": baseline,
                        },
                        {
                            "task": task,
                            "model": "Q",
                            "n_qubits": qubits,
                            "repeat_seed": seed,
                            "fold": 1,
                            "f1_macro": baseline + 0.02,
                        },
                    ]
                )

    result = primary_hypothesis_tests(
        pd.DataFrame(rows), seed=123, n_permutations=500, n_bootstrap=500
    )

    assert set(zip(result["task"], result["n_qubits"])) == {
        ("binary", 4),
        ("binary", 8),
        ("three_class", 4),
        ("three_class", 8),
    }
    assert (result["model_a"] == "Q").all()
    assert (result["model_b"] == "B2").all()
    assert (result["holm_p_value"] >= result["raw_p_value"]).all()
    assert (result["difference_std"] >= 0).all()


def test_primary_result_matrix_rejects_missing_duplicate_and_extra_runs() -> None:
    """Catches aggregation over anything other than 2×10×8 primary results."""
    rows = []
    models = (("B0", None), ("B1", None)) + tuple(
        (model, qubits) for model in ("B2", "QF", "Q") for qubits in (4, 8)
    )
    for task in ("binary", "three_class"):
        for seed in range(10):
            for model, qubits in models:
                rows.append(
                    {
                        "task": task,
                        "model": model,
                        "n_qubits": qubits,
                        "repeat_seed": seed,
                        "fold": 1,
                    }
                )
    complete = pd.DataFrame(rows)
    assert validate_primary_result_matrix(complete, repeat_seeds=range(10)) == {"runs": 160}

    with np.testing.assert_raises_regex(ValueError, "incompleta"):
        validate_primary_result_matrix(complete.iloc[:-1], repeat_seeds=range(10))
    duplicated = pd.concat([complete, complete.iloc[[0]]], ignore_index=True)
    with np.testing.assert_raises_regex(ValueError, "duplicati"):
        validate_primary_result_matrix(duplicated, repeat_seeds=range(10))
    extra = complete.copy()
    extra.loc[len(extra)] = {
        "task": "binary",
        "model": "B3",
        "n_qubits": np.nan,
        "repeat_seed": 0,
        "fold": 1,
    }
    with np.testing.assert_raises_regex(ValueError, "extra"):
        validate_primary_result_matrix(extra, repeat_seeds=range(10))
