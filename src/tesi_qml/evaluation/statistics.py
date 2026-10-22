"""Aggregazione e confronti appaiati tra esperimenti ripetuti."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Applica la correzione step-down di Holm e ripristina l'ordine originale."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Servono uno o più p-value in un vettore")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("I p-value devono essere finiti e compresi tra zero e uno")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        sorted_values * np.arange(len(values), 0, -1, dtype=np.float64)
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def validate_primary_result_matrix(
    frame: pd.DataFrame, *, repeat_seeds: Sequence[int]
) -> dict[str, int]:
    """Verifica le 160 celle complete task×seed×modello del confronto primario."""
    required = {"task", "model", "n_qubits", "repeat_seed", "fold"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Colonne identità mancanti: {sorted(missing)}")
    key_columns = ["task", "model", "n_qubits", "repeat_seed", "fold"]
    if frame.duplicated(key_columns).any():
        raise ValueError("La matrice primaria contiene run duplicati")

    model_cells = (
        ("B0", None),
        ("B1", None),
        ("B2", 4),
        ("B2", 8),
        ("QF", 4),
        ("QF", 8),
        ("Q", 4),
        ("Q", 8),
    )
    expected = {
        (task, model, qubits, int(seed), 1)
        for task in ("binary", "three_class")
        for seed in repeat_seeds
        for model, qubits in model_cells
    }
    observed = {
        (
            str(row.task),
            str(row.model),
            None if pd.isna(row.n_qubits) else int(row.n_qubits),
            int(row.repeat_seed),
            int(row.fold),
        )
        for row in frame[list(key_columns)].itertuples(index=False)
    }
    missing_cells = expected - observed
    extra_cells = observed - expected
    if missing_cells or extra_cells:
        raise ValueError(
            "Matrice primaria incompleta o con run extra: "
            f"mancanti={len(missing_cells)}, extra={len(extra_cells)}"
        )
    return {"runs": len(frame)}


def _metric_inference_values(
    group: pd.DataFrame,
    metric: str,
    group_columns: Sequence[str],
) -> tuple[np.ndarray, str, int]:
    raw_values = group[metric].dropna()
    observation_count = len(raw_values)
    for unit in ("repeat_seed", "seed"):
        if unit in group.columns and unit not in group_columns and group[unit].notna().any():
            metric_frame = group[[unit, metric]].dropna(subset=[metric])
            if metric_frame[unit].isna().any():
                raise ValueError(f"Colonna {unit} mancante per alcune osservazioni di {metric}")
            values = metric_frame.groupby(unit, sort=True)[metric].mean().to_numpy(dtype=np.float64)
            return values, unit, observation_count
    return raw_values.to_numpy(dtype=np.float64), "run", observation_count


def validate_experiment_matrix(
    frame: pd.DataFrame,
    *,
    repeat_seeds: Sequence[int],
    folds: Sequence[int],
) -> dict[str, int]:
    """Verifica che ogni configurazione abbia esattamente tutti i repeat/fold attesi."""
    required = {"experiment_fingerprint", "repeat_seed", "fold"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Colonne identità mancanti: {sorted(missing)}")
    key_columns = ["experiment_fingerprint", "repeat_seed", "fold"]
    if frame.duplicated(key_columns).any():
        raise ValueError("La matrice contiene run duplicati per configurazione/repeat/fold")
    expected = {(int(seed), int(fold)) for seed in repeat_seeds for fold in folds}
    report: dict[str, int] = {}
    for fingerprint, group in frame.groupby("experiment_fingerprint", sort=True):
        observed = set(zip(group["repeat_seed"].astype(int), group["fold"].astype(int)))
        if observed != expected:
            missing_pairs = sorted(expected - observed)
            extra_pairs = sorted(observed - expected)
            raise ValueError(
                f"Matrice sperimentale incompleta per {fingerprint}: "
                f"mancanti={missing_pairs}, extra={extra_pairs}"
            )
        report[str(fingerprint)] = len(group)
    return report


def aggregate_runs(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("accuracy", "balanced_accuracy", "f1_macro"),
    group_columns: Sequence[str] = ("model",),
) -> pd.DataFrame:
    missing = set(group_columns).union(metrics) - set(frame.columns)
    if missing:
        raise ValueError(f"Colonne mancanti per l'aggregazione: {sorted(missing)}")
    records: list[dict[str, object]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for keys, group in frame.groupby(grouper, sort=True, dropna=False):
        key_values = (keys,) if len(group_columns) == 1 else tuple(keys)
        row = dict(zip(group_columns, key_values))
        for metric in metrics:
            values, inference_unit, observation_count = _metric_inference_values(
                group, metric, group_columns
            )
            n = len(values)
            mean = float(values.mean()) if n else float("nan")
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            half_width = float(t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
            row.update(
                {
                    f"{metric}_n": n,
                    f"{metric}_observation_n": observation_count,
                    f"{metric}_inference_unit": inference_unit,
                    f"{metric}_mean": mean,
                    f"{metric}_std": std,
                    f"{metric}_median": float(np.median(values)) if n else float("nan"),
                    f"{metric}_ci95_low": mean - half_width,
                    f"{metric}_ci95_high": mean + half_width,
                }
            )
        records.append(row)
    return pd.DataFrame(records)


def paired_model_comparison(
    frame: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
    *,
    seed: int,
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
) -> dict[str, object]:
    """Confronta A-B usando esclusivamente fold/seed realmente appaiati."""
    seed_column = "repeat_seed" if "repeat_seed" in frame.columns else "seed"
    required = {"model", metric, seed_column, "fold"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Colonne mancanti per il confronto: {sorted(missing)}")
    if n_permutations < 1 or n_bootstrap < 1:
        raise ValueError("Il numero di ricampionamenti deve essere positivo")
    pairing_keys = [
        key for key in (seed_column, "fold", "repeat", "subset_size") if key in frame.columns
    ]
    selected = frame.loc[frame["model"].isin([model_a, model_b]), pairing_keys + ["model", metric]]
    if selected.duplicated(pairing_keys + ["model"]).any():
        raise ValueError("Esistono più risultati per la stessa coppia modello/fold/seed")
    paired = selected.pivot(index=pairing_keys, columns="model", values=metric).dropna()
    if model_a not in paired or model_b not in paired or paired.empty:
        raise ValueError("Nessuna osservazione completa appaiata tra i due modelli")
    differences = (paired[model_a] - paired[model_b]).to_numpy(dtype=np.float64)
    paired_differences = (paired[model_a] - paired[model_b]).rename("difference").reset_index()
    inference_values = (
        paired_differences.groupby(seed_column, sort=True)["difference"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    rng = np.random.default_rng(seed)

    bootstrap_indices = rng.integers(
        0, len(inference_values), size=(n_bootstrap, len(inference_values))
    )
    bootstrap_means = inference_values[bootstrap_indices].mean(axis=1)
    confidence_interval = np.quantile(bootstrap_means, [0.025, 0.975])

    observed = abs(float(inference_values.mean()))
    sign_flips = rng.choice((-1.0, 1.0), size=(n_permutations, len(inference_values)))
    null_means = np.abs((sign_flips * inference_values).mean(axis=1))
    p_value = (float(np.count_nonzero(null_means >= observed)) + 1.0) / (n_permutations + 1.0)
    standard_deviation = float(inference_values.std(ddof=1)) if len(inference_values) > 1 else 0.0
    effect_size = (
        float(inference_values.mean() / standard_deviation)
        if standard_deviation > np.finfo(float).eps
        else None
    )
    wilcoxon_p = (
        1.0
        if np.allclose(inference_values, 0.0)
        else float(wilcoxon(inference_values, alternative="two-sided").pvalue)
    )
    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "difference_definition": f"{model_a} - {model_b}",
        "pairing_keys": pairing_keys,
        "n_pairs": len(differences),
        "n_inference_blocks": len(inference_values),
        "inference_unit": seed_column,
        "mean_difference": float(inference_values.mean()),
        "difference_std": standard_deviation,
        "median_difference": float(np.median(differences)),
        "confidence_interval_95": confidence_interval.astype(float).tolist(),
        "permutation_p_value": p_value,
        "effect_size_cohen_dz": effect_size,
        "wilcoxon_p_value": wilcoxon_p,
        "random_seed": int(seed),
        "n_permutations": int(n_permutations),
        "n_bootstrap": int(n_bootstrap),
    }


def primary_hypothesis_tests(
    frame: pd.DataFrame,
    *,
    seed: int,
    metric: str = "f1_macro",
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
) -> pd.DataFrame:
    """Esegue e corregge esclusivamente i quattro confronti primari Q-B2."""
    required = {"task", "model", "n_qubits", metric}
    if missing := required - set(frame.columns):
        raise ValueError(f"Colonne mancanti per la famiglia primaria: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for index, (task, qubits) in enumerate(
        (("binary", 4), ("binary", 8), ("three_class", 4), ("three_class", 8))
    ):
        selected = frame.loc[(frame["task"] == task) & (frame["n_qubits"] == qubits)]
        comparison = paired_model_comparison(
            selected,
            "Q",
            "B2",
            metric,
            seed=seed + index,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
        )
        records.append(
            {
                "task": task,
                "n_qubits": qubits,
                **comparison,
                "raw_p_value": comparison["permutation_p_value"],
            }
        )
    corrected = holm_adjust([float(record["raw_p_value"]) for record in records])
    for record, value in zip(records, corrected):
        record["holm_p_value"] = float(value)
    return pd.DataFrame(records)
