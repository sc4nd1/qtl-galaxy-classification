"""Costruzione deterministica del manifest sperimentale bilanciato."""

from __future__ import annotations

from typing import Any

import pandas as pd

CLASS_TO_INDEX = {"elliptical": 0, "merger": 1, "spiral": 2}


def build_balanced_manifest(
    validated: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rimuove duplicati visivi e bilancia sul minimo multiplo di quattro."""
    required = {"objid", "label", "valid", "sha256", "path"}
    missing = sorted(required - set(validated.columns))
    if missing:
        raise ValueError(f"Colonne mancanti nel manifest validato: {missing}")

    working = validated.loc[validated["valid"].fillna(False)].copy()
    excluded_samples: list[dict[str, str]] = []
    working["objid"] = working["objid"].astype("string")
    working["label"] = working["label"].astype("string")
    if not set(working["label"]).issubset(CLASS_TO_INDEX):
        raise ValueError("Il manifest contiene classi non supportate")
    if working["sha256"].isna().any():
        raise ValueError("Ogni immagine valida deve avere SHA-256")

    hash_class_count = working.groupby("sha256", observed=True)["label"].nunique()
    cross_class_hashes = set(hash_class_count[hash_class_count > 1].index)
    cross_class_mask = working["sha256"].isin(cross_class_hashes)
    cross_class_removed = int(cross_class_mask.sum())
    for row in working.loc[cross_class_mask, ["objid", "label", "sha256"]].to_dict("records"):
        excluded_samples.append(
            {
                **{key: str(value) for key, value in row.items()},
                "reason": "cross_class_duplicate_hash",
            }
        )
    working = working.loc[~cross_class_mask].copy()

    working = working.sort_values(["label", "objid"], kind="stable")
    before_intra = len(working)
    intra_mask = working.duplicated(subset=["label", "sha256"], keep="first")
    for row in working.loc[intra_mask, ["objid", "label", "sha256"]].to_dict("records"):
        excluded_samples.append(
            {
                **{key: str(value) for key, value in row.items()},
                "reason": "intra_class_duplicate_hash",
            }
        )
    working = working.drop_duplicates(subset=["label", "sha256"], keep="first")
    intra_class_removed = before_intra - len(working)

    counts_before = working["label"].value_counts().to_dict()
    absent = sorted(set(CLASS_TO_INDEX) - set(counts_before))
    if absent:
        raise ValueError(f"Classi senza immagini valide: {absent}")
    per_class = (min(counts_before.values()) // 4) * 4
    if per_class < 4:
        raise ValueError("Servono almeno quattro immagini valide per classe")

    selected = []
    for label, class_index in CLASS_TO_INDEX.items():
        candidates = working.loc[working["label"] == label]
        selected.append(
            candidates.sample(n=per_class, random_state=seed + class_index, replace=False)
        )
    final = pd.concat(selected, ignore_index=True)
    selected_ids = set(final["objid"].astype(str))
    not_selected = working.loc[~working["objid"].astype(str).isin(selected_ids)]
    for row in not_selected[["objid", "label", "sha256"]].to_dict("records"):
        excluded_samples.append(
            {**{key: str(value) for key, value in row.items()}, "reason": "class_balance_sampling"}
        )
    final["class_index"] = final["label"].map(CLASS_TO_INDEX).astype("int64")
    final = final.sort_values(["label", "objid"], kind="stable").reset_index(drop=True)
    final.insert(0, "sample_index", range(len(final)))

    if not final["objid"].is_unique:
        raise ValueError("ObjID duplicati nel manifest finale")
    if not final["sha256"].is_unique:
        raise ValueError("Hash immagine duplicati nel manifest finale")

    summary: dict[str, Any] = {
        "valid_rows_input": int(validated["valid"].fillna(False).sum()),
        "cross_class_hash_rows_removed": cross_class_removed,
        "intra_class_hash_rows_removed": intra_class_removed,
        "valid_counts_after_dedup": counts_before,
        "per_class": per_class,
        "total": len(final),
        "seed": seed,
        "excluded_samples": excluded_samples,
    }
    return final, summary
