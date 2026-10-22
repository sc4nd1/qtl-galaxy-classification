"""Split stratificati 2/1/1, persistibili e condivisi da tutti i modelli."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_repeated_four_fold_splits(
    manifest: pd.DataFrame,
    *,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Genera quattro rotazioni train/validation/test per ciascun repeat seed."""
    required = {"objid", "label", "class_index"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Colonne mancanti nel manifest: {missing}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("I repeat seed devono essere unici")

    ordered = manifest.sort_values("objid", kind="stable").reset_index(drop=True)
    labels = ordered["label"].to_numpy()
    chunks: list[pd.DataFrame] = []

    for repeat_index, seed in enumerate(seeds):
        splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=int(seed))
        base_fold = np.empty(len(ordered), dtype=np.int64)
        for fold_id, (_, held_out) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            base_fold[held_out] = fold_id

        for test_fold in range(4):
            validation_fold = (test_fold - 1) % 4
            split_name = np.full(len(ordered), "train", dtype=object)
            split_name[base_fold == validation_fold] = "validation"
            split_name[base_fold == test_fold] = "test"
            chunk = ordered[["objid", "label", "class_index"]].copy()
            chunk.insert(0, "split", split_name)
            chunk.insert(0, "base_fold", base_fold)
            chunk.insert(0, "iteration", test_fold + 1)
            chunk.insert(0, "repeat_seed", int(seed))
            chunk.insert(0, "repeat_index", repeat_index)
            chunks.append(chunk)

    result = pd.concat(chunks, ignore_index=True)
    split_order = pd.CategoricalDtype(["train", "validation", "test"], ordered=True)
    result["split"] = result["split"].astype(split_order)
    result = result.sort_values(
        ["repeat_index", "iteration", "split", "label", "objid"], kind="stable"
    ).reset_index(drop=True)
    result["split"] = result["split"].astype("string")
    validate_split_table(result, manifest)
    return result


def validate_split_table(splits: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Fallisce prima del training se uno split è corrotto o introduce leakage."""
    required = {"repeat_seed", "iteration", "split", "objid", "label"}
    missing = sorted(required - set(splits.columns))
    if missing:
        raise ValueError(f"Colonne mancanti nella tabella split: {missing}")

    manifest_ids = set(manifest["objid"].astype(str))
    manifest_counts = manifest["label"].value_counts().to_dict()
    if any(count % 4 for count in manifest_counts.values()):
        raise ValueError("Il manifest deve essere divisibile per quattro in ogni classe")

    for (repeat_seed, iteration), group in splits.groupby(["repeat_seed", "iteration"], sort=False):
        if group["objid"].duplicated().any():
            raise ValueError(
                f"ObjID duplicato/overlap nel repeat {repeat_seed}, iterazione {iteration}"
            )
        if set(group["split"]) != {"train", "validation", "test"}:
            raise ValueError("Ogni iterazione deve contenere train, validation e test")
        if set(group["objid"].astype(str)) != manifest_ids:
            raise ValueError("La copertura dello split non coincide con il manifest")

        observed = group.groupby(["split", "label"], observed=True).size().to_dict()
        for label, total in manifest_counts.items():
            expected = {"train": total // 2, "validation": total // 4, "test": total // 4}
            for split_name, expected_count in expected.items():
                if observed.get((split_name, label), 0) != expected_count:
                    raise ValueError("Split non stratificato o sbilanciato")

    for repeat_seed, group in splits.groupby("repeat_seed", sort=False):
        counts = group.groupby(["objid", "split"], observed=True).size().unstack(fill_value=0)
        if not (
            (counts.get("train", 0) == 2).all()
            and (counts.get("validation", 0) == 1).all()
            and (counts.get("test", 0) == 1).all()
        ):
            raise ValueError(f"Rotazione incompleta nel repeat {repeat_seed}")
