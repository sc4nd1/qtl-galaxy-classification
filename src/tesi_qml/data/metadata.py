"""Pulizia scientifica dei candidati SDSS e assegnazione delle hard label."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pandas as pd
from pandas.api.types import is_float_dtype

REQUIRED_COLUMNS = (
    "objid",
    "ra",
    "dec",
    "prob_elliptical",
    "prob_spiral",
    "prob_merger",
    "redshift",
)
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]


@dataclass(frozen=True)
class CleanMetadataResult:
    clean: pd.DataFrame
    rejected: pd.DataFrame
    summary: dict[str, object]


def read_candidate_csv(source: Path | str | IO[str]) -> pd.DataFrame:
    """Legge il catalogo preservando gli identificativi SDSS come stringhe esatte."""
    frame = pd.read_csv(source, dtype={"objid": "string"})
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Colonne obbligatorie mancanti: {missing}")
    return frame


def _empty_with_reason(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[0:0].assign(reason=pd.Series(dtype="string"))


def clean_metadata(
    frame: pd.DataFrame,
    thresholds: Mapping[str, float],
    *,
    min_redshift: float = 0.005,
    max_redshift: float = 0.1,
) -> CleanMetadataResult:
    """Valida il catalogo e assegna una sola hard label per campione."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Colonne obbligatorie mancanti: {missing}")
    expected_thresholds = {"elliptical", "spiral", "merger"}
    if set(thresholds) != expected_thresholds:
        raise ValueError(f"Le soglie devono essere definite per {sorted(expected_thresholds)}")
    if is_float_dtype(frame["objid"]):
        raise ValueError("objid deve essere letto come string per evitare perdita di precisione")

    working = frame.loc[:, REQUIRED_COLUMNS].copy()
    working.insert(0, "source_row", frame.index.to_numpy())
    working["objid"] = working["objid"].astype("string").str.strip()
    for column in NUMERIC_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    original_count = len(working)
    before_exact = len(working)
    working = working.drop_duplicates(subset=list(REQUIRED_COLUMNS), keep="first")
    exact_duplicates_removed = before_exact - len(working)

    rejected_parts: list[pd.DataFrame] = []

    duplicate_ids = working.loc[working.duplicated("objid", keep=False), "objid"].dropna().unique()
    if len(duplicate_ids):
        conflict = working.loc[working["objid"].isin(duplicate_ids)].copy()
        conflict["reason"] = "duplicate_conflict"
        rejected_parts.append(conflict)
        working = working.loc[~working["objid"].isin(duplicate_ids)].copy()

    reason = pd.Series(pd.NA, index=working.index, dtype="string")
    reason.loc[working["objid"].isna() | (working["objid"] == "")] = "invalid_objid"

    invalid_numeric = working[list(NUMERIC_COLUMNS)].isna().any(axis=1)
    reason.loc[reason.isna() & invalid_numeric] = "invalid_numeric"

    invalid_probability = ~working[["prob_elliptical", "prob_spiral", "prob_merger"]].apply(
        lambda col: col.between(0.0, 1.0)
    ).all(axis=1)
    reason.loc[reason.isna() & invalid_probability] = "invalid_probability"

    invalid_coordinates = ~working["ra"].between(0.0, 360.0) | ~working["dec"].between(-90.0, 90.0)
    reason.loc[reason.isna() & invalid_coordinates] = "invalid_coordinates"

    invalid_redshift = ~working["redshift"].between(min_redshift, max_redshift, inclusive="neither")
    reason.loc[reason.isna() & invalid_redshift] = "redshift_out_of_range"

    label_flags = pd.DataFrame(
        {
            "elliptical": working["prob_elliptical"] > float(thresholds["elliptical"]),
            "spiral": working["prob_spiral"] > float(thresholds["spiral"]),
            "merger": working["prob_merger"] > float(thresholds["merger"]),
        },
        index=working.index,
    )
    label_count = label_flags.sum(axis=1)
    reason.loc[reason.isna() & (label_count == 0)] = "no_hard_label"
    reason.loc[reason.isna() & (label_count > 1)] = "class_conflict"

    invalid_rows = working.loc[reason.notna()].copy()
    invalid_rows["reason"] = reason.loc[reason.notna()]
    if not invalid_rows.empty:
        rejected_parts.append(invalid_rows)

    clean = working.loc[reason.isna()].copy()
    if not clean.empty:
        clean["label"] = label_flags.loc[clean.index].idxmax(axis=1).astype("string")
    else:
        clean["label"] = pd.Series(dtype="string")

    clean = clean.sort_values("objid", kind="stable").reset_index(drop=True)
    rejected = (
        pd.concat(rejected_parts, ignore_index=True)
        if rejected_parts
        else _empty_with_reason(working)
    )
    rejected = rejected.sort_values(["objid", "source_row"], kind="stable").reset_index(drop=True)

    summary: dict[str, object] = {
        "input_rows": original_count,
        "exact_duplicates_removed": exact_duplicates_removed,
        "clean_rows": len(clean),
        "rejected_rows": len(rejected),
        "class_counts": clean["label"].value_counts().sort_index().to_dict(),
        "rejection_counts": rejected["reason"].value_counts().sort_index().to_dict(),
    }
    return CleanMetadataResult(clean=clean, rejected=rejected, summary=summary)


def select_download_pool(
    clean: pd.DataFrame,
    *,
    seed: int,
    oversample_factor: float = 1.25,
) -> pd.DataFrame:
    """Mantiene tutti i merger e seleziona riserve deterministiche per le altre classi."""
    if oversample_factor < 1.0:
        raise ValueError("oversample_factor deve essere almeno 1.0")
    if "label" not in clean.columns or "objid" not in clean.columns:
        raise ValueError("Il manifest pulito deve contenere label e objid")

    merger = clean.loc[clean["label"] == "merger"].copy()
    if merger.empty:
        raise ValueError("Nessun candidato merger disponibile")
    target = math.ceil(len(merger) * oversample_factor)

    selected = [merger]
    for offset, label in enumerate(("elliptical", "spiral"), start=1):
        candidates = clean.loc[clean["label"] == label]
        size = min(target, len(candidates))
        selected.append(candidates.sample(n=size, random_state=seed + offset, replace=False))

    result = pd.concat(selected, ignore_index=True)
    return result.sort_values(["label", "objid"], kind="stable").reset_index(drop=True)
