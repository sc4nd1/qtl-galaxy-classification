"""Task di classificazione e filtri annidati per manifest e feature bank."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
import torch

from .data.features import FeatureBank


@dataclass(frozen=True)
class TaskSpec:
    """Definizione esplicita e validata delle classi di un task."""

    name: str
    class_names: tuple[str, ...]
    class_to_index: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Il nome del task non può essere vuoto")
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("I nomi delle classi devono essere presenti e univoci")
        mapping = dict(self.class_to_index)
        if set(mapping) != set(self.class_names):
            raise ValueError("Il mapping deve contenere esattamente le classi del task")
        if sorted(mapping.values()) != list(range(len(self.class_names))):
            raise ValueError("Gli indici di classe devono essere contigui da zero")
        if tuple(sorted(mapping, key=mapping.get)) != self.class_names:
            raise ValueError("class_names e class_to_index devono avere lo stesso ordine")

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "class_names": list(self.class_names),
            "class_to_index": dict(self.class_to_index),
        }


_TASKS = {
    "binary": TaskSpec(
        name="binary",
        class_names=("elliptical", "spiral"),
        class_to_index={"elliptical": 0, "spiral": 1},
    ),
    "three_class": TaskSpec(
        name="three_class",
        class_names=("elliptical", "merger", "spiral"),
        class_to_index={"elliptical": 0, "merger": 1, "spiral": 2},
    ),
}


def get_task(name: str) -> TaskSpec:
    """Restituisce una configurazione di task nota o fallisce esplicitamente."""
    try:
        return _TASKS[name]
    except KeyError as error:
        raise ValueError(f"Task non supportato: {name}") from error


def filter_task_frame(
    frame: pd.DataFrame,
    task: TaskSpec | str,
    *,
    label_column: str = "label",
    index_column: str = "class_index",
) -> pd.DataFrame:
    """Filtra una tabella triclasse e rimappa gli indici senza rigenerare righe."""
    selected_task = get_task(task) if isinstance(task, str) else task
    required = {label_column, index_column}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Colonne task mancanti: {missing}")
    labels = frame[label_column].astype("string")
    result = frame.loc[labels.isin(selected_task.class_names)].copy()
    result[label_column] = result[label_column].astype("string")
    result[index_column] = result[label_column].map(selected_task.class_to_index)
    if result[index_column].isna().any():
        raise ValueError("Impossibile rimappare tutte le label del task")
    result[index_column] = result[index_column].astype("int64")
    observed = sorted(result[index_column].unique().tolist())
    expected = list(range(selected_task.num_classes))
    if observed != expected:
        raise ValueError(
            f"Le label filtrate non sono contigue o manca una classe: {observed}, attese {expected}"
        )
    return result.reset_index(drop=True)


def filter_feature_bank(bank: FeatureBank, task: TaskSpec | str) -> FeatureBank:
    """Crea una vista task-specific senza alterare identità o tensore sorgente."""
    selected_task = get_task(task) if isinstance(task, str) else task
    if "label_name" not in bank.metadata:
        raise ValueError("La FeatureBank deve contenere label_name per il filtro del task")
    labels = bank.metadata["label_name"].astype("string")
    mask = labels.isin(selected_task.class_names)
    metadata = bank.metadata.loc[mask].reset_index(drop=True).copy()
    metadata["label_name"] = metadata["label_name"].astype("string")
    metadata["label"] = metadata["label_name"].map(selected_task.class_to_index)
    if metadata["label"].isna().any():
        raise ValueError("Impossibile rimappare tutte le label della FeatureBank")
    metadata["label"] = metadata["label"].astype("int64")
    observed = sorted(metadata["label"].unique().tolist())
    expected = list(range(selected_task.num_classes))
    if observed != expected:
        raise ValueError(
            f"Le label filtrate non sono contigue o manca una classe: {observed}, attese {expected}"
        )
    metadata["feature_index"] = range(len(metadata))
    features = bank.features[torch.from_numpy(mask.to_numpy(dtype=bool))]
    provenance = dict(bank.provenance)
    provenance["task"] = selected_task.as_dict()
    return FeatureBank(features=features, metadata=metadata, provenance=provenance)
