"""Artefatti per-sample tracciabili fino all'ObjID e all'immagine sorgente."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def prediction_frame(
    sample_metadata: pd.DataFrame,
    logits: np.ndarray | torch.Tensor,
    class_names: Sequence[str],
    run_metadata: Mapping[str, object],
) -> pd.DataFrame:
    """Unisce logits e probabilità ai metadata senza perdere l'identità dei campioni."""
    values = _as_numpy(logits).astype(np.float64, copy=False)
    if values.ndim != 2 or values.shape[1] != len(class_names):
        raise ValueError("Shape dei logits incompatibile con le classi")
    if len(sample_metadata) != len(values):
        raise ValueError("La lunghezza dei metadata non coincide con quella dei logits")
    if "objid" not in sample_metadata or "class_index" not in sample_metadata:
        raise ValueError("I metadata devono contenere objid e class_index")
    if not np.isfinite(values).all():
        raise ValueError("I logits contengono NaN o Inf")

    stable = values - values.max(axis=1, keepdims=True)
    exp = np.exp(stable)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    truth = sample_metadata["class_index"].to_numpy(dtype=np.int64)
    names = np.asarray(list(class_names), dtype=object)
    if (truth < 0).any() or (truth >= len(names)).any():
        raise ValueError("class_index fuori range")

    result = sample_metadata.reset_index(drop=True).copy()
    result["objid"] = result["objid"].astype("string")
    result["true_class"] = names[truth]
    result["predicted_index"] = predicted
    result["predicted_class"] = names[predicted]
    result["correct"] = predicted == truth
    result["confidence"] = probabilities.max(axis=1)
    result["entropy"] = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-15, 1.0)), axis=1)
    for index, name in enumerate(class_names):
        result[f"logit_{name}"] = values[:, index]
        result[f"prob_{name}_model"] = probabilities[:, index]
    for key, value in run_metadata.items():
        if key in result.columns:
            raise ValueError(f"Il metadata di run sovrascriverebbe la colonna {key}")
        result[key] = value
    return result
