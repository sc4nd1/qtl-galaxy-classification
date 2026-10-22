"""Feature bank compatta e condivisa dalle teste classiche e quantistiche."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..reproducibility import seed_everything
from .datasets import TRACE_METADATA_COLUMNS


@dataclass
class FeatureBank:
    features: torch.Tensor
    metadata: pd.DataFrame
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("Le feature devono essere una matrice [campioni, dimensioni]")
        if len(self.features) != len(self.metadata):
            raise ValueError("Feature e metadata hanno lunghezze diverse")
        required = {"objid", "label", "view", "view_seed", "feature_index"}
        missing = sorted(required - set(self.metadata.columns))
        if missing:
            raise ValueError(f"Colonne metadata feature mancanti: {missing}")
        expected = list(range(len(self.metadata)))
        if self.metadata["feature_index"].tolist() != expected:
            raise ValueError("feature_index non allineato alla matrice")

    def save(self, directory: Path | str) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        tensor_tmp = target / "features.pt.part"
        metadata_tmp = target / "metadata.parquet.part"
        provenance_tmp = target / "provenance.json.part"
        torch.save(self.features.cpu(), tensor_tmp)
        self.metadata.to_parquet(metadata_tmp, index=False)
        provenance_tmp.write_text(
            json.dumps(self.provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tensor_tmp.replace(target / "features.pt")
        metadata_tmp.replace(target / "metadata.parquet")
        provenance_tmp.replace(target / "provenance.json")

    @classmethod
    def load(cls, directory: Path | str) -> FeatureBank:
        source = Path(directory)
        features = torch.load(source / "features.pt", map_location="cpu", weights_only=True)
        metadata = pd.read_parquet(source / "metadata.parquet")
        provenance_path = source / "provenance.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file()
            else {}
        )
        return cls(features=features, metadata=metadata, provenance=provenance)


class FeatureBankDataset(Dataset):
    def __init__(self, bank: FeatureBank) -> None:
        self.bank = bank

    def __len__(self) -> int:
        return len(self.bank.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.bank.metadata.iloc[index]
        result = {
            "inputs": self.bank.features[index],
            "label": int(row["label"]),
            "objid": str(row["objid"]),
            "view": int(row["view"]),
            "row_index": int(row["feature_index"]),
        }
        for column in TRACE_METADATA_COLUMNS:
            if column in row.index and pd.notna(row[column]):
                value = row[column]
                result[column] = value.item() if hasattr(value, "item") else value
        return result


def extract_feature_bank(
    extractor: nn.Module,
    dataset: Dataset,
    *,
    n_views: int,
    seed: int,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
    provenance: dict[str, object] | None = None,
) -> FeatureBank:
    """Estrae viste riproducibili in ordine stabile e conserva la loro identità."""
    if n_views < 1:
        raise ValueError("n_views deve essere almeno uno")
    target_device = torch.device(device)
    original_training = extractor.training
    extractor = extractor.to(target_device)
    extractor.eval()
    feature_chunks: list[torch.Tensor] = []
    records: list[dict[str, object]] = []

    try:
        with torch.no_grad():
            for view in range(n_views):
                view_seed = seed + view
                seed_everything(view_seed)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                for batch in loader:
                    output = extractor(batch["inputs"].to(target_device))
                    if not isinstance(output, torch.Tensor) or output.ndim != 2:
                        raise ValueError("L'estrattore deve restituire una matrice 2D")
                    output = output.detach().to(device="cpu", dtype=torch.float32)
                    feature_chunks.append(output)
                    labels = batch["label"].tolist()
                    row_indices = batch["row_index"].tolist()
                    for position, (objid, label, row_index) in enumerate(
                        zip(batch["objid"], labels, row_indices)
                    ):
                        record = {
                            "objid": str(objid),
                            "label": int(label),
                            "view": view,
                            "view_seed": view_seed,
                            "source_row_index": int(row_index),
                            "feature_index": len(records),
                        }
                        for column in TRACE_METADATA_COLUMNS:
                            if column in batch:
                                value = batch[column][position]
                                if isinstance(value, torch.Tensor):
                                    value = value.item()
                                record[column] = value
                        records.append(record)
    finally:
        extractor.train(original_training)

    return FeatureBank(
        features=torch.cat(feature_chunks, dim=0),
        metadata=pd.DataFrame(records),
        provenance=dict(provenance or {}),
    )
