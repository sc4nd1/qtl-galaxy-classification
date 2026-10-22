"""Dataset indicizzati esclusivamente dal manifest, senza enumerazione filesystem."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

TRACE_METADATA_COLUMNS = (
    "label_name",
    "path",
    "sha256",
    "redshift",
    "prob_elliptical",
    "prob_merger",
    "prob_spiral",
    "ra",
    "dec",
)


class ManifestImageDataset(Dataset):
    """Carica immagini seguendo l'ordine e l'identità salvati nel manifest."""

    REQUIRED: ClassVar[set[str]] = {"objid", "label", "class_index", "path"}

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        project_root: Path | str,
        transform: Callable | None = None,
    ) -> None:
        missing = sorted(self.REQUIRED - set(manifest.columns))
        if missing:
            raise ValueError(f"Colonne mancanti nel manifest immagine: {missing}")
        if manifest["objid"].duplicated().any():
            raise ValueError("ObjID duplicati nel dataset")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.manifest["objid"] = self.manifest["objid"].astype("string")
        self.manifest["path"] = self.manifest["path"].map(
            lambda value: str(value).replace("\\", "/")
        )
        self.project_root = Path(project_root).resolve()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.manifest.iloc[index]
        image_path = self._resolve_path(str(row["path"]))
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            inputs = self.transform(rgb) if self.transform is not None else rgb.copy()
        result = {
            "inputs": inputs,
            "label": int(row["class_index"]),
            "label_name": str(row["label"]),
            "objid": str(row["objid"]),
            "path": str(image_path),
            "row_index": index,
        }
        for column in TRACE_METADATA_COLUMNS:
            source_column = "label" if column == "label_name" else column
            if source_column in row.index and pd.notna(row[source_column]):
                value = row[source_column]
                result[column] = value.item() if hasattr(value, "item") else value
        result["path"] = str(image_path)
        return result
