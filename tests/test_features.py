from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from tesi_qml.data.features import FeatureBank, FeatureBankDataset, extract_feature_bank


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "inputs": torch.tensor([float(index), float(index + 1)]),
            "label": index % 3,
            "objid": f"obj-{index}",
            "row_index": index,
            "redshift": 0.01 + index * 0.001,
            "prob_merger": 0.7 if index == 1 else 0.1,
            "sha256": f"hash-{index}",
            "path": f"images/{index}.jpg",
        }


def test_feature_bank_is_reproducible_for_view_seed() -> None:
    """Catches augmentation/view RNG drift between model families."""
    extractor = torch.nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        extractor.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2))

    first = extract_feature_bank(extractor, TinyDataset(), n_views=2, seed=11, batch_size=2)
    second = extract_feature_bank(extractor, TinyDataset(), n_views=2, seed=11, batch_size=2)

    assert torch.equal(first.features, second.features)
    pd.testing.assert_frame_equal(first.metadata, second.metadata)
    assert first.features.shape == (6, 4)
    assert first.metadata.groupby("view").size().to_dict() == {0: 3, 1: 3}
    assert {"redshift", "prob_merger", "sha256", "path"} <= set(first.metadata.columns)


def test_feature_bank_round_trip_and_dataset(tmp_path: Path) -> None:
    """Catches feature/metadata row misalignment after serialization."""
    bank = FeatureBank(
        features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        metadata=pd.DataFrame(
            [
                {"objid": "a", "label": 0, "view": 0, "view_seed": 7, "feature_index": 0},
                {"objid": "b", "label": 1, "view": 0, "view_seed": 7, "feature_index": 1},
            ]
        ),
        provenance={"manifest_sha256": "abc", "backbone": "resnet18", "transform": "eval-v1"},
    )
    bank.save(tmp_path)
    loaded = FeatureBank.load(tmp_path)
    dataset = FeatureBankDataset(loaded)

    assert torch.equal(loaded.features, bank.features)
    assert loaded.provenance == bank.provenance
    assert dataset[1]["objid"] == "b"
    assert dataset[1]["label"] == 1
    assert torch.equal(dataset[1]["inputs"], torch.tensor([3.0, 4.0]))
