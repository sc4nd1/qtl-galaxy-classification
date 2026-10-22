import pandas as pd
import pytest
import torch

from tesi_qml.data.features import FeatureBank
from tesi_qml.tasks import TaskSpec, filter_feature_bank, filter_task_frame, get_task


def _feature_bank() -> FeatureBank:
    return FeatureBank(
        features=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]], dtype=torch.float32
        ),
        metadata=pd.DataFrame(
            [
                {
                    "objid": "e",
                    "label": 0,
                    "label_name": "elliptical",
                    "view": 0,
                    "view_seed": 11,
                    "feature_index": 0,
                },
                {
                    "objid": "m",
                    "label": 1,
                    "label_name": "merger",
                    "view": 0,
                    "view_seed": 11,
                    "feature_index": 1,
                },
                {
                    "objid": "s",
                    "label": 2,
                    "label_name": "spiral",
                    "view": 0,
                    "view_seed": 11,
                    "feature_index": 2,
                },
                {
                    "objid": "s",
                    "label": 2,
                    "label_name": "spiral",
                    "view": 1,
                    "view_seed": 12,
                    "feature_index": 3,
                },
            ]
        ),
        provenance={"manifest_sha256": "abc", "backbone": "resnet18"},
    )


def test_binary_filter_removes_mergers_and_remaps_spiral_without_losing_identity() -> None:
    """Catches binary banks that retain merger or keep the non-contiguous label 2."""
    filtered = filter_feature_bank(_feature_bank(), get_task("binary"))

    assert filtered.metadata[["objid", "label", "view", "view_seed"]].to_dict("records") == [
        {"objid": "e", "label": 0, "view": 0, "view_seed": 11},
        {"objid": "s", "label": 1, "view": 0, "view_seed": 11},
        {"objid": "s", "label": 1, "view": 1, "view_seed": 12},
    ]
    assert filtered.metadata["feature_index"].tolist() == [0, 1, 2]
    torch.testing.assert_close(filtered.features, _feature_bank().features[[0, 2, 3]])
    assert filtered.provenance["manifest_sha256"] == "abc"
    assert filtered.provenance["task"] == {
        "name": "binary",
        "class_names": ["elliptical", "spiral"],
        "class_to_index": {"elliptical": 0, "spiral": 1},
    }


def test_three_class_filter_preserves_canonical_labels() -> None:
    """Catches accidental relabeling of the three-class source-of-truth bank."""
    filtered = filter_feature_bank(_feature_bank(), get_task("three_class"))

    assert filtered.metadata["label"].tolist() == [0, 1, 2, 2]
    assert torch.equal(filtered.features, _feature_bank().features)


def test_binary_split_is_a_nested_filter_of_three_class_split() -> None:
    """Catches regenerating binary splits instead of filtering the three-class table."""
    source = pd.DataFrame(
        [
            {"objid": "e", "label": "elliptical", "class_index": 0, "split": "train"},
            {"objid": "m", "label": "merger", "class_index": 1, "split": "validation"},
            {"objid": "s", "label": "spiral", "class_index": 2, "split": "test"},
        ]
    )

    binary = filter_task_frame(source, get_task("binary"))
    three_class = filter_task_frame(source, get_task("three_class"))

    expected = three_class.loc[three_class["label"].isin(["elliptical", "spiral"])]
    assert binary.set_index("objid")["split"].to_dict() == expected.set_index("objid")[
        "split"
    ].to_dict()
    assert binary.set_index("objid")["class_index"].to_dict() == {"e": 0, "s": 1}


def test_task_spec_rejects_non_contiguous_mapping() -> None:
    """Catches CrossEntropy targets with gaps after a task filter."""
    with pytest.raises(ValueError, match="contigui"):
        TaskSpec(
            name="broken",
            class_names=("elliptical", "spiral"),
            class_to_index={"elliptical": 0, "spiral": 2},
        )


def test_get_task_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Task non supportato"):
        get_task("unknown")
