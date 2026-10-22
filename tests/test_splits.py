import pandas as pd
import pytest

from tesi_qml.data.splits import make_repeated_four_fold_splits, validate_split_table


def balanced_manifest(per_class: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"objid": f"{label}-{i}", "label": label, "class_index": class_index}
            for class_index, label in enumerate(("elliptical", "merger", "spiral"))
            for i in range(per_class)
        ]
    )


def test_repeated_splits_are_disjoint_complete_and_stratified() -> None:
    """Catches overlap, missing samples and class imbalance in saved experiment splits."""
    manifest = balanced_manifest()
    splits = make_repeated_four_fold_splits(manifest, seeds=[42, 314])

    validate_split_table(splits, manifest)

    for (_, iteration), group in splits.groupby(["repeat_seed", "iteration"]):
        sets = {name: set(part.objid) for name, part in group.groupby("split")}
        assert sets["train"].isdisjoint(sets["validation"])
        assert sets["train"].isdisjoint(sets["test"])
        assert sets["validation"].isdisjoint(sets["test"])
        assert set.union(*sets.values()) == set(manifest.objid)
        assert group.groupby(["split", "label"]).size().to_dict() == {
            ("test", "elliptical"): 2,
            ("test", "merger"): 2,
            ("test", "spiral"): 2,
            ("train", "elliptical"): 4,
            ("train", "merger"): 4,
            ("train", "spiral"): 4,
            ("validation", "elliptical"): 2,
            ("validation", "merger"): 2,
            ("validation", "spiral"): 2,
        }


def test_each_sample_is_test_and_validation_once_per_repeat() -> None:
    """Catches a fold rotation that overuses or never evaluates a sample."""
    splits = make_repeated_four_fold_splits(balanced_manifest(), seeds=[42])
    counts = splits.groupby(["objid", "split"]).size().unstack(fill_value=0)
    assert (counts["test"] == 1).all()
    assert (counts["validation"] == 1).all()
    assert (counts["train"] == 2).all()


def test_split_generation_is_reproducible() -> None:
    """Catches hidden global RNG state changing experiment membership."""
    manifest = balanced_manifest()
    first = make_repeated_four_fold_splits(manifest, seeds=[42])
    second = make_repeated_four_fold_splits(manifest, seeds=[42])
    pd.testing.assert_frame_equal(first, second)


def test_split_validation_rejects_overlap() -> None:
    """Catches corrupted split files before expensive training begins."""
    manifest = balanced_manifest()
    splits = make_repeated_four_fold_splits(manifest, seeds=[42])
    duplicate = splits.iloc[[0]].assign(split="test")
    corrupted = pd.concat([splits, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="duplicat|overlap"):
        validate_split_table(corrupted, manifest)
