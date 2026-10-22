import pandas as pd

from tesi_qml.data.manifests import build_balanced_manifest


def validated_frame(counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    cursor = 0
    for label, count in counts.items():
        for _ in range(count):
            cursor += 1
            rows.append(
                {
                    "objid": str(10**18 + cursor),
                    "label": label,
                    "valid": True,
                    "sha256": f"hash-{cursor}",
                    "path": f"data/images/{label}/{cursor}.jpg",
                }
            )
    return pd.DataFrame(rows)


def test_balancing_uses_smallest_class_and_multiple_of_four() -> None:
    """Catches unequal class sizes or a sample count incompatible with four folds."""
    frame = validated_frame({"elliptical": 13, "spiral": 12, "merger": 11})

    final, summary = build_balanced_manifest(frame, seed=42)

    assert summary["per_class"] == 8
    assert final.groupby("label").size().to_dict() == {
        "elliptical": 8,
        "merger": 8,
        "spiral": 8,
    }
    assert final.objid.is_unique


def test_balancing_removes_cross_class_hash_conflicts() -> None:
    """Catches the same decoded image being assigned two different ground truths."""
    frame = validated_frame({"elliptical": 9, "spiral": 9, "merger": 9})
    frame.loc[0, "sha256"] = "shared"
    frame.loc[9, "sha256"] = "shared"

    final, summary = build_balanced_manifest(frame, seed=42)

    assert "shared" not in set(final.sha256)
    assert summary["cross_class_hash_rows_removed"] == 2
    shared_exclusions = [row for row in summary["excluded_samples"] if row["sha256"] == "shared"]
    assert {row["reason"] for row in shared_exclusions} == {"cross_class_duplicate_hash"}
    assert len(shared_exclusions) == 2


def test_balancing_keeps_only_one_intra_class_hash_duplicate() -> None:
    """Catches repeated cutouts inflating one class and leaking across folds."""
    frame = validated_frame({"elliptical": 10, "spiral": 10, "merger": 10})
    frame.loc[0:1, "sha256"] = "same-elliptical"

    final, summary = build_balanced_manifest(frame, seed=11)

    assert (final.sha256 == "same-elliptical").sum() <= 1
    assert summary["intra_class_hash_rows_removed"] == 1


def test_balancing_is_reproducible_for_the_same_seed() -> None:
    """Catches sampling drift between classical and quantum experiments."""
    frame = validated_frame({"elliptical": 20, "spiral": 20, "merger": 12})
    first, _ = build_balanced_manifest(frame, seed=7)
    second, _ = build_balanced_manifest(frame, seed=7)
    assert first.objid.tolist() == second.objid.tolist()
