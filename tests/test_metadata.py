from io import StringIO

import pandas as pd
import pytest

from tesi_qml.data.metadata import clean_metadata, read_candidate_csv, select_download_pool

THRESHOLDS = {"elliptical": 0.8, "spiral": 0.8, "merger": 0.4}


def candidate(
    objid: str,
    *,
    prob_elliptical: float = 0.0,
    prob_spiral: float = 0.0,
    prob_merger: float = 0.0,
    ra: float = 120.0,
    dec: float = 20.0,
    redshift: float = 0.05,
) -> dict[str, object]:
    return {
        "objid": objid,
        "ra": ra,
        "dec": dec,
        "prob_elliptical": prob_elliptical,
        "prob_spiral": prob_spiral,
        "prob_merger": prob_merger,
        "redshift": redshift,
    }


def test_read_candidate_csv_preserves_objid_beyond_float_precision() -> None:
    """Catches parsing a 10^18 SDSS identifier through float64."""
    exact = "1237654605318717697"
    text = (
        "objid,ra,dec,prob_elliptical,prob_spiral,prob_merger,redshift\n"
        f"{exact},120,20,0.91,0.03,0,0.05\n"
    )

    frame = read_candidate_csv(StringIO(text))

    assert frame.loc[0, "objid"] == exact
    assert str(frame["objid"].dtype) == "string"


def test_clean_metadata_assigns_only_mutually_exclusive_hard_labels() -> None:
    """Catches ambiguous candidates silently entering more than one class."""
    frame = pd.DataFrame(
        [
            candidate("1", prob_elliptical=0.9),
            candidate("2", prob_spiral=0.9),
            candidate("3", prob_merger=0.5),
            candidate("4", prob_elliptical=0.9, prob_merger=0.5),
        ]
    )

    result = clean_metadata(frame, THRESHOLDS)

    assert result.clean[["objid", "label"]].values.tolist() == [
        ["1", "elliptical"],
        ["2", "spiral"],
        ["3", "merger"],
    ]
    assert result.rejected.set_index("objid").loc["4", "reason"] == "class_conflict"


def test_clean_metadata_collapses_exact_duplicates_and_rejects_discordant_objid() -> None:
    """Catches duplicated spectra producing two incompatible records for one image."""
    exact = candidate("10", prob_elliptical=0.9)
    frame = pd.DataFrame(
        [
            exact,
            exact.copy(),
            candidate("11", prob_spiral=0.9, redshift=0.04),
            candidate("11", prob_spiral=0.9, redshift=0.06),
        ]
    )

    result = clean_metadata(frame, THRESHOLDS)

    assert result.clean.objid.tolist() == ["10"]
    assert set(result.rejected.loc[result.rejected.objid == "11", "reason"]) == {
        "duplicate_conflict"
    }
    assert result.summary["exact_duplicates_removed"] == 1


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (candidate("20", prob_elliptical=0.9, redshift=0.2), "redshift_out_of_range"),
        (candidate("21", prob_elliptical=1.2), "invalid_probability"),
        (candidate("22", prob_elliptical=0.9, ra=400), "invalid_coordinates"),
        (candidate("23", prob_elliptical=0.2), "no_hard_label"),
    ],
)
def test_clean_metadata_records_a_specific_rejection_reason(row: dict, reason: str) -> None:
    """Catches invalid scientific metadata being silently retained."""
    result = clean_metadata(pd.DataFrame([row]), THRESHOLDS)
    assert result.clean.empty
    assert result.rejected.iloc[0].reason == reason


def test_select_download_pool_keeps_all_mergers_and_is_deterministic() -> None:
    """Catches accidental undersampling of the rare class or unstable candidate pools."""
    rows = [candidate(str(i), prob_merger=0.5) | {"label": "merger"} for i in range(4)]
    rows += [
        candidate(str(100 + i), prob_elliptical=0.9) | {"label": "elliptical"} for i in range(10)
    ]
    rows += [candidate(str(200 + i), prob_spiral=0.9) | {"label": "spiral"} for i in range(10)]
    clean = pd.DataFrame(rows)

    first = select_download_pool(clean, seed=7, oversample_factor=1.25)
    second = select_download_pool(clean, seed=7, oversample_factor=1.25)

    assert first.objid.tolist() == second.objid.tolist()
    assert first.groupby("label").size().to_dict() == {
        "elliptical": 5,
        "merger": 4,
        "spiral": 5,
    }
    assert set(first.loc[first.label == "merger", "objid"]) == {"0", "1", "2", "3"}


def test_clean_metadata_rejects_float_objid_that_cannot_be_exact() -> None:
    """Catches accepting an identifier that was already rounded before cleaning."""
    frame = pd.DataFrame([candidate(1.2376546053187177e18, prob_elliptical=0.9)])
    with pytest.raises(ValueError, match="objid.*string"):
        clean_metadata(frame, THRESHOLDS)
