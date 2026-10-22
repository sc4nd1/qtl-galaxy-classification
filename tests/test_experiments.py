from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from tesi_qml.config import load_yaml
from tesi_qml.experiments import RunSpec, build_run_identity, run_directory


def _configs():
    return load_yaml("configs/dataset.yaml"), load_yaml("configs/training.yaml")


def test_run_identity_records_task_freeze_state_subseeds_and_checksums() -> None:
    """Catches incomplete provenance that cannot distinguish QF from Q."""
    dataset, training = _configs()
    spec = RunSpec("binary", "QF", 4, 42, 1, 2, "full")

    identity = build_run_identity(
        spec,
        dataset_config=dataset,
        training_config=training,
        feature_bank_provenance={"train": {"manifest_sha256": "abc"}},
        initialization_checksums={"shared_projection": "p", "quantum": "q"},
    )

    assert identity["task"] == "binary"
    assert identity["class_names"] == ["elliptical", "spiral"]
    assert identity["class_to_index"] == {"elliptical": 0, "spiral": 1}
    assert identity["quantum_frozen"] is True
    assert identity["initialization_checksums"]["quantum"] == "q"
    assert set(identity["sub_seeds"]) == {
        "split",
        "feature_view",
        "model",
        "shared_initialization",
        "quantum_initialization",
        "dataloader",
        "libraries",
    }
    assert len(identity["config_fingerprint"]) == 64


def test_run_fingerprint_changes_with_every_scientific_identity_field() -> None:
    """Catches resume paths accepting checkpoints from a different experiment."""
    dataset, training = _configs()
    base = RunSpec("binary", "Q", 4, 42, 1, 2, "full")

    def fingerprint(spec, dataset_config=dataset):
        return build_run_identity(
            spec, dataset_config=dataset_config, training_config=training
        )["config_fingerprint"]

    base_fingerprint = fingerprint(base)
    for changed in (
        replace(base, task="three_class"),
        replace(base, qubits=8),
        replace(base, model="QF"),
        replace(base, repeat_seed=314),
    ):
        assert fingerprint(changed) != base_fingerprint

    changed_mapping = deepcopy(dataset)
    changed_mapping["tasks"]["binary"] = {
        "class_names": ["spiral", "elliptical"],
        "class_to_index": {"spiral": 0, "elliptical": 1},
    }
    assert fingerprint(base, changed_mapping) != base_fingerprint


def test_paired_models_share_initialization_subseeds_but_not_model_seed() -> None:
    """Catches B2/QF/Q pairing through fragile global construction order."""
    dataset, training = _configs()
    identities = {
        model: build_run_identity(
            RunSpec("three_class", model, 8, 73),
            dataset_config=dataset,
            training_config=training,
        )
        for model in ("B2", "QF", "Q")
    }

    for namespace in ("shared_initialization", "quantum_initialization"):
        assert len({identity["sub_seeds"][namespace] for identity in identities.values()}) == 1
    assert len({identity["sub_seeds"]["model"] for identity in identities.values()}) == 3


def test_run_directory_is_task_and_fingerprint_scoped(tmp_path: Path) -> None:
    spec = RunSpec("binary", "Q", 4, 42)
    path = run_directory(tmp_path, spec, "a" * 64)

    assert path == (
        tmp_path
        / "results"
        / "runs"
        / "binary"
        / "q_q4"
        / "seed_42"
        / "iteration_1"
        / "config_aaaaaaaaaaaa"
    )


@pytest.mark.parametrize(
    "spec,message",
    [
        (RunSpec("binary", "Q", None, 42), "qubit"),
        (RunSpec("binary", "B0", 4, 42), "qubit"),
        (RunSpec("binary", "unknown", None, 42), "modello"),
        (RunSpec("binary", "Q", 4, 42, iteration=2), "iterazione"),
    ],
)
def test_run_spec_rejects_out_of_scope_primary_identity(spec, message) -> None:
    with pytest.raises(ValueError, match=message):
        spec.validate_primary()
