from pathlib import Path

import pytest

from tesi_qml.config import load_yaml, resolve_project_path


def test_load_yaml_and_resolve_relative_path(tmp_path: Path) -> None:
    """Catches silent cwd-dependent path resolution and malformed config loading."""
    cfg = tmp_path / "x.yaml"
    cfg.write_text("data_dir: data/images\n", encoding="utf-8")

    assert load_yaml(cfg)["data_dir"] == "data/images"
    assert resolve_project_path(tmp_path, "data/images") == tmp_path / "data/images"


def test_load_yaml_rejects_non_mapping_root(tmp_path: Path) -> None:
    """Catches accepting a YAML list where downstream code expects named settings."""
    cfg = tmp_path / "x.yaml"
    cfg.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(TypeError, match="mapping"):
        load_yaml(cfg)


def test_resolve_project_path_keeps_absolute_path(tmp_path: Path) -> None:
    """Catches accidentally prefixing an already absolute cluster path."""
    absolute = (tmp_path / "external").resolve()
    assert resolve_project_path(tmp_path / "project", str(absolute)) == absolute


def test_training_and_augmentation_seeds_are_explicit_and_distinct() -> None:
    """Prevents two conceptually different randomness sources sharing a hidden seed."""
    config = load_yaml(Path("configs/training.yaml"))

    assert isinstance(config["training_seed"], int)
    assert isinstance(config["augmentation_seed"], int)
    assert config["training_seed"] != config["augmentation_seed"]


def test_primary_protocol_is_locked_to_approved_tasks_seeds_and_models() -> None:
    """Catches accidental expansion of the preregistered primary matrix."""
    dataset = load_yaml(Path("configs/dataset.yaml"))
    training = load_yaml(Path("configs/training.yaml"))

    assert dataset["splits"]["repeat_seeds"] == [
        42,
        314,
        2718,
        1618,
        2026,
        7,
        73,
        137,
        1729,
        4096,
    ]
    assert dataset["splits"]["primary_iterations"] == [1]
    assert dataset["tasks"] == {
        "binary": {
            "class_names": ["elliptical", "spiral"],
            "class_to_index": {"elliptical": 0, "spiral": 1},
        },
        "three_class": {
            "class_names": ["elliptical", "merger", "spiral"],
            "class_to_index": {"elliptical": 0, "merger": 1, "spiral": 2},
        },
    }
    assert training["feature_bank"]["train_views"] == 1
    assert training["quantum_init_std"] == 0.05
    assert training["optimizer"]["learning_rate"] == 0.001
    assert training["primary_matrix"]["models"] == [
        {"name": "B0"},
        {"name": "B1"},
        {"name": "B2", "qubits": 4},
        {"name": "B2", "qubits": 8},
        {"name": "QF", "qubits": 4},
        {"name": "QF", "qubits": 8},
        {"name": "Q", "qubits": 4},
        {"name": "Q", "qubits": 8},
    ]
    assert training["primary_matrix"]["tasks"] == ["binary", "three_class"]
    assert training["primary_matrix"]["depth"] == 2
    assert training["primary_matrix"]["subset_size"] == "full"
