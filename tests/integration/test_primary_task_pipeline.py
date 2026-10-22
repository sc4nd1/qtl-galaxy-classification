import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from tesi_qml.config import load_yaml
from tesi_qml.data.features import FeatureBank
from tesi_qml.matrix import is_run_complete


def _write_bank(root: Path, split: str) -> None:
    labels = [0, 0, 1, 1, 2, 2]
    names = ["elliptical", "elliptical", "merger", "merger", "spiral", "spiral"]
    bank = FeatureBank(
        features=torch.tensor(
            [
                [-2.0, 0.0, 0.0, 0.0],
                [-1.5, 0.1, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.1, 1.5, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [1.5, -0.1, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        metadata=pd.DataFrame(
            [
                {
                    "objid": f"{split}-{index}",
                    "label": label,
                    "label_name": name,
                    "view": 0,
                    "view_seed": 7,
                    "feature_index": index,
                }
                for index, (label, name) in enumerate(zip(labels, names))
            ]
        ),
        provenance={
            "manifest_sha256": "abc",
            "backbone": "resnet18-test",
            "feature_config_fingerprint": "def",
            "transform": f"{split}-test",
        },
    )
    bank.save(root / split)


@pytest.mark.parametrize(
    ("task_name", "model_name", "expected_mapping"),
    [
        ("binary", "B2", {"elliptical": 0, "spiral": 1}),
        pytest.param(
            "binary",
            "QF",
            {"elliptical": 0, "spiral": 1},
            marks=pytest.mark.quantum,
        ),
        pytest.param(
            "three_class",
            "Q",
            {"elliptical": 0, "merger": 1, "spiral": 2},
            marks=pytest.mark.quantum,
        ),
    ],
)
def test_primary_cli_pipeline_writes_dynamic_complete_artifacts(
    tmp_path: Path,
    task_name: str,
    model_name: str,
    expected_mapping: dict[str, int],
) -> None:
    """Exercise task filtering, paired init, PennyLane training and sealed test output."""
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    dataset = load_yaml("configs/dataset.yaml")
    training = load_yaml("configs/training.yaml")
    training.update({"max_epochs": 1, "patience": 1, "batch_size": 2, "comparison_batch_size": 2})
    (configs / "dataset.yaml").write_text(yaml.safe_dump(dataset), encoding="utf-8")
    (configs / "training.yaml").write_text(yaml.safe_dump(training), encoding="utf-8")
    feature_root = project / "data" / "features" / "seed_42" / "iteration_1"
    for split in ("train", "validation", "test"):
        _write_bank(feature_root, split)

    script = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(project),
            "--config",
            str(configs / "training.yaml"),
            "--dataset-config",
            str(configs / "dataset.yaml"),
            "--task",
            task_name,
            "--model",
            model_name,
            "--qubits",
            "4",
            "--repeat-seed",
            "42",
            "--iteration",
            "1",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    run_dirs = list((project / "results" / "runs" / task_name).rglob("run_config.json"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0].parent
    assert is_run_complete(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(run_dir / "test_predictions.parquet")
    metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    assert config["class_to_index"] == expected_mapping
    assert config["initialization_checksums"]["shared_projection"]
    expected_probabilities = {f"prob_{name}_model" for name in expected_mapping}
    assert expected_probabilities <= set(predictions.columns)
    assert set(predictions["class_index"]) == set(expected_mapping.values())
    if task_name == "binary":
        assert "prob_merger_model" not in predictions.columns
    if model_name == "QF":
        assert metrics["frozen_quantum_checksum"] == config["initialization_checksums"]["quantum"]
    if model_name == "Q":
        assert metrics["runtime"]["quantum_parameters"] > 0
