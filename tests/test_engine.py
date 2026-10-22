import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from tesi_qml.models.initialization import tensor_checksum
from tesi_qml.training.engine import Trainer


class TinyDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor(
            [[-2.0, 0.0], [-1.5, 0.1], [0.0, 2.0], [0.1, 1.5], [2.0, 0.0], [1.5, -0.1]]
        )
        self.y = torch.tensor([0, 0, 1, 1, 2, 2])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            "inputs": self.x[index],
            "label": self.y[index],
            "objid": str(1000 + index),
            "row_index": index,
            "redshift": 0.02 + index * 0.001,
            "sha256": f"hash-{index}",
            "path": f"images/{index}.jpg",
        }


class FrozenMutationModel(torch.nn.Module):
    def __init__(self, *, mutate_during_training: bool) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(2, 3)
        self.circuit_weight = torch.nn.Parameter(torch.tensor([0.25]), requires_grad=False)
        self.mutate_during_training = mutate_during_training

    def forward(self, inputs):
        if self.training and self.mutate_during_training:
            with torch.no_grad():
                self.circuit_weight.add_(1.0)
        return self.classifier(inputs)

    def frozen_quantum_checksum(self):
        return tensor_checksum(self.circuit_weight)


def _trainer(run_dir: Path) -> Trainer:
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 3)
    return Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.08),
        criterion=torch.nn.CrossEntropyLoss(),
        class_names=("elliptical", "spiral", "merger"),
        run_dir=run_dir,
        device="cpu",
    )


def test_trainer_rejects_optimizer_with_frozen_parameters(tmp_path: Path) -> None:
    """Catches a QF circuit tensor accidentally included in AdamW."""
    model = FrozenMutationModel(mutate_during_training=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="congelati"):
        Trainer(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            class_names=("elliptical", "merger", "spiral"),
            run_dir=tmp_path,
        )


def test_trainer_detects_frozen_quantum_weight_mutation(tmp_path: Path) -> None:
    """Catches hidden circuit state drift even when it is absent from the optimizer."""
    model = FrozenMutationModel(mutate_during_training=True)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.01)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        class_names=("elliptical", "merger", "spiral"),
        run_dir=tmp_path,
    )
    loader = DataLoader(TinyDataset(), batch_size=3, shuffle=False)

    with pytest.raises(RuntimeError, match="pesi quantistici congelati"):
        trainer.fit(loader, loader, max_epochs=1)


def test_resume_continues_epoch_history_and_optimizer_state(tmp_path: Path):
    loader = DataLoader(TinyDataset(), batch_size=3, shuffle=False)
    first = _trainer(tmp_path)
    result_two = first.fit(loader, loader, max_epochs=2)
    assert result_two.last_epoch == 2

    resumed = _trainer(tmp_path)
    result_four = resumed.fit(loader, loader, max_epochs=4, resume=True)

    assert result_four.last_epoch == 4
    assert [row["epoch"] for row in result_four.history] == [1, 2, 3, 4]
    checkpoint = torch.load(tmp_path / "checkpoints" / "last.pt", weights_only=False)
    assert checkpoint["epoch"] == 4
    assert checkpoint["optimizer_state"]["state"]


def test_mini_training_writes_traceable_artifacts(tmp_path: Path):
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
    trainer = _trainer(tmp_path)
    result = trainer.fit(loader, loader, max_epochs=8, run_metadata={"fold": 0, "seed": 7})

    assert result.best_epoch >= 1
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "history.jsonl").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "predictions.parquet").is_file()

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert metrics["n_samples"] == len(TinyDataset())
    assert {
        "objid",
        "predicted_class",
        "confidence",
        "fold",
        "seed",
        "redshift",
        "sha256",
        "path",
    } <= set(predictions.columns)
    assert len(predictions) == len(TinyDataset())
    expected_epoch_fields = {
        "gradient_norm",
        "quantum_gradient_norm",
        "quantum_gradient_variance",
        "train_precision_macro",
        "train_recall_macro",
        "validation_precision_macro",
        "validation_recall_macro",
        "is_best",
        "early_stopping_bad_epochs",
        "elapsed_seconds",
    }
    assert all(expected_epoch_fields <= set(row) for row in result.history)
    assert {
        "training_seconds",
        "inference_seconds",
        "total_parameters",
        "trainable_parameters",
        "best_epoch",
    } <= set(metrics["runtime"])


def test_resume_rejects_incompatible_run_identity(tmp_path: Path):
    loader = DataLoader(TinyDataset(), batch_size=3, shuffle=False)
    _trainer(tmp_path).fit(loader, loader, max_epochs=1, run_metadata={"model": "B0"})
    with pytest.raises(ValueError, match="incompatibile"):
        _trainer(tmp_path).fit(
            loader,
            loader,
            max_epochs=2,
            resume=True,
            run_metadata={"model": "QISKIT", "n_qubits": 8},
        )


def test_resume_restores_rng_and_dataloader_generator(tmp_path: Path):
    def make_loader(seed):
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(TinyDataset(), batch_size=2, shuffle=True, generator=generator)

    uninterrupted_dir = tmp_path / "uninterrupted"
    interrupted_dir = tmp_path / "interrupted"
    _trainer(uninterrupted_dir).fit(
        make_loader(101), make_loader(202), max_epochs=4, run_metadata={"model": "B0"}
    )
    _trainer(interrupted_dir).fit(
        make_loader(101), make_loader(202), max_epochs=2, run_metadata={"model": "B0"}
    )
    _trainer(interrupted_dir).fit(
        make_loader(999),
        make_loader(888),
        max_epochs=4,
        resume=True,
        run_metadata={"model": "B0"},
    )
    uninterrupted = torch.load(uninterrupted_dir / "checkpoints" / "last.pt", weights_only=False)[
        "model_state"
    ]
    resumed = torch.load(interrupted_dir / "checkpoints" / "last.pt", weights_only=False)[
        "model_state"
    ]
    for key in uninterrupted:
        assert torch.equal(uninterrupted[key], resumed[key])
