from pathlib import Path

import pytest
import torch

from tesi_qml.training.checkpointing import CheckpointManager, EarlyStopping


class _CudaMappedRngState:
    """Simula un RNG tensor spostato da ``torch.load(map_location='cuda')``."""

    def __init__(self, cpu_state: torch.Tensor) -> None:
        self.cpu_state = cpu_state

    def cpu(self) -> torch.Tensor:
        return self.cpu_state


def test_restore_rng_state_moves_map_location_state_back_to_cpu():
    """Catches CUDA checkpoint resumes passing a CUDA RNG tensor to the CPU generator."""
    saved = CheckpointManager._rng_state()
    mapped = {**saved, "torch": _CudaMappedRngState(saved["torch"])}

    CheckpointManager._restore_rng_state(mapped)

    assert torch.equal(torch.get_rng_state(), saved["torch"])


def test_best_checkpoint_restores_best_epoch_not_last(tmp_path: Path):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path)

    for epoch, loss in enumerate([1.0, 0.8, 0.9], start=1):
        with torch.no_grad():
            model.weight.fill_(epoch)
        manager.save(
            epoch=epoch,
            validation_loss=loss,
            model=model,
            optimizer=optimizer,
            history=[{"epoch": epoch}],
        )

    assert manager.best_epoch == 2
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    manager.restore_best(model)
    assert model.weight.item() == pytest.approx(2.0)


def test_early_stopping_uses_strict_improvement_and_patience():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert not stopper.step(1.0)
    assert not stopper.step(0.995)
    assert stopper.step(0.994)
    assert stopper.best == pytest.approx(1.0)


def test_fresh_manager_does_not_reuse_existing_best(tmp_path: Path):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    first = CheckpointManager(tmp_path)
    with torch.no_grad():
        model.weight.fill_(123.0)
    first.save(
        epoch=1,
        validation_loss=0.1,
        model=model,
        optimizer=optimizer,
        history=[],
    )

    fresh = CheckpointManager(tmp_path)
    with pytest.raises(FileExistsError, match="--resume"):
        fresh.start_fresh()
    assert fresh.best_epoch is None
