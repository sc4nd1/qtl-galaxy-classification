"""Checkpoint atomici e arresto anticipato riproducibile."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    best: float = float("inf")
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience deve essere almeno uno")
        if self.min_delta < 0:
            raise ValueError("min_delta non può essere negativo")

    def step(self, value: float) -> bool:
        """Aggiorna lo stato e restituisce True quando occorre fermarsi."""
        if value < self.best - self.min_delta:
            self.best = float(value)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, float | int]:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best = float(state["best"])
        self.bad_epochs = int(state["bad_epochs"])


class CheckpointManager:
    """Gestisce checkpoint ``best`` e ``last`` senza lasciare file parziali."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_path = self.directory / "best.pt"
        self.last_path = self.directory / "last.pt"
        self.best_loss = float("inf")
        self.best_epoch: int | None = None

    def start_fresh(self) -> None:
        if self.best_path.exists() or self.last_path.exists():
            raise FileExistsError(
                f"La directory {self.directory} contiene checkpoint: usare --resume o un nuovo run"
            )
        self.best_loss = float("inf")
        self.best_epoch = None

    def prepare_resume(self) -> None:
        if not self.best_path.is_file() or not self.last_path.is_file():
            raise FileNotFoundError("Per il resume servono sia best.pt sia last.pt")
        state = torch.load(self.best_path, map_location="cpu", weights_only=False)
        self.best_loss = float(state["validation_loss"])
        self.best_epoch = int(state["epoch"])

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_state(state: dict[str, Any] | None) -> None:
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])

    @staticmethod
    def _atomic_save(state: dict[str, Any], path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".part")
        torch.save(state, temporary)
        temporary.replace(path)

    def save(
        self,
        *,
        epoch: int,
        validation_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        history: list[dict[str, Any]],
        scheduler: Any = None,
        early_stopping: EarlyStopping | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        state = {
            "epoch": int(epoch),
            "validation_loss": float(validation_loss),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "early_stopping_state": (
                early_stopping.state_dict() if early_stopping is not None else None
            ),
            "history": history,
            "extra": extra or {},
            "rng_state": self._rng_state(),
        }
        self._atomic_save(state, self.last_path)
        improved = validation_loss < self.best_loss
        if improved:
            self.best_loss = float(validation_loss)
            self.best_epoch = int(epoch)
            self._atomic_save(state, self.best_path)
        return improved

    @staticmethod
    def _restore(
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        early_stopping: EarlyStopping | None = None,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = False,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(state["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        if early_stopping is not None and state.get("early_stopping_state") is not None:
            early_stopping.load_state_dict(state["early_stopping_state"])
        if restore_rng:
            CheckpointManager._restore_rng_state(state.get("rng_state"))
        return state

    def restore_best(
        self, model: nn.Module, *, map_location: str | torch.device = "cpu"
    ) -> dict[str, Any]:
        return self._restore(self.best_path, model, map_location=map_location, restore_rng=False)

    def restore_last(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Any = None,
        early_stopping: EarlyStopping | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        return self._restore(
            self.last_path,
            model,
            optimizer,
            scheduler,
            early_stopping,
            map_location,
            True,
        )
