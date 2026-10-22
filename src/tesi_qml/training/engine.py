"""Motore di training comune a baseline e modelli quantistici."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.datasets import TRACE_METADATA_COLUMNS
from ..evaluation.predictions import prediction_frame
from ..reproducibility import config_fingerprint
from .checkpointing import CheckpointManager, EarlyStopping
from .metrics import compute_classification_metrics
from .parameters import validate_optimizer_parameters


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    last_epoch: int
    history: list[dict[str, Any]]
    stopped_early: bool


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _run_fingerprint(metadata: Mapping[str, object] | None) -> str:
    return config_fingerprint(dict(metadata or {}))


def _generator_state(loader: DataLoader) -> torch.Tensor | None:
    generator = getattr(loader, "generator", None)
    return generator.get_state() if generator is not None else None


def _restore_generator(loader: DataLoader, state: torch.Tensor | None) -> None:
    generator = getattr(loader, "generator", None)
    if generator is not None and state is not None:
        generator.set_state(state)


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        class_names: Sequence[str],
        run_dir: Path | str,
        device: torch.device | str = "cpu",
        scheduler: Any = None,
        early_stopping: EarlyStopping | None = None,
        gradient_clip_norm: float | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        validate_optimizer_parameters(model, optimizer)
        self.criterion = criterion
        self.class_names = tuple(class_names)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoints = CheckpointManager(self.run_dir / "checkpoints")
        self.stop_requested = False
        checksum = getattr(self.model, "frozen_quantum_checksum", None)
        self._frozen_quantum_checksum = checksum() if callable(checksum) else None

    def _verify_frozen_quantum_weights(self) -> None:
        if self._frozen_quantum_checksum is None:
            return
        current = self.model.frozen_quantum_checksum()
        if current != self._frozen_quantum_checksum:
            raise RuntimeError("I pesi quantistici congelati sono cambiati")

    def request_stop(self) -> None:
        """Chiede un arresto pulito al termine dell'epoca corrente."""
        self.stop_requested = True

    @staticmethod
    def _gradient_norm(model: nn.Module, *, quantum_only: bool = False) -> float:
        squared = 0.0
        for name, parameter in model.named_parameters():
            if quantum_only and "quantum" not in name.lower():
                continue
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().norm(2).item()) ** 2
        return math.sqrt(squared)

    def _epoch(
        self, loader: DataLoader, *, training: bool
    ) -> tuple[float, torch.Tensor, torch.Tensor, float, float, float]:
        self.model.train(training)
        total_loss = 0.0
        total_samples = 0
        logits_parts: list[torch.Tensor] = []
        labels_parts: list[torch.Tensor] = []
        gradient_norms: list[float] = []
        quantum_gradient_norms: list[float] = []

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                labels = batch["label"].to(self.device, dtype=torch.long)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise FloatingPointError("Logits o loss contengono NaN/Inf")
                if training:
                    loss.backward()
                    gradient_norms.append(self._gradient_norm(self.model))
                    quantum_gradient_norms.append(
                        self._gradient_norm(self.model, quantum_only=True)
                    )
                    if any(
                        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                        for parameter in self.model.parameters()
                    ):
                        raise FloatingPointError("I gradienti contengono NaN/Inf")
                    if self.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                    self._verify_frozen_quantum_weights()
                count = int(labels.shape[0])
                total_loss += float(loss.detach().item()) * count
                total_samples += count
                logits_parts.append(logits.detach().cpu())
                labels_parts.append(labels.detach().cpu())
        if total_samples == 0:
            raise ValueError("Il DataLoader non contiene campioni")
        mean_gradient = float(sum(gradient_norms) / len(gradient_norms)) if gradient_norms else 0.0
        mean_quantum_gradient = (
            float(sum(quantum_gradient_norms) / len(quantum_gradient_norms))
            if quantum_gradient_norms
            else 0.0
        )
        quantum_gradient_variance = (
            float(torch.tensor(quantum_gradient_norms, dtype=torch.float64).var(unbiased=False))
            if quantum_gradient_norms
            else 0.0
        )
        return (
            total_loss / total_samples,
            torch.cat(logits_parts),
            torch.cat(labels_parts),
            mean_gradient,
            mean_quantum_gradient,
            quantum_gradient_variance,
        )

    def _metric_summary(self, labels: torch.Tensor, logits: torch.Tensor) -> dict[str, Any]:
        probabilities = torch.softmax(logits, dim=1).numpy()
        return compute_classification_metrics(labels.numpy(), probabilities, self.class_names)

    def _write_history(self, history: list[dict[str, Any]]) -> None:
        lines = "".join(json.dumps(row, sort_keys=True) + "\n" for row in history)
        _atomic_text(self.run_dir / "history.jsonl", lines)

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        *,
        max_epochs: int,
        resume: bool = False,
        run_metadata: Mapping[str, object] | None = None,
    ) -> TrainingResult:
        if max_epochs < 1:
            raise ValueError("max_epochs deve essere almeno uno")
        history: list[dict[str, Any]] = []
        start_epoch = 1
        fingerprint = _run_fingerprint(run_metadata)
        if resume:
            self.checkpoints.prepare_resume()
            state = self.checkpoints.restore_last(
                self.model,
                self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
                map_location=self.device,
            )
            stored_fingerprint = state.get("extra", {}).get("run_fingerprint")
            if stored_fingerprint != fingerprint:
                raise ValueError("Checkpoint incompatibile con l'identità/configurazione del run")
            loader_states = state.get("extra", {}).get("loader_generator_states", {})
            _restore_generator(train_loader, loader_states.get("train"))
            _restore_generator(validation_loader, loader_states.get("validation"))
            history = list(state.get("history", []))
            start_epoch = int(state["epoch"]) + 1
        else:
            self.checkpoints.start_fresh()

        stopped_early = False
        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, max_epochs + 1):
            started = time.perf_counter()
            (
                train_loss,
                train_logits,
                train_labels,
                gradient_norm,
                quantum_gradient_norm,
                quantum_gradient_variance,
            ) = self._epoch(train_loader, training=True)
            validation_loss, validation_logits, validation_labels, _, _, _ = self._epoch(
                validation_loader, training=False
            )
            train_metrics = self._metric_summary(train_labels, train_logits)
            validation_metrics = self._metric_summary(validation_labels, validation_logits)
            should_stop = (
                self.early_stopping.step(validation_loss) if self.early_stopping else False
            )
            is_best = validation_loss < self.checkpoints.best_loss
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "validation_f1_macro": validation_metrics["f1"]["macro"],
                "train_precision_macro": train_metrics["precision"]["macro"],
                "train_recall_macro": train_metrics["recall"]["macro"],
                "train_f1_macro": train_metrics["f1"]["macro"],
                "validation_precision_macro": validation_metrics["precision"]["macro"],
                "validation_recall_macro": validation_metrics["recall"]["macro"],
                "gradient_norm": gradient_norm,
                "quantum_gradient_norm": quantum_gradient_norm,
                "quantum_gradient_variance": quantum_gradient_variance,
                "is_best": is_best,
                "early_stopping_bad_epochs": self.early_stopping.bad_epochs
                if self.early_stopping
                else 0,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            self._write_history(history)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(validation_loss)
                else:
                    self.scheduler.step()
            self.checkpoints.save(
                epoch=epoch,
                validation_loss=validation_loss,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
                history=history,
                extra={
                    "run_metadata": dict(run_metadata or {}),
                    "run_fingerprint": fingerprint,
                    "loader_generator_states": {
                        "train": _generator_state(train_loader),
                        "validation": _generator_state(validation_loader),
                    },
                },
            )
            last_epoch = epoch
            if should_stop or self.stop_requested:
                stopped_early = should_stop
                break

        if self.checkpoints.best_epoch is None:
            raise RuntimeError("Nessun checkpoint best disponibile")
        self.checkpoints.restore_best(self.model, map_location=self.device)
        self._verify_frozen_quantum_weights()
        inference_started = time.perf_counter()
        predictions = self.predict(validation_loader, run_metadata=run_metadata)
        inference_seconds = time.perf_counter() - inference_started
        probability_columns = [f"prob_{name}_model" for name in self.class_names]
        metrics = compute_classification_metrics(
            predictions["class_index"].to_numpy(),
            predictions[probability_columns].to_numpy(),
            self.class_names,
        )
        total_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        signature_method = getattr(self.model, "circuit_signature", None)
        metrics["runtime"] = {
            "training_seconds": float(sum(row["elapsed_seconds"] for row in history)),
            "inference_seconds": float(inference_seconds),
            "inference_samples_per_second": float(len(predictions) / inference_seconds)
            if inference_seconds > 0
            else None,
            "total_parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
            "best_epoch": int(self.checkpoints.best_epoch),
            "last_epoch": int(last_epoch),
            "stopped_early": bool(stopped_early),
            "stop_requested": bool(self.stop_requested),
            "cuda_max_memory_bytes": int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0,
        }
        if callable(signature_method):
            metrics["model_signature"] = signature_method()
        _atomic_text(
            self.run_dir / "metrics.json",
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        )
        temporary_predictions = self.run_dir / "predictions.parquet.part"
        predictions.to_parquet(temporary_predictions, index=False)
        temporary_predictions.replace(self.run_dir / "predictions.parquet")
        return TrainingResult(
            best_epoch=int(self.checkpoints.best_epoch),
            last_epoch=last_epoch,
            history=history,
            stopped_early=stopped_early,
        )

    def predict(
        self,
        loader: DataLoader,
        *,
        run_metadata: Mapping[str, object] | None = None,
    ) -> pd.DataFrame:
        self.model.eval()
        logits_parts: list[torch.Tensor] = []
        records: list[dict[str, object]] = []
        with torch.no_grad():
            for batch in loader:
                logits_parts.append(self.model(batch["inputs"].to(self.device)).detach().cpu())
                labels = batch["label"].tolist()
                objids = batch["objid"]
                row_indices = batch.get("row_index", range(len(labels)))
                if isinstance(row_indices, torch.Tensor):
                    row_indices = row_indices.tolist()
                for position, (objid, label, row_index) in enumerate(
                    zip(objids, labels, row_indices)
                ):
                    record = {
                        "objid": str(objid),
                        "class_index": int(label),
                        "row_index": int(row_index),
                    }
                    for column in TRACE_METADATA_COLUMNS:
                        if column in batch:
                            value = batch[column][position]
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                            record[column] = value
                    records.append(record)
        if not logits_parts:
            raise ValueError("Il DataLoader non contiene campioni")
        return prediction_frame(
            pd.DataFrame(records),
            torch.cat(logits_parts),
            self.class_names,
            run_metadata or {},
        )
