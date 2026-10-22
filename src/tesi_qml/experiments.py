"""Identità minime, stabili e riproducibili dei run primari."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reproducibility import config_fingerprint, derive_seed
from .tasks import TaskSpec, get_task

PRIMARY_MODELS = ("B0", "B1", "B2", "QF", "Q")
LEGACY_MODELS = ("PL", "PL_REUP", "PL_AMP")


@dataclass(frozen=True)
class RunSpec:
    task: str
    model: str
    qubits: int | None
    repeat_seed: int
    iteration: int = 1
    n_layers: int = 2
    subset_size: str = "full"

    def validate(self) -> None:
        get_task(self.task)
        if self.model not in PRIMARY_MODELS + LEGACY_MODELS:
            raise ValueError(f"modello non supportato: {self.model}")
        if self.model not in {"B0", "B1"} and self.qubits not in (4, 8):
            raise ValueError("Il modello richiede 4 o 8 qubit")
        if self.model in {"B0", "B1"} and self.qubits is not None:
            raise ValueError("B0 e B1 non hanno una dimensione in qubit")
        if self.repeat_seed < 0:
            raise ValueError("Il repeat seed deve essere non negativo")
        if self.iteration not in (1, 2, 3, 4):
            raise ValueError("L'iterazione deve essere compresa tra 1 e 4")
        if self.n_layers not in (1, 2, 4):
            raise ValueError("La profondità deve essere 1, 2 o 4")

    def validate_primary(self) -> None:
        if self.model not in PRIMARY_MODELS:
            raise ValueError(f"modello primario non supportato: {self.model}")
        self.validate()
        if self.iteration != 1:
            raise ValueError("Il confronto primario usa soltanto iterazione 1")
        if self.n_layers != 2:
            raise ValueError("Il confronto primario usa profondità 2")
        if self.subset_size != "full":
            raise ValueError("Il confronto primario usa il dataset completo")

    @property
    def quantum_frozen(self) -> bool:
        return self.model == "QF"

    @property
    def model_tag(self) -> str:
        suffix = f"_q{self.qubits}" if self.qubits is not None else ""
        return f"{self.model.lower()}{suffix}"

    @property
    def run_id(self) -> str:
        return config_fingerprint(
            {
                "task": self.task,
                "model": self.model,
                "qubits": self.qubits,
                "repeat_seed": self.repeat_seed,
                "iteration": self.iteration,
                "n_layers": self.n_layers,
                "subset_size": self.subset_size,
            }
        )


def _configured_task(dataset_config: dict[str, Any], task_name: str) -> TaskSpec:
    configured = dataset_config.get("tasks", {}).get(task_name)
    if configured is None:
        return get_task(task_name)
    return TaskSpec(
        name=task_name,
        class_names=tuple(configured["class_names"]),
        class_to_index={str(key): int(value) for key, value in configured["class_to_index"].items()},
    )


def _sub_seeds(spec: RunSpec) -> dict[str, int]:
    paired_namespace = f"{spec.task}:q{spec.qubits}"
    return {
        "split": derive_seed(spec.repeat_seed, "split"),
        "feature_view": derive_seed(spec.repeat_seed, "feature-view"),
        "model": derive_seed(spec.repeat_seed, f"model:{spec.task}:{spec.model}:q{spec.qubits}"),
        "shared_initialization": derive_seed(
            spec.repeat_seed, f"shared-initialization:{paired_namespace}"
        ),
        "quantum_initialization": derive_seed(
            spec.repeat_seed, f"quantum-initialization:{paired_namespace}"
        ),
        "dataloader": derive_seed(spec.repeat_seed, f"dataloader:{spec.task}"),
        "libraries": derive_seed(spec.repeat_seed, f"libraries:{spec.task}:{spec.model}"),
    }


def build_run_identity(
    spec: RunSpec,
    *,
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    feature_bank_provenance: dict[str, Any] | None = None,
    initialization_checksums: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Materializza provenance e fingerprint usati da resume e aggregazione."""
    spec.validate()
    task = _configured_task(dataset_config, spec.task)
    scientific_identity = {
        "task": task.name,
        "class_names": list(task.class_names),
        "class_to_index": dict(task.class_to_index),
        "model": spec.model,
        "n_qubits": spec.qubits,
        "n_layers": spec.n_layers,
        "quantum_frozen": spec.quantum_frozen,
        "subset_size": spec.subset_size,
        "training_config": training_config,
        "dataset_protocol": {
            "thresholds": dataset_config.get("thresholds"),
            "splits": dataset_config.get("splits"),
            "task": task.as_dict(),
        },
    }
    identity = {
        **scientific_identity,
        "experiment_fingerprint": config_fingerprint(scientific_identity),
        "repeat_seed": spec.repeat_seed,
        "iteration": spec.iteration,
        "fold": spec.iteration,
        "sub_seeds": _sub_seeds(spec),
        "feature_bank_provenance": dict(feature_bank_provenance or {}),
        "initialization_checksums": dict(initialization_checksums or {}),
    }
    identity["config_fingerprint"] = config_fingerprint(identity)
    return identity


def run_directory(project_root: Path | str, spec: RunSpec, fingerprint: str) -> Path:
    """Restituisce il path deterministico degli artefatti di un run."""
    return (
        Path(project_root)
        / "results"
        / "runs"
        / spec.task
        / spec.model_tag
        / f"seed_{spec.repeat_seed}"
        / f"iteration_{spec.iteration}"
        / f"config_{fingerprint[:12]}"
    )
