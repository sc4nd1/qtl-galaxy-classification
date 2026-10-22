"""Specifica immutabile dei circuiti PennyLane."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AngleCircuitSpec:
    n_qubits: int
    n_layers: int = 2

    def __post_init__(self) -> None:
        if self.n_qubits not in (4, 8):
            raise ValueError("Il confronto approvato usa 4 o 8 qubit")
        if self.n_layers not in (1, 2, 4):
            raise ValueError("La profondità approvata è 1, 2 o 4 layer")

    @property
    def n_input_parameters(self) -> int:
        return self.n_qubits

    @property
    def n_weight_parameters(self) -> int:
        return self.n_layers * self.n_qubits * 2

    @property
    def weight_shape(self) -> tuple[int, int, int]:
        return (self.n_layers, self.n_qubits, 2)

    @property
    def entangling_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((wire, wire + 1) for wire in range(self.n_qubits - 1))

    @property
    def observables(self) -> tuple[str, ...]:
        return tuple(f"Z{wire}" for wire in range(self.n_qubits))

    def signature(self, *, framework: str) -> dict[str, object]:
        return {
            "framework": framework,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "encoding": "RY",
            "trainable_gates_per_layer": ("RZ", "RY"),
            "entangling_gate": "CNOT",
            "entangling_pairs": self.entangling_pairs,
            "observables": self.observables,
            "n_input_parameters": self.n_input_parameters,
            "n_weight_parameters": self.n_weight_parameters,
            "weight_shape": self.weight_shape,
        }
