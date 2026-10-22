"""QNN angle encoding implementata con PennyLane."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn.functional as F
from torch import nn

from .circuit_spec import AngleCircuitSpec
from .classical import scale_angles
from .initialization import tensor_checksum


class PennylaneAngleQNN(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 512,
        n_qubits: int = 4,
        n_layers: int = 2,
        num_classes: int = 3,
        angle_scale: str = "tanh_pi",
        init_std: float = 0.05,
        backend: str = "default.qubit",
        shots: int | None = None,
        quantum_frozen: bool = False,
    ) -> None:
        super().__init__()
        self.spec = AngleCircuitSpec(n_qubits=n_qubits, n_layers=n_layers)
        self.angle_scale = angle_scale
        self.quantum_frozen = bool(quantum_frozen)
        self.pre_quantum = nn.Linear(input_dim, n_qubits)
        device = qml.device(backend, wires=n_qubits, shots=shots)
        diff_method = (
            "backprop" if shots is None and backend == "default.qubit" else "parameter-shift"
        )
        spec = self.spec

        @qml.qnode(device, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for wire in range(spec.n_qubits):
                qml.RY(inputs[..., wire], wires=wire)
            for layer in range(spec.n_layers):
                for wire in range(spec.n_qubits):
                    qml.RZ(weights[layer, wire, 0], wires=wire)
                    qml.RY(weights[layer, wire, 1], wires=wire)
                for control, target in spec.entangling_pairs:
                    qml.CNOT(wires=[control, target])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(spec.n_qubits)]

        def initialise(tensor: torch.Tensor) -> torch.Tensor:
            return nn.init.normal_(tensor, mean=0.0, std=init_std)

        self.quantum_layer = qml.qnn.TorchLayer(
            circuit,
            {"weights": self.spec.weight_shape},
            init_method={"weights": initialise},
        )
        self.quantum_weight().requires_grad_(not self.quantum_frozen)
        self.post_quantum = nn.Linear(n_qubits, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        angles = scale_angles(self.pre_quantum(features), self.angle_scale)
        quantum_features = self.quantum_layer(angles).to(dtype=features.dtype)
        return self.post_quantum(quantum_features)

    def circuit_signature(self) -> dict[str, object]:
        signature = self.spec.signature(framework="pennylane")
        signature["angle_scale"] = self.angle_scale
        return signature

    def quantum_weight(self) -> nn.Parameter:
        """Restituisce l'unico tensore variazionale con shape circuitale canonica."""
        return self.quantum_layer.weights

    def quantum_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.quantum_weight(),)

    def frozen_quantum_checksum(self) -> str | None:
        return tensor_checksum(self.quantum_weight()) if self.quantum_frozen else None


class PennylaneReuploadingQNN(nn.Module):
    """Angle encoding ripetuto a ogni layer variazionale."""

    def __init__(
        self,
        *,
        input_dim: int = 512,
        n_qubits: int = 4,
        n_layers: int = 2,
        num_classes: int = 3,
        angle_scale: str = "tanh_pi",
        init_std: float = 0.05,
        backend: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        super().__init__()
        self.spec = AngleCircuitSpec(n_qubits=n_qubits, n_layers=n_layers)
        self.angle_scale = angle_scale
        self.pre_quantum = nn.Linear(input_dim, n_layers * n_qubits)
        device = qml.device(backend, wires=n_qubits, shots=shots)
        diff_method = (
            "backprop" if shots is None and backend == "default.qubit" else "parameter-shift"
        )
        spec = self.spec

        @qml.qnode(device, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for layer in range(spec.n_layers):
                for wire in range(spec.n_qubits):
                    qml.RY(inputs[..., layer * spec.n_qubits + wire], wires=wire)
                    qml.Rot(
                        weights[layer, wire, 0],
                        weights[layer, wire, 1],
                        weights[layer, wire, 2],
                        wires=wire,
                    )
                for control, target in spec.entangling_pairs:
                    qml.CNOT(wires=[control, target])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(spec.n_qubits)]

        def initialise(tensor: torch.Tensor) -> torch.Tensor:
            return nn.init.normal_(tensor, mean=0.0, std=init_std)

        self.weight_shape = (n_layers, n_qubits, 3)
        self.quantum_layer = qml.qnn.TorchLayer(
            circuit,
            {"weights": self.weight_shape},
            init_method={"weights": initialise},
        )
        self.post_quantum = nn.Linear(n_qubits, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        angles = scale_angles(self.pre_quantum(features), self.angle_scale)
        quantum_features = self.quantum_layer(angles).to(dtype=features.dtype)
        return self.post_quantum(quantum_features)

    def circuit_signature(self) -> dict[str, object]:
        signature = self.spec.signature(framework="pennylane")
        signature.update(
            {
                "encoding": "RY data re-uploading",
                "input_shape": (self.spec.n_layers * self.spec.n_qubits,),
                "trainable_gates_per_layer": ("Rot",),
                "weight_shape": self.weight_shape,
                "angle_scale": self.angle_scale,
            }
        )
        return signature


class PennylaneAmplitudeQNN(nn.Module):
    """Prepara uno stato di ampiezze normalizzato prima dell'ansatz variazionale."""

    def __init__(
        self,
        *,
        input_dim: int = 512,
        n_qubits: int = 4,
        n_layers: int = 2,
        num_classes: int = 3,
        init_std: float = 0.05,
        backend: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        super().__init__()
        self.spec = AngleCircuitSpec(n_qubits=n_qubits, n_layers=n_layers)
        self.state_dimension = 2**n_qubits
        self.pre_quantum = nn.Linear(input_dim, self.state_dimension)
        device = qml.device(backend, wires=n_qubits, shots=shots)
        diff_method = (
            "backprop" if shots is None and backend == "default.qubit" else "parameter-shift"
        )
        spec = self.spec
        wires = tuple(range(n_qubits))

        @qml.qnode(device, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=wires, normalize=False)
            qml.StronglyEntanglingLayers(weights, wires=wires)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(spec.n_qubits)]

        def initialise(tensor: torch.Tensor) -> torch.Tensor:
            return nn.init.normal_(tensor, mean=0.0, std=init_std)

        self.weight_shape = (n_layers, n_qubits, 3)
        self.quantum_layer = qml.qnn.TorchLayer(
            circuit,
            {"weights": self.weight_shape},
            init_method={"weights": initialise},
        )
        self.post_quantum = nn.Linear(n_qubits, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        amplitudes = F.normalize(self.pre_quantum(features), p=2, dim=-1, eps=1e-12)
        quantum_features = self.quantum_layer(amplitudes).to(dtype=features.dtype)
        return self.post_quantum(quantum_features)

    def circuit_signature(self) -> dict[str, object]:
        return {
            "framework": "pennylane",
            "n_qubits": self.spec.n_qubits,
            "n_layers": self.spec.n_layers,
            "encoding": "amplitude",
            "state_dimension": self.state_dimension,
            "ansatz": "StronglyEntanglingLayers",
            "observables": self.spec.observables,
            "weight_shape": self.weight_shape,
        }
