"""Inizializzazioni esplicite e appaiate per B2, QF e Q."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch
from torch import nn


def tensor_checksum(tensor: torch.Tensor) -> str:
    """Calcola uno SHA-256 sensibile a shape, dtype e byte del tensore."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _linear_tensors(
    out_features: int,
    in_features: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    bound = 1.0 / math.sqrt(in_features)
    weight = torch.empty(out_features, in_features, dtype=torch.float32)
    bias = torch.empty(out_features, dtype=torch.float32)
    weight.uniform_(-bound, bound, generator=generator)
    bias.uniform_(-bound, bound, generator=generator)
    return weight, bias


@dataclass(frozen=True)
class PairedInitialization:
    """Tensori condivisi rigenerabili indipendentemente dall'ordine dei moduli."""

    pre_weight: torch.Tensor
    pre_bias: torch.Tensor
    post_weight: torch.Tensor
    post_bias: torch.Tensor
    quantum_weight: torch.Tensor

    @classmethod
    def create(
        cls,
        input_dim: int,
        bottleneck_dim: int,
        num_classes: int,
        n_layers: int,
        shared_seed: int,
        quantum_seed: int,
        quantum_init_std: float = 0.05,
    ) -> PairedInitialization:
        if min(input_dim, bottleneck_dim, num_classes, n_layers) < 1:
            raise ValueError("Le dimensioni di inizializzazione devono essere positive")
        if shared_seed < 0 or quantum_seed < 0:
            raise ValueError("I seed di inizializzazione devono essere non negativi")
        if quantum_init_std <= 0:
            raise ValueError("quantum_init_std deve essere positivo")
        shared_generator = torch.Generator(device="cpu").manual_seed(shared_seed)
        quantum_generator = torch.Generator(device="cpu").manual_seed(quantum_seed)
        pre_weight, pre_bias = _linear_tensors(
            bottleneck_dim, input_dim, generator=shared_generator
        )
        post_weight, post_bias = _linear_tensors(
            num_classes, bottleneck_dim, generator=shared_generator
        )
        quantum_weight = torch.empty(n_layers, bottleneck_dim, 2, dtype=torch.float32).normal_(
            mean=0.0, std=quantum_init_std, generator=quantum_generator
        )
        return cls(pre_weight, pre_bias, post_weight, post_bias, quantum_weight)

    def checksums(self) -> dict[str, str]:
        return {
            "shared_projection": tensor_checksum(self.pre_weight),
            "shared_projection_bias": tensor_checksum(self.pre_bias),
            "shared_head": tensor_checksum(self.post_weight),
            "shared_head_bias": tensor_checksum(self.post_bias),
            "quantum": tensor_checksum(self.quantum_weight),
        }

    @staticmethod
    def _copy_linear(layer: nn.Linear, weight: torch.Tensor, bias: torch.Tensor) -> None:
        if tuple(layer.weight.shape) != tuple(weight.shape) or layer.bias is None:
            raise ValueError("Shape del layer lineare incompatibile con l'inizializzazione")
        if tuple(layer.bias.shape) != tuple(bias.shape):
            raise ValueError("Shape del bias incompatibile con l'inizializzazione")
        with torch.no_grad():
            layer.weight.copy_(weight.to(device=layer.weight.device, dtype=layer.weight.dtype))
            layer.bias.copy_(bias.to(device=layer.bias.device, dtype=layer.bias.dtype))

    def apply(self, model: nn.Module) -> None:
        """Applica i tensori ai nomi standard B2 o PennyLane QF/Q."""
        if hasattr(model, "pre_head") and hasattr(model, "classifier"):
            pre_layer = model.pre_head
            post_layer = model.classifier
        elif hasattr(model, "pre_quantum") and hasattr(model, "post_quantum"):
            pre_layer = model.pre_quantum
            post_layer = model.post_quantum
        else:
            raise TypeError("Il modello non espone layer condivisi B2/QF/Q")
        if not isinstance(pre_layer, nn.Linear) or not isinstance(post_layer, nn.Linear):
            raise TypeError("I layer condivisi devono essere torch.nn.Linear")
        self._copy_linear(pre_layer, self.pre_weight, self.pre_bias)
        self._copy_linear(post_layer, self.post_weight, self.post_bias)

        quantum_weight_method = getattr(model, "quantum_weight", None)
        if callable(quantum_weight_method):
            target = quantum_weight_method()
            if tuple(target.shape) != tuple(self.quantum_weight.shape):
                raise ValueError("Shape dei pesi quantistici incompatibile")
            with torch.no_grad():
                target.copy_(self.quantum_weight.to(device=target.device, dtype=target.dtype))
