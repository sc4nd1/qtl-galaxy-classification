"""Costruzione e validazione degli optimizer per modelli classici e ibridi."""

from __future__ import annotations

from torch import nn
from torch.optim import Optimizer


def trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Restituisce soltanto parametri che partecipano al gradiente."""
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("Il modello non contiene parametri addestrabili")
    return parameters


def _optimizer_parameters(optimizer: Optimizer) -> tuple[nn.Parameter, ...]:
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    )


def validate_optimizer_parameters(model: nn.Module, optimizer: Optimizer) -> None:
    """Fallisce se l'optimizer include congelati o omette addestrabili."""
    optimizer_parameters = _optimizer_parameters(optimizer)
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    frozen = [parameter for parameter in optimizer_parameters if not parameter.requires_grad]
    if frozen:
        raise ValueError("L'optimizer contiene parametri congelati")
    expected_ids = {id(parameter) for parameter in trainable_parameters(model)}
    missing = expected_ids - optimizer_ids
    extra = optimizer_ids - {id(parameter) for parameter in model.parameters()}
    if missing:
        raise ValueError("L'optimizer omette parametri addestrabili del modello")
    if extra:
        raise ValueError("L'optimizer contiene parametri estranei al modello")
