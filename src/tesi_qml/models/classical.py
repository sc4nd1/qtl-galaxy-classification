"""Baseline classiche generali e capacity-matched per 4/8 qubit."""

from __future__ import annotations

import math

import torch
from torch import nn
from torchvision import models


def scale_angles(values: torch.Tensor, mode: str = "tanh_pi") -> torch.Tensor:
    """Applica la stessa compressione usata prima dei circuiti angle encoding."""
    if mode == "tanh_pi_half":
        return torch.tanh(values) * (math.pi / 2.0)
    if mode == "tanh_pi":
        return torch.tanh(values) * math.pi
    if mode == "sigmoid_2pi":
        return torch.sigmoid(values) * (2.0 * math.pi)
    raise ValueError(f"Modalità di scaling non supportata: {mode}")


class LinearHead(nn.Module):
    def __init__(self, input_dim: int = 512, num_classes: int = 3) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class CapacityMatchedHead(nn.Module):
    """Baseline che attraversa lo stesso collo di bottiglia 4D/8D della QNN."""

    def __init__(
        self,
        input_dim: int = 512,
        bottleneck_dim: int = 4,
        num_classes: int = 3,
        angle_scale: str = "tanh_pi",
    ) -> None:
        super().__init__()
        if bottleneck_dim not in (4, 8):
            raise ValueError("Il bottleneck principale deve avere 4 o 8 dimensioni")
        self.pre_head = nn.Linear(input_dim, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        self.angle_scale = angle_scale

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        latent = scale_angles(self.pre_head(features), self.angle_scale)
        return self.classifier(latent)


class ImageClassifier(nn.Module):
    """Combina una backbone ResNet-like e una testa feature-based."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        if not hasattr(backbone, "layer4"):
            raise ValueError("La backbone deve esporre layer4 per il fine-tuning progressivo")
        self.backbone = backbone
        self.head = head
        self._backbone_stage = "full"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))

    def train(self, mode: bool = True) -> ImageClassifier:
        super().train(mode)
        if self._backbone_stage == "frozen":
            self.backbone.eval()
        elif self._backbone_stage == "layer4":
            self.backbone.eval()
            self.backbone.layer4.train(mode)
        return self


def set_backbone_stage(model: ImageClassifier, stage: str) -> None:
    """Seleziona frozen, layer4 o full senza lasciare BatchNorm nascoste in train."""
    if stage not in {"frozen", "layer4", "full"}:
        raise ValueError("stage deve essere frozen, layer4 o full")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = stage == "full"
    if stage == "layer4":
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    model._backbone_stage = stage
    model.train(model.training)


def resnet18_backbone(*, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()
    return backbone
