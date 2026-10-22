import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from tesi_qml.models.classical import (
    CapacityMatchedHead,
    ImageClassifier,
    LinearHead,
    MLPHead,
    set_backbone_stage,
)


@torch.no_grad()
@pytest.mark.parametrize("num_classes", [2, 3])
def test_feature_heads_return_task_logits(num_classes: int) -> None:
    """Catches hard-coded three-class outputs in binary comparison baselines."""
    inputs = torch.randn(5, 512)
    for head in (
        LinearHead(num_classes=num_classes),
        MLPHead(num_classes=num_classes),
        CapacityMatchedHead(bottleneck_dim=4, num_classes=num_classes),
        CapacityMatchedHead(bottleneck_dim=8, num_classes=num_classes),
    ):
        assert head(inputs).shape == (5, num_classes)


def test_capacity_matched_head_exposes_requested_bottleneck() -> None:
    """Catches a classical baseline with more latent features than its QNN comparator."""
    four = CapacityMatchedHead(bottleneck_dim=4)
    eight = CapacityMatchedHead(bottleneck_dim=8)
    assert four.pre_head.out_features == 4
    assert four.classifier.in_features == 4
    assert eight.pre_head.out_features == 8
    assert eight.classifier.in_features == 8


def test_frozen_backbone_has_no_trainable_parameters_and_stays_in_eval() -> None:
    """Catches BatchNorm statistics drifting while a frozen backbone trains."""
    backbone = resnet18(weights=None)
    backbone.fc = nn.Identity()
    model = ImageClassifier(backbone, LinearHead())

    set_backbone_stage(model, "frozen")
    model.train()

    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert not model.backbone.training
    assert model.head.training


def test_layer4_stage_trains_only_layer4_and_head() -> None:
    """Catches accidental full-ResNet fine-tuning on small subsets."""
    backbone = resnet18(weights=None)
    backbone.fc = nn.Identity()
    model = ImageClassifier(backbone, MLPHead())

    set_backbone_stage(model, "layer4")
    model.train()

    assert any(parameter.requires_grad for parameter in model.backbone.layer4.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.layer3.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert model.backbone.layer4.training
    assert not model.backbone.layer3.training


def test_image_classifier_forward_uses_backbone_features() -> None:
    """Catches bypassing the configured head during end-to-end fine-tuning."""
    backbone = resnet18(weights=None)
    backbone.fc = nn.Identity()
    model = ImageClassifier(backbone, CapacityMatchedHead(bottleneck_dim=4))
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 3)
