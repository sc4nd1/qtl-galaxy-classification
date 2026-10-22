import pytest
import torch

from scripts.train import build_model


@pytest.mark.parametrize("num_classes", [2, 3])
def test_build_model_passes_task_class_count_to_classical_heads(num_classes: int) -> None:
    """Catches the train CLI silently constructing three-logit binary models."""
    inputs = torch.randn(2, 16)

    b0 = build_model("B0", input_dim=16, num_classes=num_classes)
    b1 = build_model("B1", input_dim=16, num_classes=num_classes)
    b2 = build_model("B2", input_dim=16, num_classes=num_classes, qubits=4)

    assert b0(inputs).shape == (2, num_classes)
    assert b1(inputs).shape == (2, num_classes)
    assert b2(inputs).shape == (2, num_classes)


def test_build_model_rejects_unknown_primary_model() -> None:
    with pytest.raises(ValueError, match="Modello non supportato"):
        build_model("UNKNOWN", input_dim=16, num_classes=2)
