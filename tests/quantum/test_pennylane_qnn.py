import pytest
import torch
from torch.optim import AdamW

pytest.importorskip("pennylane")

from tesi_qml.models.initialization import PairedInitialization, tensor_checksum
from tesi_qml.models.pennylane_qnn import (
    PennylaneAmplitudeQNN,
    PennylaneAngleQNN,
    PennylaneReuploadingQNN,
)
from tesi_qml.training.parameters import trainable_parameters, validate_optimizer_parameters


@pytest.mark.quantum
@pytest.mark.parametrize("qubits", [4, 8])
def test_pennylane_angle_qnn_forward_backward_for_approved_qubits(qubits: int) -> None:
    """Catches broken batching or disconnected classical/quantum gradients."""
    torch.manual_seed(3)
    model = PennylaneAngleQNN(input_dim=16, n_qubits=qubits, n_layers=2, init_std=0.05)
    inputs = torch.randn(2, 16)
    targets = torch.tensor([0, 2])

    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert model.pre_quantum.weight.grad is not None
    assert model.pre_quantum.weight.grad.norm().item() > 0
    q_grads = [p.grad for p in model.quantum_layer.parameters()]
    assert q_grads and all(grad is not None and torch.isfinite(grad).all() for grad in q_grads)


@pytest.mark.quantum
@pytest.mark.parametrize("model_cls", [PennylaneReuploadingQNN, PennylaneAmplitudeQNN])
def test_pennylane_alternative_qnns_forward_backward(model_cls) -> None:
    """Catches alternative PennyLane models with disconnected feature or circuit gradients."""
    torch.manual_seed(11)
    model = model_cls(input_dim=16, n_qubits=4, n_layers=2, init_std=0.05)
    inputs = torch.randn(2, 16)
    targets = torch.tensor([1, 2])

    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert model.pre_quantum.weight.grad is not None
    assert model.pre_quantum.weight.grad.norm().item() > 0
    q_grads = [p.grad for p in model.quantum_layer.parameters()]
    assert q_grads and all(grad is not None and torch.isfinite(grad).all() for grad in q_grads)


@pytest.mark.quantum
def test_pennylane_reuploading_uses_one_angle_block_per_layer() -> None:
    """Catches re-uploading variants that accidentally reuse one encoding block."""
    model = PennylaneReuploadingQNN(input_dim=16, n_qubits=4, n_layers=2)

    assert model.pre_quantum.out_features == 8
    assert model.circuit_signature()["encoding"] == "RY data re-uploading"


@pytest.mark.quantum
def test_pennylane_amplitude_qnn_uses_power_of_two_state_vector() -> None:
    """Catches amplitude variants that cannot prepare a valid 4-qubit state."""
    model = PennylaneAmplitudeQNN(input_dim=16, n_qubits=4, n_layers=1)

    assert model.pre_quantum.out_features == 16
    assert model.circuit_signature()["encoding"] == "amplitude"


@pytest.mark.quantum
def test_frozen_qnn_keeps_classical_gradients_and_quantum_weights_bit_identical() -> None:
    """Catches QF implementations that detach inputs or update frozen circuit weights."""
    torch.manual_seed(23)
    model = PennylaneAngleQNN(
        input_dim=8,
        n_qubits=4,
        n_layers=2,
        num_classes=2,
        quantum_frozen=True,
    )
    initialization = PairedInitialization.create(8, 4, 2, 2, 30, 31, 0.05)
    initialization.apply(model)
    before = model.quantum_weight().detach().clone()
    optimizer = AdamW(trainable_parameters(model), lr=1e-3)

    logits = model(torch.randn(2, 8))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()

    assert logits.shape == (2, 2)
    assert model.pre_quantum.weight.grad is not None
    assert model.pre_quantum.weight.grad.norm().item() > 0
    assert model.post_quantum.weight.grad is not None
    assert model.post_quantum.weight.grad.norm().item() > 0
    assert model.quantum_weight().grad is None
    optimizer.step()
    torch.testing.assert_close(before, model.quantum_weight(), rtol=0, atol=0)
    assert tensor_checksum(before) == model.frozen_quantum_checksum()


@pytest.mark.quantum
def test_frozen_and_trainable_qnn_receive_identical_quantum_initialization() -> None:
    """Catches QF and Q controls starting from different circuit states."""
    initialization = PairedInitialization.create(8, 4, 3, 2, 40, 41, 0.05)
    qf = PennylaneAngleQNN(input_dim=8, quantum_frozen=True)
    q = PennylaneAngleQNN(input_dim=8, quantum_frozen=False)

    initialization.apply(qf)
    initialization.apply(q)

    torch.testing.assert_close(qf.quantum_weight(), q.quantum_weight(), rtol=0, atol=0)
    assert not qf.quantum_weight().requires_grad
    assert q.quantum_weight().requires_grad


@pytest.mark.quantum
def test_optimizer_validation_rejects_frozen_quantum_parameters() -> None:
    """Catches QF optimizers that silently carry the frozen circuit tensor."""
    model = PennylaneAngleQNN(input_dim=8, quantum_frozen=True)
    invalid = AdamW(model.parameters(), lr=1e-3)
    valid = AdamW(trainable_parameters(model), lr=1e-3)

    with pytest.raises(ValueError, match="congelati"):
        validate_optimizer_parameters(model, invalid)
    validate_optimizer_parameters(model, valid)
