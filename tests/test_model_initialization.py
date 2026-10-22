import torch

from tesi_qml.models.classical import CapacityMatchedHead
from tesi_qml.models.initialization import PairedInitialization, tensor_checksum


def test_paired_initialization_is_reproducible_without_global_rng_order() -> None:
    """Catches module construction order changing paired comparison weights."""
    first = PairedInitialization.create(
        input_dim=16,
        bottleneck_dim=4,
        num_classes=3,
        n_layers=2,
        shared_seed=7,
        quantum_seed=8,
        quantum_init_std=0.05,
    )
    torch.manual_seed(999)
    _ = torch.randn(100)
    second = PairedInitialization.create(
        input_dim=16,
        bottleneck_dim=4,
        num_classes=3,
        n_layers=2,
        shared_seed=7,
        quantum_seed=8,
        quantum_init_std=0.05,
    )

    assert first.checksums() == second.checksums()
    assert tensor_checksum(first.quantum_weight) == tensor_checksum(second.quantum_weight)
    torch.testing.assert_close(first.pre_weight, second.pre_weight, rtol=0, atol=0)


def test_paired_initialization_applies_exact_shared_layers_to_b2_and_q() -> None:
    """Catches B2 and Q starting from different projection or head parameters."""
    from tesi_qml.models.pennylane_qnn import PennylaneAngleQNN

    bundle = PairedInitialization.create(
        input_dim=16,
        bottleneck_dim=4,
        num_classes=2,
        n_layers=2,
        shared_seed=17,
        quantum_seed=18,
        quantum_init_std=0.05,
    )
    b2 = CapacityMatchedHead(input_dim=16, bottleneck_dim=4, num_classes=2)
    q = PennylaneAngleQNN(input_dim=16, n_qubits=4, n_layers=2, num_classes=2)

    bundle.apply(b2)
    bundle.apply(q)

    torch.testing.assert_close(b2.pre_head.weight, q.pre_quantum.weight, rtol=0, atol=0)
    torch.testing.assert_close(b2.pre_head.bias, q.pre_quantum.bias, rtol=0, atol=0)
    torch.testing.assert_close(b2.classifier.weight, q.post_quantum.weight, rtol=0, atol=0)
    torch.testing.assert_close(b2.classifier.bias, q.post_quantum.bias, rtol=0, atol=0)
    torch.testing.assert_close(q.quantum_weight(), bundle.quantum_weight, rtol=0, atol=0)
    assert bundle.checksums()["shared_projection"] == tensor_checksum(q.pre_quantum.weight)
    assert bundle.checksums()["quantum"] == tensor_checksum(q.quantum_weight())


def test_paired_initialization_changes_with_scientific_seed_or_task_shape() -> None:
    """Catches reused initial states across independent repeat seeds or tasks."""
    binary = PairedInitialization.create(16, 4, 2, 2, 1, 2, 0.05)
    binary_other_seed = PairedInitialization.create(16, 4, 2, 2, 3, 4, 0.05)
    three_class = PairedInitialization.create(16, 4, 3, 2, 1, 2, 0.05)

    assert binary.checksums()["shared_projection"] != binary_other_seed.checksums()[
        "shared_projection"
    ]
    assert binary.checksums()["shared_head"] != three_class.checksums()["shared_head"]
