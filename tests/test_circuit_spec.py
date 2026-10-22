import pytest

from tesi_qml.models.circuit_spec import AngleCircuitSpec


@pytest.mark.parametrize("qubits", [4, 8])
def test_angle_spec_counts_parameters_and_linear_entanglement(qubits: int) -> None:
    """Catches hidden capacity/topology differences in the framework comparison."""
    spec = AngleCircuitSpec(n_qubits=qubits, n_layers=2)
    assert spec.n_input_parameters == qubits
    assert spec.n_weight_parameters == 2 * 2 * qubits
    assert spec.weight_shape == (2, qubits, 2)
    assert spec.entangling_pairs == tuple((i, i + 1) for i in range(qubits - 1))
    assert spec.observables == tuple(f"Z{i}" for i in range(qubits))


@pytest.mark.parametrize("qubits", [3, 5, 6])
def test_angle_spec_rejects_out_of_scope_qubit_counts(qubits: int) -> None:
    """Catches accidental experiments outside the approved 4-vs-8 comparison."""
    with pytest.raises(ValueError, match="4 o 8"):
        AngleCircuitSpec(n_qubits=qubits, n_layers=2)


def test_angle_spec_rejects_unsupported_depth() -> None:
    """Catches unplanned depth changes that invalidate matched comparisons."""
    with pytest.raises(ValueError, match="1, 2 o 4"):
        AngleCircuitSpec(n_qubits=4, n_layers=3)
