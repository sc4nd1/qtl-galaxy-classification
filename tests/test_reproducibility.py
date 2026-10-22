import random

import numpy as np
import torch

from tesi_qml.reproducibility import config_fingerprint, derive_seed, seed_everything


def test_seed_everything_repeats_python_numpy_and_torch() -> None:
    """Catches partial seeding that makes model comparisons non-reproducible."""
    seed_everything(17)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    seed_everything(17)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_config_fingerprint_is_order_independent_and_change_sensitive():
    first = config_fingerprint({"model": "PL", "training": {"lr": 0.01, "epochs": 10}})
    reordered = config_fingerprint({"training": {"epochs": 10, "lr": 0.01}, "model": "PL"})
    changed = config_fingerprint({"model": "PL", "training": {"lr": 0.02, "epochs": 10}})
    assert first == reordered
    assert first != changed
    assert len(first) == 64


def test_derived_seed_is_stable_and_namespace_sensitive() -> None:
    """Catches Python hash randomization leaking into recorded experiment sub-seeds."""
    assert derive_seed(42, "model-shared-q4") == derive_seed(42, "model-shared-q4")
    assert derive_seed(42, "model-shared-q4") != derive_seed(42, "loader")
    assert derive_seed(42, "loader") != derive_seed(314, "loader")
    assert 0 <= derive_seed(42, "loader") < 2**31


def test_derived_seed_rejects_invalid_inputs() -> None:
    with np.testing.assert_raises_regex(ValueError, "non negativo"):
        derive_seed(-1, "loader")
    with np.testing.assert_raises_regex(ValueError, "namespace"):
        derive_seed(42, "")
