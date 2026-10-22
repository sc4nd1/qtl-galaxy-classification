from torchvision.transforms import InterpolationMode

from tesi_qml.data.transforms import build_eval_transform, build_train_transform


def test_rotation_precedes_center_crop() -> None:
    """Catches black rotation corners entering the 224px training crop."""
    transform = build_train_transform(224)
    names = [type(step).__name__ for step in transform.transforms]
    assert names.index("RandomRotation") < names.index("CenterCrop")


def test_rotation_uses_bilinear_interpolation() -> None:
    """Catches nearest-neighbor artifacts that alter galaxy morphology."""
    transform = build_train_transform(224)
    rotation = next(
        step for step in transform.transforms if type(step).__name__ == "RandomRotation"
    )
    assert rotation.interpolation is InterpolationMode.BILINEAR


def test_eval_transform_contains_no_random_operation() -> None:
    """Catches stochastic validation/test metrics."""
    names = [type(step).__name__ for step in build_eval_transform(224).transforms]
    assert names == ["CenterCrop", "ToTensor", "Normalize"]
