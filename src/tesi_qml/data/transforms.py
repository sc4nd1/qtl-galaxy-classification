"""Trasformazioni coerenti con il buffer geometrico dei cutout SDSS 320px."""

from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """Ruota prima del crop per eliminare i bordi introdotti dalla rotazione."""
    return transforms.Compose(
        [
            transforms.RandomRotation(
                degrees=(0.0, 360.0),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Trasformazione deterministica per validation e test."""
    return transforms.Compose(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
