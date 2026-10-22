"""Validazione reale dei cutout astronomici prima del training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class ImageValidation:
    valid: bool
    reason: str
    path: str
    sha256: str | None = None
    byte_size: int | None = None
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    mean_luminance: float | None = None
    std_luminance: float | None = None


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _invalid(path: Path, reason: str, **values) -> ImageValidation:
    return ImageValidation(valid=False, reason=reason, path=str(path), **values)


def validate_image(
    path: Path | str,
    *,
    expected_size: tuple[int, int] = (320, 320),
    min_bytes: int = 512,
    min_std_luminance: float = 1.0,
) -> ImageValidation:
    """Decodifica e controlla dimensioni, modo e contenuto informativo di un JPEG."""
    image_path = Path(path)
    if not image_path.is_file():
        return _invalid(image_path, "missing_file")

    byte_size = image_path.stat().st_size
    try:
        with Image.open(image_path) as probe:
            probe.verify()
        with Image.open(image_path) as decoded:
            width, height = decoded.size
            mode = decoded.mode
            if decoded.format != "JPEG":
                return _invalid(
                    image_path,
                    "wrong_format",
                    byte_size=byte_size,
                    width=width,
                    height=height,
                    mode=mode,
                )
            if (width, height) != expected_size:
                return _invalid(
                    image_path,
                    "wrong_dimensions",
                    byte_size=byte_size,
                    width=width,
                    height=height,
                    mode=mode,
                )
            if mode != "RGB":
                return _invalid(
                    image_path,
                    "wrong_mode",
                    byte_size=byte_size,
                    width=width,
                    height=height,
                    mode=mode,
                )
            rgb = np.asarray(decoded.convert("RGB"), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError):
        return _invalid(image_path, "decode_error", byte_size=byte_size)

    if byte_size < min_bytes:
        return _invalid(
            image_path,
            "too_small",
            byte_size=byte_size,
            width=width,
            height=height,
            mode=mode,
        )

    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    mean_luminance = float(luminance.mean())
    std_luminance = float(luminance.std())
    digest = sha256_file(image_path)
    if std_luminance < min_std_luminance:
        return _invalid(
            image_path,
            "low_contrast",
            sha256=digest,
            byte_size=byte_size,
            width=width,
            height=height,
            mode=mode,
            mean_luminance=mean_luminance,
            std_luminance=std_luminance,
        )

    return ImageValidation(
        valid=True,
        reason="valid",
        path=str(image_path),
        sha256=digest,
        byte_size=byte_size,
        width=width,
        height=height,
        mode=mode,
        mean_luminance=mean_luminance,
        std_luminance=std_luminance,
    )
