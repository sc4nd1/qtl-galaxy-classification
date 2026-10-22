from pathlib import Path

import numpy as np
from PIL import Image

from tesi_qml.data.validation import sha256_file, validate_image


def write_valid_jpeg(path: Path, size: tuple[int, int] = (256, 256)) -> None:
    y, x = np.indices((size[1], size[0]))
    pixels = np.stack(((x + y) % 256, x % 256, y % 256), axis=-1).astype(np.uint8)
    Image.fromarray(pixels).save(path, format="JPEG", quality=90)


def test_validate_image_accepts_decodable_rgb_256_jpeg(tmp_path: Path) -> None:
    """Catches validators that trust only filename or HTTP status."""
    path = tmp_path / "valid.jpg"
    write_valid_jpeg(path)

    result = validate_image(path, expected_size=(256, 256))

    assert result.valid
    assert result.reason == "valid"
    assert (result.width, result.height, result.mode) == (256, 256, "RGB")
    assert result.sha256 == sha256_file(path)
    assert result.std_luminance > 1.0


def test_validate_image_public_default_matches_320_download_contract(tmp_path: Path) -> None:
    path = tmp_path / "default.jpg"
    write_valid_jpeg(path, size=(320, 320))
    assert validate_image(path).valid


def test_validate_image_rejects_html_named_jpg(tmp_path: Path) -> None:
    """Catches saving an SDSS/HTTP error page as a training image."""
    path = tmp_path / "x.jpg"
    path.write_text("<html>error</html>", encoding="utf-8")

    result = validate_image(path)

    assert not result.valid
    assert result.reason == "decode_error"


def test_validate_image_rejects_wrong_dimensions(tmp_path: Path) -> None:
    """Catches inconsistent cutout dimensions entering a shared transform pipeline."""
    path = tmp_path / "small.jpg"
    write_valid_jpeg(path, size=(128, 128))
    assert validate_image(path).reason == "wrong_dimensions"


def test_validate_image_rejects_uniform_placeholder(tmp_path: Path) -> None:
    """Catches blank cutouts that decode correctly but contain no useful morphology."""
    path = tmp_path / "blank.jpg"
    Image.new("RGB", (256, 256), color=(10, 10, 10)).save(path, format="JPEG")
    assert validate_image(path, expected_size=(256, 256)).reason == "low_contrast"
