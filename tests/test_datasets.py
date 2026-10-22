from pathlib import Path

import pandas as pd
from PIL import Image

from tesi_qml.data.datasets import ManifestImageDataset


def test_manifest_dataset_uses_manifest_identity_not_filesystem_order(tmp_path: Path) -> None:
    """Catches train/eval index drift caused by separate os.listdir calls."""
    for name, color in (("z.jpg", "red"), ("a.jpg", "blue")):
        Image.new("RGB", (320, 320), color=color).save(tmp_path / name, format="JPEG")
    manifest = pd.DataFrame(
        [
            {"objid": "Z", "label": "spiral", "class_index": 2, "path": "z.jpg"},
            {"objid": "A", "label": "elliptical", "class_index": 0, "path": "a.jpg"},
        ]
    )

    dataset = ManifestImageDataset(manifest, project_root=tmp_path, transform=lambda image: image)

    assert dataset[0]["objid"] == "Z"
    assert dataset[0]["label"] == 2
    assert dataset[0]["path"].endswith("z.jpg")
    assert dataset[1]["objid"] == "A"


def test_manifest_dataset_rejects_duplicate_objid(tmp_path: Path) -> None:
    """Catches duplicate samples before they can cross split boundaries."""
    manifest = pd.DataFrame(
        [
            {"objid": "1", "label": "merger", "class_index": 1, "path": "x.jpg"},
            {"objid": "1", "label": "merger", "class_index": 1, "path": "y.jpg"},
        ]
    )
    try:
        ManifestImageDataset(manifest, project_root=tmp_path)
    except ValueError as exc:
        assert "ObjID" in str(exc)
    else:
        raise AssertionError("duplicate ObjID accepted")


def test_manifest_dataset_normalizes_windows_paths_for_linux(tmp_path: Path) -> None:
    """Catches Windows separators making portable manifests unreadable on Colab."""
    image_path = tmp_path / "data" / "images" / "spiral" / "1.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), color="blue").save(image_path, format="JPEG")
    manifest = pd.DataFrame(
        [
            {
                "objid": "1",
                "label": "spiral",
                "class_index": 2,
                "path": r"data\images\spiral\1.jpg",
            }
        ]
    )

    dataset = ManifestImageDataset(manifest, project_root=tmp_path)

    assert dataset.manifest.loc[0, "path"] == "data/images/spiral/1.jpg"
    assert dataset[0]["objid"] == "1"
