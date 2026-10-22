import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from scripts.colab import create_colab_bundles, import_physical_dataset


def test_colab_environments_pin_the_pennylane_compatible_jax_pair():
    """Catches Colab's preinstalled JAX breaking the pinned PennyLane import."""
    project_root = Path(__file__).resolve().parents[1]

    for filename in ("requirements-cluster.txt", "requirements-colab.txt"):
        requirements = {
            line.split("==", maxsplit=1)[0]: line.split("==", maxsplit=1)[1]
            for line in (project_root / filename).read_text(encoding="utf-8").splitlines()
            if "==" in line and not line.lstrip().startswith("#")
        }
        assert requirements.get("jax") == "0.6.2"
        assert requirements.get("jaxlib") == "0.6.2"


def _write_dataset_config(root: Path) -> None:
    config = {
        "project_seed": 42,
        "source_csv": "data/source/GalaxyCandidates.csv",
        "candidate_manifest": "data/manifests/candidates.parquet",
        "download_manifest": "data/manifests/downloads.parquet",
        "final_manifest": "data/manifests/final.parquet",
        "image_dir": "data/images",
        "split_dir": "data/splits",
        "classes": {"elliptical": 0, "merger": 1, "spiral": 2},
        "download": {"width": 320, "height": 320},
        "splits": {"n_folds": 4, "repeat_seeds": [42, 314, 2718, 1618, 2026]},
    }
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "dataset.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _write_source_csv(root: Path, objids_by_class: dict[str, list[str]]) -> None:
    rows = []
    for label, objids in objids_by_class.items():
        for offset, objid in enumerate(objids):
            rows.append(
                {
                    "objid": objid,
                    "ra": 120.0 + offset,
                    "dec": 2.0 + offset,
                    "redshift": 0.03,
                    "prob_elliptical": 0.9 if label == "elliptical" else 0.05,
                    "prob_merger": 0.6 if label == "merger" else 0.05,
                    "prob_spiral": 0.9 if label == "spiral" else 0.05,
                }
            )
    (root / "data" / "source").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / "data" / "source" / "GalaxyCandidates.csv", index=False)


def _write_physical_images(
    root: Path,
    objids_by_class: dict[str, list[str]],
    *,
    image_size: tuple[int, int] = (320, 320),
) -> None:
    for label, objids in objids_by_class.items():
        class_dir = root / label
        class_dir.mkdir(parents=True)
        for objid in objids:
            rng = np.random.default_rng(int(objid))
            width, height = image_size
            pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(class_dir / f"{objid}.jpg", quality=95)


def test_import_physical_dataset_builds_manifest_splits_and_relative_image_paths(tmp_path):
    """Catches Colab imports that copy images but leave the pipeline without manifests/splits."""
    project = tmp_path / "project"
    physical = tmp_path / "physical"
    objids_by_class = {
        "elliptical": ["1000", "1001", "1002", "1003"],
        "merger": ["2000", "2001", "2002", "2003"],
        "spiral": ["3000", "3001", "3002", "3003"],
    }
    _write_dataset_config(project)
    _write_source_csv(project, objids_by_class)
    _write_physical_images(physical, objids_by_class)

    summary = import_physical_dataset(project, physical)

    final = pd.read_parquet(project / "data" / "manifests" / "final.parquet")
    splits = pd.read_parquet(project / "data" / "splits" / "all.parquet")
    assert summary["imported_images"] == 12
    assert final["label"].value_counts().to_dict() == {
        "elliptical": 4,
        "merger": 4,
        "spiral": 4,
    }
    assert final["path"].map(lambda value: not Path(value).is_absolute()).all()
    assert not final["path"].str.contains("\\", regex=False).any()
    assert final["path"].map(lambda value: (project / value).is_file()).all()
    assert len(splits) == 12 * 5 * 4


def test_import_physical_dataset_accepts_real_image_size_instead_of_download_config(tmp_path):
    """Catches Colab imports rejecting pre-downloaded 256px datasets from the old notebooks."""
    project = tmp_path / "project"
    physical = tmp_path / "physical"
    objids_by_class = {
        "elliptical": ["1000", "1001", "1002", "1003"],
        "merger": ["2000", "2001", "2002", "2003"],
        "spiral": ["3000", "3001", "3002", "3003"],
    }
    _write_dataset_config(project)
    _write_source_csv(project, objids_by_class)
    _write_physical_images(physical, objids_by_class, image_size=(256, 256))

    summary = import_physical_dataset(project, physical)

    validated = pd.read_parquet(project / "data" / "manifests" / "validated.parquet")
    assert summary["valid_images"] == 12
    assert set(validated["width"]) == {256}
    assert set(validated["height"]) == {256}


def test_import_physical_dataset_can_use_class_folders_when_catalog_is_incomplete(tmp_path):
    """Catches old physical datasets whose image ObjIDs are not all present in the CSV."""
    project = tmp_path / "project"
    physical = tmp_path / "physical"
    objids_by_class = {
        "elliptical": ["1000", "1001", "1002", "1003"],
        "merger": ["2000", "2001", "2002", "2003"],
        "spiral": ["3000", "3001", "3002", "3003"],
    }
    _write_dataset_config(project)
    _write_source_csv(project, {"elliptical": ["1000"]})
    _write_physical_images(physical, objids_by_class)

    summary = import_physical_dataset(project, physical, allow_missing_catalog=True)

    final = pd.read_parquet(project / "data" / "manifests" / "final.parquet")
    assert summary["missing_catalog_rows"] == 11
    assert len(final) == 12
    assert final[["objid", "label", "path", "sha256", "class_index"]].notna().all().all()


def test_import_physical_dataset_tolerates_duplicate_source_catalog_rows(tmp_path):
    """Catches raw source CSV duplicates blocking an otherwise usable physical import."""
    project = tmp_path / "project"
    physical = tmp_path / "physical"
    objids_by_class = {
        "elliptical": ["1000", "1001", "1002", "1003"],
        "merger": ["2000", "2001", "2002", "2003"],
        "spiral": ["3000", "3001", "3002", "3003"],
    }
    _write_dataset_config(project)
    _write_source_csv(project, objids_by_class)
    source_path = project / "data" / "source" / "GalaxyCandidates.csv"
    duplicate = pd.read_csv(source_path).iloc[[0]]
    pd.concat([pd.read_csv(source_path), duplicate], ignore_index=True).to_csv(
        source_path, index=False
    )
    _write_physical_images(physical, objids_by_class)

    summary = import_physical_dataset(project, physical)

    assert summary["imported_images"] == 12


def test_create_colab_bundles_keeps_code_and_dataset_archives_small(tmp_path):
    """Catches Colab bundles that include virtualenv/cache/results or omit dataset artifacts."""
    project = tmp_path / "project"
    output = tmp_path / "bundles"
    for directory in (
        "src/tesi_qml",
        "scripts",
        "configs",
        "notebooks",
        "data/source",
        "data/images/spiral",
    ):
        (project / directory).mkdir(parents=True)
    (project / "src" / "tesi_qml" / "__init__.py").write_text("", encoding="utf-8")
    (project / "scripts" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (project / "configs" / "training.yaml").write_text("batch_size: 32\n", encoding="utf-8")
    (project / "notebooks" / "colab_quickstart.ipynb").write_text("{}", encoding="utf-8")
    (project / "notebooks" / "colab_runner.ipynb").write_text("{}", encoding="utf-8")
    for notebook_name in (
        "01_dataset_audit.ipynb",
        "02_experiment_analysis.ipynb",
        "03_thesis_figures.ipynb",
    ):
        (project / "notebooks" / notebook_name).write_text("{}", encoding="utf-8")
    (project / "data" / "source" / "GalaxyCandidates.csv").write_text("objid\n1\n", encoding="utf-8")
    (project / "data" / "images" / "spiral" / "1.jpg").write_bytes(b"fake-image")
    (project / ".venv" / "lib").mkdir(parents=True)
    (project / ".venv" / "lib" / "ignored.py").write_text("", encoding="utf-8")
    (project / "results" / "runs").mkdir(parents=True)
    (project / "results" / "runs" / "ignored.json").write_text("{}", encoding="utf-8")

    bundles = create_colab_bundles(project, output)

    with tarfile.open(bundles["code"], "r:gz") as archive:
        code_members = set(archive.getnames())
    with tarfile.open(bundles["dataset"], "r:gz") as archive:
        dataset_members = set(archive.getnames())
    assert "src/tesi_qml/__init__.py" in code_members
    assert "scripts/train.py" in code_members
    assert "configs/training.yaml" in code_members
    assert "notebooks/colab_quickstart.ipynb" in code_members
    assert "notebooks/01_dataset_audit.ipynb" in code_members
    assert "notebooks/02_experiment_analysis.ipynb" in code_members
    assert "notebooks/03_thesis_figures.ipynb" in code_members
    assert not any(member.startswith(".venv/") for member in code_members)
    assert not any(member.startswith("results/") for member in code_members)
    assert "data/source/GalaxyCandidates.csv" in dataset_members
    assert "data/images/spiral/1.jpg" in dataset_members
    assert (output / "colab_quickstart.ipynb").is_file()
    assert (output / "colab_runner.ipynb").is_file()
