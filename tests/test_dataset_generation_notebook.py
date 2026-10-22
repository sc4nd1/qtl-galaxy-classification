import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "01_Dataset_Generation.ipynb"


def _make_dataset(project_root: Path, *, with_images: bool = False) -> None:
    manifests = project_root / "data" / "manifests"
    split_dir = project_root / "data" / "splits"
    manifests.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    labels = ("elliptical", "merger", "spiral")
    probabilities = {
        "elliptical": (0.92, 0.03, 0.02),
        "merger": (0.08, 0.06, 0.71),
        "spiral": (0.04, 0.93, 0.03),
    }
    rows = []
    for label_index, label in enumerate(labels):
        for index in range(12):
            p_el, p_sp, p_mg = probabilities[label]
            rows.append({
                "objid": f"{label_index}{index:04d}",
                "label": label,
                "prob_elliptical": p_el - index * 0.001,
                "prob_spiral": p_sp - index * 0.001,
                "prob_merger": p_mg - index * 0.001,
                "redshift": 0.01 + label_index * 0.015 + index * 0.001,
                "path": f"data/images/{label}/{label_index}{index:04d}.jpg",
                "valid": True,
                "width": 320,
                "height": 320,
            })
    candidates = pd.DataFrame(rows)
    downloads = candidates.groupby("label", group_keys=False).head(10).reset_index(drop=True)
    validated = downloads.copy()
    final = downloads.groupby("label", group_keys=False).head(8).reset_index(drop=True)
    if with_images:
        from PIL import Image

        for row in final.itertuples():
            path = project_root / row.path
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (320, 320), color=(40, 60, 80)).save(path)
    candidates.to_parquet(manifests / "candidates.parquet", index=False)
    downloads.to_parquet(manifests / "downloads.parquet", index=False)
    validated.to_parquet(manifests / "validated.parquet", index=False)
    final.to_parquet(manifests / "final.parquet", index=False)
    (manifests / "metadata_summary.json").write_text(json.dumps({
        "input_rows": 40, "clean_rows": len(candidates),
        "exact_duplicates_removed": 2, "rejected_rows": 2,
    }), encoding="utf-8")

    split_rows = []
    for seed in range(10):
        for row in final.itertuples():
            local_index = int(row.objid[-2:])
            split = "train" if local_index < 4 else "validation" if local_index < 6 else "test"
            split_rows.append({
                "repeat_seed": seed, "iteration": 1, "split": split,
                "objid": row.objid, "label": row.label,
            })
    pd.DataFrame(split_rows).to_parquet(split_dir / "all.parquet", index=False)


def test_notebook_uses_manifests_without_images_or_downloads(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "tesi_finale"
    output = tmp_path / "dataset_analysis"
    _make_dataset(project)
    monkeypatch.setenv("TESI_PROJECT_ROOT", str(project))
    monkeypatch.setenv("TESI_DATASET_OUTPUT", str(output))
    monkeypatch.setenv("TESI_STRICT_DATASET", "0")

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    original_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                exec(compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(original_cwd)

    expected = {
        "01_data_attrition",
        "02_class_balance",
        "03_vote_fractions",
        "04_redshift_distribution",
        "05_repeated_holdout",
    }
    for name in expected:
        for suffix in (".png", ".pdf"):
            path = output / "figures" / f"{name}{suffix}"
            assert path.is_file() and path.stat().st_size > 500, path
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["image_gallery_generated"] is False
    assert manifest["figure_count"] == 5


def test_notebook_generates_optional_gallery_from_existing_images(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "tesi_finale"
    output = tmp_path / "dataset_analysis"
    _make_dataset(project, with_images=True)
    monkeypatch.setenv("TESI_PROJECT_ROOT", str(project))
    monkeypatch.setenv("TESI_DATASET_OUTPUT", str(output))
    monkeypatch.setenv("TESI_STRICT_DATASET", "0")

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    original_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                exec(compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(original_cwd)

    gallery = output / "figures" / "06_galaxy_samples.png"
    assert gallery.is_file() and gallery.stat().st_size > 500
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["image_gallery_generated"] is True
    assert manifest["figure_count"] == 6


def test_notebook_contains_no_legacy_download_or_fold_pipeline() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    forbidden = (
        "download_images", "requests", "BASE_URL", "create_flat_dataset",
        "MAX_IMAGES_PER_CLASS", "1752", "1754", "256x256", "4-Fold",
    )
    assert not any(token in source for token in forbidden)
    assert "repeated holdout" in source.lower()
    assert "iteration=1" in source
    assert "320×320" in source
