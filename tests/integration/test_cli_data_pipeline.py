import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_cli_data_pipeline_uses_persisted_intermediate_manifests(tmp_path: Path):
    rows = []
    for class_index, label in enumerate(("elliptical", "merger", "spiral")):
        for sample in range(4):
            objid = str(9_000_000_000_000_000_000 + class_index * 10 + sample)
            probability = {"elliptical": 0.05, "merger": 0.05, "spiral": 0.05}
            probability[label] = 0.9 if label != "merger" else 0.6
            rows.append(
                {
                    "objid": objid,
                    "ra": 100 + sample,
                    "dec": 5 + class_index,
                    "prob_elliptical": probability["elliptical"],
                    "prob_merger": probability["merger"],
                    "prob_spiral": probability["spiral"],
                    "redshift": 0.03,
                }
            )
            y, x = np.indices((320, 320))
            pixels = np.stack(
                ((x + sample * 11) % 256, (y + class_index * 23) % 256, (x + y) % 256),
                axis=-1,
            ).astype(np.uint8)
            target = tmp_path / "data" / "images" / label / f"{objid}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(target, format="JPEG", quality=90)
    source = tmp_path / "data" / "source" / "GalaxyCandidates.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(source, index=False)
    config = {
        "project_seed": 42,
        "source_csv": "data/source/GalaxyCandidates.csv",
        "candidate_manifest": "data/manifests/candidates.parquet",
        "download_manifest": "data/manifests/downloads.parquet",
        "final_manifest": "data/manifests/final.parquet",
        "image_dir": "data/images",
        "split_dir": "data/splits",
        "thresholds": {"elliptical": 0.8, "spiral": 0.8, "merger": 0.4},
        "redshift": {"minimum": 0.005, "maximum": 0.1},
        "download": {
            "oversample_factor": 1.0,
            "width": 320,
            "height": 320,
            "scale": 0.396,
            "timeout_seconds": 20,
            "max_retries": 1,
            "requests_per_second": 4,
        },
        "splits": {"repeat_seeds": [42]},
    }
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT / "src")
    for script in (
        "clean_metadata.py",
        "validate_images.py",
        "build_manifest.py",
        "make_splits.py",
    ):
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT / "scripts" / script),
                "--project-root",
                str(tmp_path),
                "--config",
                str(config_path),
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, f"{script}: {completed.stderr}"

    final = pd.read_parquet(tmp_path / "data/manifests/final.parquet")
    splits = pd.read_parquet(tmp_path / "data/splits/all.parquet")
    exclusions = pd.read_parquet(tmp_path / "data/manifests/manifest_exclusions.parquet")
    assert len(final) == 12
    assert final["path"].map(lambda value: not Path(value).is_absolute()).all()
    assert not final["path"].str.contains("\\", regex=False).any()
    assert splits.groupby(["repeat_seed", "iteration"]).ngroups == 4
    assert set(splits["split"]) == {"train", "validation", "test"}
    assert set(exclusions.columns) == {"objid", "label", "sha256", "reason"}
