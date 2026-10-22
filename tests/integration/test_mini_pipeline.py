from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

matplotlib.use("Agg")

from tesi_qml.data.datasets import ManifestImageDataset
from tesi_qml.data.features import FeatureBankDataset, extract_feature_bank
from tesi_qml.data.manifests import build_balanced_manifest
from tesi_qml.data.metadata import clean_metadata, read_candidate_csv
from tesi_qml.data.splits import make_repeated_four_fold_splits
from tesi_qml.data.validation import validate_image
from tesi_qml.evaluation.plots import plot_confusion_matrix, plot_learning_curves, save_figure
from tesi_qml.models.classical import LinearHead
from tesi_qml.training.engine import Trainer


def _write_fixture_catalog_and_images(root: Path) -> Path:
    rows = []
    labels = ("elliptical", "merger", "spiral")
    for class_index, label in enumerate(labels):
        for sample in range(4):
            objid = str(10_000_000_000_000_000_000 + class_index * 100 + sample)
            probabilities = {"elliptical": 0.05, "merger": 0.05, "spiral": 0.05}
            probabilities[label] = 0.9 if label != "merger" else 0.6
            rows.append(
                {
                    "objid": objid,
                    "ra": 120.0 + sample,
                    "dec": 10.0 + class_index,
                    "prob_elliptical": probabilities["elliptical"],
                    "prob_spiral": probabilities["spiral"],
                    "prob_merger": probabilities["merger"],
                    "redshift": 0.02 + sample * 0.001,
                }
            )
            y, x = np.mgrid[:64, :64]
            pixels = np.stack(
                (
                    (x * (class_index + 1) + sample * 17) % 256,
                    (y * (sample + 1) + class_index * 31) % 256,
                    ((x + y) * (class_index + sample + 1)) % 256,
                ),
                axis=-1,
            ).astype(np.uint8)
            path = root / "images" / label / f"{objid}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(path, format="JPEG", quality=95)
    source = root / "candidates.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    return source


def _bank_for_split(manifest, split_rows, root, extractor):
    selected = split_rows[["objid"]].merge(manifest, on="objid", validate="one_to_one")
    dataset = ManifestImageDataset(
        selected,
        project_root=root,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    return extract_feature_bank(extractor, dataset, n_views=1, seed=19, batch_size=3)


def test_offline_mini_pipeline_produces_traceable_artifacts(tmp_path: Path):
    source = _write_fixture_catalog_and_images(tmp_path)
    clean = clean_metadata(
        read_candidate_csv(source),
        {"elliptical": 0.8, "spiral": 0.8, "merger": 0.4},
    ).clean
    validations = []
    for row in clean.to_dict("records"):
        path = tmp_path / "images" / row["label"] / f"{row['objid']}.jpg"
        validations.append({**row, **asdict(validate_image(path, expected_size=(64, 64)))})
    manifest, summary = build_balanced_manifest(pd.DataFrame(validations), seed=19)
    splits = make_repeated_four_fold_splits(manifest, seeds=[19])
    rotation = splits.loc[splits["iteration"] == 1]

    extractor = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    train_bank = _bank_for_split(
        manifest, rotation.loc[rotation["split"] == "train"], tmp_path, extractor
    )
    validation_bank = _bank_for_split(
        manifest, rotation.loc[rotation["split"] == "validation"], tmp_path, extractor
    )
    train_loader = DataLoader(FeatureBankDataset(train_bank), batch_size=3, shuffle=False)
    validation_loader = DataLoader(FeatureBankDataset(validation_bank), batch_size=3, shuffle=False)

    torch.manual_seed(19)
    model = LinearHead(input_dim=3, num_classes=3)
    run_dir = tmp_path / "results" / "mini"
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.05),
        criterion=nn.CrossEntropyLoss(),
        class_names=("elliptical", "merger", "spiral"),
        run_dir=run_dir,
    )
    result = trainer.fit(
        train_loader,
        validation_loader,
        max_epochs=3,
        run_metadata={"model": "mini", "seed": 19, "fold": 1},
    )
    metrics = __import__("json").loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = pd.read_json(run_dir / "history.jsonl", lines=True)
    save_figure(
        plot_confusion_matrix(
            metrics["confusion_matrix"]["raw"], ("elliptical", "merger", "spiral")
        ),
        run_dir / "figures" / "confusion",
    )
    save_figure(plot_learning_curves(history), run_dir / "figures" / "learning")

    assert summary["per_class"] == 4
    assert len(manifest) == 12 and manifest["sha256"].is_unique
    assert set(rotation["split"]) == {"train", "validation", "test"}
    assert result.last_epoch == 3
    assert (run_dir / "predictions.parquet").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "figures" / "confusion.pdf").is_file()
    assert (run_dir / "figures" / "learning.png").is_file()
