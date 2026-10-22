#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

from tesi_qml.cli import base_parser, dry_run, load_cli_config
from tesi_qml.config import load_yaml, resolve_project_path
from tesi_qml.data.datasets import ManifestImageDataset
from tesi_qml.data.features import extract_feature_bank
from tesi_qml.data.transforms import build_eval_transform, build_train_transform
from tesi_qml.data.validation import sha256_file
from tesi_qml.models.classical import resnet18_backbone
from tesi_qml.reproducibility import config_fingerprint, derive_seed


def main() -> None:
    parser = base_parser(
        "Estrae feature ResNet18 per uno split sperimentale", default_config="configs/dataset.yaml"
    )
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--repeat-seed", type=int)
    parser.add_argument("--iteration", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    root, config = load_cli_config(args)
    if dry_run(args, "cache feature train/validation/test con identità dei campioni"):
        return
    training_path = (
        args.training_config if args.training_config.is_absolute() else root / args.training_config
    )
    training = load_yaml(training_path)
    repeat_seed = args.repeat_seed or config["splits"]["repeat_seeds"][0]
    feature_view_seed = derive_seed(repeat_seed, "feature-view")
    manifest_path = resolve_project_path(root, config["final_manifest"])
    manifest = pd.read_parquet(manifest_path)
    splits = pd.read_parquet(resolve_project_path(root, config["split_dir"]) / "all.parquet")
    selected = splits.loc[
        (splits["repeat_seed"] == repeat_seed) & (splits["iteration"] == args.iteration)
    ]
    extractor = resnet18_backbone(pretrained=not args.no_pretrained)
    base = root / "data" / "features" / f"seed_{repeat_seed}" / f"iteration_{args.iteration}"
    for split_name in ("train", "validation", "test"):
        identities = selected.loc[selected["split"] == split_name, ["objid"]]
        split_manifest = identities.merge(manifest, on="objid", how="left", validate="one_to_one")
        transform = build_train_transform() if split_name == "train" else build_eval_transform()
        dataset = ManifestImageDataset(split_manifest, project_root=root, transform=transform)
        views = training["feature_bank"]["train_views"] if split_name == "train" else 1
        bank = extract_feature_bank(
            extractor,
            dataset,
            n_views=views,
            seed=feature_view_seed,
            batch_size=training["batch_size"],
            device=args.device,
            provenance={
                "manifest_sha256": sha256_file(manifest_path),
                "backbone": "torchvision.ResNet18_Weights.IMAGENET1K_V1"
                if not args.no_pretrained
                else "resnet18_random_initialization",
                "transform": "train_rotation_crop_flip_imagenet_v1"
                if split_name == "train"
                else "eval_center_crop_imagenet_v1",
                "repeat_seed": repeat_seed,
                "fold": args.iteration,
                "split": split_name,
                "views": views,
                "augmentation_seed": feature_view_seed,
                "configured_augmentation_seed": training["augmentation_seed"],
                "feature_config_fingerprint": config_fingerprint(
                    {
                        "feature_bank": training["feature_bank"],
                        "augmentation_seed": feature_view_seed,
                    }
                ),
            },
        )
        bank.save(base / split_name)
        print(split_name, bank.features.shape)


if __name__ == "__main__":
    main()
