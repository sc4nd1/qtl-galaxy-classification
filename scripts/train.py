#!/usr/bin/env python3
import json
import signal
import time
from pathlib import Path

from tesi_qml.cli import atomic_json, atomic_parquet, base_parser, dry_run, load_cli_config
from tesi_qml.config import load_yaml


def build_model(
    name: str,
    input_dim: int,
    qubits: int | None = None,
    layers: int = 2,
    *,
    num_classes: int = 3,
    quantum_init_std: float = 0.05,
):
    from tesi_qml.models.classical import CapacityMatchedHead, LinearHead, MLPHead

    if name == "B0":
        return LinearHead(input_dim=input_dim, num_classes=num_classes)
    if name == "B1":
        return MLPHead(input_dim=input_dim, num_classes=num_classes)
    if name == "B2":
        if qubits is None:
            raise ValueError("B2 richiede --qubits")
        return CapacityMatchedHead(
            input_dim=input_dim, bottleneck_dim=qubits, num_classes=num_classes
        )
    if name in {"Q", "QF", "PL"}:
        from tesi_qml.models.pennylane_qnn import PennylaneAngleQNN

        if qubits is None:
            raise ValueError(f"{name} richiede --qubits")
        return PennylaneAngleQNN(
            input_dim=input_dim,
            n_qubits=qubits,
            n_layers=layers,
            num_classes=num_classes,
            init_std=quantum_init_std,
            quantum_frozen=name == "QF",
        )
    if name == "PL_REUP":
        from tesi_qml.models.pennylane_qnn import PennylaneReuploadingQNN

        return PennylaneReuploadingQNN(
            input_dim=input_dim, n_qubits=qubits, n_layers=layers, num_classes=num_classes
        )
    if name == "PL_AMP":
        from tesi_qml.models.pennylane_qnn import PennylaneAmplitudeQNN

        return PennylaneAmplitudeQNN(
            input_dim=input_dim, n_qubits=qubits, n_layers=layers, num_classes=num_classes
        )
    raise ValueError(f"Modello non supportato: {name}")


def select_training_subset(bank, subset_size: str, seed: int):
    import torch
    from sklearn.model_selection import train_test_split

    from tesi_qml.data.features import FeatureBank
    from tesi_qml.reproducibility import config_fingerprint as subset_fingerprint

    if subset_size == "full":
        return bank
    requested = int(subset_size)
    identities = bank.metadata[["objid", "label"]].drop_duplicates("objid")
    if requested >= len(identities):
        return bank
    selected, _ = train_test_split(
        identities, train_size=requested, stratify=identities["label"], random_state=seed
    )
    mask = bank.metadata["objid"].isin(selected["objid"])
    metadata = bank.metadata.loc[mask].reset_index(drop=True).copy()
    features = bank.features[torch.from_numpy(mask.to_numpy())]
    metadata["feature_index"] = range(len(metadata))
    subset_objids = sorted(metadata["objid"].astype(str).unique())
    provenance = {
        **bank.provenance,
        "training_subset_seed": int(seed),
        "training_subset_size": len(subset_objids),
        "training_subset_objid_sha256": subset_fingerprint(subset_objids),
    }
    return FeatureBank(features=features, metadata=metadata, provenance=provenance)


def train_b3(args, root, config, dataset_config, repeat_seed, run_dir, metadata) -> None:
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    from tesi_qml.cli import atomic_json
    from tesi_qml.config import resolve_project_path
    from tesi_qml.data.datasets import ManifestImageDataset
    from tesi_qml.data.transforms import build_eval_transform, build_train_transform
    from tesi_qml.models.classical import (
        ImageClassifier,
        MLPHead,
        resnet18_backbone,
        set_backbone_stage,
    )
    from tesi_qml.training.checkpointing import EarlyStopping
    from tesi_qml.training.engine import Trainer
    from tesi_qml.training.metrics import compute_classification_metrics

    manifest = pd.read_parquet(resolve_project_path(root, dataset_config["final_manifest"]))
    splits = pd.read_parquet(
        resolve_project_path(root, dataset_config["split_dir"]) / "all.parquet"
    )
    rotation = splits.loc[
        (splits["repeat_seed"] == repeat_seed) & (splits["iteration"] == args.iteration)
    ]

    def split_manifest(name: str):
        identities = rotation.loc[rotation["split"] == name, ["objid"]]
        result = identities.merge(manifest, on="objid", validate="one_to_one")
        if name == "train" and args.subset_size != "full" and int(args.subset_size) < len(result):
            result, _ = train_test_split(
                result,
                train_size=int(args.subset_size),
                stratify=result["label"],
                random_state=config["training_seed"],
            )
        return result.reset_index(drop=True)

    train_dataset = ManifestImageDataset(
        split_manifest("train"), project_root=root, transform=build_train_transform()
    )
    validation_dataset = ManifestImageDataset(
        split_manifest("validation"), project_root=root, transform=build_eval_transform()
    )
    test_dataset = ManifestImageDataset(
        split_manifest("test"), project_root=root, transform=build_eval_transform()
    )
    generator = torch.Generator().manual_seed(config["augmentation_seed"])
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, generator=generator
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    model = ImageClassifier(resnet18_backbone(pretrained=True), MLPHead())
    active_trainer = [None]
    if hasattr(signal, "SIGUSR1"):
        signal.signal(
            signal.SIGUSR1,
            lambda _signum, _frame: (
                active_trainer[0].request_stop() if active_trainer[0] is not None else None
            ),
        )

    set_backbone_stage(model, "frozen")
    frozen_optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    frozen_trainer = Trainer(
        model=model,
        optimizer=frozen_optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        class_names=("elliptical", "merger", "spiral"),
        run_dir=run_dir / "phases" / "frozen",
        device=args.device,
        early_stopping=EarlyStopping(config["patience"], config["min_delta"]),
        gradient_clip_norm=config["gradient_clip_norm"],
    )
    active_trainer[0] = frozen_trainer
    frozen_resume = (
        args.resume and (run_dir / "phases" / "frozen" / "checkpoints" / "last.pt").is_file()
    )
    frozen_trainer.fit(
        train_loader,
        validation_loader,
        max_epochs=config["b3"]["frozen_epochs"],
        resume=frozen_resume,
        run_metadata={**metadata, "phase": "frozen"},
    )

    set_backbone_stage(model, "layer4")
    layer4_optimizer = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": config["optimizer"]["learning_rate"]},
            {
                "params": model.backbone.layer4.parameters(),
                "lr": config["optimizer"]["backbone_learning_rate"],
            },
        ],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    layer4_trainer = Trainer(
        model=model,
        optimizer=layer4_optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        class_names=("elliptical", "merger", "spiral"),
        run_dir=run_dir,
        device=args.device,
        early_stopping=EarlyStopping(config["patience"], config["min_delta"]),
        gradient_clip_norm=config["gradient_clip_norm"],
    )
    active_trainer[0] = layer4_trainer
    layer4_resume = args.resume and (run_dir / "checkpoints" / "last.pt").is_file()
    layer4_trainer.fit(
        train_loader,
        validation_loader,
        max_epochs=config["b3"]["layer4_epochs"],
        resume=layer4_resume,
        run_metadata={**metadata, "phase": "layer4"},
    )
    predictions = layer4_trainer.predict(test_loader, run_metadata=metadata)
    probability_columns = ["prob_elliptical_model", "prob_merger_model", "prob_spiral_model"]
    metrics = compute_classification_metrics(
        predictions["class_index"].to_numpy(),
        predictions[probability_columns].to_numpy(),
        ("elliptical", "merger", "spiral"),
    )
    predictions.to_parquet(run_dir / "test_predictions.parquet", index=False)
    atomic_json(run_dir / "test_metrics.json", metrics)


def main() -> None:
    parser = base_parser(
        "Addestra baseline o angle-QNN su feature precomputate",
        default_config="configs/training.yaml",
    )
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument(
        "--model",
        choices=(
            "B0",
            "B1",
            "B2",
            "B3",
            "QF",
            "Q",
            "PL",
            "PL_REUP",
            "PL_AMP",
        ),
        default="B1",
    )
    parser.add_argument("--task", choices=("binary", "three_class"), default="three_class")
    parser.add_argument("--qubits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--layers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--repeat-seed", type=int)
    parser.add_argument("--iteration", type=int, default=1, choices=range(1, 5))
    parser.add_argument(
        "--subset-size", choices=("50", "100", "300", "900", "full"), default="full"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, config = load_cli_config(args)
    if dry_run(
        args,
        f"training {args.task}/{args.model}, {args.qubits} qubit, {args.layers} layer",
    ):
        return
    import torch
    from torch.utils.data import DataLoader

    from tesi_qml.data.features import FeatureBank, FeatureBankDataset
    from tesi_qml.experiments import RunSpec, build_run_identity, run_directory
    from tesi_qml.models.initialization import PairedInitialization
    from tesi_qml.reproducibility import config_fingerprint, environment_metadata, seed_everything
    from tesi_qml.tasks import filter_feature_bank, get_task
    from tesi_qml.training.checkpointing import EarlyStopping
    from tesi_qml.training.engine import Trainer
    from tesi_qml.training.metrics import compute_classification_metrics
    from tesi_qml.training.parameters import trainable_parameters

    dataset_path = (
        args.dataset_config if args.dataset_config.is_absolute() else root / args.dataset_config
    )
    dataset_config = load_yaml(dataset_path)
    task = get_task(args.task)
    repeat_seed = args.repeat_seed or dataset_config["splits"]["repeat_seeds"][0]
    feature_root = (
        root / "data" / "features" / f"seed_{repeat_seed}" / f"iteration_{args.iteration}"
    )
    feature_provenance = None
    feature_extractor_identity = None
    if args.model != "B3":
        feature_provenance = {
            split: json.loads(
                (feature_root / split / "provenance.json").read_text(encoding="utf-8")
            )
            for split in ("train", "validation", "test")
        }
        feature_extractor_identity = {
            "manifest_sha256": feature_provenance["train"]["manifest_sha256"],
            "backbone": feature_provenance["train"]["backbone"],
            "feature_config_fingerprint": feature_provenance["train"]["feature_config_fingerprint"],
            "transforms": sorted({item["transform"] for item in feature_provenance.values()}),
        }
    if args.model == "B3":
        if args.task != "three_class":
            raise ValueError("B3 legacy è disponibile soltanto per three_class")
        identity = {
            "task": task.name,
            "class_names": list(task.class_names),
            "class_to_index": dict(task.class_to_index),
            "model": "B3",
            "n_qubits": None,
            "n_layers": None,
            "subset_size": args.subset_size,
            "repeat_seed": repeat_seed,
            "fold": args.iteration,
            "training_config": config,
            "dataset_config": dataset_config,
        }
        identity["experiment_fingerprint"] = config_fingerprint(
            {key: value for key, value in identity.items() if key not in {"repeat_seed", "fold"}}
        )
        identity["config_fingerprint"] = config_fingerprint(identity)
        metadata = {
            key: identity[key]
            for key in (
                "task",
                "model",
                "repeat_seed",
                "fold",
                "experiment_fingerprint",
                "config_fingerprint",
            )
        }
        run_dir = (
            root
            / "results"
            / "runs"
            / task.name
            / "b3"
            / f"seed_{repeat_seed}"
            / f"iteration_{args.iteration}"
            / f"config_{identity['config_fingerprint'][:12]}"
        )
        atomic_json(run_dir / "run_config.json", identity)
        train_b3(args, root, config, dataset_config, repeat_seed, run_dir, metadata)
        return

    effective_qubits = None if args.model in {"B0", "B1"} else args.qubits
    spec = RunSpec(
        task=args.task,
        model=args.model,
        qubits=effective_qubits,
        repeat_seed=repeat_seed,
        iteration=args.iteration,
        n_layers=args.layers,
        subset_size=args.subset_size,
    )
    provisional_identity = build_run_identity(
        spec,
        dataset_config=dataset_config,
        training_config=config,
        feature_bank_provenance=feature_provenance,
    )
    seed_everything(provisional_identity["sub_seeds"]["libraries"])

    train_bank = filter_feature_bank(FeatureBank.load(feature_root / "train"), task)
    train_bank = select_training_subset(
        train_bank, args.subset_size, provisional_identity["sub_seeds"]["dataloader"]
    )
    validation_bank = filter_feature_bank(FeatureBank.load(feature_root / "validation"), task)
    test_bank = filter_feature_bank(FeatureBank.load(feature_root / "test"), task)

    model = build_model(
        args.model,
        input_dim=train_bank.features.shape[1],
        qubits=args.qubits,
        layers=args.layers,
        num_classes=task.num_classes,
        quantum_init_std=config["quantum_init_std"],
    )
    initialization_checksums: dict[str, str] = {}
    if args.model in {"B2", "QF", "Q"}:
        paired = PairedInitialization.create(
            input_dim=train_bank.features.shape[1],
            bottleneck_dim=args.qubits,
            num_classes=task.num_classes,
            n_layers=args.layers,
            shared_seed=provisional_identity["sub_seeds"]["shared_initialization"],
            quantum_seed=provisional_identity["sub_seeds"]["quantum_initialization"],
            quantum_init_std=config["quantum_init_std"],
        )
        paired.apply(model)
        initialization_checksums = paired.checksums()

    filtered_provenance = {
        "train": train_bank.provenance,
        "validation": validation_bank.provenance,
        "test": test_bank.provenance,
    }
    identity = build_run_identity(
        spec,
        dataset_config=dataset_config,
        training_config=config,
        feature_bank_provenance=filtered_provenance,
        initialization_checksums=initialization_checksums,
    )
    identity["feature_extractor"] = feature_extractor_identity
    requested_device = args.device
    effective_device = requested_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        effective_device = "cpu"
        print("CUDA non disponibile: il run prosegue su CPU")
    identity["environment"] = environment_metadata()
    identity["runtime"] = {
        "requested_device": requested_device,
        "effective_device": effective_device,
        "pennylane_backend": "default.qubit" if args.model in {"QF", "Q", "PL"} else None,
    }
    run_dir = run_directory(root, spec, identity["config_fingerprint"])
    atomic_json(run_dir / "run_config.json", identity)
    metadata = {
        "task": task.name,
        "model": args.model,
        "n_qubits": effective_qubits,
        "n_layers": args.layers,
        "quantum_frozen": args.model == "QF",
        "subset_size": args.subset_size,
        "repeat_seed": repeat_seed,
        "fold": args.iteration,
        "experiment_fingerprint": identity["experiment_fingerprint"],
        "config_fingerprint": identity["config_fingerprint"],
    }

    if args.model in {"B2", "QF", "Q"}:
        batch_size = config["comparison_batch_size"]
    elif args.model in {"PL", "PL_REUP", "PL_AMP"}:
        batch_size = config["quantum_batch_size"]
    else:
        batch_size = config["batch_size"]
    generator = torch.Generator().manual_seed(identity["sub_seeds"]["dataloader"])
    train_loader = DataLoader(
        FeatureBankDataset(train_bank), batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(
        FeatureBankDataset(validation_bank), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(FeatureBankDataset(test_bank), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"]
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        class_names=task.class_names,
        run_dir=run_dir,
        device=effective_device,
        scheduler=scheduler,
        early_stopping=EarlyStopping(config["patience"], config["min_delta"]),
        gradient_clip_norm=config["gradient_clip_norm"],
    )
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda _signum, _frame: trainer.request_stop())
    resume_available = args.resume and (run_dir / "checkpoints" / "last.pt").is_file()
    result = trainer.fit(
        train_loader,
        validation_loader,
        max_epochs=config["max_epochs"],
        resume=resume_available,
        run_metadata=metadata,
    )
    test_started = time.perf_counter()
    test_predictions = trainer.predict(test_loader, run_metadata=metadata)
    test_inference_seconds = time.perf_counter() - test_started
    probability_columns = [f"prob_{name}_model" for name in task.class_names]
    test_metrics = compute_classification_metrics(
        test_predictions["class_index"].to_numpy(),
        test_predictions[probability_columns].to_numpy(),
        task.class_names,
    )
    validation_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    test_metrics["task"] = task.as_dict()
    test_metrics["runtime"] = {
        **validation_metrics.get("runtime", {}),
        "test_inference_seconds": test_inference_seconds,
        "test_samples_per_second": len(test_predictions) / test_inference_seconds
        if test_inference_seconds > 0
        else None,
        "quantum_parameters": sum(
            parameter.numel() for parameter in getattr(model, "quantum_parameters", lambda: ())()
        ),
    }
    if args.model == "QF":
        test_metrics["frozen_quantum_checksum"] = model.frozen_quantum_checksum()
    atomic_parquet(run_dir / "test_predictions.parquet", test_predictions)
    atomic_json(run_dir / "test_metrics.json", test_metrics)
    print(result)


if __name__ == "__main__":
    main()
