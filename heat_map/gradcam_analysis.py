"""Read-only reconstruction of thesis heads and class-targeted Grad-CAM.

All caches, extracted inputs and outputs live beside this file in heat_map.
No optimizer, training, RNG restoration or modification of the source snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["MPLCONFIGDIR"] = str(HERE / "cache/matplotlib")
os.environ["TORCH_HOME"] = str(HERE / "cache/torch")
os.environ["XDG_CACHE_HOME"] = str(HERE / "cache")
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

MODELS = ("B2", "Q", "QF")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
LABELS = {"elliptical": "Ellittica", "spiral": "Spirale", "merger": "Merger"}


def safe_name(value):
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or any(":" in p for p in path.parts):
        raise ValueError(f"Unsafe archive/path name: {value}")
    return str(path)


class Snapshot:
    """Folder or tar.gz reader. Never invokes extractall or trusts member paths."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.tar = None
        if self.path.is_dir():
            self.members = {
                safe_name(p.relative_to(self.path).as_posix()): p
                for p in self.path.rglob("*")
                if p.is_file()
            }
        elif self.path.is_file():
            self.tar = tarfile.open(self.path, "r:*")
            self.members = {}
            try:
                for member in self.tar.getmembers():
                    name = safe_name(member.name)
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError(f"Archive links/special files are not allowed: {name}")
                    if name in self.members:
                        raise ValueError(f"Duplicate archive member: {name}")
                    self.members[name] = member
            except Exception:
                self.tar.close()
                raise
        else:
            raise FileNotFoundError(f"Snapshot not found: {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.tar:
            self.tar.close()

    def read(self, name):
        name = safe_name(name)
        member = self.members[name]
        if self.tar:
            with self.tar.extractfile(member) as stream:
                return stream.read()
        if not member.resolve().is_relative_to(self.path):
            raise ValueError("Source file points outside snapshot")
        return member.read_bytes()

    def suffix(self, suffix):
        suffix = safe_name(suffix)
        matches = [n for n in self.members if n == suffix or n.endswith("/" + suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one source for {suffix}, found {len(matches)}")
        return matches[0]

    def run(self, task, subset, seed, qubits, model):
        found = []
        for name in self.members:
            if not name.endswith("run_config.json"):
                continue
            config = json.loads(self.read(name))
            if (
                config.get("task") == task
                and str(config.get("subset_size")) == str(subset)
                and config.get("repeat_seed") == seed
                and config.get("n_qubits") == qubits
                and config.get("model") == model
                and config.get("iteration") == 1
            ):
                found.append((name, config))
        if len(found) != 1:
            raise ValueError(
                f"{task}/{subset}/{seed}/{qubits}/{model}: {len(found)} matching runs; select an unambiguous snapshot"
            )
        return found[0]


def make_head(config, angle_scale):
    """Exact final B2/Q/QF ansatz, with checkpoint-compatible state_dict names.

    Mirrors classical.py and PennylaneAngleQNN, but does not import/patch the
    project. Unsupported architecture signatures are rejected by load_runs.
    """
    if angle_scale != "tanh_pi":
        raise ValueError(f"Unsupported final-matrix angle scale: {angle_scale}")
    q = int(config["n_qubits"])
    layers = int(config["n_layers"])
    classes = len(config["class_names"])
    dim = int(config["training_config"]["feature_bank"]["feature_dimension"])
    if q not in (4, 8) or layers != 2 or dim != 512:
        raise ValueError("This analysis supports the final 512D, 4/8-qubit, 2-layer matrix only")

    class B2(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_head = nn.Linear(dim, q)
            self.classifier = nn.Linear(q, classes)

        def forward(self, features):
            return self.classifier(torch.tanh(self.pre_head(features)) * np.pi)

    if config["model"] == "B2":
        model = B2()
    elif config["model"] in ("Q", "QF"):
        import pennylane as qml

        @qml.qnode(
            qml.device("default.qubit", wires=q, shots=None),
            interface="torch",
            diff_method="backprop",
        )
        def circuit(inputs, weights):
            for wire in range(q):
                qml.RY(inputs[..., wire], wires=wire)
            for layer in range(layers):
                for wire in range(q):
                    qml.RZ(weights[layer, wire, 0], wires=wire)
                    qml.RY(weights[layer, wire, 1], wires=wire)
                for wire in range(q - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(q)]

        class Quantum(nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_quantum = nn.Linear(dim, q)
                self.quantum_layer = qml.qnn.TorchLayer(circuit, {"weights": (layers, q, 2)})
                self.post_quantum = nn.Linear(q, classes)

            def forward(self, features):
                angles = torch.tanh(self.pre_quantum(features)) * np.pi
                values = self.quantum_layer(angles).to(dtype=features.dtype)
                return self.post_quantum(values)

        model = Quantum()
    else:
        raise ValueError("Only B2, Q and QF are supported")
    # Attribution differentiates inputs, not trained weights; no optimization.
    model.eval()
    model.requires_grad_(False)
    return model


def gradcam_from_spatial(spatial, head, target):
    """Gradient of a fixed class logit w.r.t. post-layer4 activations.

    Detaching here is exact for d(logit)/d(layer4): GAP and the whole head
    remain in the graph, including the quantum circuit for Q and QF.
    """
    if spatial.ndim != 4 or spatial.shape[0] != 1:
        raise ValueError("One [1,C,H,W] activation tensor is required")
    with torch.enable_grad():
        activation = spatial.detach().clone().requires_grad_(True)
        features = F.adaptive_avg_pool2d(activation, (1, 1)).flatten(1)
        logits = head(features)
        if target < 0 or target >= logits.shape[1]:
            raise ValueError("Target class is outside the classifier")
        (gradient,) = torch.autograd.grad(logits[0, target], activation)
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        signed = (weights * activation).sum(dim=1)[0]
        cam = signed.relu()
    if not torch.isfinite(cam).all() or not torch.isfinite(gradient).all():
        raise ValueError("Non-finite attribution; no figure will be accepted")
    return {
        "cam": cam.detach().cpu().numpy(),
        "signed_cam": signed.detach().cpu().numpy(),
        "gradient": gradient.detach().cpu().numpy(),
        "logits": logits.detach().cpu().numpy()[0],
        "probabilities": logits.detach().double().softmax(dim=1).cpu().numpy()[0],
    }


def normalize_maps(maps):
    if any(not np.isfinite(m).all() or (m < 0).any() for m in maps):
        raise ValueError("Grad-CAM display expects finite positive contributions")
    maximum = max(float(np.max(m)) for m in maps)
    return ([m / maximum if maximum > 0 else np.zeros_like(m) for m in maps], maximum)


def check_prediction(actual, saved, tolerance=1e-4):
    actual, saved = np.asarray(actual, float), np.asarray(saved, float)
    if actual.shape != saved.shape or not np.isfinite(actual).all() or not np.isfinite(saved).all():
        raise ValueError("Invalid reconstructed/saved probabilities")
    difference = float(np.max(np.abs(actual - saved)))
    if difference > tolerance or np.argmax(actual) != np.argmax(saved):
        raise ValueError(
            f"Prediction mismatch: max probability error={difference:.6g}, tolerance={tolerance}; check weights, preprocessing and checkpoint"
        )
    return difference


def select_examples(frame, classes, per_class, selection_seed, extra_disagreement):
    frame = frame.copy()
    frame["objid"] = frame.objid.astype(str)
    if frame.objid.duplicated().any():
        raise ValueError("Duplicate test objid")
    frame["_selection_hash"] = [
        hashlib.sha256(f"{selection_seed}|{oid}".encode()).hexdigest() for oid in frame.objid
    ]
    selected = []
    for name in classes:
        group = frame[frame.true_class.eq(name)].sort_values(["_selection_hash", "objid"])
        if len(group) < per_class:
            raise ValueError(f"Not enough test examples for {name}")
        selected.append(group.head(per_class).assign(selection_reason="hash_sample_by_true_class"))
    result = pd.concat(selected, ignore_index=True)
    if extra_disagreement:
        disagreement = frame[[f"pred_{m}" for m in MODELS]].nunique(axis=1).gt(1)
        candidates = frame[disagreement & ~frame.objid.isin(result.objid)]
        # Explicitly selected difficult case, never described as representative.
        if not candidates.empty:
            extra = candidates.sort_values(["_selection_hash", "objid"]).head(1)
            result = pd.concat(
                [result, extra.assign(selection_reason="selected_model_disagreement")],
                ignore_index=True,
            )
    return result.drop(columns="_selection_hash")


def load_runs(snapshot, args, output):
    configs, heads, predictions, audit = {}, {}, {}, {}
    for model in MODELS:
        name, config = snapshot.run(args.task, args.subset, args.seed, args.qubits, model)
        prefix = str(PurePosixPath(name).parent)
        metrics = json.loads(snapshot.read(prefix + "/metrics.json"))
        if model != "B2":
            signature = metrics.get("model_signature", {})
            expected = {
                "encoding": "RY",
                "entangling_gate": "CNOT",
                "entangling_pairs": [[i, i + 1] for i in range(args.qubits - 1)],
                "trainable_gates_per_layer": ["RZ", "RY"],
                "observables": [f"Z{i}" for i in range(args.qubits)],
                "weight_shape": [2, args.qubits, 2],
                "angle_scale": "tanh_pi",
            }
            if any(signature.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Unsupported circuit signature for {model}: {signature}")
        provenance = config["feature_bank_provenance"]["test"]
        if (
            provenance["backbone"] != "torchvision.ResNet18_Weights.IMAGENET1K_V1"
            or provenance["transform"] != "eval_center_crop_imagenet_v1"
            or provenance["views"] != 1
        ):
            raise ValueError(f"Unsupported feature pipeline for {model}")
        checkpoint_bytes = snapshot.read(prefix + "/checkpoints/best.pt")
        # Only use archives created by the user: legacy checkpoints include RNG
        # objects and require pickle deserialization, never restore those RNGs.
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
        )
        fingerprint = checkpoint["extra"]["run_metadata"]["config_fingerprint"]
        if fingerprint != config["config_fingerprint"]:
            raise ValueError(f"Checkpoint/config fingerprint mismatch for {model}")
        head = make_head(config, "tanh_pi").to(args.device)
        head.load_state_dict(checkpoint["model_state"], strict=True)
        head.eval()
        frame = pd.read_parquet(io.BytesIO(snapshot.read(prefix + "/test_predictions.parquet")))
        frame["objid"] = frame.objid.astype(str)
        if not frame.objid.str.fullmatch(r"\d+").all():
            raise ValueError("Invalid SDSS object identifier")
        if frame.objid.duplicated().any():
            raise ValueError(f"Multiple test predictions for one object in {model}")
        expected_meta = {
            "model": model,
            "task": args.task,
            "repeat_seed": args.seed,
            "n_qubits": args.qubits,
            "config_fingerprint": config["config_fingerprint"],
        }
        if any(not frame[k].eq(v).all() for k, v in expected_meta.items()):
            raise ValueError(f"Prediction metadata does not match run {model}")
        if not frame.subset_size.astype(str).eq(args.subset).all():
            raise ValueError("Prediction subset mismatch")
        configs[model], heads[model], predictions[model] = (
            config,
            head,
            frame.set_index("objid", drop=False),
        )
        input_dir = output / "inputs" / model
        input_dir.mkdir(parents=True)
        (input_dir / "run_config.json").write_bytes(snapshot.read(name))
        (input_dir / "best.pt").write_bytes(checkpoint_bytes)
        audit[model] = {
            "source": str(snapshot.path),
            "run": prefix,
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "config_fingerprint": fingerprint,
        }
        print(
            f"Loaded {model}: best epoch {checkpoint['epoch']}, {len(frame)} test objects",
            flush=True,
        )
    base = configs["B2"]
    for model in MODELS[1:]:
        for field in [
            "class_names",
            "class_to_index",
            "initialization_checksums",
            "feature_bank_provenance",
        ]:
            if configs[model][field] != base[field]:
                raise ValueError(f"Unpaired {model}/B2 field: {field}")
        if set(predictions[model].index) != set(predictions["B2"].index):
            raise ValueError("Test object sets differ between heads")
        for field in ["true_class", "class_index", "sha256"]:
            other = predictions[model].reindex(predictions["B2"].index)[field]
            if not other.eq(predictions["B2"][field]).all():
                raise ValueError(f"Test identity/label mismatch: {field}")
    common = predictions["B2"][["objid", "true_class", "class_index", "path", "sha256"]].copy()
    for model in MODELS:
        common[f"pred_{model}"] = predictions[model].predicted_class
    return configs, heads, predictions, common.reset_index(drop=True), audit


def load_backbone(args):
    from torchvision.models import ResNet18_Weights, resnet18

    if args.weights:
        weights_path = Path(args.weights).expanduser().resolve()
    else:
        weights_path = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
        if not weights_path.is_file():
            print(
                "Downloading the official ImageNet ResNet18 weights into heat_map/cache (no galaxy downloads).",
                flush=True,
            )
            torch.hub.load_state_dict_from_url(
                ResNet18_Weights.IMAGENET1K_V1.url,
                progress=True,
                check_hash=True,
                map_location="cpu",
            )
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"):
        raise ValueError("ResNet18 file is not the expected IMAGENET1K_V1 weight artifact")
    model = resnet18(weights=None)
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True), strict=True
    )
    model.fc = nn.Identity()
    model.requires_grad_(False)
    return model.eval().to(args.device), digest


def spatial_features(backbone, tensor):
    with torch.no_grad():
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(tensor))))
        x = backbone.layer4(backbone.layer3(backbone.layer2(backbone.layer1(x))))
    return x


def load_image(snapshot, row, output):
    from torchvision import transforms

    original_path = str(row["path"]).replace("\\", "/")
    if "/data/images/" in original_path:
        suffix = "data/images/" + original_path.split("/data/images/", 1)[1]
    elif original_path.startswith("data/images/"):
        suffix = original_path
    else:
        raise ValueError(f"Unsupported recorded image path: {original_path}")
    name = snapshot.suffix(suffix)
    content = snapshot.read(name)
    digest = hashlib.sha256(content).hexdigest()
    if digest != row["sha256"]:
        raise ValueError(f"Image checksum mismatch for {row['objid']}")
    with Image.open(io.BytesIO(content)) as image:
        if min(image.size) < 224:
            raise ValueError("Image is smaller than the expected crop")
        crop = transforms.CenterCrop(224)(image.convert("RGB"))
    tensor = transforms.Normalize(MEAN, STD)(transforms.ToTensor()(crop)).unsqueeze(0)
    image_dir = output / "inputs/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{row['objid']}.jpg").write_bytes(content)
    crop.save(image_dir / f"{row['objid']}_crop.png")
    return crop, tensor, digest


def plot_example(crop, row, results, classes, destination, context=""):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    resized = [
        F.interpolate(
            torch.tensor(results[m]["cam"])[None, None],
            (224, 224),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        for m in MODELS
    ]
    normalized, scale = normalize_maps(resized)
    with plt.rc_context(
        {"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42, "savefig.dpi": 250}
    ):
        fig, axes = plt.subplots(1, 4, figsize=(13.8, 5.0))
        fig.subplots_adjust(left=0.02, right=0.97, bottom=0.18, top=0.70, wspace=0.08)
        for ax in axes:
            ax.imshow(crop)
            ax.axis("off")
        axes[0].set_title("Immagine di test\nCrop effettivo 224 × 224", fontsize=11)
        target = classes.index(row["true_class"])
        for ax, model, heat in zip(axes[1:], MODELS, normalized):
            result = results[model]
            ax.imshow(heat, cmap="inferno", vmin=0, vmax=1, alpha=0.70 * heat)
            predicted = int(np.argmax(result["probabilities"]))
            p_target = result["probabilities"][target]
            color = "#166534" if classes[predicted] == row["true_class"] else "#9F1239"
            ax.set_title(
                f"{model} · pred.: {LABELS[classes[predicted]]}\n"
                f"p(classe spiegata) = {p_target:.3f}",
                color=color,
                fontsize=11,
            )
            if not np.any(result["cam"] > 0):
                ax.text(
                    0.5,
                    0.04,
                    "Nessun contributo positivo",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=8,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.7},
                )
        reason = (
            "campione deterministico per classe"
            if row["selection_reason"] == "hash_sample_by_true_class"
            else "caso selezionato di disaccordo"
            if row["selection_reason"] == "selected_model_disagreement"
            else "esempio scelto manualmente"
        )
        fig.suptitle(
            f"{context}\n"
            f"Classe vera e spiegata: {LABELS[row['true_class']]} · ObjID {row['objid']}\n"
            f"{reason}",
            y=0.98,
            fontsize=12,
        )
        bar_ax = fig.add_axes([0.36, 0.09, 0.38, 0.025])
        bar = fig.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap="inferno"),
            cax=bar_ax,
            orientation="horizontal",
        )
        bar.set_label(
            "Contributo positivo relativo · scala comune ai tre modelli per questa immagine",
            fontsize=9,
        )
        paths = {}
        try:
            for suffix in ("png", "pdf"):
                path = destination.with_suffix("." + suffix)
                fig.savefig(path, bbox_inches="tight", facecolor="white")
                paths[suffix] = str(path)
        finally:
            plt.close(fig)
    return paths, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", required=True, type=Path, help="Snapshot folder or trusted state tar.gz"
    )
    parser.add_argument(
        "--dataset", required=True, type=Path, help="Dataset folder or colab_dataset.tar.gz"
    )
    parser.add_argument("--task", choices=["binary", "three_class"], default="three_class")
    parser.add_argument("--subset", choices=["100", "300", "900", "full"], default="full")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--qubits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=20260907)
    parser.add_argument(
        "--objids", default="", help="Optional comma-separated test IDs; overrides sampling"
    )
    parser.add_argument("--extra-disagreement", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--weights",
        type=Path,
        help="Existing official ResNet18 IMAGENET1K_V1 checkpoint (read-only)",
    )
    parser.add_argument("--probability-tolerance", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--trust-checkpoints",
        action="store_true",
        help="Allow pickle loading of your own training checkpoints",
    )
    args = parser.parse_args()
    if not args.trust_checkpoints:
        parser.error(
            "Use --trust-checkpoints only for your own trusted training archives; torch.load uses pickle"
        )
    if args.per_class < 1 or args.threads < 1 or args.probability_tolerance <= 0:
        parser.error("per-class, threads and tolerance must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available: select cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = HERE / "outputs" / f"{args.task}_{args.subset}_q{args.qubits}_seed{args.seed}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "started",
        "output": str(output),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "method": "positive Grad-CAM of the TRUE class LOGIT at ResNet18 layer4 output",
        "normalization": "shared maximum of positive upsampled CAMs within each image; not across images",
        "spatial_resolution": "7x7 interpolated to 224x224; interpolation does not add spatial detail",
        "examples": [],
    }
    report_path = output / "audit.json"
    try:
        import pennylane
        import torchvision

        report["environment"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pennylane": pennylane.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        with Snapshot(args.state) as snapshot:
            configs, heads, predictions, common, run_audit = load_runs(snapshot, args, output)
        report["runs"] = run_audit
        classes = configs["B2"]["class_names"]
        if args.objids:
            ids = [s.strip() for s in args.objids.split(",")]
            if len(ids) != len(set(ids)) or not set(ids).issubset(set(common.objid)):
                raise ValueError("Manual IDs must be unique and present in the shared test set")
            selected = common.set_index("objid", drop=False).loc[ids].reset_index(drop=True)
            selected["selection_reason"] = "manual"
        else:
            selected = select_examples(
                common, classes, args.per_class, args.selection_seed, args.extra_disagreement
            )
        selected.to_csv(output / "selected_examples.csv", index=False)
        backbone, weight_hash = load_backbone(args)
        report["backbone_sha256"] = weight_hash
        (output / "figures").mkdir()
        (output / "maps").mkdir()
        with Snapshot(args.dataset) as dataset:
            for index, row in enumerate(selected.to_dict("records"), 1):
                print(f"[{index}/{len(selected)}] {row['true_class']} · {row['objid']}", flush=True)
                crop, tensor, image_hash = load_image(dataset, row, output)
                spatial = spatial_features(backbone, tensor.to(args.device))
                if tuple(spatial.shape) != (1, 512, 7, 7):
                    raise ValueError(f"Unexpected spatial activation shape {spatial.shape}")
                target = classes.index(row["true_class"])
                results, per_model = {}, {}
                for model in MODELS:
                    result = gradcam_from_spatial(spatial, heads[model], target)
                    saved = predictions[model].loc[row["objid"]]
                    probabilities = [saved[f"prob_{name}_model"] for name in classes]
                    error = check_prediction(
                        result["probabilities"], probabilities, args.probability_tolerance
                    )
                    stored_logits = np.array([saved[f"logit_{name}"] for name in classes])
                    logit_error = float(np.max(np.abs(result["logits"] - stored_logits)))
                    # Softmax is shift invariant: also check logits to protect the
                    # exact class-score interpretation used by this attribution.
                    if logit_error > 0.001:
                        raise ValueError(f"Logit mismatch for {model}: {logit_error}")
                    results[model] = result
                    np.savez_compressed(
                        output / "maps" / f"{row['objid']}_{model}.npz",
                        **result,
                        objid=row["objid"],
                        target_class=row["true_class"],
                    )
                    per_model[model] = {
                        "probabilities": result["probabilities"].tolist(),
                        "predicted_class": classes[int(np.argmax(result["probabilities"]))],
                        "max_probability_error": error,
                        "max_logit_error": logit_error,
                        "positive_cam_max": float(result["cam"].max()),
                        "zero_positive_cam": bool(not np.any(result["cam"] > 0)),
                        "gradient_abs_max": float(np.abs(result["gradient"]).max()),
                    }
                    print(
                        f"  {model}: prediction verified (max probability error {error:.2g})",
                        flush=True,
                    )
                task_label = "Binario" if args.task == "binary" else "Tre classi"
                context = f"{task_label} · {args.subset.capitalize()} · {args.qubits} dimensioni/qubit · seed {args.seed}"
                paths, maximum = plot_example(
                    crop,
                    row,
                    results,
                    classes,
                    output / "figures" / f"{index:02d}_{row['true_class']}_{row['objid']}",
                    context,
                )
                report["examples"].append(
                    {
                        **row,
                        "image_sha256": image_hash,
                        "models": per_model,
                        "display_maximum": maximum,
                        "figures": paths,
                    }
                )
                report_path.write_text(
                    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        report["status"] = "complete"
        print(f"COMPLETE: {output}", flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
