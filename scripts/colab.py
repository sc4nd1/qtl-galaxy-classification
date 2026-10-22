#!/usr/bin/env python3
"""Utility compatte per portare la pipeline su Google Colab senza download SDSS."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
from PIL import Image

from tesi_qml.cli import atomic_json, atomic_parquet
from tesi_qml.config import load_yaml, resolve_project_path
from tesi_qml.data.manifests import CLASS_TO_INDEX, build_balanced_manifest
from tesi_qml.data.splits import make_repeated_four_fold_splits
from tesi_qml.data.validation import validate_image

CODE_BUNDLE_ITEMS = (
    "src",
    "scripts",
    "configs",
    "tests",
    "notebooks",
    "docs",
    "README.md",
    "COLAB_GUIDE.md",
    "pyproject.toml",
    "requirements-cluster.txt",
    "requirements-colab.txt",
)
DATASET_BUNDLE_ITEMS = (
    "data/source",
    "data/images",
    "data/manifests",
    "data/splits",
)
STATE_BUNDLE_ITEMS = ("results", "data/features")
EXCLUDED_PARTS = {
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "results",
    ".git",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def _iter_bundle_files(root: Path, relative_item: str):
    source = root / relative_item
    if not source.exists():
        return
    if source.is_file():
        yield source
        return
    for path in source.rglob("*"):
        if path.is_file() and not (set(path.relative_to(root).parts) & EXCLUDED_PARTS):
            yield path


def _write_tar(root: Path, output_path: Path, items: tuple[str, ...]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        for item in items:
            for path in _iter_bundle_files(root, item):
                archive.add(path, arcname=path.relative_to(root).as_posix())
    return output_path


def _iter_state_files(root: Path):
    excluded = EXCLUDED_PARTS - {"results"}
    for item in STATE_BUNDLE_ITEMS:
        source = root / item
        if not source.exists():
            continue
        paths = (source,) if source.is_file() else source.rglob("*")
        for path in paths:
            relative_parts = set(path.relative_to(root).parts)
            if (
                path.is_file()
                and not (relative_parts & excluded)
                and not path.name.endswith(".part")
            ):
                yield path


def create_state_archive(project_root: Path | str, archive_path: Path | str) -> Path:
    """Salva atomicamente risultati e feature bank riusabili tra sessioni Colab."""
    root = Path(project_root).resolve()
    output = Path(archive_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in _iter_state_files(root):
            archive.add(path, arcname=path.relative_to(root).as_posix())
    temporary.replace(output)
    return output


def _validated_state_path(member: tarfile.TarInfo) -> PurePosixPath:
    relative = PurePosixPath(member.name)
    allowed = any(
        relative == PurePosixPath(item) or relative.is_relative_to(PurePosixPath(item))
        for item in STATE_BUNDLE_ITEMS
    )
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not allowed
        or member.issym()
        or member.islnk()
    ):
        raise ValueError(f"Membro TAR non sicuro: {member.name}")
    return relative


def restore_state_archive(project_root: Path | str, archive_path: Path | str) -> bool:
    """Ripristina lo stato Colab senza permettere scritture fuori dalla project root."""
    root = Path(project_root).resolve()
    source = Path(archive_path).resolve()
    if not source.is_file():
        return False
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".colab_restore_", dir=root) as temporary_dir:
        staging = Path(temporary_dir)
        with tarfile.open(source, "r:gz") as archive:
            for member in archive.getmembers():
                relative = _validated_state_path(member)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Membro TAR non sicuro: {member.name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Membro TAR illeggibile: {member.name}")
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as destination:
                    shutil.copyfileobj(extracted, destination)
        for path in staging.rglob("*"):
            if path.is_file():
                destination = root / path.relative_to(staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
    return True


def create_colab_bundles(
    project_root: Path | str,
    output_dir: Path | str,
    *,
    include_features: bool = False,
) -> dict[str, Path]:
    """Crea archivi separati per codice e dataset fisico già preparato."""
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    dataset_items = DATASET_BUNDLE_ITEMS + (("data/features",) if include_features else ())
    bundles = {
        "code": _write_tar(root, output / "colab_code.tar.gz", CODE_BUNDLE_ITEMS),
        "dataset": _write_tar(root, output / "colab_dataset.tar.gz", dataset_items),
    }
    for name in ("colab_quickstart.ipynb",):
        source = root / "notebooks" / name
        if not source.is_file():
            raise FileNotFoundError(f"Notebook Colab mancante: {source}")
        destination = output / name
        shutil.copy2(source, destination)
        bundles[name.removesuffix(".ipynb")] = destination
    return bundles


def _read_source_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path, dtype={"objid": "string"})
    if "objid" not in catalog.columns:
        raise ValueError(f"Catalogo sorgente senza colonna objid: {path}")
    catalog["objid"] = catalog["objid"].astype("string")
    catalog = catalog.drop_duplicates("objid", keep="first")
    return catalog.set_index("objid", drop=False)


def _scan_physical_images(
    physical_root: Path, labels: set[str]
) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    skipped_duplicates = 0
    for label in sorted(labels):
        class_dir = physical_root / label
        if not class_dir.is_dir():
            raise ValueError(f"Cartella classe mancante nel dataset fisico: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            objid = image_path.stem
            if objid in seen:
                skipped_duplicates += 1
                continue
            seen[objid] = label
            rows.append({"objid": objid, "label": label, "source_path": str(image_path)})
    if not rows:
        raise ValueError(f"Nessuna immagine JPEG trovata in {physical_root}")
    return rows, skipped_duplicates


def import_physical_dataset(
    project_root: Path | str,
    physical_root: Path | str,
    *,
    dataset_config: Path | str = "configs/dataset.yaml",
    source_csv: Path | str | None = None,
    allow_missing_catalog: bool = False,
) -> dict[str, Any]:
    """Importa un dataset fisico ``classe/objid.jpg`` evitando il download da Colab."""
    root = Path(project_root).resolve()
    physical = Path(physical_root).resolve()
    config_path = Path(dataset_config)
    config = load_yaml(config_path if config_path.is_absolute() else root / config_path)
    labels = set(config.get("classes", CLASS_TO_INDEX).keys())
    unsupported = sorted(labels - set(CLASS_TO_INDEX))
    if unsupported:
        raise ValueError(f"Classi non supportate: {unsupported}")

    configured_source_csv = resolve_project_path(root, config["source_csv"])
    source_csv_path = (
        Path(source_csv).resolve() if source_csv is not None else configured_source_csv
    )
    catalog = _read_source_catalog(source_csv_path)
    if source_csv is not None and source_csv_path != configured_source_csv.resolve():
        configured_source_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_csv_path, configured_source_csv)
    image_dir = resolve_project_path(root, config["image_dir"])
    manifest_dir = resolve_project_path(root, config["download_manifest"]).parent
    split_dir = resolve_project_path(root, config["split_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_catalog: list[str] = []
    scanned_images, skipped_duplicate_objids = _scan_physical_images(physical, labels)
    for item in scanned_images:
        objid = item["objid"]
        label = item["label"]
        if objid not in catalog.index:
            if not allow_missing_catalog:
                missing_catalog.append(objid)
                continue
            catalog_row = {"objid": str(objid)}
            missing_catalog.append(objid)
        else:
            catalog_row = catalog.loc[objid].to_dict()
        destination = image_dir / label / f"{objid}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source_path"], destination)
        rows.append(
            {
                **catalog_row,
                "objid": str(objid),
                "label": label,
                "path": destination.relative_to(root).as_posix(),
            }
        )
    if missing_catalog and not allow_missing_catalog:
        preview = ", ".join(missing_catalog[:5])
        raise ValueError(f"Immagini senza ObjID nel CSV sorgente: {preview}")

    downloads = pd.DataFrame(rows).sort_values(["label", "objid"], kind="stable")
    downloads_path = resolve_project_path(root, config["download_manifest"])
    atomic_parquet(downloads_path, downloads)

    if downloads.empty:
        raise ValueError("Nessuna immagine importata dal dataset fisico")
    with Image.open(root / str(downloads.iloc[0]["path"])) as first_image:
        expected_size = first_image.size
    validated_rows = []
    for record in downloads.to_dict("records"):
        validation = asdict(validate_image(root / str(record["path"]), expected_size=expected_size))
        validation["path"] = record["path"]
        validated_rows.append({**record, **validation})
    validated = pd.DataFrame(validated_rows)
    validated_path = downloads_path.with_name("validated.parquet")
    atomic_parquet(validated_path, validated)

    final, manifest_summary = build_balanced_manifest(validated, seed=config["project_seed"])
    final_path = resolve_project_path(root, config["final_manifest"])
    atomic_parquet(final_path, final)
    exclusions = pd.DataFrame(
        manifest_summary["excluded_samples"], columns=["objid", "label", "sha256", "reason"]
    )
    atomic_parquet(final_path.with_name("manifest_exclusions.parquet"), exclusions)
    atomic_json(final_path.with_name("manifest_summary.json"), manifest_summary)

    splits = make_repeated_four_fold_splits(final, seeds=config["splits"]["repeat_seeds"])
    splits_path = split_dir / "all.parquet"
    atomic_parquet(splits_path, splits)

    summary = {
        "imported_images": len(downloads),
        "valid_images": int(validated["valid"].fillna(False).sum()),
        "missing_catalog_rows": len(missing_catalog),
        "skipped_duplicate_objids": int(skipped_duplicate_objids),
        "final_samples": len(final),
        "per_class": final["label"].value_counts().sort_index().astype(int).to_dict(),
        "split_rows": len(splits),
        "downloads_manifest": str(downloads_path.relative_to(root)),
        "validated_manifest": str(validated_path.relative_to(root)),
        "final_manifest": str(final_path.relative_to(root)),
        "splits": str(splits_path.relative_to(root)),
    }
    atomic_json(manifest_dir / "physical_import_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import-physical", help="importa dataset fisico classe/objid.jpg"
    )
    import_parser.add_argument("--physical-root", type=Path, required=True)
    import_parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    import_parser.add_argument("--source-csv", type=Path)
    import_parser.add_argument("--allow-missing-catalog", action="store_true")

    bundle_parser = subparsers.add_parser("bundle", help="crea archivi colab_code/dataset")
    bundle_parser.add_argument("--output-dir", type=Path, default=Path("colab_bundle"))
    bundle_parser.add_argument("--include-features", action="store_true")

    args = parser.parse_args()
    if args.command == "import-physical":
        summary = import_physical_dataset(
            args.project_root,
            args.physical_root,
            dataset_config=args.dataset_config,
            source_csv=args.source_csv,
            allow_missing_catalog=args.allow_missing_catalog,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "bundle":
        bundles = create_colab_bundles(
            args.project_root,
            args.output_dir,
            include_features=args.include_features,
        )
        print(json.dumps({key: str(path) for key, path in bundles.items()}, indent=2))


if __name__ == "__main__":
    main()
