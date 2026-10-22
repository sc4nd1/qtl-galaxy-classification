#!/usr/bin/env python3
from dataclasses import asdict

import pandas as pd

from tesi_qml.cli import atomic_parquet, base_parser, dry_run, load_cli_config
from tesi_qml.config import resolve_project_path
from tesi_qml.data.validation import validate_image


def main() -> None:
    parser = base_parser(
        "Valida decodifica, dimensione, contrasto e hash delle immagini",
        default_config="configs/dataset.yaml",
    )
    args = parser.parse_args()
    root, config = load_cli_config(args)
    if dry_run(args, "validazione completa dei cutout"):
        return
    manifest = pd.read_parquet(resolve_project_path(root, config["download_manifest"]))
    image_dir = resolve_project_path(root, config["image_dir"])
    expected = (config["download"]["width"], config["download"]["height"])
    rows = []
    for record in manifest.to_dict("records"):
        path = image_dir / str(record["label"]) / f"{record['objid']}.jpg"
        validation = asdict(validate_image(path, expected_size=expected))
        validation["path"] = path.relative_to(root).as_posix()
        rows.append({**record, **validation})
    output = resolve_project_path(root, config["download_manifest"]).with_name("validated.parquet")
    atomic_parquet(output, pd.DataFrame(rows))
    print(pd.DataFrame(rows)["reason"].value_counts().to_dict())


if __name__ == "__main__":
    main()
