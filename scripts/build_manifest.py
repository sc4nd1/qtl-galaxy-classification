#!/usr/bin/env python3
import pandas as pd

from tesi_qml.cli import atomic_json, atomic_parquet, base_parser, dry_run, load_cli_config
from tesi_qml.config import resolve_project_path
from tesi_qml.data.manifests import build_balanced_manifest


def main() -> None:
    parser = base_parser(
        "Costruisce il manifest finale deduplicato e bilanciato",
        default_config="configs/dataset.yaml",
    )
    args = parser.parse_args()
    root, config = load_cli_config(args)
    if dry_run(args, "deduplicazione visiva e bilanciamento sul minimo valido"):
        return
    validated_path = resolve_project_path(root, config["download_manifest"]).with_name(
        "validated.parquet"
    )
    final, summary = build_balanced_manifest(
        pd.read_parquet(validated_path), seed=config["project_seed"]
    )
    output = resolve_project_path(root, config["final_manifest"])
    atomic_parquet(output, final)
    exclusions = pd.DataFrame(
        summary["excluded_samples"], columns=["objid", "label", "sha256", "reason"]
    )
    atomic_parquet(output.with_name("manifest_exclusions.parquet"), exclusions)
    atomic_json(output.with_name("manifest_summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
