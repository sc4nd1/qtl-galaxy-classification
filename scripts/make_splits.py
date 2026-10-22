#!/usr/bin/env python3
import pandas as pd

from tesi_qml.cli import atomic_parquet, base_parser, dry_run, load_cli_config
from tesi_qml.config import resolve_project_path
from tesi_qml.data.splits import make_repeated_four_fold_splits


def main() -> None:
    parser = base_parser(
        "Genera split stratificati ripetuti 2/1/1", default_config="configs/dataset.yaml"
    )
    args = parser.parse_args()
    root, config = load_cli_config(args)
    if dry_run(args, "generazione split per tutti i repeat seed"):
        return
    manifest = pd.read_parquet(resolve_project_path(root, config["final_manifest"]))
    splits = make_repeated_four_fold_splits(manifest, seeds=config["splits"]["repeat_seeds"])
    output = resolve_project_path(root, config["split_dir"]) / "all.parquet"
    atomic_parquet(output, splits)
    print(
        {"rows": len(splits), "experiments": splits.groupby(["repeat_seed", "iteration"]).ngroups}
    )


if __name__ == "__main__":
    main()
