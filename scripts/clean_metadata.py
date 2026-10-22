#!/usr/bin/env python3

from tesi_qml.cli import atomic_json, atomic_parquet, base_parser, dry_run, load_cli_config
from tesi_qml.config import resolve_project_path
from tesi_qml.data.metadata import clean_metadata, read_candidate_csv, select_download_pool


def main() -> None:
    parser = base_parser(
        "Pulisce GalaxyCandidates e applica le hard label", default_config="configs/dataset.yaml"
    )
    args = parser.parse_args()
    root, config = load_cli_config(args)
    if dry_run(args, "pulizia metadata e creazione pool di download"):
        return
    source = resolve_project_path(root, config["source_csv"])
    result = clean_metadata(
        read_candidate_csv(source),
        config["thresholds"],
        min_redshift=config["redshift"]["minimum"],
        max_redshift=config["redshift"]["maximum"],
    )
    pool = select_download_pool(
        result.clean,
        seed=config["project_seed"],
        oversample_factor=config["download"]["oversample_factor"],
    )
    candidate_path = resolve_project_path(root, config["candidate_manifest"])
    download_path = resolve_project_path(root, config["download_manifest"])
    atomic_parquet(candidate_path, result.clean)
    atomic_parquet(download_path, pool)
    atomic_parquet(candidate_path.with_name("rejected.parquet"), result.rejected)
    atomic_json(candidate_path.with_name("metadata_summary.json"), result.summary)
    print(result.summary)


if __name__ == "__main__":
    main()
