#!/usr/bin/env python3
import time

import pandas as pd
import requests

from tesi_qml.cli import atomic_parquet, base_parser, dry_run, load_cli_config
from tesi_qml.config import resolve_project_path
from tesi_qml.data.download import DownloadPolicy, download_one


def main() -> None:
    parser = base_parser(
        "Scarica cutout SDSS DR17 in modo atomico e riprendibile",
        default_config="configs/dataset.yaml",
    )

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    root, config = load_cli_config(args)

    if dry_run(args, f"download immagini nell'intervallo [{args.start}, {args.stop or 'fine'})"):
        return

    source = pd.read_parquet(resolve_project_path(root, config["download_manifest"]))
    source = source.iloc[args.start : args.stop]
    settings = config["download"]

    policy = DownloadPolicy(
        width=settings["width"],
        height=settings["height"],
        scale=settings["scale"],
        read_timeout=settings["timeout_seconds"],
        max_retries=settings["max_retries"],
    )

    image_dir = resolve_project_path(root, config["image_dir"])
    interval = 1.0 / float(settings["requests_per_second"])
    records = []

    with requests.Session() as session:
        for row in source.to_dict("records"):
            destination = image_dir / str(row["label"]) / f"{row['objid']}.jpg"
            records.append(download_one(row, destination, session, policy).to_dict())
            time.sleep(interval)

    output = resolve_project_path(root, config["download_manifest"]).with_name(
        f"download_results_{args.start}_{args.stop or 'end'}.parquet"
    )

    atomic_parquet(output, pd.DataFrame(records))
    print(pd.DataFrame(records)["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
