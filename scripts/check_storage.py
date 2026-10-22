#!/usr/bin/env python3
from tesi_qml.cli import base_parser, dry_run, load_cli_config
from tesi_qml.storage import check_storage


def main() -> None:
    parser = base_parser(
        "Verifica spazio e inode prima della pipeline", default_config="configs/dataset.yaml"
    )
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--min-free-inodes", type=int, default=10_000)
    args = parser.parse_args()
    root, _ = load_cli_config(args)
    if dry_run(args, f"controllo soglie {args.min_free_gb} GB e {args.min_free_inodes} inode"):
        return
    status = check_storage(
        root,
        min_free_bytes=int(args.min_free_gb * 1024**3),
        min_free_inodes=args.min_free_inodes,
    )
    print(status)


if __name__ == "__main__":
    main()
