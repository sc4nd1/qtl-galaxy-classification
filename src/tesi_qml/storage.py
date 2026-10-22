"""Controlli preventivi di spazio e inode per il cluster."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_STATVFS = getattr(os, "statvfs", None)


def check_storage(
    path: Path | str,
    *,
    min_free_bytes: int,
    min_free_inodes: int,
    disk_usage_fn: Callable[[Path], Any] = shutil.disk_usage,
    statvfs_fn: Callable[[Path], Any] | None = DEFAULT_STATVFS,
) -> dict[str, int | None]:
    target = Path(path).resolve()
    usage = disk_usage_fn(target)
    free_bytes = int(usage.free)
    if free_bytes < min_free_bytes:
        raise RuntimeError(
            f"spazio libero insufficiente: {free_bytes} byte, richiesti {min_free_bytes}"
        )
    free_inodes: int | None = None
    if statvfs_fn is not None:
        stats = statvfs_fn(target)
        free_inodes = int(stats.f_favail)
        if free_inodes < min_free_inodes:
            raise RuntimeError(
                f"Numero di inode liberi insufficiente: {free_inodes}, richiesti {min_free_inodes}"
            )
    return {"free_bytes": free_bytes, "free_inodes": free_inodes}
