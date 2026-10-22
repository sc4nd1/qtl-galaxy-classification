"""Seed e metadata necessari alla riproducibilità degli esperimenti."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from importlib import metadata
from typing import Any

import numpy as np
import torch


def config_fingerprint(config: Any) -> str:
    """Hash stabile di una configurazione JSON-serializzabile."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def derive_seed(repeat_seed: int, namespace: str) -> int:
    """Deriva un sub-seed stabile senza dipendere da ``hash()`` di Python."""
    if repeat_seed < 0:
        raise ValueError("Il repeat seed deve essere non negativo")
    if not namespace:
        raise ValueError("Il namespace del sub-seed non può essere vuoto")
    digest = hashlib.sha256(f"{repeat_seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Imposta i generatori casuali usati dalla pipeline."""
    if seed < 0:
        raise ValueError("Il seed deve essere non negativo")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass


def environment_metadata() -> dict[str, Any]:
    """Restituisce una fotografia JSON-serializzabile dell'ambiente corrente."""
    packages = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "torchvision",
        "pennylane",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }
