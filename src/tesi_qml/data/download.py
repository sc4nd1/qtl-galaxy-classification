"""Download SDSS riprendibile con scrittura atomica e audit per campione."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import requests

from .validation import validate_image


class HTTPSession(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        timeout: tuple[float, float],
        headers: Mapping[str, str],
    ): ...


@dataclass(frozen=True)
class DownloadPolicy:
    base_url: str = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
    width: int = 320
    height: int = 320
    scale: float = 0.396
    connect_timeout: float = 5.0
    read_timeout: float = 20.0
    max_retries: int = 5
    backoff_seconds: float = 1.0
    user_agent: str = "tesi-qml/0.1 (academic SDSS DR17 image study)"


@dataclass(frozen=True)
class DownloadRecord:
    objid: str
    label: str
    path: str
    status: str
    attempts: int
    http_status: int | None = None
    content_type: str | None = None
    sha256: str | None = None
    byte_size: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}


def download_one(
    record: Mapping[str, object],
    destination: Path | str,
    session: HTTPSession | requests.Session,
    policy: DownloadPolicy,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> DownloadRecord:
    """Scarica un singolo cutout e lo pubblica solo dopo validazione completa."""
    objid = str(record["objid"])
    label = str(record["label"])
    final_path = Path(destination)
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    existing = validate_image(final_path, expected_size=(policy.width, policy.height))
    if existing.valid:
        return DownloadRecord(
            objid=objid,
            label=label,
            path=str(final_path),
            status="skipped_valid",
            attempts=0,
            sha256=existing.sha256,
            byte_size=existing.byte_size,
        )

    params = {
        "ra": float(record["ra"]),
        "dec": float(record["dec"]),
        "scale": policy.scale,
        "width": policy.width,
        "height": policy.height,
    }
    last_status: int | None = None
    last_content_type: str | None = None
    last_error: str | None = None

    for attempt in range(1, policy.max_retries + 1):
        try:
            response = session.get(
                policy.base_url,
                params=params,
                timeout=(policy.connect_timeout, policy.read_timeout),
                headers={"User-Agent": policy.user_agent, "Accept": "image/jpeg"},
            )
            last_status = int(response.status_code)
            last_content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()

            if last_status in TRANSIENT_STATUS:
                last_error = f"http_status:{last_status}"
            elif last_status != 200:
                return DownloadRecord(
                    objid,
                    label,
                    str(final_path),
                    "http_error",
                    attempt,
                    http_status=last_status,
                    content_type=last_content_type,
                    error=f"http_status:{last_status}",
                )
            elif not last_content_type.startswith("image/"):
                return DownloadRecord(
                    objid,
                    label,
                    str(final_path),
                    "invalid_response",
                    attempt,
                    http_status=last_status,
                    content_type=last_content_type,
                    error=f"content_type:{last_content_type}",
                )
            else:
                with part_path.open("wb") as stream:
                    stream.write(response.content)
                    stream.flush()
                    os.fsync(stream.fileno())
                validation = validate_image(part_path, expected_size=(policy.width, policy.height))
                if validation.valid:
                    part_path.replace(final_path)
                    return DownloadRecord(
                        objid,
                        label,
                        str(final_path),
                        "downloaded",
                        attempt,
                        http_status=last_status,
                        content_type=last_content_type,
                        sha256=validation.sha256,
                        byte_size=validation.byte_size,
                    )
                last_error = validation.reason
                part_path.unlink(missing_ok=True)
        except requests.RequestException as exc:
            last_error = f"request_error:{type(exc).__name__}"
            part_path.unlink(missing_ok=True)

        if attempt < policy.max_retries:
            sleep_fn(policy.backoff_seconds * (2 ** (attempt - 1)))

    part_path.unlink(missing_ok=True)
    status = "http_error" if last_status in TRANSIENT_STATUS else "invalid_image"
    return DownloadRecord(
        objid,
        label,
        str(final_path),
        status,
        policy.max_retries,
        http_status=last_status,
        content_type=last_content_type,
        error=last_error,
    )
