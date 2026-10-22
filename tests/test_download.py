from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from tesi_qml.data.download import DownloadPolicy, download_one


def jpeg_bytes() -> bytes:
    y, x = np.indices((320, 320))
    pixels = np.stack(((x + y) % 256, x % 256, y % 256), axis=-1).astype(np.uint8)
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="JPEG", quality=90)
    return stream.getvalue()


class StaticResponse:
    def __init__(self, content: bytes, status_code: int = 200, content_type: str = "image/jpeg"):
        self.content = content
        self.status_code = status_code
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(content))}
        self.url = "https://example.invalid/cutout"


class StaticSession:
    def __init__(self, responses: list[StaticResponse]):
        self.responses = list(responses)
        self.last_headers = None

    def get(self, url, *, params, timeout, headers):
        self.last_headers = headers
        if not self.responses:
            raise AssertionError("unexpected network request")
        return self.responses.pop(0)


def record(objid: str = "1237654605318717697") -> dict[str, object]:
    return {"objid": objid, "label": "merger", "ra": 120.0, "dec": 20.0}


def test_download_one_writes_atomically_and_preserves_exact_objid(tmp_path: Path) -> None:
    """Catches rounded filenames and partial JPEGs being treated as complete."""
    destination = tmp_path / "1237654605318717697.jpg"
    session = StaticSession([StaticResponse(jpeg_bytes())])
    result = download_one(
        record(),
        destination,
        session,
        DownloadPolicy(max_retries=1),
        sleep_fn=lambda _: None,
    )

    assert result.objid == "1237654605318717697"
    assert result.status == "downloaded"
    assert destination.is_file()
    assert not destination.with_suffix(".jpg.part").exists()
    assert result.sha256
    assert session.last_headers["User-Agent"].startswith("tesi-qml/")


def test_download_one_resumes_without_requesting_an_existing_valid_file(tmp_path: Path) -> None:
    """Catches resume logic that wastes rate-limited SDSS requests."""
    destination = tmp_path / "x.jpg"
    destination.write_bytes(jpeg_bytes())

    result = download_one(
        record("x"),
        destination,
        StaticSession([]),
        DownloadPolicy(max_retries=1),
        sleep_fn=lambda _: None,
    )

    assert result.status == "skipped_valid"
    assert result.attempts == 0


def test_download_one_rejects_non_image_response_without_final_file(tmp_path: Path) -> None:
    """Catches a successful HTTP code masking an HTML error payload."""
    destination = tmp_path / "x.jpg"
    result = download_one(
        record("x"),
        destination,
        StaticSession([StaticResponse(b"<html>blocked</html>", content_type="text/html")]),
        DownloadPolicy(max_retries=1),
        sleep_fn=lambda _: None,
    )

    assert result.status == "invalid_response"
    assert result.error == "content_type:text/html"
    assert not destination.exists()
    assert not destination.with_suffix(".jpg.part").exists()


def test_download_one_retries_transient_status_and_then_succeeds(tmp_path: Path) -> None:
    """Catches treating SDSS rate limiting as permanent data attrition."""
    destination = tmp_path / "x.jpg"
    session = StaticSession(
        [
            StaticResponse(b"busy", status_code=503, content_type="text/plain"),
            StaticResponse(jpeg_bytes()),
        ]
    )
    result = download_one(
        record("x"),
        destination,
        session,
        DownloadPolicy(max_retries=2, backoff_seconds=0),
        sleep_fn=lambda _: None,
    )

    assert result.status == "downloaded"
    assert result.attempts == 2
