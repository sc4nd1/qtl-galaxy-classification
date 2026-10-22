import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts import colab


def test_colab_state_archive_round_trip_preserves_results_and_features(tmp_path: Path) -> None:
    """Catches session backups that lose cached features or copy incomplete files."""
    project = tmp_path / "project"
    archive = tmp_path / "drive" / "session_state.tar.gz"
    (project / "results" / "matrix").mkdir(parents=True)
    (project / "data" / "features" / "seed_42").mkdir(parents=True)
    (project / "results" / "matrix" / "primary.json").write_text(
        '{"complete": 1}\n', encoding="utf-8"
    )
    (project / "data" / "features" / "seed_42" / "features.pt").write_bytes(b"features")
    (project / "results" / "checkpoint.pt.part").write_bytes(b"incomplete")

    colab.create_state_archive(project, archive)

    restored = tmp_path / "restored"
    assert colab.restore_state_archive(restored, archive)
    assert (restored / "results" / "matrix" / "primary.json").read_text(
        encoding="utf-8"
    ) == '{"complete": 1}\n'
    assert (restored / "data" / "features" / "seed_42" / "features.pt").read_bytes() == b"features"
    assert not (restored / "results" / "checkpoint.pt.part").exists()


def test_restore_state_archive_treats_missing_first_session_as_empty(tmp_path: Path) -> None:
    """Catches first-run setup failing merely because no prior state exists."""
    assert not colab.restore_state_archive(tmp_path / "project", tmp_path / "missing.tar.gz")


def test_restore_state_archive_rejects_path_traversal(tmp_path: Path) -> None:
    """Catches a malformed Drive archive writing outside the project directory."""
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        payload = b"unsafe"
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="non sicuro"):
        colab.restore_state_archive(tmp_path / "project", archive)
    assert not (tmp_path / "outside.txt").exists()


def test_colab_session_saves_failed_attempt_before_returning(tmp_path: Path) -> None:
    """Catches a failed matrix command leaving its recoverable artifacts only on the VM."""
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_matrix.py").write_text(
        """\
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--project-root', type=Path, required=True)
parser.add_argument('--max-runs')
parser.add_argument('--device')
parser.add_argument('--report-only', action='store_true')
args = parser.parse_args()
counter = args.project_root / 'results' / 'attempt.txt'
counter.parent.mkdir(parents=True, exist_ok=True)
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
raise SystemExit(7 if attempt == 2 else 0)
""",
        encoding="utf-8",
    )
    state_archive = tmp_path / "drive" / "session_state.tar.gz"
    session_script = Path(__file__).resolve().parents[1] / "scripts" / "colab_session.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(session_script),
            "--project-root",
            str(project),
            "--state-archive",
            str(state_archive),
            "run",
            "--max-runs",
            "2",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 7
    restored = tmp_path / "restored"
    assert colab.restore_state_archive(restored, state_archive)
    assert (restored / "results" / "attempt.txt").read_text(encoding="utf-8") == "2"
