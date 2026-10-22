import json
import sys
import types
from pathlib import Path

import pytest

NOTEBOOKS = {
    "01_dataset_audit.ipynb": ("# Audit del dataset", "## Distribuzione delle classi"),
    "02_experiment_analysis.ipynb": ("# Analisi degli esperimenti", "## Confronti appaiati"),
    "03_thesis_figures.ipynb": ("# Figure per la tesi", "## Esportazione PNG e PDF"),
}


@pytest.mark.parametrize("filename,headings", NOTEBOOKS.items())
def test_notebook_policy_and_expected_sections(filename, headings):
    path = Path(__file__).resolve().parents[1] / "notebooks" / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(notebook)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert "/content/drive" not in serialized
    assert "C:\\\\Users" not in serialized
    assert "Trainer.fit" not in serialized
    assert "scripts/train.py" not in serialized
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    for heading in headings:
        assert heading in markdown


def test_colab_notebook_delegates_scientific_work_to_scripts() -> None:
    """Catches model matrices or training logic being duplicated inside Colab."""
    path = Path(__file__).resolve().parents[1] / "notebooks" / "colab_quickstart.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(notebook)

    assert "scripts/colab_session.py" in serialized
    assert "MODELS =" not in serialized
    assert "for model" not in serialized
    assert "Trainer" not in serialized
    assert "PennylaneAngleQNN" not in serialized


def test_colab_quickstart_initialization_cell_executes_without_name_errors(monkeypatch) -> None:
    """Catches setup cells that use colab/drive without proper imports."""
    path = Path(__file__).resolve().parents[1] / "notebooks" / "colab_quickstart.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    setup_cell = "".join(notebook["cells"][1]["source"])
    drive = types.SimpleNamespace(mount=lambda _path: None)
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = drive
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    namespace: dict[str, object] = {}
    try:
        exec(setup_cell, namespace)
    except FileNotFoundError:
        pass

    assert namespace.get("PROJECT_ROOT") == Path("/content/tesi_finale")
