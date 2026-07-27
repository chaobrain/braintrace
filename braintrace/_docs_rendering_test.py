"""Regression tests for stored reader-facing documentation evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "docs"


def _notebook(stem: str, directory: str) -> dict:
    path = _DOCS / directory / f"{stem}.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))


def _code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def _outputs(notebook: dict) -> list[dict]:
    return [
        output
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", [])
    ]


def _reader_text(notebook: dict) -> str:
    chunks: list[str] = []
    for output in _outputs(notebook):
        if output.get("output_type") == "stream":
            text = output.get("text", "")
            chunks.append("".join(text) if isinstance(text, list) else text)
        data = output.get("data", {})
        plain = data.get("text/plain", "")
        chunks.append("".join(plain) if isinstance(plain, list) else plain)
    return "\n".join(chunks)


def test_quickstart_stores_and_references_loss_figure():
    source = (_DOCS / "quickstart" / "quickstart.rst").read_text(encoding="utf-8")
    image = _DOCS / "_static" / "quickstart_loss.png"

    assert ".. image:: /_static/quickstart_loss.png" in source
    assert image.stat().st_size > 1_000
    assert 'fig.savefig("quickstart_loss.png"' in source
    assert "initial loss:" in source
    assert "final loss:" in source


def test_core_concepts_stores_online_and_bptt_results():
    notebook = _notebook("concepts", "quickstart")
    text = _reader_text(notebook)

    for expected in (
        "initial loss:",
        "final loss:",
        "initial prediction:",
        "final prediction:",
        "parameter change:",
        "online forward loss:",
        "BPTT forward loss:",
    ):
        assert expected in text


@pytest.mark.parametrize(
    ("stem", "directory"),
    (
        ("rnn_online_learning", "quickstart"),
        ("snn_online_learning", "quickstart"),
        ("drtrl", "tutorials"),
        ("pp_prop", "tutorials"),
    ),
)
def test_training_notebook_stores_clean_plot(stem: str, directory: str):
    notebook = _notebook(stem, directory)
    outputs = _outputs(notebook)
    text = _reader_text(notebook)

    assert any("image/png" in output.get("data", {}) for output in outputs)
    assert not any(output.get("output_type") == "error" for output in outputs)
    assert "FigureCanvasAgg is non-interactive" not in text
    assert "D:\\BrainTrace" not in text
    assert "/mnt/d/" not in text


@pytest.mark.parametrize(
    ("stem", "directory"),
    (
        ("rnn_online_learning", "quickstart"),
        ("snn_online_learning", "quickstart"),
        ("pp_prop", "tutorials"),
    ),
)
def test_training_plot_is_explicitly_shown_and_stored_once(
    stem: str, directory: str
):
    notebook = _notebook(stem, directory)
    plotting_cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "plt.subplots(" in "".join(cell["source"])
    ]

    assert len(plotting_cells) == 1
    source = "".join(plotting_cells[0]["source"])
    png_outputs = [
        output
        for output in plotting_cells[0].get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    assert "plt.show()" in source
    assert source.rstrip().endswith("plt.show()")
    assert len(png_outputs) == 1


def test_rnn_comparison_stores_two_hundred_update_run():
    notebook = _notebook("rnn_online_learning", "quickstart")
    source = _code_source(notebook)
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    text = _reader_text(notebook)

    assert "n_epochs=200" in source
    assert "n_epochs=200, time_lag=20, batch_size=32" in source
    assert "same 200 fixed-seed batches" in markdown
    for checkpoint in ("Step 0", "Step 100", "Step 199"):
        assert checkpoint in text


def test_snn_example_has_learnable_fixed_seed_evidence():
    notebook = _notebook("snn_online_learning", "quickstart")
    source = _code_source(notebook)
    text = _reader_text(notebook)

    assert "brainstate.random.seed" in source
    assert "Accuracy" in source
    assert "n_updates=200" in source
    assert "lr=3e-3" in source
    assert "decay_or_rank=0.5" in source
    assert "Update 199" in source
    checkpoints = re.findall(
        r"Update\s+\d+,\s+Loss:\s+([0-9.eE+-]+),\s+Accuracy:\s+([0-9.]+)",
        text,
    )
    assert len(checkpoints) >= 2
    first_loss, first_accuracy = map(float, checkpoints[0])
    final_loss, final_accuracy = map(float, checkpoints[-1])
    assert final_loss < first_loss
    assert final_accuracy > first_accuracy
    assert final_accuracy > 0.5


@pytest.mark.parametrize("stem", ("rnn_online_learning", "snn_online_learning"))
def test_workflow_training_uses_compiled_loops_and_brainstate_random(stem: str):
    notebook = _notebook(stem, "quickstart")
    source = _code_source(notebook)

    assert "brainstate.random.seed" in source
    assert "np.random" not in source
    assert "while True" not in source
    assert not re.search(r"^\s*for\s+(?:i|epoch)\b", source, re.MULTILINE)
    assert "brainstate.transform.for_loop" in source
