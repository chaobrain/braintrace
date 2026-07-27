"""Structural regression tests for the reader-facing documentation.

These tests complement ``_docs_examples_test.py``.  They guard navigation and
notebook contracts that Sphinx cannot validate because notebook execution is
disabled in ``docs/conf.py``.
"""

from __future__ import annotations

import json
import re
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "docs"


def _notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sources(path: Path, cell_type: str | None = None) -> list[str]:
    cells = _notebook(path)["cells"]
    return [
        "".join(cell["source"])
        for cell in cells
        if cell_type is None or cell["cell_type"] == cell_type
    ]


def _all_source(path: Path) -> str:
    return "\n".join(_sources(path))


def _execute_notebook_code(path: Path) -> dict:
    namespace = {"__name__": f"docs.{path.stem}"}
    for index, source in enumerate(_sources(path, "code")):
        exec(compile(source, f"{path}::cell-{index}", "exec"), namespace)
    return namespace


def test_installation_uses_direct_pip_and_python_c_verification():
    path = _DOCS / "quickstart" / "installation.ipynb"
    source = _all_source(path)

    assert "python -m pip" not in source
    for extra in ("cpu", "cuda12", "cuda13", "tpu"):
        assert f"pip install -U braintrace[{extra}]" in source
    assert "## Verify the installation" in source
    assert 'python -c "import braintrace, jax;' in source


def test_installation_and_homepage_use_concise_platform_commands():
    installation = _all_source(_DOCS / "quickstart" / "installation.ipynb")
    homepage = (_DOCS / "index.rst").read_text(encoding="utf-8")
    development = homepage[
        homepage.index(".. tab-item:: Development") :
        homepage.index("For verification and platform troubleshooting")
    ]

    assert "### TPU" in installation
    assert "Google Cloud TPU" not in installation
    assert development.count("pip install") == 1
    assert "pip install -r requirements.txt" in development
    assert (
        ".. tab-item:: Development\n\n"
        "         .. code-block:: bash\n\n"
        "            pip install -r requirements.txt"
    ) in development
    assert "requirements-dev.txt" not in development
    assert "requirements-doc.txt" not in development
    assert "pip install -e ." not in development


def test_current_tutorial_titles_are_concise_and_consistent():
    homepage = (_DOCS / "index.rst").read_text(encoding="utf-8")
    quickstart = (_DOCS / "quickstart" / "quickstart.rst").read_text(
        encoding="utf-8"
    )
    etp = _all_source(_DOCS / "tutorials" / "etp_primitives.ipynb")
    concepts = _all_source(_DOCS / "quickstart" / "concepts.ipynb")
    transforms = _all_source(
        _DOCS / "tutorials" / "customizing_primitive_transforms.ipynb"
    )

    assert quickstart.splitlines()[0] == "Quickstart"
    assert "Five-minute Quickstart" not in homepage
    assert "Quickstart" in homepage
    assert etp.splitlines()[0] == "# ETP Primitives"
    assert "ETP Primitives Deep Dive" not in etp
    assert "ETP Primitives Deep Dive" not in concepts
    assert "ETP Primitives Deep Dive" not in transforms


def test_root_navigation_exposes_the_learning_path():
    source = (_DOCS / "index.rst").read_text(encoding="utf-8")
    toctrees = source[source.index(".. toctree::") :]

    expected = (
        ":caption: Get started",
        "quickstart/installation.ipynb",
        "quickstart/quickstart",
        "quickstart/concepts.ipynb",
        ":caption: Tutorial",
        "quickstart/rnn_online_learning.ipynb",
        "quickstart/snn_online_learning.ipynb",
        "tutorials/drtrl.ipynb",
        "tutorials/pp_prop.ipynb",
        "tutorials/customizing_primitive_transforms.ipynb",
        "tutorials/hidden_states.ipynb",
        "tutorials/graph_visualization.ipynb",
        "tutorials/batching.ipynb",
        ":caption: Advanced",
        "tutorials/etp_primitives.ipynb",
        "advanced/compiler_internals.ipynb",
        ":caption: Examples",
        "examples/snn_examples.rst",
        "examples/rnn_examples.rst",
        "examples/pp_prop_examples.rst",
        "examples/drtrl_examples.rst",
        ":caption: API Reference",
        "API Overview <apis/index.rst>",
        "ETP Operators <apis/concepts.rst>",
        "Compiler and Executor <apis/compiler.rst>",
        "Algorithms <apis/algorithms.rst>",
        "Neural Network Layers <apis/nn.rst>",
        "Others <apis/primitives.rst>",
    )
    positions = [toctrees.index(item) for item in expected]
    assert positions == sorted(positions)
    assert ":caption: Training workflows" not in source
    assert ":caption: Algorithm tutorials" not in source


def test_homepage_restores_historical_editorial_layout():
    source = (_DOCS / "index.rst").read_text(encoding="utf-8")

    assert "Basic usage" not in source
    assert ".. tab-set::" in source
    for tab in ("CPU", "NVIDIA GPU", "TPU", "Development"):
        assert f".. tab-item:: {tab}" in source
    assert "Learn more" in source
    assert "Learning path" in source
    assert ".. figure:: _static/braintrace-learning-map.svg" in source
    assert source.count(".. grid-item-card:: :material-regular:") == 9
    for target in (
        "quickstart/installation",
        "quickstart/quickstart",
        "quickstart/concepts",
        "quickstart/rnn_online_learning",
        "quickstart/snn_online_learning",
        "apis/algorithms",
        "advanced/compiler_internals",
        "examples/snn_examples",
        "apis/index",
    ):
        assert target in source
    for number in range(1, 11):
        assert f"**{number}**" in source


def test_learning_map_is_accessible_and_complete():
    path = _DOCS / "_static" / "braintrace-learning-map.svg"
    source = path.read_text(encoding="utf-8")
    root = ET.fromstring(source)
    namespace = {"svg": "http://www.w3.org/2000/svg"}

    labelled_by = root.attrib["aria-labelledby"].split()
    title = root.find("svg:title", namespace)
    description = root.find("svg:desc", namespace)
    assert title is not None and title.attrib["id"] in labelled_by
    assert description is not None and description.attrib["id"] in labelled_by
    assert root.attrib["viewBox"] == "0 0 760 720"
    for number in range(1, 11):
        assert f">{number}<" in source


def test_tutorial_groups_are_progressive_enhancement():
    conf = (_DOCS / "conf.py").read_text(encoding="utf-8")
    script = (_DOCS / "_static" / "js" / "tutorial-groups.js").read_text(
        encoding="utf-8"
    )

    assert '"css/braintrace-docs.css"' in conf
    assert '"js/tutorial-groups.js"' in conf
    assert "Training workflows" in script
    assert "Algorithm tutorials" in script
    assert "Compiler & runtime" in script
    for filename in (
        "rnn_online_learning.html",
        "snn_online_learning.html",
        "drtrl.html",
        "pp_prop.html",
        "customizing_primitive_transforms.html",
        "hidden_states.html",
        "graph_visualization.html",
        "batching.html",
    ):
        assert filename in script
    assert "etp_primitives.html" not in script


def test_sphinx_uses_current_notebook_options_and_excludes_internal_specs():
    conf = (_DOCS / "conf.py").read_text(encoding="utf-8")

    assert 'nb_execution_mode = "off"' in conf
    assert "nb_execution_timeout = 200" in conf
    assert "brainx_inject_base = False" in conf
    assert "jupyter_execute_notebooks" not in conf
    assert re.search(r"exclude_patterns\s*=.*['\"]specs['\"]", conf)


def test_quickstart_has_current_reproducible_online_learning_primitives():
    source = (_DOCS / "quickstart" / "quickstart.rst").read_text(encoding="utf-8")

    for required in (
        "brainstate.random.seed",
        "braintrace.nn.MiniGRU",
        "braintrace.compile",
        "brainstate.transform.grad",
        "brainstate.transform.scan",
        "initial loss:",
        "final loss:",
    ):
        assert required in source
    assert "jax.random" not in source
    assert "jax.lax.scan" not in source
    assert "online_scan" not in source
    assert not re.search(r"^\s*for\s+\w+\s+in\s+range\(", source, re.MULTILINE)


def test_quickstart_executes_and_reduces_loss(capsys, monkeypatch, tmp_path):
    source = (_DOCS / "quickstart" / "quickstart.rst").read_text(encoding="utf-8")
    directive = source.index(".. code-block:: python")
    indented = source[directive:].splitlines()[2:]
    code_lines = []
    for line in indented:
        if line.startswith("   ") or not line:
            code_lines.append(line)
        else:
            break

    monkeypatch.chdir(tmp_path)
    exec(compile(textwrap.dedent("\n".join(code_lines)), "quickstart.rst", "exec"), {})
    output = capsys.readouterr().out
    initial = float(re.search(r"initial loss:\s*([0-9.eE+-]+)", output).group(1))
    final = float(re.search(r"final loss:\s*([0-9.eE+-]+)", output).group(1))
    assert final < initial
    assert (tmp_path / "quickstart_loss.png").stat().st_size > 1_000


def test_core_concepts_has_one_step_three_and_two_column_algorithm_table():
    path = _DOCS / "quickstart" / "concepts.ipynb"
    source = _all_source(path)

    assert source.count("# Step 3:") == 1
    assert "Code comparison with BPTT" in source
    assert "initial loss" in source
    assert "final loss" in source
    assert "parameter change" in source

    algorithms_cell = next(
        cell for cell in _sources(path, "markdown")
        if "## 6. Available Algorithms" in cell
    )
    header = next(line for line in algorithms_cell.splitlines() if line.startswith("| Algorithm"))
    separator = algorithms_cell.splitlines()[algorithms_cell.splitlines().index(header) + 1]
    assert header.count("|") == 3
    assert separator.count("|") == 3
    for api_name in (
        "braintrace.D_RTRL",
        "braintrace.pp_prop",
        "braintrace.EProp",
        "braintrace.OSTLRecurrent",
        "braintrace.OSTLFeedforward",
        "braintrace.SnAp",
    ):
        assert api_name in algorithms_cell


def test_core_concepts_title_has_no_product_suffix():
    markdown = _sources(_DOCS / "quickstart" / "concepts.ipynb", "markdown")

    assert markdown[0].splitlines()[0] == "# Core Concepts"
    assert "Core Concepts of BrainTrace" not in "\n".join(markdown)


def test_algorithm_api_documents_snap():
    source = (_DOCS / "apis" / "algorithms.rst").read_text(encoding="utf-8")
    assert re.search(r"^\s+SnAp\s*$", source, re.MULTILINE)


@pytest.mark.parametrize(
    ("filename", "heading"),
    (
        ("rnn_online_learning.ipynb", "# RNN Online Learning"),
        ("snn_online_learning.ipynb", "# SNN Online Learning"),
    ),
)
def test_training_workflow_titles_are_product_agnostic(filename: str, heading: str):
    source = _sources(_DOCS / "quickstart" / filename, "markdown")
    assert source[0].splitlines()[0] == heading
    assert "with BrainTrace" not in source[0]


def test_snn_opening_has_no_standalone_introduction():
    path = _DOCS / "quickstart" / "snn_online_learning.ipynb"
    markdown = _sources(path, "markdown")

    assert not any(re.search(r"^## Introduction\s*$", cell, re.MULTILINE) for cell in markdown)
    assert markdown[1].startswith("## 1. Setup")
    assert "What you will learn" in markdown[0]


@pytest.mark.parametrize(
    ("stem", "algorithm"),
    (("drtrl", "braintrace.D_RTRL"), ("pp_prop", "braintrace.pp_prop")),
)
def test_algorithm_tutorial_is_a_minigru_notebook(stem: str, algorithm: str):
    notebook = _DOCS / "tutorials" / f"{stem}.ipynb"
    markdown = _DOCS / "tutorials" / f"{stem}.md"
    source = _all_source(notebook)

    assert not markdown.exists()
    for required in (
        "braintrace.nn.MiniGRU",
        algorithm,
        "brainstate.random.seed",
        "brainstate.transform.grad",
        "brainstate.transform.scan",
        "initial loss:",
        "final loss:",
    ):
        assert required in source
    assert "jax.random" not in source
    assert "jax.lax.scan" not in source
    assert "online_scan" not in source


def test_algorithm_notebook_stems_are_unambiguous():
    for stem in ("drtrl", "pp_prop"):
        matches = sorted((_DOCS / "tutorials").glob(f"{stem}.*"))
        assert [path.suffix for path in matches] == [".ipynb"]


@pytest.mark.parametrize(
    ("filename", "required_links"),
    (
        (
            "snn_examples.rst",
            (
                "/apis/generated/braintrace.pp_prop",
                "/apis/generated/braintrace.EProp",
                "/apis/generated/braintrace.OSTLRecurrent",
                "/apis/nn",
            ),
        ),
        (
            "rnn_examples.rst",
            (
                "/apis/generated/braintrace.D_RTRL",
                "/apis/generated/braintrace.SnAp",
                "/apis/nn",
            ),
        ),
        (
            "pp_prop_examples.rst",
            (
                "/tutorials/pp_prop",
                "/apis/generated/braintrace.pp_prop",
                "/apis/concepts",
            ),
        ),
        (
            "drtrl_examples.rst",
            (
                "/tutorials/drtrl",
                "/apis/generated/braintrace.D_RTRL",
                "/apis/concepts",
            ),
        ),
    ),
)
def test_example_tracks_link_to_current_tutorials_and_apis(
    filename: str, required_links: tuple[str, ...]
):
    source = (_DOCS / "examples" / filename).read_text(
        encoding="utf-8"
    )

    assert "Related API" in source
    for target in required_links:
        assert target in source


def test_flattened_examples_page_is_removed():
    assert not (_DOCS / "examples" / "core_examples.rst").exists()


def test_api_reference_separates_primitive_registration_as_others():
    source = (_DOCS / "apis" / "index.rst").read_text(encoding="utf-8")
    operators = source[source.index("* - **Operators**") :]
    operators = operators[: operators.index("* -", 4)]
    others = source[source.index("* - **Others**") :]

    assert ":doc:`concepts`" in operators
    assert ":doc:`primitives`" not in operators
    assert ":doc:`primitives`" in others


@pytest.mark.parametrize("stem", ("drtrl", "pp_prop"))
def test_algorithm_tutorial_executes_and_reduces_loss(stem: str, capsys):
    _execute_notebook_code(_DOCS / "tutorials" / f"{stem}.ipynb")
    output = capsys.readouterr().out
    initial = float(re.search(r"initial loss:\s*([0-9.eE+-]+)", output).group(1))
    final = float(re.search(r"final loss:\s*([0-9.eE+-]+)", output).group(1))
    assert final < initial
