"""Structural regression tests for the reader-facing documentation.

These tests complement ``_docs_examples_test.py``.  They guard navigation and
notebook contracts that Sphinx cannot validate because notebook execution is
disabled in ``docs/conf.py``.
"""

from __future__ import annotations

import ast
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest

import braintrace


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


def _conf_assignments() -> dict[str, object]:
    tree = ast.parse((_DOCS / "conf.py").read_text(encoding="utf-8"))
    assignments: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                assignments[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return assignments


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
    quickstart = _all_source(_DOCS / "quickstart" / "quickstart.ipynb")
    fundamentals = _all_source(
        _DOCS / "tutorials" / "five_primitive_functions.ipynb"
    )
    etp = _all_source(_DOCS / "tutorials" / "etp_primitives.ipynb")
    concepts = _all_source(_DOCS / "quickstart" / "concepts.ipynb")
    transforms = _all_source(
        _DOCS / "advanced" / "customizing_primitive_transforms.ipynb"
    )

    assert quickstart.splitlines()[0] == "# Quickstart"
    assert "Five-minute Quickstart" not in homepage
    assert "Quickstart" in homepage
    assert fundamentals.splitlines()[0] == "# ETP Operator Fundamentals"
    assert etp.splitlines()[0] == "# Custom ETP Primitives"
    assert "The Five Primitive Functions" not in homepage
    assert "The Five Primitive Functions" not in concepts
    assert "# ETP Primitives" not in etp
    assert "ETP Primitives Deep Dive" not in etp
    assert "ETP Primitives Deep Dive" not in concepts
    assert "ETP Primitives Deep Dive" not in transforms


def test_root_navigation_exposes_the_learning_path():
    source = (_DOCS / "index.rst").read_text(encoding="utf-8")
    toctrees = source[source.index(".. toctree::") :]

    expected = (
        ":caption: Get started",
        "quickstart/installation.ipynb",
        "quickstart/quickstart.ipynb",
        "quickstart/concepts.ipynb",
        ":caption: Tutorial",
        "quickstart/rnn_online_learning.ipynb",
        "quickstart/snn_online_learning.ipynb",
        "tutorials/drtrl.ipynb",
        "tutorials/pp_prop.ipynb",
        "tutorials/five_primitive_functions.ipynb",
        "tutorials/neural_network_layers.ipynb",
        "tutorials/hidden_states.ipynb",
        "tutorials/graph_compilation.ipynb",
        "tutorials/visualization.ipynb",
        ":caption: Advanced",
        "advanced/batching.ipynb",
        "tutorials/etp_primitives.ipynb",
        "advanced/customizing_primitive_transforms.ipynb",
        "advanced/compiler_internals.ipynb",
        "advanced/custom_algorithms.ipynb",
        "advanced/limitations.ipynb",
        ":caption: Examples",
        "examples/snn_examples.rst",
        "examples/rnn_examples.rst",
        "examples/pp_prop_examples.rst",
        "examples/drtrl_examples.rst",
        ":caption: API Reference",
        "Release Notes <changelog.md>",
        "ETP Operators <apis/concepts.rst>",
        "Compiler and Executor <apis/compiler.rst>",
        "Algorithms <apis/algorithms.rst>",
        "Neural Network Layers <apis/nn.rst>",
        "Others <apis/primitives.rst>",
    )
    positions = [toctrees.index(item) for item in expected]
    assert positions == sorted(positions)
    tutorial_caption = toctrees.index(":caption: Tutorial")
    tutorial_start = toctrees.rfind(".. toctree::", 0, tutorial_caption)
    tutorial = toctrees[tutorial_start : toctrees.index(":caption: Advanced")]
    assert ":maxdepth: 1" in tutorial
    assert "batching.ipynb" not in tutorial
    assert "tutorials/customizing_primitive_transforms.ipynb" not in tutorial
    assert "tutorials/graph_visualization.ipynb" not in tutorial

    api_caption = toctrees.index(":caption: API Reference")
    api_start = toctrees.rfind(".. toctree::", 0, api_caption)
    api_root = toctrees[api_start:]
    assert ":maxdepth: 2" in api_root
    assert api_root.count("apis/") == 5
    assert ":caption: Project" not in source
    assert "API Overview" not in source
    assert not (_DOCS / "apis" / "index.rst").exists()


def test_sphinx_api_rendering_is_strict_and_nonredundant():
    conf = _conf_assignments()

    assert conf["autodoc_typehints"] == "none"
    assert conf["napoleon_use_param"] is False
    assert conf["napoleon_use_rtype"] is False
    assert conf["napoleon_use_ivar"] is True
    assert conf["nitpicky"] is True

    suppress_warnings = conf["suppress_warnings"]
    assert isinstance(suppress_warnings, list)
    assert "ref.ref" not in suppress_warnings

    ignore_regex = conf["nitpick_ignore_regex"]
    assert isinstance(ignore_regex, list)
    assert ignore_regex
    assert all(
        isinstance(entry, tuple)
        and len(entry) == 2
        and entry[0].startswith("py:")
        and entry[1] not in {".*", ".+"}
        for entry in ignore_regex
    )


def test_public_api_has_one_canonical_autosummary_owner():
    entries: list[str] = []
    for path in (_DOCS / "apis").glob("*.rst"):
        source = path.read_text(encoding="utf-8")
        entries.extend(re.findall(r"^   ([A-Za-z_]\w*)$", source, re.MULTILINE))

    counts = Counter(entries)
    aliases = {"ES_D_RTRL"}
    non_objects = {"__version__", "__version_info__", "nn"}
    canonical = set(braintrace.__all__) - aliases - non_objects
    invalid_counts = {
        name: counts[name]
        for name in sorted(canonical)
        if counts[name] != 1
    }

    assert invalid_counts == {}
    assert counts["ES_D_RTRL"] == 0


def test_tutorial_groups_are_pure_titles_not_documents():
    index = (_DOCS / "index.rst").read_text(encoding="utf-8")
    declarations = re.findall(
        r"^\.\. braintrace-tutorial-group: (?P<title>.+)\n"
        r"   :members: (?P<members>.+)$",
        index,
        re.MULTILINE,
    )
    assert declarations == [
        (
            "Online Training",
            "RNN Online Learning | SNN Online Learning",
        ),
        (
            "Algorithm Tutorials",
            "D-RTRL: diagonal online gradient learning | "
            "pp_prop: input/output-factorized online gradients",
        ),
        (
            "Foundations",
            "ETP Operator Fundamentals | braintrace.nn Layers | "
            "Hidden State Management",
        ),
        ("Compiler & Runtime", "Graph Compilation | Visualization"),
    ]
    for stem in (
        "online_training",
        "algorithm_tutorials",
        "foundations",
        "compiler_runtime",
    ):
        assert not (_DOCS / "tutorials" / f"{stem}.rst").exists()

    conf = (_DOCS / "conf.py").read_text(encoding="utf-8")
    assert "html-page-context" in conf
    assert "generate_toctree_html" in conf
    assert "braintrace_group_tutorial_nav" not in conf
    assert not (_DOCS / "_navigation.py").exists()
    assert not (_DOCS / "_templates" / "sbt-sidebar-nav.html").exists()


def test_tutorial_content_is_split_by_reader_level():
    functions = _all_source(
        _DOCS / "tutorials" / "five_primitive_functions.ipynb"
    )
    advanced = _all_source(_DOCS / "tutorials" / "etp_primitives.ipynb")
    layers_path = _DOCS / "tutorials" / "neural_network_layers.ipynb"
    layers = _all_source(layers_path)
    graph = _all_source(_DOCS / "tutorials" / "graph_compilation.ipynb")
    visual = _all_source(_DOCS / "tutorials" / "visualization.ipynb")

    for name in (
        "braintrace.matmul",
        "braintrace.element_wise",
        "braintrace.conv",
        "braintrace.sparse_matmul",
        "braintrace.lora_matmul",
    ):
        assert name in functions
    assert "## Physical Units (`brainunit` / Quantity) Support" in functions
    assert "## JAX Compatibility" in functions
    assert "# Custom ETP Primitives" in advanced
    assert "## Physical Units (`brainunit` / Quantity) Support" not in advanced
    assert "## JAX Compatibility" not in advanced
    assert "## Rule Registries" in advanced
    assert layers_path.exists()
    for required in (
        "# braintrace.nn Layers",
        "braintrace.nn.Linear",
        "braintrace.nn.MiniGRU",
        "braintrace.compile",
        "brainstate.transform.for_loop",
        "operation",
        "non-temporal",
        "weight -> weight -> hidden",
    ):
        assert required in layers
    for source in (functions, advanced, layers):
        assert "jax.random" not in source
    for source in (functions, layers):
        assert not re.search(
            r"^\s*(?:for|while)\s+.+(?:range|model|learner).+:\s*$",
            source,
            re.MULTILINE,
        )
    assert graph.splitlines()[0] == "# Graph Compilation"
    assert "compile_etrace_graph" in graph
    assert visual.splitlines()[0] == "# Visualization"
    assert "ETraceGraph" in visual
    assert not (_DOCS / "tutorials" / "graph_visualization.ipynb").exists()
    assert (
        _DOCS / "advanced" / "customizing_primitive_transforms.ipynb"
    ).exists()
    assert not (
        _DOCS / "tutorials" / "customizing_primitive_transforms.ipynb"
    ).exists()
    assert (_DOCS / "advanced" / "batching.ipynb").exists()
    assert not (_DOCS / "tutorials" / "batching.ipynb").exists()


def test_neural_network_layers_tutorial_executes():
    _execute_notebook_code(
        _DOCS / "tutorials" / "neural_network_layers.ipynb"
    )


def test_neural_network_layers_notebook_has_stable_cell_ids():
    notebook = _notebook(
        _DOCS / "tutorials" / "neural_network_layers.ipynb"
    )
    cell_ids = [cell.get("id") for cell in notebook["cells"]]

    assert all(
        isinstance(cell_id, str)
        and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id)
        for cell_id in cell_ids
    )
    assert len(cell_ids) == len(set(cell_ids))


def test_reader_notebooks_do_not_embed_runtime_warning_outputs():
    for path in (
        _DOCS / "tutorials" / "five_primitive_functions.ipynb",
        _DOCS / "tutorials" / "etp_primitives.ipynb",
        _DOCS / "tutorials" / "neural_network_layers.ipynb",
        _DOCS / "advanced" / "batching.ipynb",
    ):
        notebook = _notebook(path)
        output_text = "\n".join(
            "".join(output.get("text", []))
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
            if output.get("output_type") == "stream"
        )
        assert "UserWarning" not in output_text
        assert "/home/" not in output_text


def test_homepage_restores_historical_editorial_layout():
    source = (_DOCS / "index.rst").read_text(encoding="utf-8")

    assert "Basic usage" not in source
    assert ".. tab-set::" in source
    for tab in ("CPU", "NVIDIA GPU", "TPU", "Development"):
        assert f".. tab-item:: {tab}" in source
    assert "Learn more" in source
    assert "Learning path" in source
    assert ".. image:: _static/braintrace-learning-map.svg" in source
    assert ".. figure:: _static/braintrace-learning-map.svg" not in source
    caption = (
        ".. container:: text-center\n\n"
        "   *A dependency guide, not a required linear syllabus.*"
    )
    assert caption in source
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
        "apis/concepts",
    ):
        assert target in source
    assert "apis/index" not in source
    assert "Training workflows" not in source
    assert "Online Training" in source
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


def test_drtrl_relies_on_one_jupyter_rendered_plot():
    notebook = _notebook(_DOCS / "tutorials" / "drtrl.ipynb")
    plot_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "D-RTRL mini-GRU training loss" in "".join(cell["source"])
    )
    source = "".join(plot_cell["source"]).rstrip()
    image_outputs = [
        output
        for output in plot_cell["outputs"]
        if "image/png" in output.get("data", {})
    ]

    assert not source.endswith("\nfig")
    assert "savefig(" not in source
    assert len(image_outputs) == 1
    assert image_outputs[0]["output_type"] == "display_data"


def test_docs_use_native_theme_navigation_without_custom_assets():
    conf = (_DOCS / "conf.py").read_text(encoding="utf-8")
    gitignore = (_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "html_static_path" in conf
    assert "html_css_files" not in conf
    assert "html_js_files" not in conf
    assert "'brainx_sphinx_header'" in conf
    assert not (_DOCS / "_static" / "css" / "braintrace-docs.css").exists()
    assert not (_DOCS / "_static" / "js" / "tutorial-groups.js").exists()
    for path in (
        "docs/_static/css/brainx-header.css",
        "docs/_static/js/brainx-header.js",
        "docs/_static/css/brainx-footer.css",
        "docs/_static/js/brainx-footer.js",
    ):
        assert path in gitignore


def test_sphinx_uses_current_notebook_options_and_excludes_internal_specs():
    conf = (_DOCS / "conf.py").read_text(encoding="utf-8")
    requirements = (_ROOT / "requirements-doc.txt").read_text(
        encoding="utf-8"
    )

    assert 'nb_execution_mode = "off"' in conf
    assert "nb_execution_timeout = 200" in conf
    assert "brainx_inject_base = False" in conf
    assert "jupyter_execute_notebooks" not in conf
    assert re.search(r"exclude_patterns\s*=.*['\"]specs['\"]", conf)
    assert "sphinx_autodoc_typehints" not in conf
    assert "sphinx-autodoc-typehints" not in requirements
    assert "napoleon_use_ivar = True" in conf


def test_quickstart_has_current_reproducible_online_learning_primitives():
    notebook = _DOCS / "quickstart" / "quickstart.ipynb"
    source = _all_source(notebook)

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
    assert "fig.savefig" not in source
    assert not re.search(r"^\s*for\s+\w+\s+in\s+range\(", source, re.MULTILINE)
    assert notebook.exists()
    assert not (_DOCS / "quickstart" / "quickstart.rst").exists()
    assert not (_DOCS / "_static" / "quickstart_loss.png").exists()


def test_quickstart_executes_and_reduces_loss(capsys):
    _execute_notebook_code(_DOCS / "quickstart" / "quickstart.ipynb")
    output = capsys.readouterr().out
    initial = float(re.search(r"initial loss:\s*([0-9.eE+-]+)", output).group(1))
    final = float(re.search(r"final loss:\s*([0-9.eE+-]+)", output).group(1))
    assert final < initial


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


def test_documentation_sources_do_not_emit_known_project_warnings():
    import braintrace

    primitives = (_DOCS / "apis" / "primitives.rst").read_text(
        encoding="utf-8"
    )
    lines = primitives.splitlines()
    registration = lines.index("Registration")
    assert len(lines[registration + 1]) >= len(lines[registration])

    conv3d_doc = braintrace.nn.Conv3d.__doc__ or ""
    assert (
        "- A tuple of three integers: (stride_h, stride_w, stride_d)\n\n"
        "        Default: 1."
        in conv3d_doc
    )

    for name in (
        "D_RTRL",
        "EProp",
        "IODimVjpAlgorithm",
        "OSTLFeedforward",
        "OSTLRecurrent",
        "ParamDimVjpAlgorithm",
        "SnAp",
        "pp_prop",
    ):
        docstring = getattr(braintrace, name).__doc__ or ""
        narrative, separator, references = docstring.partition(
            "References\n"
        )
        assert separator, f"{name} has no References section"
        labels = re.findall(r"^\s*\.\. \[(\d+)\]", references, re.MULTILINE)
        assert labels, f"{name} has no reference entries"
        for label in labels:
            assert f"[{label}]_" in narrative, (
                f"{name} does not cite reference [{label}]"
            )

    concepts = (_DOCS / "apis" / "concepts.rst").read_text(encoding="utf-8")
    algorithms = (_DOCS / "apis" / "algorithms.rst").read_text(
        encoding="utf-8"
    )
    assert "\n   EligibilityTrace\n" not in concepts
    assert "\n   EligibilityTrace\n" in algorithms


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


@pytest.mark.parametrize(
    ("filename", "script_root", "scripts"),
    (
        (
            "snn_examples.rst",
            "examples",
            (
                "000-lif-snn-for-nmnist.py",
                "001-gif-snn-for-dms.py",
                "002-coba-ei-rsnn.py",
                "003-snn-memory-and-speed-evaluation-all.py",
                "003-snn-memory-and-speed-evaluation-batched.py",
                "003-snn-memory-and-speed-evaluation-vmap.py",
                "004-feedforward-conv-snn.py",
            ),
        ),
        (
            "rnn_examples.rst",
            "examples",
            ("100-gru-on-copying-task.py", "101-integrator-rnn.py"),
        ),
        (
            "pp_prop_examples.rst",
            "examples/pp_prop",
            tuple(
                path.name
                for path in sorted((_ROOT / "examples" / "pp_prop").glob("[0-9]*.py"))
            ),
        ),
        (
            "drtrl_examples.rst",
            "examples/drtrl",
            tuple(
                path.name
                for path in sorted((_ROOT / "examples" / "drtrl").glob("[0-9]*.py"))
            ),
        ),
    ),
)
def test_every_documented_example_has_source_and_api_links(
    filename: str, script_root: str, scripts: tuple[str, ...]
):
    source = (_DOCS / "examples" / filename).read_text(encoding="utf-8")

    assert not re.search(r"``[^`\n]*\.py``", source)
    assert "*." not in source
    for script in scripts:
        local = _ROOT / script_root / script
        url = f"https://github.com/chaobrain/braintrace/blob/main/{script_root}/{script}"
        assert local.exists()
        assert f"`{script} <{url}>`__" in source

        bullet = next(
            block
            for block in re.split(r"\n(?=\* )", source)
            if f"`{script} <{url}>`__" in block
        )
        assert "API:" in bullet
        assert ":doc:" in bullet


def test_flattened_examples_page_is_removed():
    assert not (_DOCS / "examples" / "core_examples.rst").exists()


def test_api_reference_is_direct_and_keeps_primitive_registration_in_others():
    source = (_DOCS / "index.rst").read_text(encoding="utf-8")
    api = source[source.index(":caption: API Reference") :]
    entries = (
        "Release Notes <changelog.md>",
        "ETP Operators <apis/concepts.rst>",
        "Compiler and Executor <apis/compiler.rst>",
        "Algorithms <apis/algorithms.rst>",
        "Neural Network Layers <apis/nn.rst>",
        "Others <apis/primitives.rst>",
    )

    assert [api.index(entry) for entry in entries] == sorted(
        api.index(entry) for entry in entries
    )
    assert api.count("apis/primitives.rst") == 1
    assert "API Overview" not in api
    assert ":caption: Project" not in source


def test_documentation_links_follow_the_new_information_architecture():
    homepage = (_DOCS / "index.rst").read_text(encoding="utf-8")
    concepts = _all_source(_DOCS / "quickstart" / "concepts.ipynb")
    fundamentals = _all_source(
        _DOCS / "tutorials" / "five_primitive_functions.ipynb"
    )

    assert "tutorials/batching.ipynb" not in homepage
    assert "../tutorials/batching.ipynb" not in concepts
    assert "../advanced/batching.ipynb" in concepts
    assert "Custom ETP Primitives" in concepts
    assert "[Custom ETP Primitives](etp_primitives.ipynb)" in fundamentals


@pytest.mark.parametrize("stem", ("drtrl", "pp_prop"))
def test_algorithm_tutorial_executes_and_reduces_loss(stem: str, capsys):
    _execute_notebook_code(_DOCS / "tutorials" / f"{stem}.ipynb")
    output = capsys.readouterr().out
    initial = float(re.search(r"initial loss:\s*([0-9.eE+-]+)", output).group(1))
    final = float(re.search(r"final loss:\s*([0-9.eE+-]+)", output).group(1))
    assert final < initial
