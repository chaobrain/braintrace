# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Chaoming Wang <chao.brain@qq.com>
# Date: 2024-04-03
# Copyright: 2024, Chaoming Wang
# ==============================================================================

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# a_list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

import braintrace
import shutil

shutil.copy('../changelog.md', './changelog.md')

# -- Project information -----------------------------------------------------

project = 'BrainTrace'
copyright = '2024, BrainTrace'
author = 'BrainTrace Developer'

# The full version, including alpha/beta/rc tags
release = braintrace.__version__

from highlight_lexer import fix_ipython2_lexer_in_notebooks

fix_ipython2_lexer_in_notebooks(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_nb',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_thebe',
    'sphinx_design',
    'sphinx_math_dollar',
    'brainx_sphinx_header',
]


html_baseurl = 'https://brainx.chaobrain.com/braintrace/'
# Keep relative documentation assets local when previewing a built index page.
# Production already serves the canonical path with a trailing slash.
brainx_inject_base = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']

mathjax3_config = {
    "tex": {
        "inlineMath": [['\\(', '\\)']],
        "displayMath": [["\\[", "\\]"]],
    }
}

# source_suffix = '.rst'
autosummary_generate = True

# The master toctree document.
master_doc = 'index'

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.13", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}
nitpicky = True
nitpick_ignore = [
    ("py:class", "docutils.nodes.document"),
    ("py:class", "docutils.parsers.rst.directives.body.Sidebar"),
]
nitpick_ignore_regex = [
    # Ecosystem projects do not publish inventories as build dependencies.
    ("py:class", r"brainevent\.(?:CSR|DataRepresentation)"),
    (
        "py:class",
        (
            r"brainstate\.(?:HiddenState|LongTermState|ParamState|"
            r"ShortTermState|State|mixin\.Mode|nn\.(?:LoRA|Module)|"
            r"transform\.StatefulFunction|util\.FlattedDict)"
        ),
    ),
    ("py:class", r"brainunit\.(?:Quantity|sparse\.SparseMatrix)"),
    ("py:class", r"jax\.Array"),
    # Private implementation annotations expose these aliases, but BrainTrace
    # does not own public API pages for them.
    (
        "py:class",
        (
            r"(?:ArrayLike|ClosedJaxpr|ControlFlowPolicy|GroupDescent|"
            r"HiddenInVar|HiddenOutVar|HiddenState|Jaxpr|ParamState|Path|"
            r"Primitive|PyTree|RelationDescent|SnapPattern|Var|sequence)"
        ),
    ),
]

suppress_warnings = ["myst.domains"]

numfig = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'specs', 'Thumbs.db', '.DS_Store']

html_theme = "sphinx_book_theme"
html_logo = "https://brainx.chaobrain.com/images/braintrace.webp"
html_title = "BrainTrace"
html_copy_source = True
html_sourcelink_suffix = ""
html_favicon = "https://brainx.chaobrain.com/images/braintrace.webp"
html_last_updated_fmt = ""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
nb_execution_mode = "off"
thebe_config = {
    "repository_url": "https://github.com/binder-examples/jupyter-stacks-datascience",
    "repository_branch": "master",
}

html_theme_options = {
    'show_toc_level': 2,
}

# -- Options for myst ----------------------------------------------
# Notebook cell execution timeout; defaults to 30.
nb_execution_timeout = 200

autodoc_default_options = {
    'exclude-members': '....,default_rng',
}

# NumPy docstrings are the canonical owner of API type presentation.
autodoc_typehints = "none"
napoleon_use_param = False
napoleon_use_rtype = False

# Keep NumPy-style ``Attributes`` sections as field-list documentation.
# ``autoclass :members:`` remains the single owner of member targets.
napoleon_use_ivar = True
