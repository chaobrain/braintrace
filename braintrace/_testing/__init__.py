# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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
# ==============================================================================

"""Test-support fixtures. Not shipped, not part of the public API.

Everything here exists to be imported *by tests*: reference models, the BPTT
gradient oracle, and the compiler scenario catalog. None of it is reachable
from ``braintrace.__all__`` or from any shipped code path, and the package is
excluded from the built wheel (see ``[tool.setuptools.packages.find]`` in
``pyproject.toml``).

Keeping these modules here rather than beside the code they exercise buys two
things:

1. **No shipped module imports a test module.** ``oracle_models`` previously
   imported layer classes from ``braintrace/_etrace_model_test.py`` -- shipped
   code reaching into a pytest-collected file. That file is now
   :mod:`braintrace._testing.models` and the whole cluster sits on one side of
   the ship/no-ship line.
2. **The wheel can exclude them by package name.** ``packages.find`` excludes
   packages, not loose modules, so a fixture living next to shipped code cannot
   be pruned; one living in its own package can.

Modules
-------
models
    Reference layer/model classes (LIF, ALIF, dense/conv variants) shared by
    the algorithm and compiler test suites.
oracle
    BPTT gradient oracles that online algorithms are compared against,
    including the finite-window ``chunked_online_param_gradients`` path.
oracle_models
    Model factories wired for the oracle comparisons.
compiler_models
    Small models exercising specific compiler paths.
scenario_catalog
    The parameterised scenario matrix the compiler suites enumerate.
"""
