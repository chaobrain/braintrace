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

"""Version string and install-metadata invariants.

The metadata assertions here read the repository's ``pyproject.toml`` /
``requirements.txt`` rather than ``importlib.metadata``: a source checkout
routinely shadows an older installed distribution, so the installed
``dist-info`` can describe a different release than the code under test.

Both files ship in the sdist -- which is what downstream packagers build and run
this suite from -- but not in the wheel, so each test skips when its file is
absent. ``.github/`` is in neither artifact, so the CI cross-check skips outside
a repo checkout.

Rationale for the JAX floor: ``docs/specs/2026-08-07-e05-declare-jax-dependency.md``.
"""

import re
import tomllib
from pathlib import Path
from typing import Optional, Tuple

import pytest

from braintrace._version import __version__, __version_info__

#: Repository root as seen from an sdist/checkout layout (``<root>/braintrace``).
REPO_ROOT = Path(__file__).resolve().parents[1]

PYPROJECT = REPO_ROOT / 'pyproject.toml'
REQUIREMENTS = REPO_ROOT / 'requirements.txt'
CI_WORKFLOW = REPO_ROOT / '.github' / 'workflows' / 'CI.yml'

#: Backend-selector extras that must keep requesting an accelerator-flavoured
#: ``jax``. Adding a bare ``jax`` to the core dependency list must not degrade
#: these into plain installs.
BACKEND_EXTRAS = ('cpu', 'cuda12', 'cuda13', 'tpu')


def _load_pyproject() -> dict:
    if not PYPROJECT.is_file():
        pytest.skip('pyproject.toml not available (wheel install)')
    return tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))


def _jax_requirement(dependencies) -> Optional[str]:
    """Return the requirement string naming the ``jax`` project, if any.

    Matches the distribution name at the start of the requirement so that
    ``jaxlib`` or ``jax-cuda12-plugin`` are not mistaken for it.
    """
    for dep in dependencies:
        if re.match(r'^\s*jax\s*(\[|[<>=!~]|$)', dep):
            return dep
    return None


def _brainevent_requirement(dependencies, extra: Optional[str] = None) -> Optional[str]:
    """Return the requirement string naming BrainEvent and an optional extra."""
    suffix = rf'\[{re.escape(extra)}\]' if extra is not None else r'(?:\[[^\]]+\])?'
    pattern = rf'^\s*brainevent\s*{suffix}\s*([<>=!~]|$)'
    for dep in dependencies:
        if re.match(pattern, dep):
            return dep
    return None


def _floor(requirement: str) -> Tuple[int, ...]:
    """Extract the ``>=`` floor of a requirement as a comparable tuple."""
    match = re.search(r'>=\s*([0-9]+(?:\.[0-9]+)*)', requirement)
    assert match is not None, f'no >= floor in {requirement!r}'
    return tuple(int(part) for part in match.group(1).split('.'))


class TestVersionString:
    def test_version_info_matches_version(self):
        assert __version_info__ == tuple(int(p) for p in __version__.split('.'))

    def test_version_is_dotted_numeric(self):
        # `[tool.setuptools.dynamic] version` reads this attribute verbatim, and
        # `__version_info__` int-parses every component, so a suffix such as
        # "0.3.0rc1" would raise at import time rather than at build time.
        assert re.fullmatch(r'[0-9]+(\.[0-9]+)*', __version__), __version__

    def test_pyproject_reads_this_module_for_its_version(self):
        dynamic = _load_pyproject()['tool']['setuptools']['dynamic']
        assert dynamic['version'] == {'attr': 'braintrace._version.__version__'}


class TestJaxDependencyDeclaration:
    """`jax` is imported directly by the package, so it must be declared (E-05).

    Before this was declared, the floor was borrowed from ``brainevent``'s
    metadata and could move without a braintrace commit.
    """

    def test_jax_is_a_core_dependency_with_a_floor(self):
        deps = _load_pyproject()['project']['dependencies']
        requirement = _jax_requirement(deps)
        assert requirement is not None, f'jax missing from core dependencies: {deps}'
        assert _floor(requirement) >= (0, 8, 0)

    def test_declared_floor_matches_the_lowest_tested_jax(self):
        """Metadata must not promise a version the CI matrix never runs.

        This is E-05 stated as an assertion: lowering the matrix without
        lowering the floor (or raising the floor without extending the matrix)
        means the published range and the tested range have diverged.
        """
        if not CI_WORKFLOW.is_file():
            pytest.skip('CI workflow not available outside a repo checkout')
        text = CI_WORKFLOW.read_text(encoding='utf-8')
        match = re.search(r'jax-version:\s*\[([^\]]*)\]', text)
        assert match is not None, 'no jax-version matrix found in CI.yml'
        # `""` means "latest" and carries no lower bound; non-numeric entries
        # (a pre-release, say) are not comparable as dotted tuples.
        versions = [
            tuple(int(p) for p in raw.split('.'))
            for raw in re.findall(r'"([^"]*)"', match.group(1))
            if re.fullmatch(r'[0-9]+(\.[0-9]+)*', raw)
        ]
        assert versions, 'CI matrix pins no explicit JAX version'

        deps = _load_pyproject()['project']['dependencies']
        assert _floor(_jax_requirement(deps)) == min(versions)

    def test_no_upper_cap_on_jax(self):
        # A cap published today constrains JAX releases that do not exist yet,
        # for every braintrace artifact already on PyPI. Breakage is answered
        # with a targeted exclusion plus a fix instead.
        requirement = _jax_requirement(_load_pyproject()['project']['dependencies'])
        assert '<' not in requirement, requirement

    def test_requirements_txt_states_the_same_floor(self):
        if not REQUIREMENTS.is_file():
            pytest.skip('requirements.txt not available (wheel install)')
        lines = [
            line.split('#')[0].strip()
            for line in REQUIREMENTS.read_text(encoding='utf-8').splitlines()
        ]
        requirement = _jax_requirement([line for line in lines if line])
        assert requirement is not None, 'jax missing from requirements.txt'

        deps = _load_pyproject()['project']['dependencies']
        assert _floor(requirement) == _floor(_jax_requirement(deps))


class TestBackendExtrasSurviveTheCoreDependency:
    """`braintrace[cuda12]` must still resolve to a CUDA-capable JAX.

    Extras of the same distribution are additive, so the core `jax>=0.8.0`
    intersects with `jax[cuda12]` rather than replacing it -- but only for as
    long as the extras keep naming a backend.
    """

    @pytest.mark.parametrize('extra', BACKEND_EXTRAS)
    def test_extra_requests_the_backend_flavoured_jax(self, extra):
        optional = _load_pyproject()['project']['optional-dependencies']
        assert f'jax[{extra}]' in optional[extra]

    @pytest.mark.parametrize('extra', BACKEND_EXTRAS)
    def test_extra_requests_the_backend_flavoured_brainevent(self, extra):
        metadata = _load_pyproject()['project']
        core = _brainevent_requirement(metadata['dependencies'])
        backend = _brainevent_requirement(metadata['optional-dependencies'][extra], extra)
        assert core is not None
        assert backend is not None
        assert _floor(backend) == _floor(core)

    @pytest.mark.parametrize('extra', ('testing', 'dev'))
    def test_cpu_test_extras_install_brainevent_cpu_backend(self, extra):
        optional = _load_pyproject()['project']['optional-dependencies']
        assert _brainevent_requirement(optional[extra], 'cpu') is not None

    def test_extras_do_not_restate_the_floor(self):
        # The floor lives in exactly one place so it cannot drift; an extra that
        # grew its own `>=` would be a second source of truth.
        optional = _load_pyproject()['project']['optional-dependencies']
        for extra in BACKEND_EXTRAS:
            requirement = _jax_requirement(optional[extra])
            assert requirement is not None
            assert '>=' not in requirement, (extra, requirement)
