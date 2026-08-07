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

"""Build shim that keeps the test payload out of the **wheel** but in the sdist.

All packaging metadata lives in ``pyproject.toml``; this file exists for one
reason only.

This project co-locates tests: ``foo.py`` is tested by a sibling ``foo_test.py``
(AGENTS.md rule 9), and the shared fixtures live in ``braintrace._testing`` and
in ``tests`` subpackages. ``[tool.setuptools.packages.find] exclude`` matches
*package* names, so it cannot see a loose module -- which meant every one of the
76 ``*_test.py`` files was copied into the wheel. Measured before this shim:
2.90 MB of uncompressed wheel content, 1.63 MB (56%) of it test code that no
installed user can run without pytest, hypothesis, and the fixture packages.

Two artifacts, two answers
--------------------------
The wheel is what users install; it has no business carrying tests. The sdist is
what downstream packagers (conda-forge, distro packaging) build **and run the
suite from**, so pruning it would take away their only way to verify a build.

Excluding at discovery time -- in ``pyproject.toml`` -- cannot make that
distinction: it happens before either artifact exists, so it hits both. The
filtering therefore lives here, in ``build_py``, which is the step that copies
modules into the build tree:

- ``find_package_modules`` drops co-located ``*_test.py`` modules and returns
  nothing at all for a test-only package, so neither reaches the build tree.
- ``get_source_files`` -- the list ``sdist`` builds its manifest from -- turns
  that filtering off, so the sdist stays complete.

This split only holds with ``include-package-data = false`` (set in
``pyproject.toml``). Left on, setuptools re-adds every file in the sdist
manifest to the wheel as *package data*, which put the whole test payload back:
measured, 17 ``*/tests/*`` modules and all 9 ``braintrace/_testing/`` modules.
"""

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

#: Suffix identifying a co-located test module. Matches the convention in
#: AGENTS.md rule 9 (``foo.py`` -> ``foo_test.py``); the ``test_*`` prefix form
#: is deliberately not matched, since this package does not use it.
TEST_MODULE_SUFFIX = '_test'

#: Fully-qualified name of the test-support fixture package (reference models,
#: the BPTT oracle, the compiler scenario catalog). Imported only by tests.
TEST_SUPPORT_PACKAGE = 'braintrace._testing'

#: Name of the test-only subpackages under ``braintrace``.
TEST_SUBPACKAGE = 'tests'


def is_test_package(package: str) -> bool:
    """Return whether ``package`` exists only to serve the test suite.

    Parameters
    ----------
    package : str
        A fully-qualified package name, e.g. ``'braintrace._algorithm.tests'``.

    Returns
    -------
    bool
        ``True`` for the test-support fixture package, for any subpackage of it,
        and for any ``tests`` subpackage; ``False`` otherwise. The top-level
        ``braintrace`` package is never matched.

    Examples
    --------
    .. code-block:: python

        >>> is_test_package('braintrace._testing')
        True
        >>> is_test_package('braintrace._algorithm.tests')
        True
        >>> is_test_package('braintrace._algorithm')
        False
    """
    if package == TEST_SUPPORT_PACKAGE or package.startswith(TEST_SUPPORT_PACKAGE + '.'):
        return True
    return TEST_SUBPACKAGE in package.split('.')[1:]


class build_py(_build_py):
    """``build_py`` that drops the test payload from the wheel only.

    ``find_package_modules`` is the hook setuptools uses to enumerate a
    package's modules before copying them, so filtering there means the files
    never reach the build tree and no post-hoc pruning is needed. Returning an
    empty list for a test-only package leaves its directory uncreated.

    ``sdist`` does not go through that hook to build its manifest -- it calls
    ``get_source_files`` -- so the filtering is switched off for the duration of
    that call and the source distribution keeps everything.
    """

    #: When ``False``, ``find_package_modules`` filters nothing. Flipped for the
    #: duration of ``get_source_files`` so the sdist keeps its test payload.
    _prune = True

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if not self._prune:
            return modules
        if is_test_package(package):
            return []
        return [
            (pkg, mod, path)
            for (pkg, mod, path) in modules
            if not mod.endswith(TEST_MODULE_SUFFIX)
        ]

    def get_source_files(self):
        # ``sdist`` builds its manifest from this list. Answer it unpruned.
        self._prune = False
        try:
            return super().get_source_files()
        finally:
            self._prune = True


setup(cmdclass={'build_py': build_py})
