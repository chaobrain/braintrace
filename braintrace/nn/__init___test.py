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

"""Tests for the ``braintrace.nn`` package root (``braintrace/nn/__init__.py``).

The module is a deprecation dispatcher: 8 names forward to :mod:`brainpy.state`
and 40 to :mod:`brainstate.nn`, each with a :class:`DeprecationWarning` naming
the replacement, and anything else raises :class:`AttributeError`.

That final ``raise`` is the contract this file exists for. A module-level
``__getattr__`` that falls off the end returns ``None``, so without it
``braintrace.nn.Typo`` would silently *be* ``None`` and blow up much later,
somewhere unrelated. ``braintrace/__init___test.py`` covers ``__dir__`` and two
sample forwards from the package-root side; the dispatcher's own contract --
every forwarded name, the warning text, the non-memoisation, and the
fallthrough -- is pinned here.

Spec: ``docs/specs/2026-08-07-e06-colocated-tests.md``.
"""

import warnings

import brainpy.state
import brainstate
import pytest

import braintrace.nn as nn


# ===========================================================================
# Forwarding: every deprecated name resolves, warns, and lands on the right object
# ===========================================================================

@pytest.mark.parametrize('name', nn._DEPRECATED_TO_BRAINPY_STATE)
def test_brainpy_state_forward_warns_and_resolves(name):
    with pytest.warns(DeprecationWarning) as record:
        obj = getattr(nn, name)
    assert obj is getattr(brainpy.state, name)
    assert f'Use brainpy.state.{name} instead.' in str(record[0].message)
    assert f'braintrace.nn.{name} is deprecated' in str(record[0].message)


@pytest.mark.parametrize('name', nn._DEPRECATED_TO_BRAINSTATE_NN)
def test_brainstate_nn_forward_warns_and_resolves(name):
    with pytest.warns(DeprecationWarning) as record:
        obj = getattr(nn, name)
    assert obj is getattr(brainstate.nn, name)
    assert f'Use brainstate.nn.{name} instead.' in str(record[0].message)
    assert f'braintrace.nn.{name} is deprecated' in str(record[0].message)


def test_marker_sentinel_the_deprecated_lists_are_not_empty():
    # Guard against an emptied tuple making every parametrised test above
    # vacuous (zero cases collected still counts as a green run).
    assert len(nn._DEPRECATED_TO_BRAINPY_STATE) >= 8
    assert len(nn._DEPRECATED_TO_BRAINSTATE_NN) >= 40


def test_warning_is_attributed_to_the_caller_not_to_the_dispatcher():
    """``stacklevel=2``. Without it the warning points at
    ``braintrace/nn/__init__.py``, which tells the user nothing about which of
    their lines to change."""
    with pytest.warns(DeprecationWarning) as record:
        _ = nn.ReLU
    assert record[0].filename == __file__


def test_forward_is_not_memoised():
    """A second access must warn again.

    ``__getattr__`` deliberately does not write the resolved object into the
    module ``__dict__``: if it did, Python's normal attribute lookup would find
    it from then on and every later caller -- in a different file, possibly a
    different package -- would get no deprecation notice at all.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _ = nn.ReLU
        _ = nn.ReLU
    assert len(caught) == 2
    assert 'ReLU' not in vars(nn)


@pytest.mark.parametrize('name', nn.__all__)
def test_real_exports_never_reach_the_dispatcher(name):
    """``__all__`` names are bound at import time, so normal attribute lookup
    finds them and ``__getattr__`` is never consulted. They must therefore
    resolve without any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        obj = getattr(nn, name)
    assert obj is not None


# ===========================================================================
# The dispatch table itself
# ===========================================================================

def test_the_two_target_lists_are_disjoint():
    """A name in both tuples would resolve via whichever branch runs first, and
    the second entry would be unreachable -- and misleading, since it names a
    different replacement package."""
    overlap = set(nn._DEPRECATED_TO_BRAINPY_STATE) & set(nn._DEPRECATED_TO_BRAINSTATE_NN)
    assert not overlap, f'name listed under two replacement packages: {sorted(overlap)}'


def test_deprecated_names_do_not_shadow_real_exports():
    """``__getattr__`` only runs when normal lookup fails, so a deprecated entry
    that collides with an ``__all__`` name is dead code: the real class always
    wins and no warning is ever emitted for it."""
    deprecated = set(nn._DEPRECATED_TO_BRAINPY_STATE) | set(nn._DEPRECATED_TO_BRAINSTATE_NN)
    collisions = deprecated & set(nn.__all__)
    assert not collisions, f'deprecated entries shadowed by real exports: {sorted(collisions)}'


@pytest.mark.parametrize(
    'names', [nn._DEPRECATED_TO_BRAINPY_STATE, nn._DEPRECATED_TO_BRAINSTATE_NN]
)
def test_no_duplicates_within_a_list(names):
    assert len(set(names)) == len(names)


# ===========================================================================
# __dir__
# ===========================================================================

def test_dir_is_exactly_all_plus_both_deprecated_lists():
    expected = (
        set(nn.__all__)
        | set(nn._DEPRECATED_TO_BRAINPY_STATE)
        | set(nn._DEPRECATED_TO_BRAINSTATE_NN)
    )
    assert set(dir(nn)) == expected


def test_dir_is_sorted_and_duplicate_free():
    names = dir(nn)
    assert names == sorted(names)
    assert len(set(names)) == len(names)


def test_every_advertised_name_actually_resolves():
    """``__dir__`` drives tab-completion and ``dir()``-based tooling. Every name
    it advertises must be gettable, or completion offers names that raise."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        unresolvable = [name for name in dir(nn) if not hasattr(nn, name)]
    assert not unresolvable, f'advertised by __dir__ but not gettable: {unresolvable}'


# ===========================================================================
# The fallthrough (``__init__.py`` L119) -- the reason this file exists
# ===========================================================================

def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = nn.ThisNameDoesNotExist


def test_unknown_name_does_not_silently_return_none():
    """The regression the explicit ``raise`` prevents.

    Delete the ``raise`` and ``__getattr__`` returns ``None`` implicitly; this
    assertion is what tells the two apart, since a bare
    ``pytest.raises(AttributeError)`` on a missing name would also pass against
    a module that never defined ``__getattr__`` at all.
    """
    sentinel = object()
    assert getattr(nn, 'ThisNameDoesNotExist', sentinel) is sentinel


def test_fallthrough_message_names_the_module_and_the_attribute():
    with pytest.raises(AttributeError) as excinfo:
        _ = nn.ThisNameDoesNotExist
    message = str(excinfo.value)
    assert 'braintrace.nn' in message
    assert 'ThisNameDoesNotExist' in message


def test_fallthrough_does_not_warn():
    """An unknown name is a typo, not a deprecation. Emitting a
    ``DeprecationWarning`` on the way to the ``AttributeError`` would be
    actively misleading."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(AttributeError):
            _ = nn.ThisNameDoesNotExist
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_hasattr_is_false_for_an_unknown_name():
    """``hasattr`` is ``True`` for anything that returns, including ``None``;
    it only reports ``False`` because the fallthrough raises."""
    assert not hasattr(nn, 'ThisNameDoesNotExist')


def test_from_import_of_an_unknown_name_raises_import_error():
    """``from ... import`` converts the dispatcher's ``AttributeError`` into an
    ``ImportError``. Without the raise, the name would import as ``None``."""
    with pytest.raises(ImportError):
        from braintrace.nn import ThisNameDoesNotExist  # noqa: F401


@pytest.mark.parametrize(
    'name',
    [
        '__wrapped__',  # probed by inspect.signature / functools
        '__deepcopy__',  # probed by copy.deepcopy
        '__setstate__',  # probed by pickle (``object`` provides no default)
        '_ipython_canary_method_should_not_exist_',  # probed by IPython completion
    ],
)
def test_dunder_and_private_probes_raise(name):
    """Protocol probes must fail, not resolve.

    ``inspect``, ``copy``, ``pickle`` and REPL completion all test for optional
    hooks with ``getattr``. A dispatcher that returned ``None`` for them would
    hand each of those a non-callable hook, turning a clean "not supported" into
    a ``TypeError`` from inside the stdlib.
    """
    with pytest.raises(AttributeError):
        getattr(nn, name)


@pytest.mark.parametrize('name', ['Sequential', 'Module'])
def test_unlisted_brainstate_names_are_not_forwarded(name):
    """The dispatcher is an allowlist, not a blanket ``brainstate.nn`` proxy.

    ``Sequential`` and ``Module`` both exist in ``brainstate.nn`` but are not in
    either deprecated tuple, so they reach the fallthrough. This is what keeps
    ``braintrace.nn``'s surface a reviewed set rather than whatever
    ``brainstate.nn`` happens to export today.
    """
    assert hasattr(brainstate.nn, name)
    with pytest.raises(AttributeError):
        getattr(nn, name)
