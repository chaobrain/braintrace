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

"""Tests for ``braintrace/_typing.py``.

The module is almost entirely ``TypeAlias`` declarations, which have no runtime
behaviour to test, plus one executable function: :func:`as_size_tuple`. That
function is on the hot path of the public layer API -- every RNN cell in
``braintrace/nn/_rnn.py`` and :class:`~braintrace.nn.LeakyRateReadout` route
their ``in_size``/``out_size`` constructor arguments through it -- so its
normalisation contract, and the exception type behind each rejection, are worth
pinning.

Spec: ``docs/specs/2026-08-07-e06-colocated-tests.md``.
"""

import brainstate
import numpy as np
import pytest

from braintrace import _typing
from braintrace._typing import as_size_tuple


# ===========================================================================
# Normalisation: the happy paths
# ===========================================================================

def test_scalar_int_becomes_one_tuple():
    assert as_size_tuple(3) == (3,)


def test_sequence_is_preserved_in_order():
    assert as_size_tuple((2, 3, 4)) == (2, 3, 4)
    assert as_size_tuple([2, 3, 4]) == (2, 3, 4)


def test_result_is_a_plain_tuple_not_a_list_or_array():
    assert type(as_size_tuple([2, 3])) is tuple


@pytest.mark.parametrize(
    'dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32]
)
def test_numpy_integer_scalar_is_accepted(dtype):
    assert as_size_tuple(dtype(5)) == (5,)


def test_numpy_integer_array_is_accepted():
    assert as_size_tuple(np.array([2, 3])) == (2, 3)


@pytest.mark.parametrize(
    'size',
    [
        3,  # int
        np.int64(3),  # np.integer
        (3, 4),  # Sequence[int]
        (np.int64(3), np.int64(4)),  # Sequence[np.integer]
    ],
)
def test_every_arm_of_the_declared_size_union_is_accepted(size):
    """``Size`` is ``int | np.integer | Sequence[int] | Sequence[np.integer]``.

    Each arm must survive normalisation, otherwise the annotation on the
    callers (``size: Size``) promises more than the function delivers.
    """
    result = as_size_tuple(size)
    assert isinstance(result, tuple)
    assert all(type(s) is int for s in result)


# ===========================================================================
# The element-type contract: exactly ``int``, never ``np.integer``
# ===========================================================================

def test_numpy_scalars_are_downcast_to_builtin_int():
    """The point of the helper is a concrete ``tuple[int, ...]``.

    A ``np.int64`` compares equal to an ``int`` but is a different type, so an
    ``isinstance``-based check would pass on a leaked numpy scalar. Compare the
    type exactly.
    """
    result = as_size_tuple(np.int32(7))
    assert result == (7,)
    assert type(result[0]) is int
    assert not isinstance(result[0], np.integer)


def test_numpy_array_elements_are_downcast_to_builtin_int():
    result = as_size_tuple(np.array([2, 3], dtype=np.int16))
    assert [type(s) for s in result] == [int, int]


def test_bool_is_accepted_as_an_int_subclass():
    """``bool`` is a subclass of ``int``, so it takes the scalar branch.

    Pinned as a fact, not endorsed: nothing rejects it, and it normalises to 1.
    """
    assert as_size_tuple(True) == (1,)
    assert type(as_size_tuple(True)[0]) is int


# ===========================================================================
# Idempotence and the round-trip that motivates the helper
# ===========================================================================

@pytest.mark.parametrize('size', [3, np.int64(3), (2, 3), [2, 3], ()])
def test_idempotent(size):
    """``_rnn.py`` calls the helper on values a size setter already normalised.

    Applying it twice must be indistinguishable from applying it once.
    """
    once = as_size_tuple(size)
    assert as_size_tuple(once) == once


def test_result_round_trips_through_a_brainstate_size_setter():
    """The stated reason the helper exists.

    ``brainstate``'s size getters are typed as the broad ``Size`` union, which
    is not indexable to a type checker. Routing through ``as_size_tuple`` must
    give a value that both assigns cleanly and reads back unchanged.
    """
    module = brainstate.nn.Module()
    module.in_size = as_size_tuple(8)
    module.out_size = as_size_tuple([4, 5])
    assert module.in_size == (8,)
    assert module.out_size == (4, 5)
    assert as_size_tuple(module.out_size)[-1] == 5


def test_trailing_dimension_lookup_works_on_the_result():
    assert as_size_tuple((2, 3, 4))[-1] == 4
    assert as_size_tuple(6)[-1] == 6


# ===========================================================================
# No validation: the helper normalises, it does not police
# ===========================================================================

def test_zero_is_accepted():
    assert as_size_tuple(0) == (0,)
    assert as_size_tuple((0, 3)) == (0, 3)


def test_negative_is_accepted():
    """No sign check. A negative size is nonsense as a shape, but rejecting it
    is the layer's job, not the normaliser's; pinned so that adding a check
    later is a visible, deliberate change."""
    assert as_size_tuple(-1) == (-1,)
    assert as_size_tuple([-1, 2]) == (-1, 2)


@pytest.mark.parametrize('empty', [(), [], np.array([], dtype=np.int32)])
def test_empty_sequence_gives_empty_tuple(empty):
    """An empty size produces an empty tuple, which then makes ``size[-1]``
    raise ``IndexError`` at the call site rather than here."""
    assert as_size_tuple(empty) == ()


def test_empty_result_makes_trailing_lookup_fail_at_the_call_site():
    with pytest.raises(IndexError):
        _ = as_size_tuple(())[-1]


# ===========================================================================
# Rejections, pinned by exception type
# ===========================================================================

@pytest.mark.parametrize('size', [3.0, 2.7, None, object()])
def test_non_integer_scalar_raises_type_error(size):
    """A bare ``float`` (integral or not) and ``None`` fall through the scalar
    branch into ``tuple(int(s) for s in size)`` and fail as non-iterables.

    Note in particular that ``3.0`` is *not* accepted despite being integral --
    the scalar branch tests ``isinstance(..., (int, np.integer))`` only.
    """
    with pytest.raises(TypeError):
        as_size_tuple(size)


def test_zero_dimensional_numpy_array_raises_type_error():
    """Divergence from ``brainstate.nn._module._format_size_arg``, pinned.

    ``brainstate``'s own normaliser accepts a 0-d integer array
    (``np.array(3)`` -> ``(3,)``); ``as_size_tuple`` does not, because a 0-d
    array is neither ``int`` nor ``np.integer`` and iterating it raises. A 0-d
    array is not a member of the declared ``Size`` union, and the ``brainstate``
    layers that consume such a size fail the same way, so the divergence is
    documented rather than fixed.
    """
    with pytest.raises(TypeError):
        as_size_tuple(np.array(3))


def test_nested_sequence_raises_type_error():
    with pytest.raises(TypeError):
        as_size_tuple(((1, 2), (3, 4)))


def test_multidimensional_array_raises_type_error():
    with pytest.raises(TypeError):
        as_size_tuple(np.ones((2, 2), dtype=np.int32))


def test_non_numeric_string_raises_value_error():
    """A ``str`` is a ``Sequence``, so it reaches ``int()`` per character and
    fails with ``ValueError`` -- a different exception type from the
    non-iterable rejections above."""
    with pytest.raises(ValueError):
        as_size_tuple('ab')


def test_float_array_raises_nothing_but_truncates():
    """A float array is iterable and each element converts, so it is accepted.

    Pinned alongside the scalar-float rejection to show the asymmetry: the
    element path has no dtype check at all.
    """
    assert as_size_tuple(np.array([2.0, 3.0])) == (2, 3)


# ===========================================================================
# Sharp edges: pinned as facts, not endorsed
# ===========================================================================

@pytest.mark.parametrize(
    'size, expected',
    [
        ((2.7,), (2,)),  # truncates toward zero, not rounds
        ((-2.7,), (-2,)),  # ... in both directions
        ((3.0, 4.0), (3, 4)),  # integral floats pass through silently
    ],
)
def test_float_inside_a_sequence_is_silently_truncated(size, expected):
    """There is no integral-float check on sequence elements: ``int()`` is
    applied unconditionally, so ``2.7`` becomes ``2`` with no error or warning.
    A caller that meant ``round`` gets a different layer width."""
    assert as_size_tuple(size) == expected


def test_numeric_string_is_iterated_character_by_character():
    """``as_size_tuple('12')`` is ``(1, 2)``, not ``(12,)``.

    A string satisfies the ``Sequence`` branch, so a size accidentally passed
    as text produces a plausible-looking multi-dimensional size instead of an
    error. This is the most surprising accepted input; pinned deliberately.
    """
    assert as_size_tuple('12') == (1, 2)
    assert as_size_tuple('7') == (7,)


# ===========================================================================
# Alias identity
# ===========================================================================

@pytest.mark.parametrize('name', ['ArrayLike', 'DType', 'DTypeLike', 'Size'])
def test_brainstate_aliases_are_reexported_not_redeclared(name):
    """These four are re-exports. If one were ever re-declared locally it could
    drift from ``brainstate.typing`` without any import error to flag it."""
    assert getattr(_typing, name) is getattr(brainstate.typing, name)


def test_size_union_is_what_as_size_tuple_documents():
    """Guard the premise of ``test_every_arm_of_the_declared_size_union_is_accepted``:
    if ``brainstate`` widens ``Size``, the union-coverage test above silently
    stops covering the whole union."""
    assert str(_typing.Size) == (
        'typing.Union[int, typing.Sequence[int], numpy.integer, '
        'typing.Sequence[numpy.integer]]'
    )
