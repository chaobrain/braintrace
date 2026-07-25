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

"""Tests for the SnAp-n position-adjacency analysis.

Every assertion here is *structural*: the analysis reads a transition jaxpr and
returns boolean patterns, so nothing in this file needs a gradient, an oracle or
a model. Gradient-level acceptance lives in ``_algorithm/snap_n_test.py``.

The governing soundness rule, restated once so every test below can be read
against it: the returned adjacency may be a **superset** of the true one-step
position dependency, never a subset. A superset costs memory and moves the
learning rule *towards* exact; a subset would silently compute a different rule
than the one the user asked for.
"""

import itertools

import brainevent
import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._compiler.position_graph import (
    AxisAdjacency,
    SnapPattern,
    analyze_position_adjacency,
    build_snap_pattern,
    close_adjacency,
    flat_adjacency,
)
from braintrace._misc import NotSupportedError


# -----------------------------------------------------------------------------
# fixtures: transition jaxprs built directly, without the full compiler
# -----------------------------------------------------------------------------

def _jaxpr(fn, *args):
    """The open jaxpr of *fn*, with closed-over arrays as constvars."""
    return jax.make_jaxpr(fn)(*args).jaxpr


def _dense_rnn_jaxpr(varshape=(1, 4), seed=0):
    """``h -> tanh(etp_mm(h, W))`` — one top-level ETP mixing equation."""
    n = varshape[-1]
    with brainstate.random.seed_context(seed):
        w = brainstate.random.randn(n, n)
    return _jaxpr(lambda h: jnp.tanh(braintrace.matmul(h, w)), jnp.zeros(varshape))


def _sparse_pattern(n=6, density=0.4, seed=0):
    rng = np.random.RandomState(seed)
    dense = (rng.rand(n, n) < density).astype(np.float32)
    # guarantee at least one entry per row so the pattern is not vacuous
    dense[np.arange(n), (np.arange(n) + 1) % n] = 1.0
    return dense


def _sparse_rnn_jaxpr(dense):
    n = dense.shape[0]
    csr = brainevent.CSR.fromdense(jnp.asarray(dense))
    data = jnp.asarray(csr.data)
    return _jaxpr(
        lambda h: jnp.tanh(braintrace.sparse_matmul(h, data, sparse_mat=csr)),
        jnp.zeros((n,)),
    )


def _identity_pattern(d):
    return np.eye(d, dtype=bool)


def _full_pattern(d):
    return np.ones((d, d), dtype=bool)


def _matrix_closure(a: np.ndarray, n: int) -> np.ndarray:
    """``⋁_{k=0..n-1} a^k`` computed on the *flat* (un-factorised) matrix."""
    p = np.eye(a.shape[0], dtype=bool)
    out = p.copy()
    for _ in range(n - 1):
        p = (p @ a) > 0
        out |= p
    return out


# -----------------------------------------------------------------------------
# one-step adjacency
# -----------------------------------------------------------------------------

class TestAnalyzeOneStepAdjacency:

    def test_scalar_varshape_has_no_axes(self):
        jaxpr = _jaxpr(lambda h: h * 0.9, jnp.zeros(()))
        adj = analyze_position_adjacency(jaxpr, ())
        assert isinstance(adj, AxisAdjacency)
        assert adj.axes == ()
        assert not adj.conservative

    def test_dense_recurrent_matmul_is_identity_times_full(self):
        # tanh_rnn's own shape: the leading axis is a size-1 batch axis and must
        # stay `identity`, otherwise K would carry a wasted factor of `batch`.
        adj = analyze_position_adjacency(_dense_rnn_jaxpr((1, 4)), (1, 4))
        assert not adj.conservative
        assert adj.reason == ''
        assert len(adj.axes) == 2
        np.testing.assert_array_equal(adj.axes[0], _identity_pattern(1))
        np.testing.assert_array_equal(adj.axes[1], _full_pattern(4))

    def test_batch_axis_stays_identity_when_larger_than_one(self):
        adj = analyze_position_adjacency(_dense_rnn_jaxpr((3, 4)), (3, 4))
        assert not adj.conservative
        np.testing.assert_array_equal(adj.axes[0], _identity_pattern(3))
        np.testing.assert_array_equal(adj.axes[1], _full_pattern(4))

    def test_no_mixing_equation_is_identity_everywhere(self):
        # two_state_rnn's shape: the v/a coupling is hand-written arithmetic,
        # so no position-mixing equation exists at all.
        jaxpr = _jaxpr(
            lambda v, a: (0.9 * v - 0.1 * a, 0.95 * a + v),
            jnp.zeros((1, 3)), jnp.zeros((1, 3)),
        )
        adj = analyze_position_adjacency(jaxpr, (1, 3))
        assert not adj.conservative
        np.testing.assert_array_equal(adj.axes[0], _identity_pattern(1))
        np.testing.assert_array_equal(adj.axes[1], _identity_pattern(3))

    def test_sparse_adjacency_is_the_pattern_transposed(self):
        # forward is y = x @ W, so h_p depends on h_q iff W[q, p] != 0.
        dense = _sparse_pattern(6)
        adj = analyze_position_adjacency(_sparse_rnn_jaxpr(dense), (6,))
        assert not adj.conservative
        assert len(adj.axes) == 1
        np.testing.assert_array_equal(adj.axes[0], (dense != 0).T)
        # and it is genuinely not the untransposed pattern (guards the direction)
        assert not np.array_equal(dense != 0, (dense != 0).T)

    def test_sparse_adjacency_ignores_stored_numeric_zeros(self):
        # the pattern is a *static* structure; a stored zero is still an edge.
        dense = _sparse_pattern(5)
        csr = brainevent.CSR.fromdense(jnp.asarray(dense))
        zeros = jnp.zeros_like(csr.data)
        jaxpr = _jaxpr(
            lambda h: braintrace.sparse_matmul(h, zeros, sparse_mat=csr),
            jnp.zeros((5,)),
        )
        adj = analyze_position_adjacency(jaxpr, (5,))
        np.testing.assert_array_equal(adj.axes[0], (dense != 0).T)


class TestConservativeFallbacks:
    """Every case where the analysis must widen to all-to-all, and say why."""

    def _assert_conservative(self, adj, varshape, needle):
        assert adj.conservative
        assert needle in adj.reason, adj.reason
        for axis, d in zip(adj.axes, varshape):
            np.testing.assert_array_equal(axis, _full_pattern(d))

    def test_two_mixing_equations(self):
        w1 = jnp.ones((4, 4))
        w2 = jnp.ones((4, 4))
        jaxpr = _jaxpr(
            lambda h: braintrace.matmul(jnp.tanh(braintrace.matmul(h, w1)), w2),
            jnp.zeros((1, 4)),
        )
        self._assert_conservative(
            analyze_position_adjacency(jaxpr, (1, 4)), (1, 4), 'more than one'
        )

    def test_unregistered_mixing_primitive(self):
        w = jnp.ones((4, 4))
        jaxpr = _jaxpr(lambda h: jnp.tanh(h @ w), jnp.zeros((1, 4)))
        self._assert_conservative(
            analyze_position_adjacency(jaxpr, (1, 4)), (1, 4), 'dot_general'
        )

    def test_non_elementwise_equation_on_the_hidden_path(self):
        # a reduction couples every position without being a "mixing" primitive
        jaxpr = _jaxpr(lambda h: h * 0.9 + jnp.sum(h), jnp.zeros((1, 4)))
        self._assert_conservative(
            analyze_position_adjacency(jaxpr, (1, 4)), (1, 4), 'reduce_sum'
        )

    def test_reshape_on_the_hidden_path(self):
        # a reshape relabels the axes an axis-indexed pattern is attached to
        jaxpr = _jaxpr(
            lambda h: jnp.reshape(jnp.reshape(h, (4, 1)) * 0.5, (1, 4)),
            jnp.zeros((1, 4)),
        )
        self._assert_conservative(
            analyze_position_adjacency(jaxpr, (1, 4)), (1, 4), 'reshape'
        )

    def test_control_flow_on_the_hidden_path(self):
        jaxpr = _jaxpr(
            lambda h: jax.lax.cond(True, lambda z: z * 2., lambda z: z * 3., h),
            jnp.zeros((1, 4)),
        )
        self._assert_conservative(
            analyze_position_adjacency(jaxpr, (1, 4)), (1, 4), 'cond'
        )

    def test_scan_hiding_a_registered_mixing_equation(self):
        # a single dot inside a length-L scan transfers A^L, not A: the
        # registered rule would be wrong, so the scan alone forces conservative.
        w = jnp.ones((4, 4))

        def body(carry, _):
            return braintrace.matmul(carry, w), None

        jaxpr = _jaxpr(
            lambda h: jax.lax.scan(body, h, None, length=3)[0], jnp.zeros((1, 4))
        )
        self._assert_conservative(
            analyze_position_adjacency(jaxpr, (1, 4)), (1, 4), 'scan'
        )

    def test_mixing_output_shape_differs_from_varshape(self):
        # y is not the group's position layout, so the rule's axis indices do
        # not refer to the group's axes
        w = jnp.ones((4, 7))
        jaxpr = _jaxpr(lambda h: braintrace.matmul(h, w), jnp.zeros((1, 4)))
        self._assert_conservative(
            analyze_position_adjacency(jaxpr, (1, 4)), (1, 4), 'shape'
        )

    def test_every_result_covers_the_true_dependency(self):
        """The soundness property, read against a *differentiated* reference.

        Comparing an all-ones matrix with anything is vacuously true, so the
        reference here is the real thing: the nonzero pattern of the transition's
        own Jacobian, obtained by differentiating the same function the analysis
        read symbolically. Every case is checked in the direction that can
        actually fail — the returned pattern must **cover** the true dependency.

        The ``tightness`` column then says what the coverage costs, which is
        what separates "sound" from "sound but useless":

        - ``'exact'``   the analysis reproduced the Jacobian's own pattern;
        - ``'lossy'``   the fallback fired *and* retained edges that do not
          exist, so the strict-superset assertion has teeth;
        - ``'free'``    the fallback fired but this transition really is
          all-to-all, so widening cost nothing. Kept to document that a
          conservative result is not the same claim as an imprecise one.
        """
        w_full = jnp.ones((4, 4))
        dense = _sparse_pattern(6)
        csr = brainevent.CSR.fromdense(jnp.asarray(dense))
        data = jnp.asarray(csr.data)

        cases = [
            # (label, transition fn, varshape, conservative?, tightness)
            ('reduce_sum', lambda h: h * 0.9 + jnp.sum(h), (1, 4), True, 'free'),
            ('dot_general', lambda h: jnp.tanh(h @ w_full), (1, 4), True, 'free'),
            ('cond', lambda h: jax.lax.cond(
                True, lambda z: z * 2., lambda z: z * 3., h), (1, 4), True, 'lossy'),
            ('reshape', lambda h: jnp.reshape(
                jnp.reshape(h, (4, 1)) * 0.5, (1, 4)), (1, 4), True, 'lossy'),
            ('sparse', lambda h: jnp.tanh(
                braintrace.sparse_matmul(h, data, sparse_mat=csr)), (6,), False, 'exact'),
            ('dense', lambda h: jnp.tanh(
                braintrace.matmul(h, w_full)), (1, 4), False, 'exact'),
        ]
        for label, fn, varshape, conservative, tightness in cases:
            adj = analyze_position_adjacency(_jaxpr(fn, jnp.zeros(varshape)), varshape)
            got = flat_adjacency(adj.axes)
            assert adj.conservative == conservative, (
                f'{label}: expected conservative={conservative}, got {adj.reason!r}')
            # differentiate at a generic point so no entry vanishes by accident
            with brainstate.random.seed_context(3):
                h0 = brainstate.random.randn(*varshape)
            jac = jax.jacobian(lambda h: jnp.ravel(fn(h)))(h0)
            true_dep = np.abs(np.asarray(jac).reshape(got.shape)) > 1e-6
            assert np.any(true_dep), f'{label}: the reference Jacobian is empty'
            assert np.all(got >= true_dep), f'{label}: analysis lost a dependency'
            if tightness == 'lossy':
                assert np.any(got > true_dep), (
                    f'{label}: expected the fallback to over-retain, but it was '
                    f'exact — the fixture no longer exercises the widening')
            elif tightness == 'exact':
                np.testing.assert_array_equal(got, true_dep, err_msg=label)


# -----------------------------------------------------------------------------
# closure
# -----------------------------------------------------------------------------

class TestCloseAdjacency:

    def test_n_equals_one_is_the_identity(self):
        # SnAp-1 retains the instantaneous pattern only -- no propagation.
        axes = (_full_pattern(4),)
        closed = close_adjacency(axes, 1)
        np.testing.assert_array_equal(closed[0], _identity_pattern(4))

    def test_n_equals_two_is_identity_or_adjacency(self):
        a = _sparse_pattern(6) != 0
        closed = close_adjacency((a,), 2)
        np.testing.assert_array_equal(closed[0], a | _identity_pattern(6))

    def test_closure_is_monotone_in_n(self):
        a = _sparse_pattern(7, density=0.25, seed=3) != 0
        prev = close_adjacency((a,), 1)[0]
        for n in range(2, 10):
            cur = close_adjacency((a,), n)[0]
            assert np.all(cur >= prev), f'closure shrank at n={n}'
            prev = cur

    def test_closure_saturates_and_is_then_a_fixed_point(self):
        a = _sparse_pattern(7, density=0.25, seed=3) != 0
        big = close_adjacency((a,), 64)[0]
        for n in (65, 128):
            np.testing.assert_array_equal(close_adjacency((a,), n)[0], big)

    def test_identity_axis_never_grows(self):
        axes = (_identity_pattern(3), _full_pattern(4))
        for n in (1, 2, 5, 50):
            np.testing.assert_array_equal(close_adjacency(axes, n)[0],
                                          _identity_pattern(3))

    def test_zero_or_negative_n_rejected(self):
        with pytest.raises(ValueError):
            close_adjacency((_full_pattern(2),), 0)

    def test_many_parallel_paths_do_not_wrap(self):
        """A path *count* that is a multiple of 256 must still read reachable.

        The closure is a reachability question, not a counting one. Accumulating
        it in a narrow integer type answers the counting question and then
        thresholds it, so a pair joined by exactly 256 parallel two-hop paths
        wraps to zero and is reported unreachable — SnAp would then drop that
        influence from the neighbourhood, silently, with no diagnostic and no
        change to K.
        """
        fan = 256
        size = fan + 2
        a = np.zeros((size, size), dtype=bool)
        for m in range(fan):
            a[m + 1, 0] = True          # 0 -> each of the fan-out positions
            a[size - 1, m + 1] = True   # each of them -> the far position
        closed = close_adjacency((a,), 3)[0]
        assert closed[size - 1, 0], 'a 256-fold two-hop path was lost to wraparound'
        np.testing.assert_array_equal(closed, _matrix_closure(a, 3))

    def test_factorised_closure_contains_the_flat_closure(self):
        # The per-axis (Kronecker) form is EXACT when at most one axis is
        # non-identity, and a conservative superset otherwise. Both halves are
        # asserted here, because the superset direction is the soundness
        # property and the exactness condition is what the shipped rules rely on.
        # Pure cyclic shifts, deliberately WITHOUT self-loops: a reflexive axis
        # pattern makes A^k monotone in k and the factorisation exact, so the
        # strict case only appears once an axis can move without staying put.
        a0 = np.roll(np.eye(3, dtype=bool), 1, axis=1)
        a1 = np.roll(np.eye(2, dtype=bool), 1, axis=1)

        # two non-identity axes: superset, and strictly so for some n
        strict_somewhere = False
        for n in (2, 3, 4):
            factorised = flat_adjacency(close_adjacency((a0, a1), n))
            flat = _matrix_closure(flat_adjacency((a0, a1)), n)
            assert np.all(factorised >= flat), f'factorised closure lost an edge at n={n}'
            strict_somewhere |= bool(np.any(factorised > flat))
        assert strict_somewhere, 'fixture does not exercise the strict case'

        # one non-identity axis: exact
        for n in (1, 2, 3, 4, 9):
            axes = (_identity_pattern(3), a1)
            factorised = flat_adjacency(close_adjacency(axes, n))
            flat = _matrix_closure(flat_adjacency(axes), n)
            np.testing.assert_array_equal(factorised, flat)


class TestFlatAdjacency:

    def test_kron_order_matches_c_order_flat_indexing(self):
        a0 = np.array([[1, 0], [1, 1]], dtype=bool)
        a1 = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
        flat = flat_adjacency((a0, a1))
        shape = (2, 3)
        for p in itertools.product(*[range(d) for d in shape]):
            for q in itertools.product(*[range(d) for d in shape]):
                pi = np.ravel_multi_index(p, shape)
                qi = np.ravel_multi_index(q, shape)
                want = a0[p[0], q[0]] and a1[p[1], q[1]]
                assert bool(flat[pi, qi]) == bool(want)

    def test_no_axes_is_the_one_by_one_identity(self):
        np.testing.assert_array_equal(flat_adjacency(()), np.ones((1, 1), dtype=bool))


# -----------------------------------------------------------------------------
# the neighbourhood index
# -----------------------------------------------------------------------------

class TestBuildSnapPattern:

    def test_self_is_always_the_first_neighbour(self):
        dense = _sparse_pattern(6)
        pat = build_snap_pattern(_sparse_rnn_jaxpr(dense), (6,), 3)
        np.testing.assert_array_equal(pat.neighbours[:, 0], np.arange(6))
        assert np.all(pat.valid[:, 0])

    def test_neighbours_agree_with_the_closed_flat_adjacency(self):
        dense = _sparse_pattern(6)
        pat = build_snap_pattern(_sparse_rnn_jaxpr(dense), (6,), 3)
        flat = flat_adjacency(pat.axes)
        for p in range(pat.num_position):
            listed = set(int(q) for q, ok in zip(pat.neighbours[p], pat.valid[p]) if ok)
            # ``flat[r, p]`` reads "r depends on p"; the trace slot anchored at p
            # has to cover the positions p *influences*, which is column p. See
            # TestNeighbourhoodDirection below for a fixture where the two differ.
            expected = set(np.flatnonzero(flat[:, p]).tolist())
            assert listed == expected

    def test_padded_slots_repeat_self_and_are_marked_invalid(self):
        dense = _sparse_pattern(6)
        pat = build_snap_pattern(_sparse_rnn_jaxpr(dense), (6,), 2)
        for p in range(pat.num_position):
            for k in range(pat.num_neighbour):
                if not pat.valid[p, k]:
                    # a dummy index that is always in range, so a gather on it
                    # can never read out of bounds
                    assert pat.neighbours[p, k] == p

    def test_neighbour_rows_have_no_duplicate_valid_entries(self):
        dense = _sparse_pattern(6)
        pat = build_snap_pattern(_sparse_rnn_jaxpr(dense), (6,), 3)
        for p in range(pat.num_position):
            valid = pat.neighbours[p][pat.valid[p]]
            assert len(set(valid.tolist())) == len(valid)

    def test_n_equals_one_degenerates_to_a_single_neighbour(self):
        # SnAp-1 == the un-widened per-parameter trace
        pat = build_snap_pattern(_dense_rnn_jaxpr((1, 4)), (1, 4), 1)
        assert pat.num_neighbour == 1
        np.testing.assert_array_equal(pat.neighbours[:, 0], np.arange(4))

    def test_dense_rnn_saturates_at_n_two(self):
        pat1 = build_snap_pattern(_dense_rnn_jaxpr((1, 4)), (1, 4), 1)
        pat2 = build_snap_pattern(_dense_rnn_jaxpr((1, 4)), (1, 4), 2)
        pat3 = build_snap_pattern(_dense_rnn_jaxpr((1, 4)), (1, 4), 3)
        assert (pat1.num_neighbour, pat2.num_neighbour, pat3.num_neighbour) == (1, 4, 4)
        assert pat2.num_neighbour == pat2.num_position  # saturated
        np.testing.assert_array_equal(pat2.valid, np.ones((4, 4), dtype=bool))

    def test_no_mixing_gives_one_neighbour_at_every_n(self):
        jaxpr = _jaxpr(lambda h: 0.9 * h, jnp.zeros((1, 3)))
        for n in (1, 2, 5, 17):
            pat = build_snap_pattern(jaxpr, (1, 3), n)
            assert pat.num_neighbour == 1
            assert not pat.conservative

    def test_sparse_k_sequence_is_exact_and_monotone(self):
        dense = _sparse_pattern(6, density=0.25, seed=1)
        jaxpr = _sparse_rnn_jaxpr(dense)
        a = (dense != 0).T
        ks = []
        for n in range(1, 8):
            pat = build_snap_pattern(jaxpr, (6,), n)
            # K is the widest *influence* set, i.e. the largest column sum of the
            # closure. On this fixture the row maxima happen to agree, so the
            # column is spelled out rather than left to coincidence.
            want = int(_matrix_closure(a, n).sum(axis=0).max())
            assert pat.num_neighbour == want, f'n={n}'
            ks.append(pat.num_neighbour)
        assert ks == sorted(ks), ks
        assert ks[0] == 1
        assert ks[-1] == 6  # saturates on this fixture

    def test_neighbourhoods_are_nested_in_n(self):
        dense = _sparse_pattern(6, density=0.25, seed=1)
        jaxpr = _sparse_rnn_jaxpr(dense)
        prev = None
        for n in range(1, 8):
            pat = build_snap_pattern(jaxpr, (6,), n)
            cur = [
                set(int(q) for q, ok in zip(pat.neighbours[p], pat.valid[p]) if ok)
                for p in range(pat.num_position)
            ]
            if prev is not None:
                for p, (a_set, b_set) in enumerate(zip(prev, cur)):
                    assert a_set <= b_set, f'neighbourhood shrank at n={n}, p={p}'
            prev = cur

    def test_scalar_varshape_pattern(self):
        jaxpr = _jaxpr(lambda h: h * 0.9, jnp.zeros(()))
        pat = build_snap_pattern(jaxpr, (), 4)
        assert pat.num_position == 1
        assert pat.num_neighbour == 1
        np.testing.assert_array_equal(pat.neighbours, np.zeros((1, 1), dtype=pat.neighbours.dtype))

    def test_conservative_flag_and_reason_propagate(self):
        w = jnp.ones((4, 4))
        jaxpr = _jaxpr(lambda h: jnp.tanh(h @ w), jnp.zeros((1, 4)))
        pat = build_snap_pattern(jaxpr, (1, 4), 2)
        assert isinstance(pat, SnapPattern)
        assert pat.conservative
        assert pat.reason
        assert pat.num_neighbour == 4

    def test_budget_guard_raises_with_the_numbers(self):
        w = jnp.ones((64, 64))
        jaxpr = _jaxpr(lambda h: braintrace.matmul(h, w), jnp.zeros((1, 64)))
        with pytest.raises(NotSupportedError) as exc:
            build_snap_pattern(jaxpr, (1, 64), 2, max_jacobian_elements=1024)
        msg = str(exc.value)
        assert '64' in msg and 'sparse_n' in msg

    def test_budget_guard_admits_what_fits(self):
        w = jnp.ones((8, 8))
        jaxpr = _jaxpr(lambda h: braintrace.matmul(h, w), jnp.zeros((1, 8)))
        # P*(K*S)**2 = 8 * 64 = 512
        pat = build_snap_pattern(jaxpr, (1, 8), 2, max_jacobian_elements=1024)
        assert pat.num_neighbour == 8

    def test_budget_guard_counts_the_block_jacobian_not_the_trace(self):
        """The ceiling is on ``P*(K*S)**2``, which is the dominant allocation.

        A ``P*K`` budget is linear in the widening and so cannot see the term
        that actually blows up: the widened block Jacobian is ``P*(K*S)**2``.
        The fixture below saturates at ``K = P = 8``, where ``P*K = 64`` but
        ``P*K**2 = 512`` — a budget of 100 must reject it.
        """
        w = jnp.ones((8, 8))
        jaxpr = _jaxpr(lambda h: braintrace.matmul(h, w), jnp.zeros((1, 8)))
        with pytest.raises(NotSupportedError) as exc:
            build_snap_pattern(jaxpr, (1, 8), 2, max_jacobian_elements=100)
        assert '512' in str(exc.value)

    def test_budget_guard_charges_for_states_per_position(self):
        """``num_state`` enters squared, so a multi-state group costs ``S**2``."""
        w = jnp.ones((8, 8))
        jaxpr = _jaxpr(lambda h: braintrace.matmul(h, w), jnp.zeros((1, 8)))
        build_snap_pattern(jaxpr, (1, 8), 2, num_state=1, max_jacobian_elements=512)
        with pytest.raises(NotSupportedError) as exc:
            build_snap_pattern(jaxpr, (1, 8), 2, num_state=2, max_jacobian_elements=512)
        assert '2048' in str(exc.value)  # 8 * (8 * 2) ** 2


# -----------------------------------------------------------------------------
# the direction of the neighbourhood
# -----------------------------------------------------------------------------

class TestNeighbourhoodDirection:
    """A trace slot covers what its anchor *influences*, not what it reads from.

    On an undirected (or symmetric-by-accident) graph the two readings coincide
    and every size-based check -- K, saturation, nestedness -- passes under
    either one. The fixture here is a strict shift, whose in- and out-neighbours
    are disjoint apart from the anchor itself, so only membership separates them.
    Getting this backwards is not a relabelling: it fills the trace with
    positions whose contribution to the anchor's gradient is exactly zero, and
    the error curve in ``n`` then stays flat until the neighbourhood saturates.
    """

    @staticmethod
    def _shift_pattern(n_rec):
        # dense[q, r] != 0 means h_new[r] reads h[q], so this is q -> q+1
        dense = np.zeros((n_rec, n_rec), dtype='float32')
        dense[np.arange(n_rec), (np.arange(n_rec) + 1) % n_rec] = 1.0
        return dense

    def test_neighbours_run_downstream_of_the_anchor(self):
        n_rec = 6
        jaxpr = _sparse_rnn_jaxpr(self._shift_pattern(n_rec))
        for n in (1, 2, 3, 4):
            pat = build_snap_pattern(jaxpr, (n_rec,), n)
            assert pat.num_neighbour == n, f'n={n}'
            for p in range(n_rec):
                listed = set(int(q) for q, ok in zip(pat.neighbours[p], pat.valid[p]) if ok)
                downstream = {(p + k) % n_rec for k in range(n)}
                upstream = {(p - k) % n_rec for k in range(n)}
                assert listed == downstream, f'n={n}, p={p}'
                if n > 1:  # the two only coincide at the anchor itself
                    assert listed != upstream

    def test_adjacency_reads_as_dependency(self):
        # the convention the direction argument rests on: A[p, q] is "p depends
        # on q", so the shift q -> q+1 has to show up as A[q + 1, q]. `pat.axes`
        # is the closure, and at n=2 that is exactly `I | A`.
        n_rec = 5
        pat = build_snap_pattern(_sparse_rnn_jaxpr(self._shift_pattern(n_rec)), (n_rec,), 2)
        flat = flat_adjacency(pat.axes)
        for q in range(n_rec):
            assert flat[q, q]
            assert flat[(q + 1) % n_rec, q]
            assert not flat[q, (q + 1) % n_rec]
