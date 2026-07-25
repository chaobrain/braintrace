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

"""Acceptance suite for ``recurrence_scope='sparse_n'`` (SnAp-n) — P3.

Two rules govern every gradient assertion here, both inherited from the
axis-decomposition roadmap:

* **Finite window only.** The full-window multi-step VJP path returns BPTT for
  every algorithm at every coordinate (finding F-23), so a criterion phrased
  there is vacuous. Everything below goes through
  :func:`~braintrace._algorithm.oracle.chunked_online_param_gradients` with
  ``chunk_size < T``.
* **Structural pins.** A gradient criterion is read against the compiled
  neighbourhood width ``K``, the widened state axis ``M = K * S``, the
  relation's primitive, whether the position analysis was conservative, and the
  number of hidden groups. Without them a criterion can pass at ``K == 1``
  while believing it exercised the scale.
"""

import unittest

import brainevent
import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm import oracle_models as om
from braintrace._algorithm.axes import ETraceConfig
from braintrace._algorithm.oracle import (
    assert_gradients_differ,
    assert_model_is_live,
    assert_param_gradients_close,
    bptt_param_gradients,
    chunked_online_param_gradients,
    flat_gradient_leaves,
    relative_deviation,
)
from braintrace._compiler.diagnostics import DiagnosticKind, DiagnosticLevel

T = 8
N_IN = 3
#: Relative deviation below which a finite-window gradient counts as equal to
#: BPTT. Calibrated on an algorithm that *is* exact on its model (D_RTRL on
#: ``leaky_linear``), which lands at ~6e-8 for every chunk size and both trace
#: implementations; 1e-6 leaves two orders of headroom without approaching the
#: 6e-2 that the diagonal approximation costs on a recurrent model.
EXACT_RTOL = 1e-6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _snap(n: int, **kwargs):
    """Algorithm factory for ``SnAp`` on the finite-window oracle path."""
    return lambda m: braintrace.SnAp(m, n=n, vjp_method='multi-step', **kwargs)


def _compiled(spec, algo_factory, xs):
    """Build + compile one algorithm, without running any gradient."""
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = algo_factory(model)
    algo.compile_graph(xs[0])
    return algo


def _pins(algo) -> dict:
    """The structural facts a gradient criterion has to be read against."""
    groups = algo.graph.hidden_groups
    return dict(
        n_groups=len(groups),
        K=tuple(1 if g.snap is None else g.snap.num_neighbour for g in groups),
        M=tuple(g.trace_state_width for g in groups),
        S=tuple(g.num_state for g in groups),
        P=tuple(int(np.prod(g.varshape)) if g.varshape else 1 for g in groups),
        conservative=tuple(
            False if g.snap is None else g.snap.conservative for g in groups),
        saturated=tuple(
            False if g.snap is None else g.snap.is_saturated for g in groups),
        primitives=tuple(sorted({
            r.primitive.name for r in algo.graph.hidden_param_op_relations})),
    )


def _grads(spec, algo_factory, xs, chunk_size):
    return chunked_online_param_gradients(
        spec.factory, xs, algo_factory=algo_factory, chunk_size=chunk_size)


def _rel_etp(actual, expected, spec) -> float:
    """Relative deviation restricted to the spec's **ETP** parameters.

    A comparison against BPTT has to be read on the eligibility-trace
    parameters only. A model's *plain* parameters (``win`` here) are not carried
    by any trace: under chunking their gradient is exactly truncated BPTT, and
    it truncates by construction at every coordinate of every axis. Measuring
    the whole tree therefore reports the window length, not the learning rule —
    on the ring at ``chunk_size=1`` the plain projection alone contributes a
    4.5e-01 deviation while the ETP weight is exact to 7e-08.

    Algorithm-vs-algorithm comparisons stay on the *full* tree: two rules run at
    the same window truncate identically, so agreeing everywhere is the stronger
    statement.
    """
    keys = set(spec.etp_param_keys)
    assert keys, 'the spec declares no ETP parameters; the comparison is empty'
    return relative_deviation(
        {k: v for k, v in actual.items() if k in keys},
        {k: v for k, v in expected.items() if k in keys},
    )


def _trace_element_count(algo) -> int:
    """Total number of scalars held in the algorithm's eligibility traces."""
    algo.init_etrace_state()
    return sum(
        int(np.prod(jnp.shape(leaf)))
        for state in algo.etrace_bwg.values()
        for leaf in jax.tree.leaves(state.value)
    )


# The reference model for the whole scale: a sparse ring, whose position graph
# has diameter ``n_rec - 1``, so ``K(n) == min(n, n_rec)`` and every order
# between the two endpoints is structurally distinct. A *dense* recurrent weight
# has diameter 1 and saturates at n=2, which would collapse the scale into its
# two endpoints and hide every intermediate behaviour.
RING = 6


def _ring_spec():
    return om.sparse_ring_rnn(n_in=N_IN, n_rec=RING, seed=0)


def _ring_inputs(seed=0):
    return _ring_spec().make_inputs(T, N_IN, seed=seed)


# ---------------------------------------------------------------------------
# criterion 11 + 2: coordinates, and the near end of the squeeze
# ---------------------------------------------------------------------------

class TestCoordinates(unittest.TestCase):
    """Criterion 11 — the preset's axis coordinates, asserted so they cannot drift."""

    def test_snap_n_ge_2_is_the_sparse_n_coordinate(self):
        algo = braintrace.SnAp(_ring_spec().factory(), n=3)
        cfg = algo.config
        assert cfg.trace_factorization == 'per_param'
        assert cfg.temporal_recursion == 'jacobian'
        assert cfg.recurrence_scope == 'sparse_n'
        assert cfg.sparse_n == 3
        assert cfg.learning_signal == 'symmetric'
        assert cfg.trace_filter == 'none'

    def test_snap_1_canonicalises_to_the_coupled_coordinate(self):
        """``n = 1`` is ``coupled``, not a second spelling of it."""
        algo = braintrace.SnAp(_ring_spec().factory(), n=1)
        assert algo.config.recurrence_scope == 'coupled'
        assert algo.config.sparse_n is None
        # ...which is exactly OSTLRecurrent's coordinate.
        ostl = braintrace.OSTLRecurrent(_ring_spec().factory())
        assert algo.config.recurrence_scope == ostl.config.recurrence_scope
        assert algo.config.temporal_recursion == ostl.config.temporal_recursion
        assert algo.config.trace_factorization == ostl.config.trace_factorization

    def test_preset_records_the_requested_order(self):
        assert braintrace.SnAp(_ring_spec().factory(), n=4).n == 4

    def test_include_recurrent_mixing_is_on(self):
        cfg = ETraceConfig(recurrence_scope='sparse_n', sparse_n=2)
        assert cfg.include_recurrent_mixing is True

    def test_describe_round_trips(self):
        """``describe()`` emits keyword arguments, so feed them back in.

        A substring check would pass on a description that named the order but
        lost the scope, or vice versa. Reconstructing the config from its own
        text and comparing the whole dataclass checks the pair, and the ``n=1``
        row additionally pins canonicalisation: the description of a SnAp-1
        config says ``coupled``, and rebuilding from that text lands back on the
        very config that produced it.
        """
        for cfg in [
            ETraceConfig(recurrence_scope='sparse_n', sparse_n=3),
            ETraceConfig(recurrence_scope='sparse_n', sparse_n=2),
            ETraceConfig(recurrence_scope='sparse_n', sparse_n=1),
            ETraceConfig(recurrence_scope='coupled'),
            ETraceConfig(recurrence_scope='diagonal'),
        ]:
            text = cfg.describe()
            rebuilt = eval(f'ETraceConfig({text})', {'ETraceConfig': ETraceConfig})
            assert rebuilt == cfg, f'{text!r} rebuilt as {rebuilt}'


class TestSnAp1EqualsOSTLRecurrent(unittest.TestCase):
    """Criterion 2 — the near end of the squeeze."""

    def test_snap_1_equals_ostl_recurrent_elementwise(self):
        spec = _ring_spec()
        xs = _ring_inputs()
        assert_model_is_live(spec.factory, xs)

        g_snap = _grads(spec, _snap(1), xs, chunk_size=2)
        g_ostl = _grads(
            spec, lambda m: braintrace.OSTLRecurrent(m, vjp_method='multi-step'),
            xs, chunk_size=2)
        g_drtrl = _grads(
            spec, lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
            xs, chunk_size=2)

        # Negative control first: the comparison has content only if the
        # coupled and diagonal scopes are actually distinguishable here.
        assert_gradients_differ(g_drtrl, g_ostl, min_rel=1e-6)
        assert relative_deviation(g_snap, g_ostl) < 1e-12

    def test_snap_1_allocates_the_unwidened_trace(self):
        """Structural pin: ``n = 1`` must not widen anything."""
        algo = _compiled(_ring_spec(), _snap(1), _ring_inputs())
        pins = _pins(algo)
        assert pins['n_groups'] == 1
        assert pins['K'] == (1,)
        assert pins['M'] == pins['S'] == (1,)
        assert all(g.snap is None for g in algo.graph.hidden_groups)


# ---------------------------------------------------------------------------
# criterion 3: the far end of the squeeze
# ---------------------------------------------------------------------------

class TestSaturationEqualsBPTT(unittest.TestCase):
    """Criterion 3 — a saturated within-group SnAp is full RTRL, hence BPTT.

    Valid only because the model has **exactly one** hidden group (asserted,
    not chosen quietly) and an elementwise ``y -> hidden`` tail. On a
    multi-group model saturation is full RTRL *within* each group, which is not
    BPTT.
    """

    def _assert_single_group_saturated(self, algo):
        pins = _pins(algo)
        assert pins['n_groups'] == 1, 'criterion 3 requires one hidden group'
        assert pins['saturated'] == (True,), pins
        assert pins['K'] == (RING,), pins
        assert pins['M'] == (RING,), pins   # S == 1
        assert pins['conservative'] == (False,), pins
        assert pins['primitives'] == ('etp_sp_mv',), pins
        return pins

    def test_saturation_matches_bptt_across_chunk_sizes(self):
        spec = _ring_spec()
        xs = _ring_inputs()
        assert_model_is_live(spec.factory, xs)
        bptt = bptt_param_gradients(spec.factory, xs)

        self._assert_single_group_saturated(
            _compiled(spec, _snap(RING), xs))

        # chunk sizes: the most aggressive window, a non-divisor of T, and T-1.
        for chunk_size in (1, 3, T - 1):
            for chunked_trace in (True, False):
                g = _grads(spec, _snap(RING, chunked_trace=chunked_trace),
                           xs, chunk_size)
                rel = _rel_etp(g, bptt, spec)
                assert rel < EXACT_RTOL, (
                    f'saturated SnAp deviates from BPTT by {rel:.3e} at '
                    f'chunk_size={chunk_size}, chunked_trace={chunked_trace}')

    def test_the_far_end_is_not_trivially_reachable(self):
        """Negative control: SnAp-1 on the same model is *not* BPTT."""
        spec = _ring_spec()
        xs = _ring_inputs()
        bptt = bptt_param_gradients(spec.factory, xs)
        g1 = _grads(spec, _snap(1), xs, chunk_size=3)
        assert _rel_etp(g1, bptt, spec) > 1e-3

    def test_saturation_matches_bptt_with_two_states_per_position(self):
        """``S > 1, K > 1``: the widened axis is ``M = K * S``, not ``K``."""
        n_rec = 5
        spec = om.sparse_ring_two_state_rnn(n_in=N_IN, n_rec=n_rec, seed=0)
        xs = spec.make_inputs(T, N_IN, seed=0)
        assert_model_is_live(spec.factory, xs)

        algo = _compiled(spec, _snap(n_rec), xs)
        pins = _pins(algo)
        assert pins['n_groups'] == 1, pins
        assert pins['S'] == (2,), pins
        assert pins['K'] == (n_rec,), pins
        assert pins['M'] == (2 * n_rec,), pins
        assert pins['conservative'] == (False,), pins

        bptt = bptt_param_gradients(spec.factory, xs)
        g = _grads(spec, _snap(n_rec), xs, chunk_size=3)
        rel = _rel_etp(g, bptt, spec)
        assert rel < EXACT_RTOL, f'S=2 saturated SnAp vs BPTT: {rel:.3e}'

    def test_dense_recurrence_saturates_at_n_2(self):
        """A dense recurrent weight has diameter 1, so ``n = 2`` is already full RTRL."""
        spec = om.tanh_rnn(n_in=N_IN, n_rec=4, seed=0)
        xs = spec.make_inputs(T, N_IN, seed=0)
        assert_model_is_live(spec.factory, xs)

        algo = _compiled(spec, _snap(2), xs)
        pins = _pins(algo)
        # varshape is (batch, n_rec): the batch axis is structurally
        # uncoupled, so K is n_rec and not batch * n_rec.
        assert pins['n_groups'] == 1, pins
        assert pins['K'] == (4,), pins
        assert pins['P'] == (4,), pins    # batch == 1 here
        assert pins['saturated'] == (True,), pins
        assert pins['primitives'] == ('etp_mm',), pins

        bptt = bptt_param_gradients(spec.factory, xs)
        g = _grads(spec, _snap(2), xs, chunk_size=2)
        rel = _rel_etp(g, bptt, spec)
        assert rel < EXACT_RTOL, f'dense saturated SnAp vs BPTT: {rel:.3e}'


class TestTracePathAgreement(unittest.TestCase):
    """All three trace-update implementations must produce one gradient.

    ``chunked_trace=False`` fuses the roll into the executor's over-time scan;
    ``chunked_trace=True`` takes the closed-form chunk factorisation (with a
    legacy scan fallback for relations whose primitive has no chunk kernel).
    A widening applied on one path and not another would show up here.
    """

    def test_chunked_and_fused_paths_agree(self):
        spec = _ring_spec()
        xs = _ring_inputs()
        a = _grads(spec, _snap(3, chunked_trace=True), xs, chunk_size=3)
        b = _grads(spec, _snap(3, chunked_trace=False), xs, chunk_size=3)
        assert relative_deviation(a, b) < 1e-6

    def test_dense_chunk_kernel_agrees_with_the_scan(self):
        """``etp_mm`` *does* have a chunk kernel, so this pins the third path."""
        spec = om.tanh_rnn(n_in=N_IN, n_rec=4, seed=0)
        xs = spec.make_inputs(T, N_IN, seed=0)
        a = _grads(spec, _snap(2, chunked_trace=True), xs, chunk_size=3)
        b = _grads(spec, _snap(2, chunked_trace=False), xs, chunk_size=3)
        assert relative_deviation(a, b) < 1e-6


# ---------------------------------------------------------------------------
# criterion 4: nestedness (not accuracy)
# ---------------------------------------------------------------------------

class TestNestedness(unittest.TestCase):
    """Criterion 4 — the neighbourhoods nest and saturation is a fixed point.

    The *gradient error* curve in ``n`` is reported, never asserted monotone:
    nested masks do not imply monotone error, because a newly retained path can
    overshoot terms the truncation was previously cancelling against. Only the
    two endpoints are asserted (they are criteria 2 and 3).
    """

    def _neighbour_sets(self, n):
        algo = _compiled(_ring_spec(), _snap(n), _ring_inputs())
        group = algo.graph.hidden_groups[0]
        if group.snap is None:            # n == 1 canonicalises to 'coupled'
            return [{p} for p in range(RING)]
        nbr, valid = group.snap.neighbours, group.snap.valid
        return [set(nbr[p][valid[p]].tolist()) for p in range(nbr.shape[0])]

    def test_neighbourhoods_are_nested_in_n(self):
        sets = {n: self._neighbour_sets(n) for n in range(1, RING + 2)}
        for n in range(1, RING + 1):
            for p in range(RING):
                assert sets[n][p] <= sets[n + 1][p], (n, p, sets[n][p], sets[n + 1][p])

    def test_self_is_always_a_neighbour(self):
        for n in range(1, RING + 2):
            for p, s in enumerate(self._neighbour_sets(n)):
                assert p in s

    def test_saturation_is_a_fixed_point(self):
        at_diameter = self._neighbour_sets(RING)
        beyond = self._neighbour_sets(RING + 3)
        assert at_diameter == beyond
        assert all(s == set(range(RING)) for s in at_diameter)

    def test_error_curve_endpoints(self):
        """Assert the two ends; report the middle in the failure message."""
        spec = _ring_spec()
        xs = _ring_inputs()
        bptt = bptt_param_gradients(spec.factory, xs)
        curve = {
            n: _rel_etp(_grads(spec, _snap(n), xs, chunk_size=3), bptt, spec)
            for n in range(1, RING + 1)
        }
        report = ', '.join(f'n={n}: {v:.3e}' for n, v in curve.items())
        assert all(np.isfinite(v) for v in curve.values()), report
        assert curve[1] > 1e-3, f'SnAp-1 should be visibly approximate — {report}'
        assert curve[RING] < EXACT_RTOL, f'saturation should be exact — {report}'


# ---------------------------------------------------------------------------
# the interior of the scale, against an independent reference
# ---------------------------------------------------------------------------

#: Ring width for the masked-RTRL comparison. Smaller than ``RING`` so the
#: hand-written reference stays cheap; diameter 4, so n = 2, 3, 4 are all
#: strictly between the two endpoints.
MASKED_RING = 5


def _masked_rtrl_trace(model, xs, order, snap, n_rec=MASKED_RING):
    """Roll the SnAp-*order* influence matrix by hand, in numpy, in float64.

    This is the independent reference the interior of the scale is read against.
    It shares no code with the compiler: the adjacency, the mask, the transition
    operator and the instantaneous term are all written out from the model's own
    arithmetic (``h_t = tanh(x_t W_in + h_{t-1} W)``), and only the neighbour
    *index* comes from the compiled pattern — because that index is precisely
    the layout of the array being compared, not part of the value being checked.

    Returns
    -------
    list of numpy.ndarray
        Per step, the reference trace laid out as ``(nnz, K)`` — reference
        influence ``J[e, p]`` read at the position each trace slot stands for.
    """
    w = np.asarray(model.w.value)
    win = np.asarray(model.win.value)
    dense = np.zeros((n_rec, n_rec), dtype='float32')
    for q in range(n_rec):
        for off in (0, 1):
            dense[q, (q + off) % n_rec] = 1.0
    rows, cols = np.nonzero(dense)         # row-major, i.e. CSR data order
    weight = np.zeros((n_rec, n_rec))
    weight[rows, cols] = w
    nnz = len(rows)

    # The mask: positions the anchor *influences* within ``order - 1`` steps.
    # `adjacency[p, q]` is "h_p depends on h_q", so influence is the column.
    adjacency = (dense != 0).T
    reach = np.eye(n_rec, dtype=bool)
    closure = reach.copy()
    for _ in range(order - 1):
        reach = reach @ adjacency
        closure |= reach
    mask = np.stack([closure[:, cols[e]] for e in range(nnz)])

    h = np.zeros(n_rec)
    influence = np.zeros((nnz, n_rec))
    out = []
    for t in range(len(xs)):
        x = np.asarray(xs[t], dtype='float64')
        z = x @ win + h @ weight
        h_new = np.tanh(z)
        dphi = 1.0 - h_new ** 2
        transition = dphi[:, None] * weight.T          # D[p, s] = phi'(z_p) W[s, p]
        instant = np.zeros((nnz, n_rec))
        # weight entry e = (row q, col r) enters h[r] only, scaled by h_{t-1}[q]
        instant[np.arange(nnz), cols] = dphi[cols] * h[rows]
        influence = mask * (influence @ transition.T + instant)
        h = h_new
        out.append(np.stack(
            [influence[np.arange(nnz), snap.neighbours[cols, k]] * snap.valid[cols, k]
             for k in range(snap.num_neighbour)],
            axis=1,
        ))
    return out


class TestInteriorAgainstMaskedRTRL(unittest.TestCase):
    """The interior of the scale, pinned against a hand-written recursion.

    The endpoint criteria (SnAp-1 == coupled, saturation == BPTT) leave the
    whole middle of the scale unasserted, and the middle is where this feature's
    one real bug lived: a neighbourhood taken in the wrong direction produced a
    gradient error curve that was *flat* in ``n`` and only collapsed at
    saturation, with every structural pin — ``K``, nestedness, saturation,
    padding — still passing. An implementation that quietly returned ``coupled``
    for every ``1 < n < P`` and full RTRL at saturation would pass the endpoints
    too. So the interior gets its own reference.
    """

    def _run(self, order, steps=6):
        spec = om.sparse_ring_rnn(n_in=N_IN, n_rec=MASKED_RING, seed=0)
        xs = spec.make_inputs(steps, N_IN, seed=0)
        model = spec.factory()
        brainstate.nn.init_all_states(model)
        algo = braintrace.SnAp(model, n=order)
        algo.compile_graph(xs[0])
        algo.init_etrace_state(xs[0])
        snap = algo.graph.hidden_groups[0].snap
        # A Python step loop is deliberate here: the reference is compared to
        # the trace *after every step*, which a compiled scan does not expose.
        traces = []
        for t in range(steps):
            algo(xs[t])
            traces.append(np.asarray(
                jax.tree.leaves(list(algo.etrace_bwg.values())[0].value)[0]))
        return model, xs, snap, traces

    @staticmethod
    def _deviation(got, want):
        num = max(float(np.abs(g - w).max()) for g, w in zip(got, want))
        den = max(float(np.abs(w).max()) for w in want)
        assert den > 1e-6, 'the reference trace is ~zero; the comparison is vacuous'
        return num / den

    def test_interior_orders_match_the_masked_recursion(self):
        for order in (2, 3, 4):
            model, xs, snap, traces = self._run(order)
            assert snap.num_neighbour == order, (order, snap.num_neighbour)
            want = _masked_rtrl_trace(model, xs, order, snap)
            rel = self._deviation(traces, want)
            assert rel < EXACT_RTOL, f'SnAp-{order} trace deviates by {rel:.3e}'

    def test_a_narrower_reference_does_not_match(self):
        """Negative control: the comparison above has content.

        Rolling the reference with a mask one order too narrow must disagree —
        otherwise the test would pass against an implementation stuck at
        ``coupled``. It only discriminates *downward* on this fixture: the ring
        edges run one way, so widening the mask adds positions strictly
        downstream of the ones already retained, and those cannot feed back into
        the slots being compared.
        """
        for order in (3, 4):
            model, xs, snap, traces = self._run(order)
            want = _masked_rtrl_trace(model, xs, order - 1, snap)
            rel = self._deviation(traces, want)
            assert rel > 1e-3, (
                f'SnAp-{order} agrees with an order-{order - 1} reference to '
                f'{rel:.3e}; the interior comparison cannot see the order')

    def test_the_widened_slots_carry_mass(self):
        """The neighbour slots are not decoration: they hold real influence."""
        for order in (2, 3, 4):
            _, _, _, traces = self._run(order)
            last = traces[-1]
            anchor = float(np.abs(last[:, 0]).max())
            neighbours = float(np.abs(last[:, 1:]).max())
            assert neighbours > 1e-3 * anchor, (
                f'SnAp-{order}: neighbour slots hold {neighbours:.3e} against an '
                f'anchor slot of {anchor:.3e} — the widening is inert')


# ---------------------------------------------------------------------------
# criterion 5: trace-state storage
# ---------------------------------------------------------------------------

class TestTraceStateStorage(unittest.TestCase):
    """Criterion 5 — the *trace* metric only; ``Dg`` is budgeted separately."""

    def test_exact_k_sequence_on_the_ring(self):
        observed = []
        for n in range(1, RING + 2):
            algo = _compiled(_ring_spec(), _snap(n), _ring_inputs())
            group = algo.graph.hidden_groups[0]
            observed.append(1 if group.snap is None else group.snap.num_neighbour)
        assert observed == [1, 2, 3, 4, 5, 6, 6], observed

    def test_trace_element_count_is_non_decreasing(self):
        counts = [
            _trace_element_count(_compiled(_ring_spec(), _snap(n), _ring_inputs()))
            for n in range(1, RING + 2)
        ]
        assert counts == sorted(counts), counts
        # and it genuinely grows: a flat sequence would mean nothing widened.
        assert counts[-1] > counts[0]

    def test_trace_width_is_k_times_num_state(self):
        for n in (2, 4, RING):
            algo = _compiled(_ring_spec(), _snap(n), _ring_inputs())
            group = algo.graph.hidden_groups[0]
            assert group.trace_state_width == group.snap.num_neighbour * group.num_state


# ---------------------------------------------------------------------------
# criterion 6: model-agnostic
# ---------------------------------------------------------------------------

def _lora_rec_net(n_in=N_IN, n_rec=4, rank=2, seed=0):
    """Recurrence *through* ``lora_matmul`` — a primitive with no fast path
    and no registered ``snap_adjacency`` rule, so the pattern is conservative
    (all-to-all) and therefore saturated at any ``n >= 2``."""

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = brainstate.random.RandomState(seed)
                self.B = brainstate.ParamState(0.3 * rng.randn(n_rec, rank))
                self.A = brainstate.ParamState(0.3 * rng.randn(rank, n_rec))
                self.win = brainstate.ParamState(
                    0.5 * rng.randn(n_in, n_rec))
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                rec = braintrace.lora_matmul(self.h.value, self.B.value, self.A.value)
                self.h.value = jax.nn.tanh(x @ self.win.value + rec)
                return self.h.value

        return Net()

    return om.ModelSpec(factory=factory, etp_param_keys=(('B',), ('A',)),
                        plain_param_keys=(('win',),))


def _lora_mv_rec_net(n_in=N_IN, n_rec=4, rank=2, seed=0):
    """``etp_lora_mv`` — the unbatched twin, reached by an unbatched hidden state.

    It shares every rule with ``etp_lora_mm`` and is dispatched purely on
    ``x.ndim``, so a fixture that only ever presents a batch axis never reaches
    it. Same for ``etp_gmv`` below.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = brainstate.random.RandomState(seed)
                self.B = brainstate.ParamState(0.3 * rng.randn(n_rec, rank))
                self.A = brainstate.ParamState(0.3 * rng.randn(rank, n_rec))
                self.win = brainstate.ParamState(0.5 * rng.randn(n_in, n_rec))
                self.h = brainstate.HiddenState(jnp.zeros((n_rec,)))

            def update(self, x):
                rec = braintrace.lora_matmul(self.h.value, self.B.value, self.A.value)
                self.h.value = jax.nn.tanh(x @ self.win.value + rec)
                return self.h.value

        return Net()

    return om.ModelSpec(factory=factory, etp_param_keys=(('B',), ('A',)),
                        plain_param_keys=(('win',),))


def _gmv_rec_net(G=2, K=3, n_in=N_IN, seed=0):
    """``etp_gmv`` — the unbatched twin of ``etp_gmm``."""

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = brainstate.random.RandomState(seed)
                self.w = brainstate.ParamState(0.4 * rng.randn(G, K, K))
                self.win = brainstate.ParamState(0.5 * rng.randn(n_in, G * K))
                self.h = brainstate.HiddenState(jnp.zeros((G, K)))

            def update(self, x):
                rec = braintrace.grouped_matmul(self.h.value, self.w.value)
                self.h.value = jax.nn.tanh((x @ self.win.value).reshape(G, K) + rec)
                return self.h.value

        return Net()

    return om.ModelSpec(factory=factory, etp_param_keys=(('w',),),
                        plain_param_keys=(('win',),))


def _elemwise_plus_dense_net(n_in=N_IN, n_rec=4, seed=0):
    """``etp_elemwise`` as a co-relation on a group whose mixing is dense.

    The elementwise gain ``a`` is a second ETP relation feeding the *same*
    hidden group, so its trace must widen to ``K * S`` even though the
    elementwise primitive itself couples nothing.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = brainstate.random.RandomState(seed)
                self.w = brainstate.ParamState(0.4 * rng.randn(n_rec, n_rec))
                self.alpha = brainstate.ParamState(jnp.ones((n_rec,)) * 0.8)
                self.win = brainstate.ParamState(
                    0.5 * rng.randn(n_in, n_rec))
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                gain = braintrace.element_wise(self.alpha.value)
                rec = braintrace.matmul(self.h.value, self.w.value)
                self.h.value = jax.nn.tanh(x @ self.win.value + gain * rec)
                return self.h.value

        return Net()

    return om.ModelSpec(factory=factory, etp_param_keys=(('w',), ('alpha',)),
                        plain_param_keys=(('win',),))


def _grouped_rec_net(G=2, K=3, n_in=N_IN, seed=0):
    """Recurrence through ``grouped_matmul`` (block diagonal, no adjacency rule)."""

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = brainstate.random.RandomState(seed)
                self.w = brainstate.ParamState(0.4 * rng.randn(G, K, K))
                self.win = brainstate.ParamState(
                    0.5 * rng.randn(n_in, G * K))
                # (1, G, K), not (1, G*K): a reshape between the ETP output
                # and the hidden state severs the relation entirely (the
                # compiler requires y broadcastable to the hidden shape), and
                # this model exists to exercise etp_gmm, not that limitation.
                self.h = brainstate.HiddenState(jnp.zeros((1, G, K)))

            def update(self, x):
                rec = braintrace.grouped_matmul(self.h.value, self.w.value)
                inp = (x @ self.win.value).reshape(1, G, K)
                self.h.value = jax.nn.tanh(inp + rec)
                return self.h.value

        return Net()

    return om.ModelSpec(factory=factory, etp_param_keys=(('w',),),
                        plain_param_keys=(('win',),))


def _conv_rec_net(length=5, ch=2, n_in=N_IN, seed=0):
    """Recurrence through a 1-D ``conv`` — an anchored primitive whose D-RTRL
    trace is kernel-shaped per output position rather than position-shaped."""

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = brainstate.random.RandomState(seed)
                self.p = brainstate.ParamState({
                    'weight': 0.2 * rng.randn(3, ch, ch),
                    'bias': jnp.zeros((ch,)),
                })
                self.win = brainstate.ParamState(
                    0.5 * rng.randn(n_in, length * ch))
                self.h = brainstate.HiddenState(jnp.zeros((1, length, ch)))

            def update(self, x):
                rec = braintrace.conv(
                    self.h.value, self.p.value['weight'], self.p.value['bias'],
                    strides=(1,), padding='SAME',
                    dimension_numbers=('NHC', 'HIO', 'NHC'),
                )
                inp = (x @ self.win.value).reshape(1, length, ch)
                self.h.value = jax.nn.tanh(inp + rec)
                return self.h.value

        return Net()

    return om.ModelSpec(factory=factory, etp_param_keys=(('p',),),
                        plain_param_keys=(('win',),))


def _einsum_rec_net(n_rec=4, n_in=N_IN, seed=0):
    """Recurrence through ``etp_einsum`` with **no shared axis** — anchored."""

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = brainstate.random.RandomState(seed)
                self.w = brainstate.ParamState(0.4 * rng.randn(n_rec, n_rec))
                self.win = brainstate.ParamState(
                    0.5 * rng.randn(n_in, n_rec))
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                rec = braintrace.einsum('bk,kn->bn', self.h.value, self.w.value)
                self.h.value = jax.nn.tanh(x @ self.win.value + rec)
                return self.h.value

        return Net()

    return om.ModelSpec(factory=factory, etp_param_keys=(('w',),),
                        plain_param_keys=(('win',),))


def _mv_rec_net(n_rec=4, n_in=N_IN, seed=0):
    """Unbatched dense recurrence — ``etp_mv``."""

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = brainstate.random.RandomState(seed)
                self.w = brainstate.ParamState(0.4 * rng.randn(n_rec, n_rec))
                self.win = brainstate.ParamState(
                    0.5 * rng.randn(n_in, n_rec))
                self.h = brainstate.HiddenState(jnp.zeros((n_rec,)))

            def update(self, x):
                rec = braintrace.matmul(self.h.value, self.w.value)
                self.h.value = jax.nn.tanh(x @ self.win.value + rec)
                return self.h.value

        return Net()

    return om.ModelSpec(factory=factory, etp_param_keys=(('w',),),
                        plain_param_keys=(('win',),))


def _sp_mm_rec_net(n_rec=5, n_in=N_IN, seed=0):
    """Batched sparse recurrence — ``etp_sp_mm``."""
    mask = np.zeros((n_rec, n_rec), dtype='float32')
    for q in range(n_rec):
        mask[q, (q + 1) % n_rec] = 1.0
    csr = brainevent.CSR.fromdense(jnp.asarray(mask))
    nnz = int(mask.sum())

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = brainstate.random.RandomState(seed)
                self.w = brainstate.ParamState(0.6 * rng.randn(nnz))
                self.win = brainstate.ParamState(
                    0.5 * rng.randn(n_in, n_rec))
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                rec = braintrace.sparse_matmul(self.h.value, self.w.value, sparse_mat=csr)
                self.h.value = jax.nn.tanh(x @ self.win.value + rec)
                return self.h.value

        return Net()

    return om.ModelSpec(factory=factory, etp_param_keys=(('w',),),
                        plain_param_keys=(('win',),))


#: One recurrent model per anchored ETP primitive — *every* one of them, the
#: unbatched twins included. ``etp_gmv`` / ``etp_lora_mv`` share their rules with
#: ``etp_gmm`` / ``etp_lora_mm`` but are dispatched on ``x.ndim``, so leaving
#: them out would leave a whole dispatch branch unexecuted. ``etp_emb`` /
#: ``etp_emb_v`` are absent on purpose: they declare *no* anchor (a gathered row
#: has no single hidden position to land on) and are covered by the default-deny
#: test in ``_op/_registries_test.py`` and by ``TestAnchorRejection``.
_ANCHORED_MODELS = {
    'etp_mm': lambda: om.tanh_rnn(n_in=N_IN, n_rec=4, seed=0),
    'etp_mv': _mv_rec_net,
    'etp_sp_mv': _ring_spec,
    'etp_sp_mm': _sp_mm_rec_net,
    'etp_elemwise': _elemwise_plus_dense_net,
    'etp_lora_mm': _lora_rec_net,
    'etp_lora_mv': _lora_mv_rec_net,
    'etp_gmm': _grouped_rec_net,
    'etp_gmv': _gmv_rec_net,
    'etp_conv': _conv_rec_net,
    'etp_einsum': _einsum_rec_net,
}

#: An order above every fixture's position-graph diameter, so each one
#: saturates and its gradient is comparable to BPTT.
SATURATING_N = 8


class TestModelAgnostic(unittest.TestCase):
    """Criterion 6 — the scale is not a property of the dense fast path."""

    def test_every_anchored_primitive_widens_its_trace(self):
        for prim_name in sorted(_ANCHORED_MODELS):
            spec = _ANCHORED_MODELS[prim_name]()
            xs = spec.make_inputs(T, N_IN, seed=0)
            algo = _compiled(spec, _snap(2), xs)
            pins = _pins(algo)
            assert prim_name in pins['primitives'], (prim_name, pins)
            # Every group the traces attach to must have widened, and every
            # trace leaf must carry the widened axis as its last dimension.
            groups = {g.index: g for g in algo.graph.hidden_groups}
            assert any(g.trace_state_width > g.num_state for g in groups.values()), pins
            algo.init_etrace_state()
            for (_, gi), state in algo.etrace_bwg.items():
                width = groups[gi].trace_state_width
                for leaf in jax.tree.leaves(state.value):
                    assert jnp.shape(leaf)[-1] == width, (
                        prim_name, gi, jnp.shape(leaf), width)

    def test_every_anchored_primitive_matches_bptt_at_saturation(self):
        """Run the widened recurrence and solve, per primitive, on both paths.

        Allocating a trace of the right shape is not evidence that anything
        consumes it correctly: the test above compiles and initialises, and
        would pass against a widened recurrence that never executes or a solve
        that reads the wrong slots. So every primitive is driven to a *saturated*
        order — where SnAp is full within-group RTRL and therefore equals BPTT —
        through both the per-primitive fast contraction and the legacy vmap
        path, which are separate implementations of the same contraction.
        """
        report = {}
        for prim_name in sorted(_ANCHORED_MODELS):
            spec = _ANCHORED_MODELS[prim_name]()
            xs = spec.make_inputs(T, N_IN, seed=0)
            assert_model_is_live(spec.factory, xs)
            algo = _compiled(spec, _snap(SATURATING_N), xs)
            pins = _pins(algo)
            assert prim_name in pins['primitives'], (prim_name, pins)
            assert all(pins['saturated']), (prim_name, pins)

            bptt = bptt_param_gradients(spec.factory, xs)
            fast = _grads(spec, _snap(SATURATING_N), xs, chunk_size=3)
            legacy = _grads(
                spec, _snap(SATURATING_N, fast_solve=False), xs, chunk_size=3)
            rel_fast = _rel_etp(fast, bptt, spec)
            rel_legacy = _rel_etp(legacy, bptt, spec)
            report[prim_name] = (rel_fast, rel_legacy)
            assert rel_fast < EXACT_RTOL, (prim_name, 'fast', rel_fast, pins)
            assert rel_legacy < EXACT_RTOL, (prim_name, 'legacy', rel_legacy, pins)
            # the two solve paths must also agree with *each other*, which is
            # the stronger statement where both happen to be wrong the same way
            assert relative_deviation(fast, legacy) < EXACT_RTOL, (
                prim_name, relative_deviation(fast, legacy))
        assert len(report) == len(_ANCHORED_MODELS), report

    def test_lora_saturates_and_matches_bptt(self):
        """The flagship no-fast-path case, pinned by its primitive."""
        spec = _lora_rec_net()
        xs = spec.make_inputs(T, N_IN, seed=0)
        assert_model_is_live(spec.factory, xs)

        algo = _compiled(spec, _snap(2), xs)
        pins = _pins(algo)
        assert pins['n_groups'] == 1, pins
        assert pins['primitives'] == ('etp_lora_mm',), pins
        assert pins['conservative'] == (True,), pins   # no adjacency rule
        assert pins['saturated'] == (True,), pins
        assert pins['K'] == (4,), pins

        bptt = bptt_param_gradients(spec.factory, xs)
        g = _grads(spec, _snap(2), xs, chunk_size=3)
        rel = _rel_etp(g, bptt, spec)
        assert rel < EXACT_RTOL, f'LoRA saturated SnAp vs BPTT: {rel:.3e}'

    def test_two_relations_on_one_group_both_widen(self):
        spec = _elemwise_plus_dense_net()
        xs = spec.make_inputs(T, N_IN, seed=0)
        algo = _compiled(spec, _snap(2), xs)
        pins = _pins(algo)
        assert pins['n_groups'] == 1, pins
        assert set(pins['primitives']) == {'etp_mm', 'etp_elemwise'}, pins
        assert pins['K'] == (4,), pins

        bptt = bptt_param_gradients(spec.factory, xs)
        g = _grads(spec, _snap(2), xs, chunk_size=3)
        rel = _rel_etp(g, bptt, spec)
        assert rel < EXACT_RTOL, f'two-relation saturated SnAp vs BPTT: {rel:.3e}'


# ---------------------------------------------------------------------------
# criterion 7: anchor rejection
# ---------------------------------------------------------------------------

def _shared_axis_einsum_net(T2=2, N=3, n_in=N_IN, seed=0):
    """``'btk,kn->btn'`` — the weight is reused across the shared ``t`` axis,
    so a trace slot has no single hidden position to anchor on."""

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = brainstate.random.RandomState(seed)
                self.w = brainstate.ParamState(0.3 * rng.randn(N, N))
                self.win = brainstate.ParamState(
                    0.3 * rng.randn(n_in, T2 * N))
                self.h = brainstate.HiddenState(jnp.zeros((1, T2, N)))

            def update(self, x):
                rec = braintrace.einsum('btk,kn->btn', self.h.value, self.w.value)
                inp = (x @ self.win.value).reshape(1, T2, N)
                self.h.value = jax.nn.tanh(inp + rec)
                return self.h.value

        return Net()

    return om.ModelSpec(factory=factory, etp_param_keys=(('w',),),
                        plain_param_keys=(('win',),))


class TestAnchorRejection(unittest.TestCase):
    """Criterion 7 — an unanchored primitive must fail loudly, not silently."""

    def _model_and_input(self):
        spec = _shared_axis_einsum_net()
        xs = spec.make_inputs(T, N_IN, seed=0)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        return model, xs[0]

    def test_shared_axis_einsum_is_rejected_under_sparse_n(self):
        model, x0 = self._model_and_input()
        algo = braintrace.SnAp(model, n=2)
        with pytest.raises(NotImplementedError) as excinfo:
            algo.compile_graph(x0)
        msg = str(excinfo.value)
        assert 'etp_einsum' in msg
        assert 'sparse_n' in msg

    def test_the_same_model_still_compiles_under_diagonal_and_coupled(self):
        for scope in ('diagonal', 'coupled'):
            model, x0 = self._model_and_input()
            algo = braintrace.D_RTRL(
                model, config=ETraceConfig(recurrence_scope=scope))
            algo.compile_graph(x0)          # must not raise
            assert algo.is_compiled

    def test_snap_1_is_rejected_too_even_though_it_canonicalises_to_coupled(self):
        """The preset is checked on what was *asked for*, not on the coordinate.

        ``SnAp(n=1)`` canonicalises to ``recurrence_scope='coupled'``, so a check
        keyed on the coordinate skips it — and the caller silently receives
        ``coupled``. On an unanchored primitive the two are not the same rule:
        SnAp-1 is defined as the instantaneous nonzero pattern of ``dh/dtheta``,
        which here spans several positions, while ``coupled`` keeps only the
        per-position block diagonal and drops the rest. The test above shows
        ``n=2`` rejected; without the provenance flag ``n=1`` would compile.
        """
        model, x0 = self._model_and_input()
        algo = braintrace.SnAp(model, n=1)
        assert algo.config.recurrence_scope == 'coupled'   # the canonical form
        with pytest.raises(NotImplementedError) as excinfo:
            algo.compile_graph(x0)
        assert 'etp_einsum' in str(excinfo.value)

    def test_snap_1_still_compiles_where_an_anchor_exists(self):
        """Negative control: the provenance flag is not a blanket rejection."""
        spec = _ring_spec()
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.SnAp(model, n=1)
        algo.compile_graph(_ring_inputs()[0])              # must not raise
        assert algo.is_compiled

    def test_embedding_is_rejected_under_sparse_n(self):
        """``etp_emb`` declares no anchor at all (default deny)."""
        n_vocab, n_rec = 5, 4

        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = brainstate.random.RandomState(0)
                self.emb = brainstate.ParamState(
                    0.3 * rng.randn(n_vocab, n_rec))
                self.w = brainstate.ParamState(
                    0.3 * rng.randn(n_rec, n_rec))
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, ids):
                e = braintrace.embedding(ids, self.emb.value)
                rec = braintrace.matmul(self.h.value, self.w.value)
                self.h.value = jax.nn.tanh(e + rec)
                return self.h.value

        model = Net()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.SnAp(model, n=2)
        with pytest.raises(NotImplementedError) as excinfo:
            algo.compile_graph(jnp.asarray([1]))
        assert 'sparse_n' in str(excinfo.value)


# ---------------------------------------------------------------------------
# criterion 8: degeneracy pins
# ---------------------------------------------------------------------------

class TestDegeneracyPins(unittest.TestCase):
    """Criterion 8 — a degenerate pattern must reproduce ``coupled`` exactly,
    *and* be degenerate for a structural reason rather than by defaulting."""

    def test_two_state_rnn_has_no_mixing_equation(self):
        spec = om.two_state_rnn(n_in=N_IN, n_rec=3, seed=0)
        xs = spec.make_inputs(T, N_IN, seed=0)
        algo = _compiled(spec, _snap(4), xs)
        group = algo.graph.hidden_groups[0]
        pins = _pins(algo)
        assert pins['n_groups'] == 1, pins
        assert pins['S'] == (2,), pins
        assert pins['K'] == (1,), pins
        assert pins['M'] == (2,), pins
        # structural, not a fallback: the analysis found no mixing equation and
        # returned the identity, so `conservative` is False.
        assert pins['conservative'] == (False,), pins
        assert group.snap.is_degenerate
        assert all(bool(np.all(a == np.eye(a.shape[0], dtype=bool)))
                   for a in group.snap.axes)

    def test_degenerate_sparse_n_equals_coupled_equals_d_rtrl(self):
        spec = om.two_state_rnn(n_in=N_IN, n_rec=3, seed=0)
        xs = spec.make_inputs(T, N_IN, seed=0)
        assert_model_is_live(spec.factory, xs)

        g_snap = _grads(spec, _snap(4), xs, chunk_size=3)
        g_coupled = _grads(
            spec, lambda m: braintrace.D_RTRL(
                m, vjp_method='multi-step',
                config=ETraceConfig(recurrence_scope='coupled')),
            xs, chunk_size=3)
        g_diag = _grads(
            spec, lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
            xs, chunk_size=3)

        assert relative_deviation(g_snap, g_coupled) < 1e-12
        # With no ETP mixing primitive at all, `diagonal` and `coupled` are the
        # same rule on this model too — that identity is the pin, and a pair
        # that *starts* differing is as much a regression as one that stops.
        assert relative_deviation(g_snap, g_diag) < 1e-12


# ---------------------------------------------------------------------------
# criterion 9: conservative fallbacks fire where they should
# ---------------------------------------------------------------------------

class TestConservativeFallbackEndToEnd(unittest.TestCase):
    """Criterion 9 — end-to-end: the diagnostic reaches the compilation report."""

    def test_unregistered_mixing_primitive_reports_conservative(self):
        spec = _lora_rec_net()
        xs = spec.make_inputs(T, N_IN, seed=0)
        algo = _compiled(spec, _snap(2), xs)
        kinds = [r.kind for r in algo.graph.diagnostics]
        assert DiagnosticKind.SNAP_PATTERN_CONSERVATIVE in kinds
        assert algo.graph.hidden_groups[0].snap.conservative
        record = next(r for r in algo.graph.diagnostics
                      if r.kind is DiagnosticKind.SNAP_PATTERN_CONSERVATIVE)
        assert record.level is DiagnosticLevel.WARNING
        assert 'etp_lora_mm' in record.message

    def test_two_mixing_equations_report_conservative(self):
        spec = om.stacked_tanh_rnn(n_in=N_IN, n_rec=4, seed=0)
        xs = spec.make_inputs(T, N_IN, seed=0)
        algo = _compiled(spec, _snap(2), xs)
        kinds = [r.kind for r in algo.graph.diagnostics]
        assert DiagnosticKind.SNAP_PATTERN_CONSERVATIVE in kinds
        assert any(g.snap.conservative for g in algo.graph.hidden_groups)

    def test_degenerate_pattern_reports_its_own_kind(self):
        spec = om.two_state_rnn(n_in=N_IN, n_rec=3, seed=0)
        xs = spec.make_inputs(T, N_IN, seed=0)
        algo = _compiled(spec, _snap(3), xs)
        kinds = [r.kind for r in algo.graph.diagnostics]
        assert DiagnosticKind.SNAP_PATTERN_DEGENERATE in kinds
        assert DiagnosticKind.SNAP_PATTERN_CONSERVATIVE not in kinds

    def test_a_conservative_pattern_is_still_correct(self):
        """Widening is never *wrong* — it costs memory and moves the rule
        towards within-group RTRL (which is BPTT only when the group's tail is
        position-preserving; see F-31 in the known-limitations list)."""
        spec = _lora_rec_net()
        xs = spec.make_inputs(T, N_IN, seed=0)
        bptt = bptt_param_gradients(spec.factory, xs)
        g = _grads(spec, _snap(2), xs, chunk_size=3)
        assert _rel_etp(g, bptt, spec) < EXACT_RTOL

    def test_a_relabelling_tail_defeats_saturation_and_says_so(self):
        """F-31: the axis presumes the tail preserves positions.

        The rolled model and its control differ by one ``jnp.roll`` and nothing
        else. On the control, saturation is BPTT to round-off and the cheaper
        scopes are visibly approximate — the axis reads as advertised. With the
        roll, the trace's position index no longer denotes the unit whose
        influence it carries, so *every* within-group scope misses, saturated
        included, and buying more neighbours does not close the gap.

        This is not a SnAp defect: `diagonal` and `coupled` are hit the same way
        and never even consult a position graph. What is asserted here is that
        the shortfall is (a) real, (b) not fixable by raising ``n``, and (c)
        warned about rather than silent.
        """
        control = om.rolled_tail_rnn(n_in=N_IN, n_rec=5, roll=0, seed=0)
        rolled = om.rolled_tail_rnn(n_in=N_IN, n_rec=5, roll=1, seed=0)
        xs = control.make_inputs(T, N_IN, seed=1)
        assert_model_is_live(rolled.factory, xs)

        # (a) the control: a dense mixing graph has diameter 1, so n=2 saturates
        ctrl_bptt = bptt_param_gradients(control.factory, xs)
        ctrl_pins = _pins(_compiled(control, _snap(2), xs))
        assert all(ctrl_pins['saturated']) and not any(ctrl_pins['conservative'])
        ctrl_rel = _rel_etp(_grads(control, _snap(2), xs, 2), ctrl_bptt, control)
        assert ctrl_rel < EXACT_RTOL, ctrl_rel

        # (b) the same order on the rolled model misses, and n=4 does not help
        bptt = bptt_param_gradients(rolled.factory, xs)
        rel_2 = _rel_etp(_grads(rolled, _snap(2), xs, 2), bptt, rolled)
        rel_4 = _rel_etp(_grads(rolled, _snap(4), xs, 2), bptt, rolled)
        assert rel_2 > 1e-2, rel_2
        np.testing.assert_allclose(rel_4, rel_2, rtol=1e-5)

        # the coordinate below it is hit too, so this is not SnAp's failure
        rel_coupled = _rel_etp(
            _grads(rolled, lambda m: braintrace.OSTLRecurrent(
                m, vjp_method='multi-step'), xs, 2), bptt, rolled)
        assert rel_coupled > 1e-2, rel_coupled

        # (c) and the compiler says the pattern could not be derived
        algo = _compiled(rolled, _snap(2), xs)
        records = [r for r in algo.graph.diagnostics
                   if r.kind == DiagnosticKind.SNAP_PATTERN_CONSERVATIVE]
        assert records, [r.kind for r in algo.graph.diagnostics]
        assert 'position-preserving' in records[0].message, records[0].message


# ---------------------------------------------------------------------------
# criterion 10: illegal combinations
# ---------------------------------------------------------------------------

class TestIllegalCombinations(unittest.TestCase):
    """Criterion 10 — every rejected pairing names the legal ones."""

    def test_io_factorized_plus_sparse_n(self):
        with pytest.raises(ValueError) as e:
            ETraceConfig(trace_factorization='io_factorized', decay=0.9,
                         recurrence_scope='sparse_n', sparse_n=2)
        assert 'per_param' in str(e.value)

    def test_sparse_n_plus_scalar_leak(self):
        with pytest.raises(ValueError) as e:
            ETraceConfig(recurrence_scope='sparse_n', sparse_n=2,
                         temporal_recursion='scalar_leak', decay=0.9)
        assert 'jacobian' in str(e.value)

    def test_sparse_n_zero(self):
        with pytest.raises(ValueError) as e:
            ETraceConfig(recurrence_scope='sparse_n', sparse_n=0)
        assert 'at least 1' in str(e.value)

    def test_sparse_n_true_is_not_an_order(self):
        with pytest.raises(TypeError) as e:
            ETraceConfig(recurrence_scope='sparse_n', sparse_n=True)
        assert 'bool' in str(e.value).lower()

    def test_sparse_n_without_the_coefficient(self):
        with pytest.raises(ValueError):
            ETraceConfig(recurrence_scope='sparse_n')

    def test_coefficient_without_the_axis_value(self):
        with pytest.raises(ValueError):
            ETraceConfig(recurrence_scope='coupled', sparse_n=3)

    def test_sparse_n_inside_a_descended_scan(self):
        spec = om.snn_scan_rnn(n_rec=4, loops=40, seed=0)
        xs = spec.make_inputs(T, 4, seed=0)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.SnAp(model, n=2)
        with pytest.raises(NotImplementedError) as e:
            algo.compile_graph(xs[0])
        msg = str(e.value)
        assert 'sparse_n' in msg
        assert 'scan' in msg


# ---------------------------------------------------------------------------
# criterion 1: the cheap end must not move
# ---------------------------------------------------------------------------

class TestDiagonalRegression(unittest.TestCase):
    """Criterion 1 — ``recurrence_scope='diagonal'`` allocates exactly what it did.

    The frozen golden values live in ``_algorithm/tests/axis_golden_test.py``;
    what this adds is the structural half: the diagonal path must not acquire a
    snap pattern, a widened trace, or a different allocation just because the
    machinery exists.
    """

    def test_diagonal_allocates_no_snap_pattern(self):
        spec = _ring_spec()
        xs = _ring_inputs()
        algo = _compiled(spec, lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'), xs)
        for g in algo.graph.hidden_groups:
            assert g.snap is None
            assert g.trace_state_width == g.num_state

    def test_diagonal_and_snap_are_distinguishable(self):
        spec = _ring_spec()
        xs = _ring_inputs()
        g_diag = _grads(
            spec, lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'), xs, 3)
        g_snap = _grads(spec, _snap(3), xs, 3)
        assert_gradients_differ(g_diag, g_snap, min_rel=1e-6)
