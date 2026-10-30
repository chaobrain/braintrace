# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""The leaves of a multi-leaf ``ParamState`` read inside a descended scan
body (issue #225).

The body reads the leaves it uses as scan constants, in the order of first
use. Each one must keep the leaf position of the ``ParamState`` leaf it stands
for (``jax.tree.leaves`` order, so dict keys sorted); before the fix the body
numbered them in scan-constant order.
"""
import warnings

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace import ControlFlowPolicy
from braintrace._compatible_imports import new_var
from braintrace._compiler.diagnostics import DiagnosticKind, diagnostic_context
from braintrace._compiler.hid_param_op import _resolve_weight_leaf_idx

UNROLL_POLICY = ControlFlowPolicy(scan_unroll_limit=64)

_ALGOS = {
    'd_rtrl': lambda m, kw: braintrace.D_RTRL(m, **kw),
    'eprop': lambda m, kw: braintrace.EProp(m, **kw),
}


class _TwoState(brainstate.nn.Module):
    """Two hidden states integrated in ``k`` Euler substeps of one
    ``lax.scan``, with a trainable time constant ``tau`` inside the body, so
    that a scan of ``k > 16`` substeps is descended."""

    def __init__(self, k=17, n_in=3, n=4):
        super().__init__()
        self.k = k
        with brainstate.random.seed_context(0):
            self.W = brainstate.ParamState(0.5 * brainstate.random.randn(n_in, n))
        self.tau = brainstate.ParamState(jnp.full(n, 2.0))
        self.V = brainstate.HiddenState(jnp.zeros(n))
        self.w = brainstate.HiddenState(jnp.zeros(n))

    def substep(self, c, I):
        V, w = c
        h = 1.0 / self.k
        tau = braintrace.element_wise(self.tau.value)
        return (V + h / tau * (-V + jnp.tanh(2 * V) - w + I), w + h * (0.5 * V - w)), None

    def scan(self, body):
        (self.V.value, self.w.value), _ = jax.lax.scan(
            body, (self.V.value, self.w.value), None, length=self.k)
        return jax.nn.sigmoid(self.V.value)


class _LinearWithBiasInBody(_TwoState):
    """The issue model: ``braintrace.nn.Linear`` with a bias called in the
    body. The body uses ``weight`` before ``bias``; the leaf order is
    ``bias``, ``weight``."""

    def __init__(self, k=17):
        super().__init__(k)
        del self.W  # the projection is the Linear
        with brainstate.random.seed_context(3):
            self.lin = braintrace.nn.Linear(3, 4, b_init=braintools.init.Normal(0.0, 0.3))

    def update(self, x):
        return self.scan(lambda c, _: self.substep(c, self.lin(x)))


class _LinearWithBiasBeforeTheScan(_LinearWithBiasInBody):
    """The ``Linear`` called before the scan, which reads its output as a loop
    constant: the relation is lifted into the scan (#211) and resolved at top
    level, where the leaves come in leaf order."""

    def update(self, x):
        I = self.lin(x)
        return self.scan(lambda c, _: self.substep(c, I))


class _Gains(_TwoState):
    """Gains in one ``ParamState`` of equal-shape leaves ``{'a', 'b', 'c'}``,
    each read in the body through its own ``element_wise``, in the order
    *reads* (a key may repeat, and a key may be left out)."""

    def __init__(self, reads, k=17, n=4):
        super().__init__(k, n=n)
        self.reads = reads
        self.g = brainstate.ParamState(
            {'a': jnp.full(n, 1.2), 'b': jnp.full(n, 0.3), 'c': jnp.full(n, -0.4)})

    def update(self, x):
        def body(c, _):
            I = braintrace.matmul(x, self.W.value)
            e = {key: 0.0 for key in 'abc'}
            for i, key in enumerate(self.reads):  # trace-time: one ETP op per read
                e[key] = e[key] + (1.0 + 0.5 * i) * braintrace.element_wise(self.g.value[key])
            return self.substep(c, (1.0 + e['a']) * I + e['b'] + 0.5 * jnp.tanh(e['c']))

        return self.scan(body)


class _WeightFnChain(_TwoState):
    """Rules that use the leaf's value: the body reads ``g['b']`` through an
    ``exp`` ``weight_fn``, ``g['c']`` through a mask, then ``g['b']`` again.
    The leaf order is ``a``, ``b``, ``c``."""

    def __init__(self, k=17, n=4):
        super().__init__(k, n=n)
        ramp = 0.1 * jnp.arange(n)
        self.g = brainstate.ParamState(
            {'c': 0.3 + ramp, 'b': -0.5 + ramp, 'a': 1.2 + ramp})

    def update(self, x):
        mask = jnp.array([1.0, 0.0, 1.0, 1.0])

        def body(c, _):
            g = self.g.value
            I = braintrace.matmul(x, self.W.value)
            e_exp = braintrace.element_wise(g['b'], weight_fn=jnp.exp)
            e_mask = braintrace.element_wise(g['c'] * mask)
            e_b = braintrace.element_wise(g['b'])
            return self.substep(c, I + e_exp * I + e_mask + 0.5 * jnp.tanh(e_b))

        return self.scan(body)


class _MaskedLinearWithBiasInBody(_LinearWithBiasInBody):
    """A masked ``Linear`` with a bias in the body: its rule applies the mask
    to the weight's value."""

    def __init__(self, k=17):
        super().__init__(k)
        with brainstate.random.seed_context(3):
            self.lin = braintrace.nn.Linear(
                3, 4, b_init=braintools.init.Normal(0.0, 0.3),
                w_mask=jnp.array([[1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]))


def _inputs(T=6, n_in=3):
    with brainstate.random.seed_context(1):
        return brainstate.random.rand(T, n_in)


_FALLBACK = 'Could not resolve weight leaf'  # PYTREE_WEIGHT_LEAF_AMBIGUOUS


def _assert_no_fallback(caught):
    # every leaf read in these models resolves: the fallback to leaf 0 must not run
    assert not [w for w in caught if _FALLBACK in str(w.message)]


def _online(make, algo, policy=None):
    from braintrace._testing.oracle import online_param_gradients_singlestep_naive
    kw = {} if policy is None else dict(control_flow=policy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        g = online_param_gradients_singlestep_naive(
            make, _inputs(), algo_factory=lambda m: _ALGOS[algo](m, kw))
    _assert_no_fallback(caught)
    leaves, _ = jax.tree_util.tree_flatten_with_path(g)
    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in leaves}


def _descended_and_unrolled(make, algo):
    with brainstate.environ.context(precision=64):
        descended = _online(make, algo)
        unrolled = _online(make, algo, UNROLL_POLICY)
    assert set(descended) == set(unrolled)
    return descended, unrolled


def _close(a, b, tol=1e-12):
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol * max(1.0, float(np.abs(b).max())))


def _check_gains(reads, algo):
    descended, unrolled = _descended_and_unrolled(lambda: _Gains(reads), algo)
    for p in descended:
        _close(descended[p], unrolled[p])
    for key in 'abc':
        g = unrolled[f"[('g',)]['{key}']"]
        if key in reads:
            assert np.abs(g).max() > 1e-3
        else:  # a leaf the body does not read gets no credit
            np.testing.assert_array_equal(g, 0.0)


def _descended_relations(make):
    from braintrace._compiler.graph import compile_etrace_graph
    model = make()
    brainstate.nn.init_all_states(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        graph = compile_etrace_graph(model, jnp.zeros(3))
    _assert_no_fallback(caught)
    return [r for r in graph.hidden_param_op_relations if r.control_flow_context is not None]


@pytest.mark.parametrize('algo', sorted(_ALGOS))
class TestLeavesReadInTheBody:

    @pytest.mark.parametrize('make', [_LinearWithBiasInBody, _LinearWithBiasBeforeTheScan],
                             ids=['in-the-body', 'before-the-scan'])
    def test_a_linear_with_a_bias_matches_the_unrolled_scan(self, algo, make):
        # main: in the body, the bias got the weight's leaf position, a custom
        # VJP type error
        descended, unrolled = _descended_and_unrolled(make, algo)
        for p in descended:
            _close(descended[p], unrolled[p])
        assert any('bias' in p for p in unrolled)
        assert all(np.abs(unrolled[p]).max() > 1e-3 for p in unrolled if 'lin' in p)

    @pytest.mark.parametrize('make', [_WeightFnChain, _MaskedLinearWithBiasInBody],
                             ids=['weight-fn-and-mask-out-of-order', 'masked-linear-with-a-bias'])
    def test_a_rule_that_uses_the_leaf_value_matches_the_unrolled_scan(self, algo, make):
        # the rules read the value of the leaf they are given: a wrong leaf is
        # 176 % off (g['b'] through exp) or a VJP shape error (the mask)
        descended, unrolled = _descended_and_unrolled(make, algo)
        for p in descended:
            _close(descended[p], unrolled[p])
        read = [p for p in unrolled if "['b']" in p or "['c']" in p or 'lin' in p]
        assert read and all(np.abs(unrolled[p]).max() > 1e-3 for p in read)

    @pytest.mark.parametrize('reads', ['ba', 'cab', 'abc'],
                             ids=['equal-shapes-out-of-order', 'three-leaves-out-of-order',
                                  'in-leaf-order'])
    def test_equal_shape_leaves_match_the_unrolled_scan(self, algo, reads):
        # main: 'ba' and 'cab' swapped the gains silently (g['a'] 158 % off)
        _check_gains(reads, algo)

    @pytest.mark.parametrize('reads', ['b', 'c', 'bab'],
                             ids=['only-b', 'only-c', 'one-leaf-read-twice'])
    def test_leaves_read_partly_or_twice_match_the_unrolled_scan(self, algo, reads):
        # main: with one leaf read, the body listed one leaf and gave it
        # position 0, so g['b'] or g['c'] credited g['a']
        _check_gains(reads, algo)


class TestLeafPositions:

    def test_each_relation_names_the_leaf_it_reads(self):
        # the body reads c, a, b (relations come in body order); main: 0, 1, 2
        relations = _descended_relations(lambda: _Gains('cab'))
        leaves = [r.trainable_leaf_indices[key] for r in relations
                  for key, path in r.trainable_paths.items() if path == ('g',)]
        assert leaves == [2, 0, 1]

    def test_a_linear_in_the_body_names_bias_and_weight(self):
        relations = _descended_relations(_LinearWithBiasInBody)
        (r,) = [r for r in relations if ('lin', 'weight') in r.trainable_paths.values()]
        names = {key: r.trainable_leaf_indices[key] for key in r.trainable_paths}
        # jax.tree.leaves({'bias', 'weight'}): bias is leaf 0, weight leaf 1
        assert sorted(names.values()) == [0, 1]
        assert names[next(k for k in names if 'bias' in str(k))] == 0


class TestResolveWeightLeafIdx:
    """:func:`_resolve_weight_leaf_idx` with a leaf position per variable."""

    @staticmethod
    def _vars(n):
        aval = jax.core.ShapedArray((4,), jnp.float32)
        return [new_var('', aval) for _ in range(n)]

    def test_the_default_is_the_position_in_the_list(self):
        a, b = self._vars(2)
        assert _resolve_weight_leaf_idx(b, ('g',), {}, {('g',): [a, b]}) == 1
        assert _resolve_weight_leaf_idx(a, ('g',), {}, {('g',): [a]}) == 0

    def test_a_leaf_index_overrides_the_list_order(self):
        b, a = self._vars(2)
        listed = {('g',): [b, a]}  # first use: b, then a
        index = {a: 0, b: 1}
        assert _resolve_weight_leaf_idx(b, ('g',), {}, listed, index) == 1
        assert _resolve_weight_leaf_idx(a, ('g',), {}, listed, index) == 0

    def test_a_single_listed_leaf_keeps_its_position(self):
        (b,) = self._vars(1)
        assert _resolve_weight_leaf_idx(b, ('g',), {}, {('g',): [b]}, {b: 1}) == 1

    def test_two_variables_of_one_leaf(self):
        a1, a2, b = self._vars(3)
        listed = {('g',): [b, a1, a2]}
        index = {a1: 0, a2: 0, b: 1}
        assert [_resolve_weight_leaf_idx(v, ('g',), {}, listed, index)
                for v in (a1, a2, b)] == [0, 0, 1]

    def test_a_leaf_of_another_param_state_is_not_taken(self):
        # one trainable input computed from leaves of two ParamStates (a
        # documented limitation: one of them gets the credit); the walk meets
        # the other ParamState's leaf first, and must not return its position
        from types import SimpleNamespace
        a, b, other, w = self._vars(4)
        producers = {w: SimpleNamespace(invars=[b, other])}
        index = {a: 0, b: 1, other: 2}
        assert _resolve_weight_leaf_idx(w, ('g',), producers, {('g',): [b, a]}, index) == 1

    def test_an_unresolved_variable_warns_and_falls_back_to_zero(self):
        a, b, other = self._vars(3)
        with diagnostic_context() as reporter, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got = _resolve_weight_leaf_idx(other, ('g',), {}, {('g',): [b, a]}, {a: 0, b: 1})
        assert got == 0
        assert [r.kind for r in reporter.records()] == [DiagnosticKind.PYTREE_WEIGHT_LEAF_AMBIGUOUS]
