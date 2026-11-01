# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""The eligibility trace of an ETP output that reaches its hidden state only by
broadcasting (issue #224).

The trace update adds ``df ⊗ x``, and ``df`` has the hidden group's shape, so
the trace has the axes that the broadcast adds. A primitive's ``init_drtrl``
rule sizes it from ``x`` and ``y`` only. Before the fix the trace grew at its
first update, so a loop over the steps (or the substep fold of a descended
scan) failed with a carry type error. The trace is now allocated at the
shape of the instantaneous term. A broadcast along an axis of the weight is a
different case (#231): it is refused at init.
"""
import warnings

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._testing.oracle import (
    bptt_param_gradients,
    chunked_online_param_gradients,
    online_param_gradients_singlestep_naive,
)

_ALGOS = {'D_RTRL': braintrace.D_RTRL, 'EProp': braintrace.EProp}


class _Leaky(brainstate.nn.Module):
    """One leaky state ``V`` of shape ``(2, 4)`` driven by ``y = x @ W``.

    *kind* picks how ``y`` reaches ``V``:

    * ``'shared'``: an input row shared by the batch, ``y`` of shape ``(4,)``;
    * ``'shared-bias'``: the same with a bias;
    * ``'batch-of-one'``: ``y`` of shape ``(1, 4)``;
    * ``'batched'``: ``y`` of shape ``(2, 4)``, no broadcast;
    * ``'batched-bias'``: the same with a bias;
    * ``'explicit'``: the shared row broadcast to the batch before the
      projection, ``y`` of shape ``(2, 4)`` (the reference for ``'shared'``);
    * ``'weight-axis'``: ``W`` of shape ``(3, 1)``, ``y`` of shape ``(2, 1)``
      broadcast along the weight's output axis.
    """

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        n_out = 1 if kind == 'weight-axis' else 4
        with brainstate.random.seed_context(0):
            self.W = brainstate.ParamState(0.5 * brainstate.random.randn(3, n_out))
            if kind in ('shared-bias', 'batched-bias'):
                self.b = brainstate.ParamState(0.3 * brainstate.random.randn(4))
        self.V = brainstate.HiddenState(jnp.zeros((2, 4)))

    def update(self, x):
        kind = self.kind
        if kind in ('shared', 'shared-bias'):
            bias = self.b.value if kind == 'shared-bias' else None
            y = braintrace.matmul(x[0], self.W.value, bias)
        elif kind == 'batch-of-one':
            y = braintrace.matmul(x[:1], self.W.value)
        elif kind == 'explicit':
            y = braintrace.matmul(jnp.broadcast_to(x[0], x.shape), self.W.value)
        elif kind == 'batched-bias':
            y = braintrace.matmul(x, self.W.value, self.b.value)
        else:
            y = braintrace.matmul(x, self.W.value)
        self.V.value = 0.9 * self.V.value + jnp.tanh(y)
        return jax.nn.sigmoid(self.V.value)


_BROADCAST = ['shared', 'shared-bias', 'batch-of-one']


def _inputs(T=6):
    with brainstate.random.seed_context(1):
        return brainstate.random.rand(T, 2, 3)


def _learner(kind, algo, **kw):
    model = _Leaky(kind)
    brainstate.nn.init_all_states(model)
    learner = _ALGOS[algo](model, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        learner.compile_graph(_inputs()[0])  # allocates the traces
    return learner


def _trace_shapes(learner):
    return [jax.tree.map(jnp.shape, s.value) for s in learner.etrace_bwg.values()]


def _flat(grads):
    leaves, _ = jax.tree_util.tree_flatten_with_path(grads)
    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in leaves}


def _close(a, b, tol=1e-12):
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol * max(1.0, float(np.abs(b).max())))


@pytest.mark.parametrize('algo', sorted(_ALGOS))
class TestTheTraceHasItsShapeAtInit:

    @pytest.mark.parametrize('kind', _BROADCAST)
    def test_the_trace_keeps_its_shape_through_an_update(self, kind, algo):
        # main: (3, 4, 1) at init, (2, 3, 4, 1) after the first update
        learner = _learner(kind, algo)
        before = _trace_shapes(learner)
        learner(_inputs()[0])
        assert _trace_shapes(learner) == before
        assert before[0]['weight'] == (2, 3, 4, 1)

    def test_a_trace_without_a_broadcast_is_unchanged(self, algo):
        assert _trace_shapes(_learner('batched', algo))[0]['weight'] == (2, 3, 4, 1)

    @pytest.mark.parametrize('kind', _BROADCAST)
    def test_a_loop_over_the_steps_equals_one_call_per_step(self, kind, algo):
        # main: the loop carry changed type, a TypeError
        xs = _inputs()
        with brainstate.environ.context(precision=64):
            looped = _learner(kind, algo)
            out_loop = brainstate.transform.for_loop(looped, xs)
            called = _learner(kind, algo)
            out_calls = jnp.stack([called(x) for x in xs])
        _close(np.asarray(out_loop), np.asarray(out_calls))
        for a, b in zip(jax.tree.leaves([s.value for s in looped.etrace_bwg.values()]),
                        jax.tree.leaves([s.value for s in called.etrace_bwg.values()])):
            _close(np.asarray(a), np.asarray(b))

    def test_a_reduced_trace_dtype_is_kept(self, algo):
        learner = _learner('shared', algo, trace_dtype=jnp.bfloat16)
        (trace,) = [s.value for s in learner.etrace_bwg.values()]
        assert trace['weight'].shape == (2, 3, 4, 1)
        assert trace['weight'].dtype == jnp.bfloat16


@pytest.mark.parametrize('algo', sorted(_ALGOS))
class TestGradients:

    @pytest.mark.parametrize('kind', _BROADCAST)
    def test_multi_step_in_chunks_matches_bptt(self, kind, algo):
        # chunks of two steps: the total goes through the trace at every boundary
        xs = _inputs()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                bptt = bptt_param_gradients(lambda: _Leaky(kind), xs)
                online = chunked_online_param_gradients(
                    lambda: _Leaky(kind), xs, chunk_size=2,
                    algo_factory=lambda m: _ALGOS[algo](m, vjp_method='multi-step'))
        bptt, online = _flat(bptt), _flat(online)
        assert set(bptt) == set(online)
        for p in bptt:
            assert np.abs(bptt[p]).max() > 1e-3
            _close(online[p], bptt[p])

    def test_single_step_equals_the_explicitly_broadcast_input(self, algo):
        # the shared row and its explicit broadcast define the same model
        xs = _inputs()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                shared = online_param_gradients_singlestep_naive(
                    lambda: _Leaky('shared'), xs, algo_factory=_ALGOS[algo])
                explicit = online_param_gradients_singlestep_naive(
                    lambda: _Leaky('explicit'), xs, algo_factory=_ALGOS[algo])
        shared, explicit = _flat(shared), _flat(explicit)
        for p in explicit:
            _close(shared[p], explicit[p])


@pytest.mark.parametrize('algo', sorted(_ALGOS))
class TestABroadcastAlongAWeightAxis:
    """``W`` of shape ``(3, 1)``: ``y`` of shape ``(2, 1)`` broadcast along the
    weight's output axis into ``V`` of shape ``(2, 4)`` (#231)."""

    def test_it_is_refused_at_init(self, algo):
        # main: a custom VJP shape error at the first gradient
        model = _Leaky('weight-axis')
        brainstate.nn.init_all_states(model)
        learner = _ALGOS[algo](model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(NotImplementedError, match='broadcasting along an axis of the weight') as err:
                learner.compile_graph(_inputs()[0])  # allocates the traces
        message = str(err.value)
        assert "'etp_mm'" in message
        assert "('W',)" in message
        assert 'shape (2, 1)' in message and 'shape (2, 4)' in message
        assert 'issues/231' in message


class _OneUnit(brainstate.nn.Module):
    """A weight with one output unit whose output drives 4 (or 2) units of
    ``V``, through the ETP operation *kind*: a broadcast along an axis of the
    weight."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        r = lambda *s: 0.5 * brainstate.random.randn(*s)
        vshape = (2, 4)
        with brainstate.random.seed_context(0):
            if kind in ('mv', 'mm', 'einsum'):
                self.W = brainstate.ParamState(r(3, 1))
            elif kind == 'grouped':
                self.W = brainstate.ParamState(r(2, 3, 1))
                vshape = (2, 2, 4)
            elif kind == 'conv':
                self.W = brainstate.ParamState(r(1, 1, 3))
                vshape = (2, 2, 3)
            elif kind == 'embedding':
                self.W = brainstate.ParamState(r(5, 1))
            elif kind == 'lora':
                self.B = brainstate.ParamState(r(3, 2))
                self.A = brainstate.ParamState(r(2, 1))
            else:
                import brainevent
                self.csr = brainevent.CSR.fromdense(jnp.array([[0.5], [0.0], [0.3]]))
                self.W = brainstate.ParamState(self.csr.data * 1.0)
        self.V = brainstate.HiddenState(jnp.zeros(vshape))

    def update(self, x):
        kind = self.kind
        if kind == 'mv':
            y = braintrace.matmul(x[0], self.W.value)
        elif kind == 'mm':
            y = braintrace.matmul(x, self.W.value)
        elif kind == 'einsum':
            y = braintrace.einsum('bi,ij->bj', x, self.W.value)
        elif kind == 'grouped':
            y = braintrace.grouped_matmul(jnp.stack([x, 0.5 * x], axis=1), self.W.value)
        elif kind == 'conv':
            y = braintrace.conv(x[:, None, :], self.W.value)
        elif kind == 'embedding':
            y = braintrace.embedding((x[:, 0] * 5).astype(jnp.int32), self.W.value)
        elif kind == 'lora':
            y = braintrace.lora_matmul(x, self.B.value, self.A.value)
        else:
            y = braintrace.sparse_matmul(x, self.W.value, sparse_mat=self.csr)
        self.V.value = 0.9 * self.V.value + jnp.tanh(y)
        return jax.nn.sigmoid(self.V.value)


class TestABroadcastAlongAWeightAxisPerOperation:
    """Where the instantaneous term takes the broadcast, the gradient solve
    cannot reduce the trace (#231); elsewhere the term itself cannot be
    computed. Either is an error when the trace is allocated."""

    @staticmethod
    def _compile(kind):
        model = _OneUnit(kind)
        brainstate.nn.init_all_states(model)
        learner = braintrace.D_RTRL(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            learner.compile_graph(_inputs()[0])

    @pytest.mark.parametrize('kind', ['mv', 'mm', 'grouped', 'conv'])
    def test_the_solve_is_refused(self, kind):
        with pytest.raises(NotImplementedError, match='issues/231'):
            self._compile(kind)

    @pytest.mark.parametrize('kind', ['einsum', 'embedding', 'lora', 'sparse'])
    def test_the_term_is_a_located_error(self, kind):
        with pytest.raises(ValueError, match='cannot be sized') as err:
            self._compile(kind)
        assert 'reaches the hidden state(s)' in str(err.value)


class TestTheSizingIsChecked:
    """Under a broadcast, a rule whose trace neither has the term's shape nor
    grows to it raises when the trace is allocated, naming the operation, the
    weight, the leaf and both shapes."""

    @pytest.mark.parametrize('lead,shape', [
        ((5,), '(5, 3, 4, 1)'),  # does not broadcast to the term
        ((1, 2), '(1, 2, 3, 4, 1)'),  # larger than the term, which broadcasts into it
    ], ids=['incompatible', 'larger-than-the-term'])
    def test_a_wrong_init_is_an_error(self, monkeypatch, lead, shape):
        from braintrace._op import ETP_RULES_INIT_DRTRL
        from braintrace._op.dense import etp_mv_p
        rule = ETP_RULES_INIT_DRTRL[etp_mv_p]

        def wrong(x_var, y_var, weight_vars, num_state, **kw):
            out = rule(x_var, y_var, weight_vars, num_state, **kw)
            return {k: jnp.zeros(lead + v.shape, v.dtype) for k, v in out.items()}

        monkeypatch.setitem(ETP_RULES_INIT_DRTRL, etp_mv_p, wrong)
        model = _Leaky('shared')
        brainstate.nn.init_all_states(model)
        learner = braintrace.D_RTRL(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match='does not broadcast') as err:
                learner.compile_graph(_inputs()[0])  # allocates the traces
        message = str(err.value)
        assert "'etp_mv'" in message and "('W',)" in message and "'weight'" in message
        assert shape in message and '(2, 3, 4, 1)' in message


class TestTheMessages:

    def test_a_solve_that_cannot_take_the_trace_is_refused(self, monkeypatch):
        from braintrace._algorithm import param_dim_vjp

        def broken(*args, **kwargs):
            raise ValueError('the solve cannot take this trace')

        monkeypatch.setattr(param_dim_vjp, 'relation_solve_to_param', broken)
        model = _Leaky('shared')
        brainstate.nn.init_all_states(model)
        learner = braintrace.D_RTRL(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(NotImplementedError, match='issues/231') as err:
                learner.compile_graph(_inputs()[0])
        assert 'cannot take this trace' in str(err.value.__cause__)

    def test_the_reached_states_are_named_with_their_shapes(self):
        from types import SimpleNamespace as NS
        from braintrace._algorithm.param_dim_vjp import _reached_states
        var = lambda *shape: NS(aval=jax.core.ShapedArray(shape, jnp.float32))
        group = NS(hidden_paths=[('a',), ('b',), ('c',)], hidden_outvars=[var(2, 4), var(2, 4, 3), var(2, 4)])
        one = NS(connected_hidden_paths=[('a',), ('c',)])
        both = NS(connected_hidden_paths=[('b',), ('a',)])
        assert _reached_states(one, group) == "the hidden state(s) [('a',), ('c',)] of shape (2, 4)"
        assert _reached_states(both, group) == "the hidden states [('a',), ('b',)] of shapes [(2, 4), (2, 4, 3)]"


class TestTheFastPath:
    """A ``y`` that reaches its state with its own shape keeps the rule's init
    without the abstract evaluation of its term."""

    @staticmethod
    def _spy(monkeypatch, force=None):
        from braintrace._algorithm import param_dim_vjp
        check, taken = param_dim_vjp._reaches_without_broadcast, []

        def spy(relation, group):
            taken.append(check(relation, group) if force is None else force)
            return taken[-1]

        monkeypatch.setattr(param_dim_vjp, '_reaches_without_broadcast', spy)
        return taken

    @staticmethod
    def _run(kind, algo):
        with brainstate.environ.context(precision=64):
            learner = _learner(kind, algo)
            out = brainstate.transform.for_loop(learner, _inputs())
        traces = jax.tree.leaves([s.value for s in learner.etrace_bwg.values()])
        return _trace_shapes(learner), np.asarray(out), [np.asarray(t) for t in traces]

    @pytest.mark.parametrize('algo', sorted(_ALGOS))
    @pytest.mark.parametrize('kind', ['batched', 'batched-bias', 'explicit'])
    def test_it_equals_the_full_path(self, kind, algo, monkeypatch):
        taken = self._spy(monkeypatch)
        fast = self._run(kind, algo)
        assert taken and all(taken)
        self._spy(monkeypatch, force=False)  # the abstract evaluation for every relation
        full = self._run(kind, algo)
        assert fast[0] == full[0]
        np.testing.assert_array_equal(fast[1], full[1])
        assert len(fast[2]) == len(full[2])
        for a, b in zip(fast[2], full[2]):
            assert np.abs(b).max() > 0
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize('kind', _BROADCAST)
    def test_a_broadcast_never_takes_it(self, kind, monkeypatch):
        taken = self._spy(monkeypatch)
        learner = _learner(kind, 'D_RTRL')
        assert taken and not any(taken)
        assert _trace_shapes(learner)[0]['weight'] == (2, 3, 4, 1)


class _ReadTwice(brainstate.nn.Module):
    """One ``W`` read by two ETP operations, whose outputs drive ``V`` of shape ``(2, 4)``.

    * ``'mm+mv'``: a batched read and an unbatched read that broadcasts;
    * ``'mm+mv-explicit'``: the same with the shared row broadcast before the
      projection (the reference for ``'mm+mv'``);
    * ``'mv+mv'``: two unbatched reads that broadcast;
    * ``'mm+mm1'``: a batched read and a batch of one;
    * ``'mm+mv-two-states'``: a batched read into ``V`` and an unbatched read
      into an unbatched state ``U`` of shape ``(4,)``, no broadcast;
    * ``'mv+mv-two-batches'``: two unbatched reads, into ``V`` and into ``U``
      of shape ``(1, 4)``, both by broadcasting.
    """

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        with brainstate.random.seed_context(0):
            self.W = brainstate.ParamState(0.5 * brainstate.random.randn(3, 4))
        self.V = brainstate.HiddenState(jnp.zeros((2, 4)))
        if kind == 'mm+mv-two-states':
            self.U = brainstate.HiddenState(jnp.zeros((4,)))
        elif kind == 'mv+mv-two-batches':
            self.U = brainstate.HiddenState(jnp.zeros((1, 4)))

    def update(self, x):
        W, kind = self.W.value, self.kind
        if kind == 'mm+mv-two-states':
            self.V.value = 0.9 * self.V.value + jnp.tanh(braintrace.matmul(x, W))
            self.U.value = 0.8 * self.U.value + jnp.tanh(braintrace.matmul(x[0], W))
            return jax.nn.sigmoid(jnp.concatenate([self.V.value, self.U.value[None]]))
        if kind == 'mv+mv-two-batches':
            self.V.value = 0.9 * self.V.value + jnp.tanh(braintrace.matmul(x[0], W))
            self.U.value = 0.8 * self.U.value + jnp.tanh(braintrace.matmul(x[1], W))
            return jax.nn.sigmoid(jnp.concatenate([self.V.value, self.U.value]))
        if kind == 'mm+mv':
            y = braintrace.matmul(x, W) + 0.5 * braintrace.matmul(x[0], W)
        elif kind == 'mm+mv-explicit':
            y = braintrace.matmul(x, W) + 0.5 * braintrace.matmul(jnp.broadcast_to(x[0], x.shape), W)
        elif kind == 'mv+mv':
            y = braintrace.matmul(x[0], W) + 0.5 * braintrace.matmul(x[1], W)
        else:
            y = braintrace.matmul(x, W) + 0.5 * braintrace.matmul(x[:1], W)
        self.V.value = 0.9 * self.V.value + jnp.tanh(y)
        return jax.nn.sigmoid(self.V.value)


def _chunked_vs_bptt(make, algo, **kw):
    xs = _inputs()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with brainstate.environ.context(precision=64):
            bptt = bptt_param_gradients(make, xs)
            online = chunked_online_param_gradients(
                make, xs, chunk_size=2,
                algo_factory=lambda m: _ALGOS[algo](m, vjp_method='multi-step', **kw))
    bptt, online = _flat(bptt), _flat(online)
    assert set(bptt) == set(online)
    for p in bptt:
        assert np.abs(bptt[p]).max() > 1e-3
        _close(online[p], bptt[p])


@pytest.mark.parametrize('algo', sorted(_ALGOS))
class TestAWeightReadTwice:
    """The gradient of a weight that two relations read is the sum of their
    gradients, each reduced over its own batch axes."""

    @pytest.mark.parametrize('kind', ['mm+mv', 'mv+mv', 'mm+mm1', 'mv+mv-two-batches'])
    def test_multi_step_in_chunks_matches_bptt(self, kind, algo):
        # 'mm+mv' before: a custom VJP shape error at the first gradient
        _chunked_vs_bptt(lambda: _ReadTwice(kind), algo)

    @pytest.mark.parametrize('fast_solve', [True, False])
    def test_a_batched_and_an_unbatched_state_match_bptt(self, fast_solve, algo):
        # before, fast_solve=False: the unbatched read's gradient was added to
        # every batch row of the batched one and summed with it, twice its value
        _chunked_vs_bptt(lambda: _ReadTwice('mm+mv-two-states'), algo, fast_solve=fast_solve)

    def test_single_step_equals_the_explicitly_broadcast_input(self, algo):
        xs = _inputs()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                shared = online_param_gradients_singlestep_naive(
                    lambda: _ReadTwice('mm+mv'), xs, algo_factory=_ALGOS[algo])
                explicit = online_param_gradients_singlestep_naive(
                    lambda: _ReadTwice('mm+mv-explicit'), xs, algo_factory=_ALGOS[algo])
        shared, explicit = _flat(shared), _flat(explicit)
        for p in explicit:
            assert np.abs(explicit[p]).max() > 1e-3
            _close(shared[p], explicit[p])


def test_the_random_projection_of_a_weight_read_twice_equals_the_explicit_broadcast():
    # UORO projects the gradient of each read; it must reduce each over its own
    # batch axes before adding them, as the parameter-dimension solve does.
    # Adding them first fails with a custom VJP shape error.
    xs = _inputs()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with brainstate.environ.context(precision=64):
            grads = [
                _flat(online_param_gradients_singlestep_naive(
                    lambda k=kind: _ReadTwice(k), xs,
                    algo_factory=lambda m: braintrace.UORO(m, projection_key=0)))
                for kind in ('mm+mv', 'mm+mv-explicit')
            ]
    shared, explicit = grads
    for p in explicit:
        assert np.abs(explicit[p]).max() > 1e-3
        _close(shared[p], explicit[p])


class _WeightAndBias(brainstate.nn.Module):
    """One dict ``ParamState`` ``{'weight': (3, 4), 'bias': (4,)}``, as in
    ``brainstate.nn.Linear``, whose leaves are read by two unbatched operations
    that broadcast into ``V`` of shape ``(2, 4)``; each relation covers one leaf."""

    def __init__(self):
        super().__init__()
        with brainstate.random.seed_context(0):
            self.P = brainstate.ParamState({
                'weight': 0.5 * brainstate.random.randn(3, 4),
                'bias': 0.3 * brainstate.random.randn(4),
            })
        self.V = brainstate.HiddenState(jnp.zeros((2, 4)))

    def update(self, x):
        P = self.P.value
        y = braintrace.matmul(x[0], P['weight']) + braintrace.element_wise(P['bias']) * (1.0 + x[:, :1])
        self.V.value = 0.9 * self.V.value + jnp.tanh(y)
        return jax.nn.sigmoid(self.V.value)


@pytest.mark.parametrize('algo', sorted(_ALGOS))
def test_the_leaves_of_one_param_state_read_by_two_broadcasts_match_bptt(algo):
    # the reduction of a relation's gradient is chosen from the leaves that
    # relation covers; requiring every leaf of the state to have grown refuses
    # this model with a false #231 error
    _chunked_vs_bptt(_WeightAndBias, algo)


class TestTheBatchReductionPerRead:
    """``add_param_grad`` and ``reduce_param_batch_axes``, shared by the
    parameter-dimension solve and the random-projection engine."""

    @staticmethod
    def _reduce(*contributions):
        from braintrace._algorithm.param_dim_vjp import add_param_grad, reduce_param_batch_axes
        weights = {('W',): jnp.zeros((3, 4))}
        sums = {}
        for grad, batched, folded in contributions:
            add_param_grad(sums, {('W',): grad}, weights, batched=batched, folded=folded)
        return np.asarray(reduce_param_batch_axes(sums, weights)[('W',)])

    def test_reads_of_different_kinds_are_reduced_before_they_are_added(self):
        g = np.random.RandomState(0).randn(2, 3, 4)
        h = np.random.RandomState(1).randn(3, 4)
        # a folded read and an unbatched read that broadcast into the batch
        np.testing.assert_allclose(self._reduce((h, True, True), (g, False, False)), h + g.sum(0), rtol=1e-12)
        # a batched read that keeps its batch axis and an unbatched read
        np.testing.assert_allclose(self._reduce((g, True, False), (h, False, False)), g.sum(0) + h, rtol=1e-12)
        # two unbatched reads, one broadcast into the batch and one not
        np.testing.assert_allclose(self._reduce((h, False, False), (g, False, False)), h + g.sum(0), rtol=1e-12)
        # two unbatched reads broadcast into batches of two and of one
        np.testing.assert_allclose(self._reduce((g, False, False), (h[None], False, False)), g.sum(0) + h, rtol=1e-12)

    def test_reads_of_one_kind_are_added_before_they_are_reduced(self):
        g = np.random.RandomState(0).randn(2, 3, 4)
        h = np.random.RandomState(1).randn(2, 3, 4)
        np.testing.assert_array_equal(self._reduce((g, True, False), (h, True, False)), (g + h).sum(0))
        np.testing.assert_array_equal(self._reduce((g, False, False), (h, False, False)), (g + h).sum(0))


class _Gain(brainstate.nn.Module):
    """``V`` of shape ``(2, 4)`` driven by an ``etp_elemwise`` gain ``g`` of
    shape *shape* times a batched drive."""

    def __init__(self, shape):
        super().__init__()
        with brainstate.random.seed_context(0):
            self.g = brainstate.ParamState(1.0 + 0.5 * brainstate.random.randn(*shape))
        self.V = brainstate.HiddenState(jnp.zeros((2, 4)))

    def update(self, x):
        y = braintrace.element_wise(self.g.value) * (1.0 + x[:, :1])
        self.V.value = 0.9 * self.V.value + jnp.tanh(y)
        return jax.nn.sigmoid(self.V.value)


@pytest.mark.parametrize('algo', sorted(_ALGOS))
class TestAnElementwiseGain:
    """An ``etp_elemwise`` trace is sized from the hidden group, so it never
    grows; a gain broadcast along one of its own axes is refused all the same
    (#231)."""

    @pytest.mark.parametrize('shape', [(1,), (1, 4)], ids=str)
    def test_a_broadcast_along_an_axis_of_the_gain_is_refused_at_init(self, shape, algo):
        # before: a custom VJP shape error at the first gradient
        model = _Gain(shape)
        brainstate.nn.init_all_states(model)
        learner = _ALGOS[algo](model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(NotImplementedError, match='broadcasting along an axis of the weight') as err:
                learner.compile_graph(_inputs()[0])  # allocates the traces
        message = str(err.value)
        assert "'etp_elemwise'" in message and "('g',)" in message
        assert f'shape {shape}' in message
        assert "the hidden state(s) [('V',)] of shape (2, 4)" in message
        assert 'issues/231' in message

    @pytest.mark.parametrize('shape', [(), (4,), (2, 4)], ids=str)
    def test_a_gain_that_spans_its_axes_matches_bptt(self, shape, algo):
        _chunked_vs_bptt(lambda: _Gain(shape), algo)


class TestATermThatCannotBeComputed:

    def test_it_is_a_located_error_at_init(self):
        # an einsum whose positions collapse into the same hidden units: the
        # hidden-shaped df cannot carry the per-position term
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(0.1 * jnp.ones((3, 3)))
                self.h = brainstate.HiddenState(jnp.zeros((1, 3)))

            def update(self, x):
                xs = jnp.stack([self.h.value * (0.5 + t) for t in range(2)], axis=1)
                rec = braintrace.einsum('btk,kn->btn', xs, self.w.value)
                self.h.value = jax.nn.tanh(x[:1] + rec.sum(axis=1))
                return self.h.value

        model = Net()
        brainstate.nn.init_all_states(model)
        learner = braintrace.D_RTRL(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match='cannot be sized') as err:
                learner.compile_graph(_inputs()[0])
        message = str(err.value)
        assert "'etp_einsum'" in message and "('w',)" in message
        assert 'shape (1, 2, 3)' in message and "the hidden state(s) [('h',)] of shape (1, 3)" in message
        assert err.value.__cause__ is not None


@pytest.fixture
def _x64():
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', False)


@pytest.mark.parametrize('algo', sorted(_ALGOS))
def test_a_grown_trace_keeps_the_model_dtype_under_x64(algo, _x64):
    # a float32 model with x64 on: a default-dtype allocation would be float64
    class Model(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.W = brainstate.ParamState(jnp.asarray(np.random.RandomState(0).randn(3, 4), jnp.float32))
            self.V = brainstate.HiddenState(jnp.zeros((2, 4), jnp.float32))

        def update(self, x):
            self.V.value = 0.9 * self.V.value + jnp.tanh(braintrace.matmul(x[0], self.W.value))
            return jax.nn.sigmoid(self.V.value)

    xs = jnp.asarray(np.random.RandomState(1).rand(4, 2, 3), jnp.float32)
    model = Model()
    brainstate.nn.init_all_states(model)
    learner = _ALGOS[algo](model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        learner.compile_graph(xs[0])
        (trace,) = [s.value['weight'] for s in learner.etrace_bwg.values()]
        assert trace.shape == (2, 3, 4, 1) and trace.dtype == jnp.float32
        brainstate.transform.for_loop(learner, xs)
        (trace,) = [s.value['weight'] for s in learner.etrace_bwg.values()]
        assert trace.dtype == jnp.float32
        grads = brainstate.transform.grad(
            lambda x: (learner(x) ** 2).sum(), model.states(brainstate.ParamState))(xs[0])
    (g,) = jax.tree.leaves(grads)
    assert g.shape == (3, 4) and g.dtype == jnp.float32
