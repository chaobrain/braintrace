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
            if kind == 'shared-bias':
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


class TestTheSizingIsChecked:
    """A rule whose trace neither has the term's shape nor grows to it by
    broadcasting raises when the trace is allocated, naming the operation,
    the weight, the leaf and both shapes."""

    @pytest.mark.parametrize('kind,lead,shape', [
        ('shared', (5,), '(5, 3, 4, 1)'),  # does not broadcast to the term
        ('batched', (1,), '(1, 2, 3, 4, 1)'),  # larger than the term, which broadcasts into it
    ], ids=['incompatible', 'larger-than-the-term'])
    def test_a_wrong_init_is_an_error(self, monkeypatch, kind, lead, shape):
        from braintrace._op import ETP_RULES_INIT_DRTRL
        rule_of = {'shared': 'etp_mv', 'batched': 'etp_mm'}[kind]
        prim = next(p for p in ETP_RULES_INIT_DRTRL if p.name == rule_of)
        rule = ETP_RULES_INIT_DRTRL[prim]

        def wrong(x_var, y_var, weight_vars, num_state, **kw):
            out = rule(x_var, y_var, weight_vars, num_state, **kw)
            return {k: jnp.zeros(lead + v.shape, v.dtype) for k, v in out.items()}

        monkeypatch.setitem(ETP_RULES_INIT_DRTRL, prim, wrong)
        model = _Leaky(kind)
        brainstate.nn.init_all_states(model)
        learner = braintrace.D_RTRL(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match='does not broadcast') as err:
                learner.compile_graph(_inputs()[0])  # allocates the traces
        message = str(err.value)
        assert f"'{rule_of}'" in message and "('W',)" in message and "'weight'" in message
        assert shape in message and '(2, 3, 4, 1)' in message
