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

"""Comprehensive tests for ``param_dim_vjp.ParamDimVjpAlgorithm`` (D-RTRL).

The parameter-dimension VJP algorithm is an *exact* online estimator: its
total-sequence gradient via the multi-step VJP path must reproduce BPTT
element-wise. Coverage:

* construction & validation (vjp_method, fast_solve, trace_dtype, D_RTRL alias);
* eligibility-trace state lifecycle (compile / init / reset / get_etrace_of);
* forward / update mechanics (shapes, running index, trace evolution);
* gradient correctness — exact match to BPTT across the model zoo, and the
  fast-solve path numerically identical to the legacy nested-vmap path;
* reduced-precision (``trace_dtype``) storage; and
* the pure module helpers (``_cast_to_dtype``, ``_remove_units``).
"""

import brainstate
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
import brainunit as u

import braintrace
from braintrace._algorithm import EligibilityTrace, oracle
from braintrace._algorithm import oracle_models as om
from braintrace._algorithm.d_rtrl import D_RTRL
from braintrace._algorithm.param_dim_vjp import (
    ParamDimVjpAlgorithm,
    _cast_to_dtype,
    _remove_units,
)

# Model factories whose ETP weights D-RTRL must learn exactly (see oracle_models).
EXACT_MODELS = {
    'tanh_rnn': om.tanh_rnn,
    'leaky_linear': om.leaky_linear,
    'stacked_tanh_rnn': om.stacked_tanh_rnn,
    'two_state_rnn': om.two_state_rnn,
}

RNN_CELLS = [
    braintrace.nn.GRUCell,
    braintrace.nn.LSTMCell,
    braintrace.nn.MGUCell,
    braintrace.nn.MinimalRNNCell,
]


def _build(spec_factory, *, batch_size=1):
    """Instantiate and initialise a model from an oracle ModelSpec factory."""
    model = spec_factory().factory()
    brainstate.nn.init_all_states(model, batch_size=batch_size)
    return model


def _compiled(model, *, x=None, **kwargs):
    """Build a compiled ParamDimVjpAlgorithm over ``model``."""
    x = jnp.ones((3,), dtype='float32') if x is None else x
    algo = ParamDimVjpAlgorithm(model, **kwargs)
    algo.compile_graph(x)
    algo.init_etrace_state()
    return algo


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_defaults(self):
        algo = ParamDimVjpAlgorithm(_build(om.tanh_rnn))
        assert algo.vjp_method == 'single-step'
        assert algo.fast_solve is True
        assert algo.trace_dtype is None
        assert algo.is_compiled is False

    @pytest.mark.parametrize('method', ['single-step', 'multi-step'])
    def test_vjp_method_stored(self, method):
        algo = ParamDimVjpAlgorithm(_build(om.tanh_rnn), vjp_method=method)
        assert algo.vjp_method == method

    def test_invalid_vjp_method_raises(self):
        with pytest.raises((AssertionError, ValueError)):
            ParamDimVjpAlgorithm(_build(om.tanh_rnn), vjp_method='nonsense')

    @pytest.mark.parametrize('flag', [True, False])
    def test_fast_solve_stored(self, flag):
        algo = ParamDimVjpAlgorithm(_build(om.tanh_rnn), fast_solve=flag)
        assert algo.fast_solve is flag

    def test_trace_dtype_stored(self):
        algo = ParamDimVjpAlgorithm(_build(om.tanh_rnn), trace_dtype=jnp.bfloat16)
        assert algo.trace_dtype == jnp.bfloat16

    def test_d_rtrl_is_param_dim_subclass(self):
        assert issubclass(D_RTRL, ParamDimVjpAlgorithm)
        algo = D_RTRL(_build(om.tanh_rnn))
        assert isinstance(algo, ParamDimVjpAlgorithm)


# ---------------------------------------------------------------------------
# Eligibility-trace state lifecycle
# ---------------------------------------------------------------------------

class TestStateLifecycle:

    def test_compile_sets_flag(self):
        algo = ParamDimVjpAlgorithm(_build(om.tanh_rnn))
        assert algo.is_compiled is False
        algo.compile_graph(jnp.ones((3,)))
        assert algo.is_compiled is True

    def test_etrace_states_are_zero_initialised(self):
        algo = _compiled(_build(om.tanh_rnn))
        assert len(algo.etrace_bwg) >= 1
        for state in algo.etrace_bwg.values():
            assert isinstance(state, EligibilityTrace)
            for leaf in jax.tree.leaves(state.value):
                npt.assert_array_equal(u.get_mantissa(leaf), jnp.zeros_like(u.get_mantissa(leaf)))

    def test_reset_zeros_traces_and_index(self):
        algo = _compiled(_build(om.tanh_rnn))
        algo.update(jnp.ones((3,)))
        assert algo.running_index.value >= 1
        algo.reset_state(batch_size=1)
        assert int(algo.running_index.value) == 0
        for state in algo.etrace_bwg.values():
            for leaf in jax.tree.leaves(state.value):
                npt.assert_array_equal(u.get_mantissa(leaf), jnp.zeros_like(u.get_mantissa(leaf)))

    def test_get_etrace_of_known_weight(self):
        model = _build(om.tanh_rnn)
        algo = _compiled(model)
        traces = algo.get_etrace_of(model.w)  # ETP recurrent weight
        assert isinstance(traces, dict)
        assert len(traces) >= 1

    def test_get_etrace_of_plain_weight_raises(self):
        model = _build(om.tanh_rnn)
        algo = _compiled(model)
        with pytest.raises(ValueError):
            algo.get_etrace_of(model.win)  # plain projection, not an ETP relation

    def test_get_etrace_of_before_compile_raises(self):
        model = _build(om.tanh_rnn)
        algo = ParamDimVjpAlgorithm(model)
        with pytest.raises(ValueError):
            algo.get_etrace_of(model.w)


# ---------------------------------------------------------------------------
# Forward / update mechanics
# ---------------------------------------------------------------------------

class TestForwardUpdate:

    def test_single_step_output_shape(self):
        algo = _compiled(_build(om.tanh_rnn))
        out = algo(jnp.ones((3,)))
        assert out.shape == (1, 4)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_running_index_increments(self):
        algo = _compiled(_build(om.tanh_rnn))
        assert int(algo.running_index.value) == 0
        algo(jnp.ones((3,)))
        algo(jnp.ones((3,)))
        assert int(algo.running_index.value) == 2

    def test_multi_step_output_leading_dim(self):
        model = _build(om.tanh_rnn)
        inputs = brainstate.random.randn(6, 3)
        algo = ParamDimVjpAlgorithm(model, vjp_method='multi-step')
        algo.compile_graph(inputs[0])
        algo.init_etrace_state()
        outs = algo(braintrace.MultiStepData(inputs))
        assert outs.shape[0] == 6
        assert bool(jnp.all(jnp.isfinite(outs)))

    def test_traces_change_after_update(self):
        algo = _compiled(_build(om.tanh_rnn))
        # Warm up once: the recurrent weight's presynaptic input is the hidden
        # state, which is zero on the first step (so the trace stays zero until
        # the hidden state becomes non-zero).
        algo(jnp.ones((3,)))
        before = [u.get_mantissa(jax.tree.leaves(v.value)[0]).copy()
                  for v in algo.etrace_bwg.values()]
        algo(jnp.ones((3,)))
        after = [u.get_mantissa(jax.tree.leaves(v.value)[0])
                 for v in algo.etrace_bwg.values()]
        assert any(not bool(jnp.allclose(b, a)) for b, a in zip(before, after))


# ---------------------------------------------------------------------------
# Gradient correctness — the exact-algorithm contract
# ---------------------------------------------------------------------------

class TestGradientCorrectness:

    @pytest.mark.parametrize('name', list(EXACT_MODELS))
    def test_multistep_matches_bptt_exactly(self, name):
        spec = EXACT_MODELS[name]()
        inputs = brainstate.random.randn(8, 3)
        bptt = oracle.bptt_param_gradients(spec.factory, inputs)
        approx = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: ParamDimVjpAlgorithm(m, vjp_method='multi-step'),
        )
        oracle.assert_param_gradients_close(
            approx, bptt, atol=1e-5, rtol=1e-5, keys=spec.etp_param_keys
        )

    @pytest.mark.parametrize('name', list(EXACT_MODELS))
    def test_fast_solve_matches_legacy_path(self, name):
        spec = EXACT_MODELS[name]()
        inputs = brainstate.random.randn(8, 3)
        fast = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: ParamDimVjpAlgorithm(m, vjp_method='multi-step', fast_solve=True),
        )
        legacy = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: ParamDimVjpAlgorithm(m, vjp_method='multi-step', fast_solve=False),
        )
        oracle.assert_param_gradients_close(
            fast, legacy, atol=1e-6, rtol=1e-6, keys=spec.etp_param_keys
        )

    def test_d_rtrl_alias_matches_base_class(self):
        spec = om.tanh_rnn()
        inputs = brainstate.random.randn(8, 3)
        base = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: ParamDimVjpAlgorithm(m, vjp_method='multi-step'),
        )
        alias = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: D_RTRL(m, vjp_method='multi-step'),
        )
        oracle.assert_param_gradients_close(alias, base, atol=1e-6, rtol=1e-6)

    def test_trace_dtype_bf16_stays_directionally_aligned(self):
        spec = om.tanh_rnn()
        inputs = brainstate.random.randn(8, 3)
        bptt = oracle.bptt_param_gradients(spec.factory, inputs)
        bf16 = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: ParamDimVjpAlgorithm(
                m, vjp_method='multi-step', trace_dtype=jnp.bfloat16),
        )
        oracle.assert_direction_aligned(
            bf16, bptt, min_cosine=0.9, min_sign_agreement=0.8, keys=spec.etp_param_keys
        )


# ---------------------------------------------------------------------------
# RNN-cell sweep — finite gradients across the public cell zoo
# ---------------------------------------------------------------------------

class TestRNNCells:

    @pytest.mark.parametrize('cls', RNN_CELLS)
    def test_cell_gradients_are_finite(self, cls):
        model = cls(4, 5)
        brainstate.nn.init_all_states(model)
        algo = ParamDimVjpAlgorithm(model)
        x = brainstate.random.rand(4)
        algo.compile_graph(x)
        algo.init_etrace_state()

        grads = brainstate.transform.grad(
            lambda inp: algo(inp).sum(),
            model.states(brainstate.ParamState),
        )(x)
        leaves = jax.tree.leaves(grads)
        assert leaves
        for leaf in leaves:
            assert bool(jnp.all(jnp.isfinite(u.get_mantissa(leaf))))


# ---------------------------------------------------------------------------
# Pure module helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_cast_to_dtype_none_is_noop(self):
        tree = {'w': jnp.ones((2, 3), dtype=jnp.float32)}
        out = _cast_to_dtype(tree, None)
        assert jax.tree.leaves(out)[0].dtype == jnp.float32

    def test_cast_to_dtype_casts_every_leaf(self):
        tree = {'w': jnp.ones((2, 3), dtype=jnp.float32), 'b': jnp.zeros((3,), dtype=jnp.float32)}
        out = _cast_to_dtype(tree, jnp.bfloat16)
        for leaf in jax.tree.leaves(out):
            assert u.get_mantissa(leaf).dtype == jnp.bfloat16

    def test_cast_to_dtype_handles_quantity_leaf(self):
        # "unit-safe" means casting a unit-carrying leaf does not crash; the
        # mantissa is still cast to the requested dtype.
        tree = {'w': jnp.ones((2,)) * u.mV}
        out = _cast_to_dtype(tree, jnp.bfloat16)
        leaf = jax.tree.leaves(out, is_leaf=u.math.is_quantity)[0]
        assert u.get_mantissa(leaf).dtype == jnp.bfloat16

    def test_remove_units_roundtrip_with_units(self):
        tree = {'w': jnp.arange(6.0).reshape(2, 3) * u.mV}
        unitless, restore = _remove_units(tree)
        restored = restore(unitless)
        npt.assert_array_equal(u.get_mantissa(restored['w']), u.get_mantissa(tree['w']))
        assert u.get_unit(restored['w']) == u.get_unit(tree['w'])

    def test_remove_units_roundtrip_plain(self):
        tree = {'w': jnp.arange(6.0).reshape(2, 3)}
        unitless, restore = _remove_units(tree)
        restored = restore(unitless)
        npt.assert_array_equal(restored['w'], tree['w'])


class _LeakyCell(brainstate.nn.Module):
    """Recurrent leaky integrator with BOTH an ETP matmul (``W``) and an ETP
    ``element_wise`` weight (``alpha``); the recurrence makes the eligibility
    trace non-trivial across time. Used by the ``Batching()``-mode tests."""

    def __init__(self, nin, nh):
        super().__init__()
        self.nh = nh
        self.W = braintrace.nn.Linear(nin + nh, nh)
        self.alpha = brainstate.ParamState(jnp.linspace(0.5, 0.95, nh))

    def init_state(self, batch_size=None, **kw):
        size = (self.nh,) if batch_size is None else (batch_size, self.nh)
        self.u = brainstate.HiddenState(jnp.zeros(size))

    def update(self, x):
        a = jnp.clip(braintrace.element_wise(self.alpha.value), 0.0, 1.0)
        wx = self.W(jnp.concatenate([x, self.u.value], axis=-1))
        u_next = a * self.u.value + (1 - a) * wx
        self.u.value = u_next
        return u_next


def _accumulate_online_grads(model, algo_ctor, inputs, targets, batched):
    """Single-step eligibility-trace gradient summed over time.

    ``inputs``/``targets`` are ``(T, B, ...)`` when ``batched`` else ``(T, ...)``.
    Returns the per-path gradient pytree.
    """
    weights = model.states(brainstate.ParamState)

    @brainstate.transform.jit
    def run(inputs, targets):
        if batched:
            online = algo_ctor(model, mode=brainstate.mixin.Batching())
            brainstate.nn.init_all_states(model, batch_size=inputs.shape[1])
        else:
            online = algo_ctor(model)
            brainstate.nn.init_all_states(model)
        online.compile_graph(inputs[0])

        def step_loss(inp, tar):
            out = online(inp)
            return ((out - tar) ** 2).mean(), out

        def grad_step(prev, x):
            inp, tar = x
            fg = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            g, _, _ = fg(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev, g), None

        init = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
        grads, _ = brainstate.transform.scan(grad_step, init, (inputs, targets))
        return grads

    return run(inputs, targets)


def _assert_batching_matches_per_example(algo_ctor, atol=1e-4, rtol=1e-3):
    """The internal ``Batching()`` gradient must equal the batch-mean of the
    per-example unbatched gradients (the loss averages over the batch), and must
    carry no leaked batch axis (each gradient leaf matches its parameter shape).
    """
    nin, nh, batch, n_time = 4, 6, 5, 7
    brainstate.random.seed(0)
    model = _LeakyCell(nin, nh)
    xs = brainstate.random.randn(n_time, batch, nin)
    ys = brainstate.random.randn(n_time, batch, nh)

    g_batched = _accumulate_online_grads(model, algo_ctor, xs, ys, batched=True)
    per = [
        _accumulate_online_grads(model, algo_ctor, xs[:, b], ys[:, b], batched=False)
        for b in range(batch)
    ]
    g_ref = jax.tree.map(lambda *gs: sum(gs) / batch, *per)

    for key in g_batched:
        for bl, rl in zip(jax.tree.leaves(g_batched[key]), jax.tree.leaves(g_ref[key])):
            assert bl.shape == rl.shape, (key, bl.shape, rl.shape)
            npt.assert_allclose(
                u.get_mantissa(bl), u.get_mantissa(rl), rtol=rtol, atol=atol
            )


class TestElemwiseBatchingMode:
    """Regression: ``etp_elemwise`` under ``brainstate.mixin.Batching()``.

    A model with a per-element ``element_wise`` weight (e.g. an SNN leak/``alpha``)
    trained in the internal ``Batching()`` mode used to crash with a custom-VJP
    shape mismatch: the elemwise eligibility trace acquired a leading batch axis
    from the batched hidden state that was never reduced, because ``etp_elemwise``
    is registered ``batched=False`` and the solve-time batch-sum keyed off
    ``is_batched_primitive``. The batched gradient must equal the batch-mean of
    the per-example unbatched gradients, with no leaked batch axis.
    """

    def test_d_rtrl_batching_matches_per_example(self):
        _assert_batching_matches_per_example(lambda m, **kw: braintrace.D_RTRL(m, **kw))


def _docstring_rnn():
    """The exact ``RNN`` model used in the ``ParamDimVjpAlgorithm`` docstring example."""

    class RNN(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = braintrace.nn.ValinaRNNCell(1, 20, activation='tanh')
            self.out = braintrace.nn.Linear(20, 1)

        def update(self, x):
            return x >> self.cell >> self.out

    return RNN()


def test_docstring_compile_example_runs():
    """Verify the ``braintrace.compile`` example in ``ParamDimVjpAlgorithm``'s docstring."""
    model = _docstring_rnn()
    x0 = brainstate.random.randn(1)
    learner = braintrace.compile(model, braintrace.D_RTRL, x0)
    y = learner(x0)
    assert y.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(y)))
    assert len(learner.graph.hidden_param_op_relations) >= 1
