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

import unittest

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._algorithm import oracle
from braintrace._algorithm.otpe import OTPE


def _otpe_net_single_layer():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.v = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            self.v.value = 0.9 * self.v.value + braintrace.matmul(x, self.w.value)
            return self.v.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


def _two_state_net():
    """Net whose two coupled hidden states form one group with num_state == 2."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.v = brainstate.HiddenState(jnp.zeros((1, 3)))
            self.a = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            v, a = self.v.value, self.a.value
            self.v.value = 0.9 * v + braintrace.matmul(x, self.w.value) - 0.1 * a
            self.a.value = 0.95 * a + v
            return self.v.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


def _lora_net():
    """Net trained through `braintrace.lora_matmul` -- LoRA relations are outside
    OTPE's supported primitive set (see finding H5)."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            brainstate.random.seed(0)
            self.B = brainstate.ParamState(0.1 * brainstate.random.normal(size=(3, 2)))
            self.A = brainstate.ParamState(0.1 * brainstate.random.normal(size=(2, 3)))
            self.v = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            self.v.value = 0.9 * self.v.value + braintrace.lora_matmul(
                x, self.B.value, self.A.value
            )
            return self.v.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


def _bias_net():
    """Net trained through a recurrent `braintrace.matmul(..., bias=...)`, whose
    recurrence coefficient (0.9) matches the `leak` OTTT/OTPE are given -- the
    "exactly diagonal" regime in which both algorithms claim BPTT-exact gradients."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            brainstate.random.seed(0)
            self.w = brainstate.ParamState(0.1 * brainstate.random.normal(size=(3, 3)))
            self.b = brainstate.ParamState(jnp.zeros(3))
            self.v = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            self.v.value = 0.9 * self.v.value + braintrace.matmul(
                x, self.w.value, self.b.value
            )
            return self.v.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


def _unbatched_net():
    """Net with an unbatched hidden state (no leading batch axis) -- exercises
    OTPE's `reset_state` on `etp_mv` relations (see finding M4)."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            brainstate.random.seed(0)
            self.w = brainstate.ParamState(0.1 * brainstate.random.normal(size=(3, 3)))
            self.v = brainstate.HiddenState(jnp.zeros((3,)))

        def update(self, x):
            self.v.value = 0.9 * self.v.value + braintrace.matmul(x, self.w.value)
            return self.v.value

    net = Net()
    brainstate.nn.init_all_states(net)  # no batch_size -> unbatched
    return net


class TestOTPEConstruction(unittest.TestCase):
    def test_default_mode_full(self):
        algo = OTPE(_otpe_net_single_layer(), leak=0.9)
        assert algo.mode == 'full'

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            OTPE(_otpe_net_single_layer(), mode='bogus', leak=0.9)

    def test_leak_is_required(self):
        """leak is never inferred from the model; omitting it is a TypeError."""
        with self.assertRaises(TypeError):
            OTPE(_otpe_net_single_layer())

    def test_leak_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            OTPE(_otpe_net_single_layer(), leak=1.5)

    def test_multi_step_vjp_method_rejected_at_construction(self):
        """N6: OTPE's trace update/weight-gradient rules are derived one step at
        a time and have no multi-step form; `vjp_method='multi-step'` must be
        rejected eagerly at construction, not silently accepted until the first
        (multi-step) call fails."""
        with self.assertRaises(ValueError):
            OTPE(_otpe_net_single_layer(), leak=0.9, vjp_method='multi-step')

    def test_num_state_gt_one_raises(self):
        """Multi-state hidden groups are outside OTPE's LIF regime.

        M6: unlike the `NotImplementedError` in
        `TestOTPEUnsupportedRelationGuard` (which fires inside
        `init_etrace_state`, *before* `base.compile_graph` ever sets
        `is_compiled = True`), this `ValueError` fires in OTPE's own
        post-`super().compile_graph()` validation -- i.e. *after* the base
        class has already set `is_compiled = True`. That validation's
        try/except must explicitly reset the flag on failure, or a failed
        compile would leave `is_compiled` stuck `True` and the base class's
        `if not self.is_compiled:` guard would silently skip re-validation
        (and `init_etrace_state`) on a subsequent `compile_graph` call."""
        algo = OTPE(_two_state_net(), leak=0.9)
        x = jnp.ones((1, 3))
        with self.assertRaises(ValueError):
            algo.compile_graph(x)
        assert algo.is_compiled is False

        # A second call on the same (permanently invalid) model must raise
        # again, not silently pass because `is_compiled` was left True.
        with self.assertRaises(ValueError):
            algo.compile_graph(x)
        assert algo.is_compiled is False

    def test_compile_allocates_R_hat(self):
        algo = OTPE(_otpe_net_single_layer(), leak=0.9)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        assert len(algo._R_hat) == len(algo.graph.hidden_param_op_relations)


class TestOTPESingleLayer(unittest.TestCase):
    def test_update_runs_and_produces_gradients(self):
        net = _otpe_net_single_layer()
        algo = OTPE(net, leak=0.9, mode='full')
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()

        def loss(x_):
            return (algo.update(x_) ** 2).sum()

        grads, _ = brainstate.transform.grad(
            loss, algo.param_states, return_value=True
        )(x)
        g = grads[next(iter(grads))]
        assert g.shape == (3, 3)
        assert jnp.any(g != 0.0)


class TestOTPEApproxMode(unittest.TestCase):
    def test_approx_uses_factored_traces(self):
        net = _otpe_net_single_layer()
        algo = OTPE(net, mode='approx', leak=0.9)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        assert len(algo._R_hat_x) == len(algo.graph.hidden_param_op_relations)
        assert len(algo._R_hat_g) == len(algo.graph.hidden_param_op_relations)

    def test_approx_runs(self):
        net = _otpe_net_single_layer()
        algo = OTPE(net, mode='approx', leak=0.9)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()

        def loss(x_):
            return (algo.update(x_) ** 2).sum()

        grads, _ = brainstate.transform.grad(
            loss, algo.param_states, return_value=True
        )(x)
        g = grads[next(iter(grads))]
        assert g.shape == (3, 3)


class TestOTPEBiasGradient(unittest.TestCase):
    """H5: the weight-gradient solvers indexed only the ``'weight'`` key of each
    relation's trainable dict, so `braintrace.matmul(x, w, bias=b)` silently got
    a zero bias gradient. Must route every key (here also ``'bias'``) through
    `relation.trainable_paths`, exactly as `_common._route_grads_by_path` does.

    OTPE's docstring claims gradient-exactness for a single hidden layer under
    the scalar-leak (LIF) assumption. In the "exactly diagonal" regime -- the
    model's recurrence coefficient equals the `leak` OTPE is given -- both
    'full' and 'approx' modes are exact for bias specifically: bias's local
    Jacobian dy/db=1 has no "in" dimension, so 'approx' mode's outer-product
    factorization (which only approximates the joint x-times-df weight trace)
    introduces no approximation error for bias -- its ``R_hat_g``/``R_hat_bias``
    companion trace follows the identical recursion in both modes. So we assert
    *element-wise* equality with the BPTT oracle for bias in both modes.
    """

    def test_bias_gradient_matches_bptt_full_mode(self):
        inputs = jnp.stack([jnp.ones((1, 3)) * (i + 1) * 0.1 for i in range(4)])
        bptt = oracle.bptt_param_gradients(_bias_net, inputs)
        online = oracle.online_param_gradients_singlestep_naive(
            _bias_net, inputs, algo_factory=lambda m: OTPE(m, leak=0.9, mode='full')
        )
        assert bool(jnp.any(online[('b',)] != 0.0)), 'bias gradient must not be zero (H5)'
        oracle.assert_param_gradients_close(
            online, bptt, atol=1e-4, keys=[('b',), ('w',)]
        )

    def test_bias_gradient_matches_bptt_approx_mode(self):
        inputs = jnp.stack([jnp.ones((1, 3)) * (i + 1) * 0.1 for i in range(4)])
        bptt = oracle.bptt_param_gradients(_bias_net, inputs)
        online = oracle.online_param_gradients_singlestep_naive(
            _bias_net, inputs, algo_factory=lambda m: OTPE(m, leak=0.9, mode='approx')
        )
        assert bool(jnp.any(online[('b',)] != 0.0)), 'bias gradient must not be zero (H5)'
        # Only the bias key is asserted exact here -- 'approx' mode's weight
        # gradient is *not* BPTT-exact (its outer-product factorization is a
        # genuine additional approximation), covered separately in
        # TestOTPEApproxMode.
        oracle.assert_param_gradients_close(online, bptt, atol=1e-4, keys=[('b',)])


class TestOTPEUnsupportedRelationGuard(unittest.TestCase):
    """H5: LoRA/conv/sparse relations don't reduce to OTPE's per-parameter
    leaky trace; they must raise a compile-time `NotImplementedError` naming
    the offending primitive rather than crashing opaquely or silently
    mishandling the relation."""

    def test_lora_relation_raises_named_primitive_at_compile_time(self):
        algo = OTPE(_lora_net(), leak=0.9, mode='full')
        with self.assertRaises(NotImplementedError) as ctx:
            algo.compile_graph(jnp.ones((1, 3)))
        assert 'etp_lora_mm' in str(ctx.exception)

    def test_is_compiled_stays_false_after_repeated_guard_failure(self):
        """M6: a failed compile-time validation must not leave `is_compiled`
        stuck True -- otherwise `base.compile_graph`'s ``if not
        self.is_compiled:`` guard would silently skip re-validation (and
        `init_etrace_state`) on a subsequent call. Calling `compile_graph`
        twice on the same (permanently invalid) model must raise both times."""
        algo = OTPE(_lora_net(), leak=0.9, mode='full')
        x = jnp.ones((1, 3))
        with self.assertRaises(NotImplementedError):
            algo.compile_graph(x)
        assert algo.is_compiled is False

        with self.assertRaises(NotImplementedError):
            algo.compile_graph(x)
        assert algo.is_compiled is False

        # A fresh, valid model must still compile normally afterward.
        ok_algo = OTPE(_otpe_net_single_layer(), leak=0.9, mode='full')
        ok_algo.compile_graph(x)
        assert ok_algo.is_compiled is True


class TestOTPEUnbatchedResetState(unittest.TestCase):
    """M4: `OTPE.reset_state` assumed a leading batch axis and corrupted
    unbatched trace shapes (e.g. replacing the real leading `in`/`out` dim with
    `batch_size`). Must derive reset shapes from the stored trace values
    instead of assuming a batch axis is present."""

    def test_reset_preserves_unbatched_trace_shapes_full_mode(self):
        net = _unbatched_net()
        algo = OTPE(net, leak=0.9, mode='full')
        x = jnp.ones((3,))
        algo.compile_graph(x)
        algo.init_etrace_state()
        algo.update(x)  # populate traces with real (non-init) values first

        before = {rid: r.value.shape for rid, r in algo._R_hat.items()}
        algo.reset_state(batch_size=4)  # unbatched relations must ignore this
        after = {rid: r.value.shape for rid, r in algo._R_hat.items()}
        assert before == after
        assert all(v.value.shape == (3, 3) for v in algo._R_hat.values())

    def test_reset_preserves_unbatched_trace_shapes_approx_mode(self):
        net = _unbatched_net()
        algo = OTPE(net, leak=0.9, mode='approx')
        x = jnp.ones((3,))
        algo.compile_graph(x)
        algo.init_etrace_state()
        algo.update(x)

        before_x = {rid: r.value.shape for rid, r in algo._R_hat_x.items()}
        before_g = {rid: r.value.shape for rid, r in algo._R_hat_g.items()}
        algo.reset_state(batch_size=4)
        after_x = {rid: r.value.shape for rid, r in algo._R_hat_x.items()}
        after_g = {rid: r.value.shape for rid, r in algo._R_hat_g.items()}
        assert before_x == after_x
        assert before_g == after_g


def _toy_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.v = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            self.v.value = jax.nn.tanh(
                0.9 * self.v.value + braintrace.matmul(x, self.w.value)
            )
            return self.v.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


def _run(algo, n_steps=10, lr=0.05, y_target=None, pass_y=False):
    x = jnp.ones((1, 3))
    algo.compile_graph(x)
    algo.init_etrace_state()

    losses = []
    for _ in range(n_steps):
        def loss_fn(x_):
            out = algo.update(x_, y_target=y_target) if pass_y else algo.update(x_)
            target = jnp.ones_like(out)
            return ((out - target) ** 2).mean()

        grads, loss_val = brainstate.transform.grad(
            loss_fn, algo.param_states, return_value=True
        )(x)
        for path, st in algo.param_states.items():
            st.value = st.value - lr * grads[path]
        losses.append(float(loss_val))
    return losses


class TestSmokeLossDecreases(unittest.TestCase):
    def test_otpe(self):
        losses = _run(OTPE(_toy_net(), leak=0.9))
        assert losses[-1] < losses[0]


def _docstring_net():
    """The exact ``Net`` model used in the ``OTPE`` docstring example."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = braintrace.nn.ValinaRNNCell(1, 20, activation='tanh')
            self.out = braintrace.nn.Linear(20, 1)

        def update(self, x):
            return x >> self.cell >> self.out

    return Net()


def test_docstring_compile_example_runs():
    """Verify the runnable ``braintrace.compile`` example in ``OTPE``'s docstring."""
    model = _docstring_net()
    x0 = brainstate.random.randn(1)
    learner = braintrace.compile(model, braintrace.OTPE, x0, mode='full', leak=0.9)
    y = learner(x0)
    assert y.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(y)))
    assert len(learner.graph.hidden_param_op_relations) >= 1
