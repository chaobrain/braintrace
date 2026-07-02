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
from braintrace._algorithm.ottt import OTTT


def _ottt_net():
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
    OTTT's supported primitive set (see finding H5)."""

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


class TestOTTTConstruction(unittest.TestCase):
    def test_default_mode_is_A(self):
        algo = OTTT(_ottt_net(), leak=0.9)
        assert algo.mode == 'A'

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            OTTT(_ottt_net(), mode='bogus', leak=0.9)

    def test_leak_is_required(self):
        """leak is never inferred from the model; omitting it is a TypeError."""
        with self.assertRaises(TypeError):
            OTTT(_ottt_net())

    def test_leak_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            OTTT(_ottt_net(), leak=1.5)

    def test_multi_step_vjp_method_rejected_at_construction(self):
        """N6: OTTT's trace update/weight-gradient rules are derived one step at
        a time and have no multi-step form; `vjp_method='multi-step'` must be
        rejected eagerly at construction, not silently accepted until the first
        (multi-step) call fails."""
        with self.assertRaises(ValueError):
            OTTT(_ottt_net(), leak=0.9, vjp_method='multi-step')

    def test_num_state_gt_one_raises(self):
        """Multi-state hidden groups have no theoretical basis for OTTT."""
        algo = OTTT(_two_state_net(), leak=0.9)
        with self.assertRaises(ValueError):
            algo.compile_graph(jnp.ones((1, 3)))

    def test_compile_allocates_presynaptic_traces(self):
        algo = OTTT(_ottt_net(), leak=0.9)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        assert len(algo._pre_traces) == len(algo.graph.hidden_param_op_relations)


class TestOTTTEnd2End(unittest.TestCase):
    def test_update_runs_and_produces_gradients(self):
        net = _ottt_net()
        algo = OTTT(net, leak=0.9)
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

    def test_mode_O_differs_from_mode_A(self):
        def compute(mode):
            net = _ottt_net()
            algo = OTTT(net, mode=mode, leak=0.9)
            x = jnp.ones((1, 3))
            algo.compile_graph(x)
            algo.init_etrace_state()
            algo.update(x)

            def loss(x_):
                return (algo.update(x_) ** 2).sum()

            grads, _ = brainstate.transform.grad(
                loss, algo.param_states, return_value=True
            )(x)
            return grads[next(iter(grads))]

        g_A = compute('A')
        g_O = compute('O')
        assert not jnp.allclose(g_A, g_O, atol=1e-4)


class TestOTTTBiasGradient(unittest.TestCase):
    """H5: the weight-gradient solver indexed only the ``'weight'`` key of each
    relation's trainable dict, so `braintrace.matmul(x, w, bias=b)` silently got
    a zero bias gradient. Must route every key (here also ``'bias'``) through
    `relation.trainable_paths`, exactly as `_common._route_grads_by_path` does.

    OTTT's docstring claims it keeps BPTT's *spatial* credit assignment exactly
    and only drops the *temporal* (hidden-to-hidden) Jacobian. In the "exactly
    diagonal" regime -- the model's recurrence coefficient equals the `leak`
    OTTT is given, mode='A' -- this is provably exact: both the outer-product
    weight term and the scalar bias term reduce, via summation-order exchange,
    to the same double sum BPTT computes. So we assert *element-wise* equality
    with the BPTT oracle, not just non-zero/same-sign -- this model is squarely
    in that exact regime.
    """

    def test_bias_gradient_matches_bptt_in_exactly_diagonal_regime(self):
        inputs = jnp.stack([jnp.ones((1, 3)) * (i + 1) * 0.1 for i in range(4)])
        bptt = oracle.bptt_param_gradients(_bias_net, inputs)
        online = oracle.online_param_gradients_singlestep_naive(
            _bias_net, inputs, algo_factory=lambda m: OTTT(m, leak=0.9)
        )
        assert bool(jnp.any(online[('b',)] != 0.0)), 'bias gradient must not be zero (H5)'
        oracle.assert_param_gradients_close(
            online, bptt, atol=1e-4, keys=[('b',), ('w',)]
        )


class TestOTTTUnsupportedRelationGuard(unittest.TestCase):
    """H5: LoRA/conv/sparse relations don't reduce to OTTT's single
    presynaptic-trace outer product; they must raise a compile-time
    `NotImplementedError` naming the offending primitive rather than crashing
    opaquely or silently mishandling the relation."""

    def test_lora_relation_raises_named_primitive_at_compile_time(self):
        algo = OTTT(_lora_net(), leak=0.9)
        with self.assertRaises(NotImplementedError) as ctx:
            algo.compile_graph(jnp.ones((1, 3)))
        assert 'etp_lora_mm' in str(ctx.exception)

    def test_is_compiled_stays_false_after_repeated_guard_failure(self):
        """M6: a failed compile-time validation must not leave `is_compiled`
        stuck True -- otherwise `base.compile_graph`'s ``if not
        self.is_compiled:`` guard would silently skip re-validation (and
        `init_etrace_state`) on a subsequent call. Calling `compile_graph`
        twice on the same (permanently invalid) model must raise both times."""
        algo = OTTT(_lora_net(), leak=0.9)
        x = jnp.ones((1, 3))
        with self.assertRaises(NotImplementedError):
            algo.compile_graph(x)
        assert algo.is_compiled is False

        with self.assertRaises(NotImplementedError):
            algo.compile_graph(x)
        assert algo.is_compiled is False

        # A fresh, valid model must still compile normally afterward.
        ok_algo = OTTT(_ottt_net(), leak=0.9)
        ok_algo.compile_graph(x)
        assert ok_algo.is_compiled is True


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
    def test_ottt(self):
        losses = _run(OTTT(_toy_net(), leak=0.9))
        assert losses[-1] < losses[0]


def _docstring_net():
    """The exact ``Net`` model used in the ``OTTT`` docstring example."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = braintrace.nn.ValinaRNNCell(1, 20, activation='tanh')
            self.out = braintrace.nn.Linear(20, 1)

        def update(self, x):
            return x >> self.cell >> self.out

    return Net()


def test_docstring_compile_example_runs():
    """Verify the runnable ``braintrace.compile`` example in ``OTTT``'s docstring."""
    model = _docstring_net()
    x0 = brainstate.random.randn(1)
    learner = braintrace.compile(model, braintrace.OTTT, x0, mode='A', leak=0.9)
    y = learner(x0)
    assert y.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(y)))
    assert len(learner.graph.hidden_param_op_relations) >= 1
