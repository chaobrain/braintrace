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
from braintrace._algorithm.e_prop import EProp
from braintrace._algorithm.oracle_models import two_state_rnn


def _lsnn_like():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.h = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            self.h.value = jax.nn.tanh(
                braintrace.matmul(self.h.value + x, self.w.value)
            )
            return self.h.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


class TestEPropConstruction(unittest.TestCase):
    def test_default_feedback_and_kappa(self):
        algo = EProp(_lsnn_like())
        assert algo.feedback == 'symmetric'
        assert algo.kappa_filter_decay == 0.0

    def test_invalid_feedback_raises(self):
        with self.assertRaises(ValueError):
            EProp(_lsnn_like(), feedback='weird')

    def test_kappa_filter_allocated_when_nonzero(self):
        algo = EProp(_lsnn_like(), kappa_filter_decay=0.9)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        # One trace-filter state per raw-trace key (matching etrace_bwg's own
        # key space), not per hidden_param_op_relation -- those only happen
        # to coincide (1-to-1) on this trivial single-weight, single-group model.
        assert len(algo._trace_filters) == len(algo.etrace_bwg)
        assert len(algo._trace_filters) > 0

    def test_kappa_filter_skipped_when_zero(self):
        algo = EProp(_lsnn_like(), kappa_filter_decay=0.0)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        assert len(algo._trace_filters) == 0


class TestEPropKappaApplied(unittest.TestCase):
    def test_kappa_zero_matches_d_rtrl(self):
        """κ=0 must reproduce D_RTRL gradients bit-for-bit on the same model."""
        from braintrace._algorithm.param_dim_vjp import ParamDimVjpAlgorithm

        def compute(algo_cls, **extra):
            net = _lsnn_like()
            algo = algo_cls(net, **extra)
            x = jnp.ones((1, 3))
            algo.compile_graph(x)
            algo.init_etrace_state()

            def loss(x_):
                out = algo.update(x_)
                return (out ** 2).sum()

            grads, _ = brainstate.transform.grad(
                loss, algo.param_states, return_value=True
            )(x)
            return grads[next(iter(grads))]

        g_drtrl = compute(ParamDimVjpAlgorithm)
        g_eprop = compute(EProp, feedback='symmetric', kappa_filter_decay=0.0)
        assert jnp.allclose(g_drtrl, g_eprop, atol=1e-6)

    def test_kappa_nonzero_differs_from_zero(self):
        """κ only accumulates history (``bar_e^t = kappa*bar_e^{t-1} + e^t``),
        so its effect is invisible on the very first backward-invoking step
        (bar_e^{t-1} starts at zero) and only shows up once a *second*
        backward step carries forward non-trivial filter history.
        """

        def compute(kappa):
            net = _lsnn_like()
            algo = EProp(net, kappa_filter_decay=kappa)
            x = jnp.ones((1, 3))
            algo.compile_graph(x)
            algo.init_etrace_state()

            def loss(x_):
                out = algo.update(x_)
                return (out ** 2).sum()

            # First backward-invoking step primes the trace-filter history.
            brainstate.transform.grad(
                loss, algo.param_states, return_value=True
            )(x)
            grads, _ = brainstate.transform.grad(
                loss, algo.param_states, return_value=True
            )(x)
            return grads[next(iter(grads))]

        g0 = compute(0.0)
        g9 = compute(0.9)
        assert not jnp.allclose(g0, g9, atol=1e-4)


class TestEPropRandomFeedback(unittest.TestCase):
    def test_random_feedback_differs_from_symmetric(self):
        def compute(feedback, **extra):
            net = _lsnn_like()
            algo = EProp(net, feedback=feedback, **extra)
            x = jnp.ones((1, 3))
            algo.compile_graph(x)
            algo.init_etrace_state()

            def loss(x_):
                out = algo.update(x_)
                return (out ** 2).sum()

            grads, _ = brainstate.transform.grad(
                loss, algo.param_states, return_value=True
            )(x)
            return grads[next(iter(grads))]

        g_sym = compute('symmetric')
        g_rnd = compute('random', random_feedback_key=jax.random.PRNGKey(123))
        assert not jnp.allclose(g_sym, g_rnd, atol=1e-4)


class TestEPropNumStateIsolation(unittest.TestCase):
    """H4: the kappa filter must not sum across the trailing ``num_state``
    axis and broadcast the sum back -- each hidden-state channel of a
    multi-state HiddenGroup must be filtered independently.
    """

    def _run(self, fast_solve):
        spec = two_state_rnn(n_in=3, n_rec=3, seed=0)
        net = spec.factory()
        brainstate.nn.init_all_states(net, batch_size=1)
        algo = EProp(net, kappa_filter_decay=0.9, fast_solve=fast_solve)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()

        def loss(x_):
            # `update` only returns `v` (state 0); `a` (state 1) only
            # influences the loss indirectly through the *next* step's `v`,
            # so the two channels are state-asymmetric.
            out = algo.update(x_)
            return (out ** 2).sum()

        # Two backward-invoking steps so the (kappa > 0) filter history is
        # non-trivial in both channels.
        brainstate.transform.grad(loss, algo.param_states, return_value=True)(x)
        brainstate.transform.grad(loss, algo.param_states, return_value=True)(x)
        return algo

    def test_runs_without_shape_error_both_paths(self):
        # Legacy path (fast_solve=False) shape-errors today (H4); fast path
        # silently contaminates state 1 with state 0's sum. Both must simply
        # run to completion post-fix.
        self._run(fast_solve=False)
        self._run(fast_solve=True)

    def test_per_state_channels_are_not_contaminated(self):
        for fast_solve in (False, True):
            algo = self._run(fast_solve=fast_solve)
            assert len(algo._trace_filters) == 1
            flt = next(iter(algo._trace_filters.values()))
            w = flt.value['weight']
            assert w.shape[-1] == 2  # num_state == 2 (v, a)
            channel_0, channel_1 = w[..., 0], w[..., 1]
            assert not jnp.allclose(channel_0, channel_1, atol=1e-4), (
                f'fast_solve={fast_solve}: per-state channels are identical -- '
                'looks like a summed-then-broadcast contamination.'
            )


class TestEPropFilterSemantics(unittest.TestCase):
    """M1: kappa must filter the *eligibility trace*
    (``bar_e^t = kappa*bar_e^{t-1} + e^t``), not the learning signal. The two
    orderings give different gradients whenever the learning signal varies in
    time, which this hand-computed 2-step reference exercises via a
    sign-flipping loss coefficient over a constant trace.
    """

    def test_kappa_zero_matches_unfiltered(self):
        """Regression anchor: kappa=0 must reduce to the unfiltered algorithm."""
        with brainstate.environ.context(precision=64):
            def make():
                class Net(brainstate.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.w = brainstate.ParamState(jnp.array([[1.0]]))
                        self.h = brainstate.HiddenState(jnp.zeros((1, 1)))

                    def update(self, x):
                        self.h.value = braintrace.matmul(x, self.w.value)
                        return self.h.value

                net = Net()
                brainstate.nn.init_all_states(net, batch_size=1)
                return net

            def compute(kappa):
                net = make()
                algo = EProp(net, kappa_filter_decay=kappa)
                x = jnp.array([[2.0]])
                algo.compile_graph(x)
                algo.init_etrace_state()

                def loss(x_):
                    out = algo.update(x_)
                    return (out ** 2).sum()

                grads, _ = brainstate.transform.grad(
                    loss, algo.param_states, return_value=True
                )(x)
                return grads[next(iter(grads))]

            g_unfiltered = compute(0.0)
            # kappa=0.0 takes the `kappa_filter_decay > 0.0` branch's *else*
            # path (no filters allocated at all), so this is a genuine
            # code-path regression anchor, not just a numerically-tiny kappa.
            assert jnp.allclose(g_unfiltered, jnp.array([[8.0]]), atol=1e-10)

    def test_hand_computed_two_step_filtered_trace_reference(self):
        """No-recurrence toy model: ``h_t = matmul(x, w)`` with constant
        ``x = [[2.0]]``, so the raw eligibility trace ``e^t = 2.0`` every
        step. With ``kappa=0.5`` and loss coefficients ``+1`` then ``-1``:

            bar_e^1 = 0.5*0 + 2.0 = 2.0;  g1 = (+1)*bar_e^1 =  2.0
            bar_e^2 = 0.5*2.0 + 2.0 = 3.0; g2 = (-1)*bar_e^2 = -3.0
            total = g1 + g2 = -1.0

        (The old signal-filtering ordering would instead filter the
        already-scaled learning signal and produce a different total.)
        """
        with brainstate.environ.context(precision=64):
            class Net(brainstate.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = brainstate.ParamState(jnp.array([[1.0]]))
                    self.h = brainstate.HiddenState(jnp.zeros((1, 1)))

                def update(self, x):
                    self.h.value = braintrace.matmul(x, self.w.value)
                    return self.h.value

            net = Net()
            brainstate.nn.init_all_states(net, batch_size=1)
            algo = EProp(net, kappa_filter_decay=0.5)
            x = jnp.array([[2.0]])
            algo.compile_graph(x)
            algo.init_etrace_state()

            def grad_with_coeff(coeff):
                def loss(x_):
                    out = algo.update(x_)
                    return (coeff * out).sum()

                grads, _ = brainstate.transform.grad(
                    loss, algo.param_states, return_value=True
                )(x)
                return grads[next(iter(grads))]

            g1 = grad_with_coeff(1.0)
            g2 = grad_with_coeff(-1.0)
            assert jnp.allclose(g1, jnp.array([[2.0]]), atol=1e-10)
            assert jnp.allclose(g2, jnp.array([[-3.0]]), atol=1e-10)
            assert jnp.allclose(g1 + g2, jnp.array([[-1.0]]), atol=1e-10)


class TestEPropRandomFeedbackInvariance(unittest.TestCase):
    """M2: 'random feedback' must remove weight-transport (independence from
    the readout weights' *magnitude*), while still depending on the
    ``brainstate.random`` seed used to draw the fixed projection matrix.
    """

    @staticmethod
    def _readout_net(readout_scale):
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
                )
                self.h = brainstate.HiddenState(jnp.zeros((1, 3)))
                # Plain (non-ETP) readout matrix -- only its *scale* matters
                # for this test, so it is not wrapped in a ParamState.
                self.w_out = readout_scale * jax.random.normal(
                    jax.random.PRNGKey(1), (3, 3)
                )

            def update(self, x):
                self.h.value = jax.nn.tanh(
                    braintrace.matmul(self.h.value + x, self.w.value)
                )
                return self.h.value @ self.w_out

        net = Net()
        brainstate.nn.init_all_states(net, batch_size=1)
        return net

    def _compute(self, feedback, readout_scale, seed=None):
        net = self._readout_net(readout_scale)
        if feedback == 'random':
            brainstate.random.seed(seed)
            key = brainstate.random.split_key()
            algo = EProp(net, feedback='random', random_feedback_key=key)
        else:
            algo = EProp(net, feedback='symmetric')
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()

        def loss(x_):
            out = algo.update(x_)
            return (out ** 2).sum()

        grads, _ = brainstate.transform.grad(
            loss, algo.param_states, return_value=True
        )(x)
        return grads[next(iter(grads))]

    def test_symmetric_feedback_scales_with_readout_weights(self):
        # Sanity check that the readout-scale knob is actually meaningful.
        g_small = self._compute('symmetric', readout_scale=1.0)
        g_large = self._compute('symmetric', readout_scale=5.0)
        assert not jnp.allclose(g_small, g_large, atol=1e-4)

    def test_random_feedback_invariant_to_readout_scale(self):
        g_small = self._compute('random', readout_scale=1.0, seed=7)
        g_large = self._compute('random', readout_scale=5.0, seed=7)
        assert jnp.allclose(g_small, g_large, atol=1e-3)

    def test_random_feedback_depends_on_seed(self):
        g_seed1 = self._compute('random', readout_scale=1.0, seed=1)
        g_seed2 = self._compute('random', readout_scale=1.0, seed=2)
        assert not jnp.allclose(g_seed1, g_seed2, atol=1e-4)


class FakeLIF(brainstate.HiddenState):
    def __init__(self, iv, leak):
        super().__init__(iv)
        self.leak = leak


def _toy_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.v = FakeLIF(jnp.zeros((1, 3)), leak=0.9)

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
    def test_eprop(self):
        losses = _run(EProp(_toy_net()))
        assert losses[-1] < losses[0]


def _docstring_rsnn():
    """The exact ``RSNN`` model used in the ``EProp`` docstring example."""

    class RSNN(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = braintrace.nn.ValinaRNNCell(1, 20, activation='tanh')
            self.out = braintrace.nn.Linear(20, 1)

        def update(self, x):
            return x >> self.cell >> self.out

    return RSNN()


def test_docstring_compile_example_runs():
    """Verify the runnable ``braintrace.compile`` example in ``EProp``'s docstring."""
    model = _docstring_rsnn()
    x0 = brainstate.random.randn(1)
    learner = braintrace.compile(model, braintrace.EProp, x0, kappa_filter_decay=0.9)
    y = learner(x0)
    assert y.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(y)))
    assert len(learner.graph.hidden_param_op_relations) >= 1
