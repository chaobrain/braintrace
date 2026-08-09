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

"""Acceptance tests for the sequence driver: ``etrace_grad`` / ``etrace_evolve``.

Spec: ``docs/specs/2026-07-27-sequence-driver-api.md``.

The driver is a *composition* layer -- it must reproduce the hand-written
scan-accumulate block exactly, and must not perturb any algorithm's numerics.
So most tests here compare the driver against a hand-written equivalent built
in the test itself, rather than against a frozen number: a test that only
pinned constants would pass even if the driver and the block drifted together.

Two claims get the sharpest treatment, because getting them wrong is silent:

- ``chunk_size=1`` is the *plain* path, matching ``dni._as_window`` (P7 in the
  spec). ``TestTheTwoDrivingModes`` pins this from three directions.
- A mask gates only the loss; the trace still crosses zero-weighted steps.
  ``TestMasking`` pins it with a positive test *and* a negative control.
"""

import inspect

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._testing import oracle_models as om
from braintrace._testing import oracle
from braintrace._algorithm.dni import _as_window, train_synthetic_gradient
from braintrace._compile import _ALGORITHM_REGISTRY

T = 6          # sequence length; divisible by 2 and 3
K = 2          # a genuine window size (>= 2)
N_IN = 3
N_REC = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _inputs(t=T, n=N_IN, *, seed=0, scale=0.5):
    rng = np.random.RandomState(seed)
    return scale * jnp.asarray(rng.randn(t, n).astype('float32'))


def _targets(t=T, n=N_REC, *, seed=1, scale=0.5):
    rng = np.random.RandomState(seed)
    return scale * jnp.asarray(rng.randn(t, 1, n).astype('float32'))


def _accepts_vjp_method(algorithm) -> bool:
    """Whether ``algorithm`` takes a ``vjp_method``, read off its signature.

    Hard-coding the list here is how the F-30 test silently stopped testing
    anything: ``pp_prop`` was omitted, so it was built at the default
    ``'single-step'`` and the driver refused ``chunk_size=K`` before the
    assertion could run. Introspection cannot fall out of date.
    """
    cls = (_ALGORITHM_REGISTRY[algorithm.lower()]
           if isinstance(algorithm, str) else algorithm)
    return 'vjp_method' in inspect.signature(cls.__init__).parameters


def _learner(vjp_method='single-step', algorithm='D_RTRL', *, spec=None, **opts):
    """A compiled, unbatched learner over a deterministic model.

    ``om.tanh_rnn`` seeds its weights from fixed PRNGKeys, so two calls give
    bitwise-identical parameters -- which is what the two-learner equivalence
    tests rely on.
    """
    spec = spec or om.tanh_rnn(N_IN, N_REC)
    model = spec.factory()
    if _accepts_vjp_method(algorithm):
        opts.setdefault('vjp_method', vjp_method)
    return braintrace.compile(model, algorithm, _inputs()[0], batch_size=1, **opts)


def _arrays(tree):
    """Path -> numpy array, mantissa only, for unit-carrying trees.

    Flattens through nested parameter values (a ``brainstate.nn.Linear``
    ``ParamState`` holds ``{'weight': ..., 'bias': ...}``), so it works on
    module-structured models as well as the flat oracle ones.
    """
    return {k: np.asarray(u.get_mantissa(v))
            for k, v in oracle.flat_gradient_leaves(tree).items()}


def _assert_trees_equal(a, b, *, atol=0.0, rtol=0.0, msg=''):
    aa, bb = _arrays(a), _arrays(b)
    assert set(aa) == set(bb), f'{msg}: key mismatch {set(aa)} vs {set(bb)}'
    for k in aa:
        np.testing.assert_allclose(aa[k], bb[k], atol=atol, rtol=rtol,
                                   err_msg=f'{msg}: at {k}')


def _max_abs_diff(a, b):
    aa, bb = _arrays(a), _arrays(b)
    return max(float(np.max(np.abs(aa[k] - bb[k]))) for k in aa)


def _sq_error(out, tar):
    return jnp.mean((out - tar) ** 2)


def _hand_written(learner, xs, ys):
    """The block the driver replaces: scan-accumulate, summed gradients.

    Returns ``(summed_grads, per_step_losses)``. This is the status quo that
    :class:`TestEquivalenceToTheStatusQuo` holds the driver to.
    """
    weights = learner.param_states

    def step_loss(inp, tar):
        return _sq_error(learner(inp), tar)

    def body(prev, pair):
        inp, tar = pair
        g, loss = brainstate.transform.grad(
            step_loss, weights, return_value=True)(inp, tar)
        return jax.tree.map(lambda a, b: a + b, prev, g), loss

    init = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
    return brainstate.transform.scan(body, init, (xs, ys))


def _plain_step(learner):
    def step_fn(inp, tar):
        return _sq_error(learner(inp), tar)
    return step_fn


def _window_step(learner, k):
    """A ``(k,)``-returning step function for window mode."""
    def step_fn(x_win, y_win):
        out = learner(braintrace.MultiStepData(x_win))
        return jnp.mean((out - y_win) ** 2, axis=tuple(range(1, out.ndim)))
    return step_fn


# ---------------------------------------------------------------------------
# 1--3. Equivalence to the status quo
# ---------------------------------------------------------------------------

class TestEquivalenceToTheStatusQuo:
    """The driver must reproduce the block it replaces, not merely approximate it."""

    def test_sum_reduction_reproduces_the_hand_written_scan_block(self):
        """Spec test 1."""
        ref = _learner()
        xs, ys = _inputs(), _targets()
        expected_grads, expected_losses = _hand_written(ref, xs, ys)

        got = _learner()
        grads, losses = got.etrace_grad(
            xs, ys, step_fn=_plain_step(got),
            reduction='sum', return_value=True)

        _assert_trees_equal(grads, expected_grads, msg='sum-reduced gradients')
        np.testing.assert_allclose(np.asarray(losses), np.asarray(expected_losses))

    def test_mean_reduction_is_sum_divided_by_the_total_mask_weight(self):
        """Spec test 2. Covers ``mask=None``, a binary mask and a weighted mask."""
        xs, ys = _inputs(), _targets()
        for mask, denom in [
            (None, float(T)),
            (jnp.asarray([1., 0., 1., 1., 0., 1.]), 4.0),
            (jnp.asarray([0.5, 2.0, 1.0, 0.0, 0.25, 1.0]), 4.75),
        ]:
            a, b = _learner(), _learner()
            g_sum = a.etrace_grad(xs, ys, step_fn=_plain_step(a),
                                  mask=mask, reduction='sum')
            g_mean = b.etrace_grad(xs, ys, step_fn=_plain_step(b),
                                   mask=mask, reduction='mean')
            scaled = jax.tree.map(lambda v: v / denom, g_sum)
            _assert_trees_equal(g_mean, scaled, rtol=1e-6,
                                msg=f'mean vs sum/{denom}')

    def test_whole_sequence_window_matches_the_multi_step_oracle(self):
        """Spec test 3.

        ``oracle.online_param_gradients`` differentiates ``(out ** 2).sum()``
        over one whole-sequence call. Driving that same objective through the
        driver at ``chunk_size=T, reduction='sum'`` must land on the same
        gradient -- a pure plumbing check, since a single whole-sequence window
        is blind to every learning-rule axis (F-23).
        """
        spec = om.tanh_rnn(N_IN, N_REC)
        xs = _inputs()
        expected = oracle.online_param_gradients(
            spec.factory, xs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))

        learner = _learner('multi-step', spec=spec)

        def step_fn(x_win):
            out = learner(braintrace.MultiStepData(x_win))
            return (out ** 2).sum(axis=tuple(range(1, out.ndim)))

        grads = learner.etrace_grad(xs, step_fn=step_fn,
                                    chunk_size=T, reduction='sum')
        _assert_trees_equal(grads, expected, rtol=1e-5, atol=1e-6,
                            msg='whole-sequence window vs oracle')


# ---------------------------------------------------------------------------
# 4--13. The two driving modes
# ---------------------------------------------------------------------------

class TestTheTwoDrivingModes:
    """``chunk_size=1`` is the plain path (P7); window mode starts at 2."""

    @pytest.mark.parametrize('vjp_method', ['single-step', 'multi-step'])
    def test_chunk_size_one_is_exactly_chunk_size_none(self, vjp_method):
        """Spec test 4 -- the load-bearing one for the P7 alignment.

        Not "agrees to round-off": the same code path, so gradients, losses and
        post-run learner state must all be identical. If someone later re-splits
        ``1`` from ``None``, this is what fails.
        """
        xs, ys = _inputs(), _targets()
        a, b = _learner(vjp_method), _learner(vjp_method)

        ga, la = a.etrace_grad(xs, ys, step_fn=_plain_step(a),
                               chunk_size=None, return_value=True)
        gb, lb = b.etrace_grad(xs, ys, step_fn=_plain_step(b),
                               chunk_size=1, return_value=True)

        _assert_trees_equal(ga, gb, msg='chunk_size 1 vs None')
        np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))
        assert a.running_index.value == b.running_index.value == T
        np.testing.assert_array_equal(
            np.asarray(list(a.hidden_states.values())[0].value),
            np.asarray(list(b.hidden_states.values())[0].value))

    def test_the_plain_path_agrees_with_a_hand_built_length_one_window(self):
        """Spec test 5 (P2: measured max abs diff 2.4e-07).

        The driver no longer exposes a length-1 ``MultiStepData`` window, so
        this constructs one directly. It is the evidence that nothing of value
        was withdrawn: the two are the same computation to float32 round-off.
        """
        xs, ys = _inputs(), _targets()

        driver = _learner('multi-step')
        g_plain = driver.etrace_grad(xs, ys, step_fn=_plain_step(driver),
                                     reduction='sum')

        manual = _learner('multi-step')
        weights = manual.param_states

        def windowed(inp, tar):
            out = manual(braintrace.MultiStepData(inp[None]))
            return jnp.mean((out[0] - tar) ** 2)

        def body(prev, pair):
            inp, tar = pair
            g = brainstate.transform.grad(windowed, weights)(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev, g), None

        init = jax.tree.map(jnp.zeros_like,
                            {k: v.value for k, v in weights.items()})
        g_window, _ = brainstate.transform.scan(body, init, (xs, ys))

        assert _max_abs_diff(g_plain, g_window) < 1e-5

    def test_a_single_step_learner_refuses_windows_but_accepts_chunk_size_one(self):
        """Spec test 6.

        The refusal must be the driver's ``ValueError``, raised before tracing --
        not the executor's ``NotImplementedError`` three frames down (P1). And
        ``chunk_size=1`` must *not* raise, which is what makes the
        ``1``-is-plain encoding observable rather than incidental.
        """
        xs, ys = _inputs(), _targets()

        refuser = _learner('single-step')
        with pytest.raises(ValueError, match='multi-step'):
            refuser.etrace_grad(xs, ys, step_fn=_window_step(refuser, K),
                                chunk_size=K)

        ok = _learner('single-step')
        ok.etrace_grad(xs, ys, step_fn=_plain_step(ok), chunk_size=1)

    def test_evolve_accepts_windows_on_a_single_step_learner(self):
        """Spec test 7 (P4).

        ``etrace_evolve`` runs no loss VJP, so windowed driving is legal where
        ``etrace_grad`` refuses it. The two compatibility matrices genuinely
        differ, and this is the test that says so.
        """
        learner = _learner('single-step')
        learner.etrace_evolve(_inputs(), chunk_size=K)
        assert learner.running_index.value == T

    def test_a_learner_without_a_vjp_method_is_refused_in_window_mode(self):
        """Spec test 8.

        No shipped algorithm lacks ``vjp_method``, so ``None`` means "an unknown
        subclass whose windowing support is unverified" -- refused, not admitted.
        """
        from braintrace._algorithm.sequence import SequenceDriverMixin

        class _Unknown(SequenceDriverMixin):
            def __init__(self, learner):
                self._learner = learner

            @property
            def _seq_call(self):
                return self._learner

            @property
            def _seq_param_states(self):
                return self._learner.param_states

            @property
            def _seq_vjp_method(self):
                return None

        stub = _Unknown(_learner('multi-step'))
        with pytest.raises(ValueError, match='multi-step'):
            stub.etrace_grad(_inputs(), step_fn=lambda x: jnp.zeros(K),
                             chunk_size=K)

    def test_chunk_size_must_divide_the_sequence_length(self):
        """Spec test 9. ``chunk_size=1`` can never trip it, and is asserted not to."""
        xs, ys = _inputs(t=5), _targets(t=5)
        learner = _learner('multi-step')
        with pytest.raises(ValueError, match='multiple|divide'):
            learner.etrace_grad(xs, ys, step_fn=_window_step(learner, K),
                                chunk_size=K)

        ok = _learner('multi-step')
        ok.etrace_grad(xs, ys, step_fn=_plain_step(ok), chunk_size=1)

    @pytest.mark.parametrize('bad', [0, -1, -3])
    def test_a_chunk_size_below_one_is_refused(self, bad):
        """Spec test 10, matching the bound in ``dni.py``."""
        learner = _learner('multi-step')
        with pytest.raises(ValueError, match='chunk_size'):
            learner.etrace_grad(_inputs(), step_fn=lambda x: 0.0, chunk_size=bad)

    @pytest.mark.parametrize('bad', [1.5, '2', jnp.asarray(2)])
    def test_a_non_integer_chunk_size_is_refused(self, bad):
        """Spec test 10, continued."""
        learner = _learner('multi-step')
        with pytest.raises(TypeError, match='chunk_size'):
            learner.etrace_grad(_inputs(), step_fn=lambda x: 0.0, chunk_size=bad)

    @pytest.mark.parametrize('chunk_size,expected', [(None, ()), (1, ()), (K, (K,))])
    def test_step_fn_sees_the_documented_slice_shape(self, chunk_size, expected):
        """Spec test 11.

        The leading window axis is present only at ``k >= 2``; ``None`` and ``1``
        both hand over ``seq[t]``. ``seen`` records what the step function was
        actually given, which is the only way to observe the slicing directly.
        """
        seen = []
        learner = _learner('multi-step')

        def step_fn(x, y):
            seen.append((x.shape, y.shape))
            if chunk_size is None or chunk_size == 1:
                return _sq_error(learner(x), y)
            out = learner(braintrace.MultiStepData(x))
            return jnp.mean((out - y) ** 2, axis=tuple(range(1, out.ndim)))

        learner.etrace_grad(_inputs(), _targets(), step_fn=step_fn,
                            chunk_size=chunk_size)

        x_shape, y_shape = seen[0]
        assert x_shape == expected + (N_IN,)
        assert y_shape == expected + (1, N_REC)

    def test_a_step_fn_returning_the_wrong_rank_is_refused(self):
        """Spec test 12."""
        plain = _learner('multi-step')
        with pytest.raises(ValueError, match='scalar'):
            plain.etrace_grad(_inputs(), step_fn=lambda x: jnp.zeros(3))

        windowed = _learner('multi-step')
        with pytest.raises(ValueError, match=r'\(2,\)|shape'):
            windowed.etrace_grad(_inputs(), step_fn=lambda x: jnp.zeros(5),
                                 chunk_size=K)

    def test_the_window_is_a_real_knob_not_a_formatting_choice(self):
        """Spec test 13.

        Plain and windowed driving must give *different* gradients: the window
        changes the learning rule (P3), and a driver that silently collapsed the
        two would be hiding that. Asserted through the finite-window path, never
        a whole-sequence VJP (F-23).
        """
        xs, ys = _inputs(), _targets()
        a = _learner('multi-step')
        g_plain = a.etrace_grad(xs, ys, step_fn=_plain_step(a), reduction='sum')
        b = _learner('multi-step')
        g_window = b.etrace_grad(xs, ys, step_fn=_window_step(b, K),
                                 chunk_size=K, reduction='sum')
        oracle.assert_gradients_differ(g_plain, g_window)

    def test_window_mode_reproduces_a_hand_written_window_loop_exactly(self):
        """Spec test 13, the exact half.

        The test above says the window *changes* the answer, which is only half
        the claim: the driver must also be pure plumbing within window mode, or
        "different" could mean "different and wrong". Since the change is real,
        the reference has to be built at the same window size rather than
        borrowed from the plain path -- so this is the window-mode counterpart
        of :meth:`TestEquivalenceToTheStatusQuo.
        test_sum_reduction_reproduces_the_hand_written_scan_block`, and it is
        an exact comparison, not a tolerance.
        """
        xs, ys = _inputs(), _targets()

        driver = _learner('multi-step')
        g_driver = driver.etrace_grad(xs, ys, step_fn=_window_step(driver, K),
                                      chunk_size=K, reduction='sum')

        manual = _learner('multi-step')
        weights = manual.param_states
        window = _window_step(manual, K)

        def body(prev, pair):
            x_win, y_win = pair
            g = brainstate.transform.grad(
                lambda a, b: jnp.sum(window(a, b)), weights)(x_win, y_win)
            return jax.tree.map(lambda p, q: p + q, prev, g), None

        init = jax.tree.map(jnp.zeros_like,
                            {k: v.value for k, v in weights.items()})
        g_manual, _ = brainstate.transform.scan(
            body, init,
            (xs.reshape(T // K, K, *xs.shape[1:]),
             ys.reshape(T // K, K, *ys.shape[1:])))

        _assert_trees_equal(g_driver, g_manual,
                            msg='windowed driver vs hand-written window loop')


# ---------------------------------------------------------------------------
# 14--17. Masking
# ---------------------------------------------------------------------------

class TestMasking:
    """A mask gates the loss only; the trace is driven at every step."""

    def test_a_zero_prefix_mask_equals_evolving_over_that_prefix(self):
        """Spec test 14 -- the spec's sharpest claim.

        Two *identically seeded* learners are essential: driving one learner
        through both arms would advance its state between them, so the two sides
        would not describe the same trajectory.
        """
        xs, ys = _inputs(), _targets()
        a_len = 2

        a = _learner()
        a.etrace_evolve(xs[:a_len])
        g1 = a.etrace_grad(xs[a_len:], ys[a_len:], step_fn=_plain_step(a))

        b = _learner()
        mask = jnp.concatenate([jnp.zeros(a_len), jnp.ones(T - a_len)])
        g2 = b.etrace_grad(xs, ys, step_fn=_plain_step(b), mask=mask)

        _assert_trees_equal(g1, g2, rtol=1e-6, atol=1e-7,
                            msg='evolve-prefix vs zero-mask-prefix')

    def test_masking_is_not_the_same_as_shortening_the_sequence(self):
        """Spec test 15 -- the negative control for test 14.

        If masking merely dropped the prefix, a learner that never *saw* the
        prefix would produce the same gradient. It must not: the trace built
        over the zero-weighted steps is exactly what the later steps consume.
        """
        xs, ys = _inputs(), _targets()
        a_len = 2

        masked = _learner()
        mask = jnp.concatenate([jnp.zeros(a_len), jnp.ones(T - a_len)])
        g_masked = masked.etrace_grad(xs, ys, step_fn=_plain_step(masked),
                                      mask=mask)

        truncated = _learner()
        g_truncated = truncated.etrace_grad(
            xs[a_len:], ys[a_len:], step_fn=_plain_step(truncated))

        oracle.assert_gradients_differ(g_masked, g_truncated)

    def test_an_interior_zero_still_drives_the_trace_through_it(self):
        """Spec test 15, the *mid-sequence* half.

        A zero *prefix* can be got right by an implementation that simply skips
        leading zeros -- the trace is still at its initial value there, so
        skipping and evolving coincide. An interior zero cannot: the trace
        arrives loaded, and the steps after it consume what crossing it
        produced. So this compares a mask with a hole against the two
        surviving spans driven as one continuous trajectory, and separately
        against the same spans driven as *independent* runs, which must differ.
        """
        xs, ys = _inputs(), _targets()
        hole = 3
        mask = jnp.ones(T).at[hole].set(0.0)

        holed = _learner()
        g_holed = holed.etrace_grad(xs, ys, step_fn=_plain_step(holed),
                                    mask=mask, reduction='sum')

        # Same trajectory, expressed as three consecutive calls: score the
        # steps before the hole, evolve across it, score the steps after.
        piecewise = _learner()
        g_before = piecewise.etrace_grad(xs[:hole], ys[:hole],
                                         step_fn=_plain_step(piecewise),
                                         reduction='sum')
        piecewise.etrace_evolve(xs[hole:hole + 1])
        g_after = piecewise.etrace_grad(xs[hole + 1:], ys[hole + 1:],
                                        step_fn=_plain_step(piecewise),
                                        reduction='sum')
        g_split = jax.tree.map(lambda a, b: a + b, g_before, g_after)
        _assert_trees_equal(g_holed, g_split, rtol=1e-5, atol=1e-7,
                            msg='interior zero vs evolve-across-the-hole')

        # The negative control: dropping the hole instead of crossing it.
        dropped = _learner()
        kept = jnp.concatenate([jnp.arange(hole), jnp.arange(hole + 1, T)])
        g_dropped = dropped.etrace_grad(xs[kept], ys[kept],
                                        step_fn=_plain_step(dropped),
                                        reduction='sum')
        oracle.assert_gradients_differ(g_holed, g_dropped)

    def test_an_all_zero_mask_gives_exactly_zero_and_stays_finite(self):
        """Spec test 16.

        The ``max(mask.sum(), 1)`` denominator is what keeps this finite; the
        guarantee is scoped to finite, differentiable step losses.
        """
        learner = _learner()
        grads = learner.etrace_grad(
            _inputs(), _targets(), step_fn=_plain_step(learner),
            mask=jnp.zeros(T))
        for k, v in _arrays(grads).items():
            assert np.all(np.isfinite(v)), f'{k} is not finite'
            np.testing.assert_array_equal(v, np.zeros_like(v))

    def test_a_weighted_mask_reweights_the_objective(self):
        """Spec test 17. ``mask`` is a weight vector; binary is just its common case.

        Asserting only "weighted differs from all-ones" is far too weak: an
        implementation that *binarised* the mask -- treating every non-zero
        weight as 1 -- also differs from all-ones, and would pass. (Measured:
        binary-vs-ones ``6.5e-03``, while binary-vs-correct is ``1.0e-01``.)

        So the arithmetic is pinned against an independent construction that
        never multiplies by a weight at all. The trace evolves identically
        whatever the mask holds, so each step's contribution is fixed and the
        weighted gradient is exactly the linear combination of the one-hot
        runs, ``sum_t w_t * g_t``. Recovering it to float32 round-off pins the
        multiply, and the binarised control is run alongside to show the
        tolerance is small enough to catch it.
        """
        xs, ys = _inputs(), _targets()
        weights_vec = jnp.asarray([0.0, 1.0, 2.0, 0.5, 1.0, 0.0])

        a = _learner()
        g_weighted = a.etrace_grad(xs, ys, step_fn=_plain_step(a),
                                   mask=weights_vec, reduction='sum')

        combination = None
        for t in range(T):
            one_hot = _learner()
            g_t = one_hot.etrace_grad(
                xs, ys, step_fn=_plain_step(one_hot),
                mask=jnp.zeros(T).at[t].set(1.0), reduction='sum')
            scaled = jax.tree.map(lambda v, c=float(weights_vec[t]): c * v, g_t)
            combination = scaled if combination is None else jax.tree.map(
                lambda p, q: p + q, combination, scaled)

        # measured 1.5e-08 against a gradient of scale 2.2e-01
        assert _max_abs_diff(g_weighted, combination) < 1e-6, (
            'a weighted mask must scale each step\'s contribution by its own '
            'weight, not merely select the non-zero steps')

        # The mutant this test exists to kill: same support, weights binarised.
        binarised = _learner()
        g_binary = binarised.etrace_grad(
            xs, ys, step_fn=_plain_step(binarised),
            mask=(weights_vec > 0).astype(jnp.float32), reduction='sum')
        assert _max_abs_diff(g_weighted, g_binary) > 1e-3, (
            'the binarised control is indistinguishable here, so the '
            'tolerance above proves nothing -- pick more separated weights')


# ---------------------------------------------------------------------------
# 18--20. vmap
# ---------------------------------------------------------------------------

class _VmapNet(brainstate.nn.Module):
    """A model whose hidden state is created in ``init_state``, not ``__init__``.

    ``om.tanh_rnn`` allocates its ``HiddenState`` in ``__init__``, so
    ``vmap_new_states`` has nothing to batch and ``compile(vmap=True)`` raises
    ``BatchAxisError`` on the first call -- the whole vmap section was
    previously written against a fixture that could not run. ``ValinaRNNCell``
    defers its state to ``init_state``, which is the property that matters.

    ``wout`` is a plain (non-ETP) parameter and receives exact local reverse-mode
    gradients. The vmap fixture runs ``'multi-step'`` to cover sequence-window
    execution as well as ETP routing.
    """

    def __init__(self, n_in=N_IN, n_rec=N_REC, seed=0):
        super().__init__()
        with brainstate.random.seed_context(seed):
            self.cell = braintrace.nn.ValinaRNNCell(n_in, n_rec, activation='tanh')
        self.wout = brainstate.ParamState(
            0.1 * jax.random.normal(jax.random.PRNGKey(seed + 1), (n_rec, n_rec)))

    def update(self, x):
        return self.cell(x) @ self.wout.value


def _vmap_learner(batch, vjp_method='multi-step', **opts):
    return braintrace.compile(_VmapNet(), 'D_RTRL', jnp.zeros((batch, N_IN)),
                              batch_size=batch, vmap=True,
                              vjp_method=vjp_method, **opts)


def _lane_learner(vjp_method='multi-step'):
    """An independently compiled, unbatched twin of one ``_VmapNet`` lane."""
    return braintrace.compile(_VmapNet(), 'D_RTRL', jnp.zeros((1, N_IN)),
                              batch_size=1, vjp_method=vjp_method)


def _lane_data(batch, *, seed=7):
    """Per-lane-*distinct* inputs and targets.

    Tiling one lane across the batch, or zeroing the targets, hides exactly the
    bug this section is for: a driver that mixed or transposed lanes would
    produce the same gradient as one that did not.
    """
    rng = np.random.RandomState(seed)
    xs = 0.5 * jnp.asarray(rng.randn(T, batch, N_IN).astype('float32'))
    ys = 0.5 * jnp.asarray(rng.randn(T, batch, N_REC).astype('float32'))
    return xs, ys


class TestVmap:
    def test_the_vmapped_learner_carries_the_driver_methods(self):
        """Spec test 18.

        ``compile(vmap=True)`` must return something that *has* ``etrace_grad``;
        before this change it returned a bare ``brainstate.nn.Vmap``, which does
        not. Reaching into ``.module`` instead would drive the unbatched learner
        and silently give per-lane-wrong results.
        """
        learner = _vmap_learner(3)
        assert hasattr(learner, 'etrace_grad')
        assert hasattr(learner, 'etrace_evolve')

    def test_the_vmapped_gradient_is_the_sum_over_independent_lanes(self):
        """Spec test 18, the part that has content.

        The parameters are shared across lanes, so the batched gradient must be
        the sum of the gradients each lane would produce on its own. Asserting
        only that the result is *finite* would pass for zeros, for an empty
        tree, and for lane-mixed gradients -- so this builds ``batch``
        independently compiled unbatched learners on per-lane-distinct data and
        adds them up. Measured agreement: ``1.2e-07`` relative.
        """
        batch = 3
        xs, ys = _lane_data(batch)

        batched = _vmap_learner(batch)

        def step_fn(inp, tar):
            return jnp.sum((batched(inp) - tar) ** 2)

        g_batched = batched.etrace_grad(xs, ys, step_fn=step_fn, reduction='sum')

        lane_total = None
        for j in range(batch):
            lane = _lane_learner()

            def lane_step(inp, tar, learner=lane):
                return jnp.sum((learner(inp) - tar) ** 2)

            g_lane = lane.etrace_grad(xs[:, j:j + 1], ys[:, j:j + 1],
                                      step_fn=lane_step, reduction='sum')
            lane_total = g_lane if lane_total is None else jax.tree.map(
                lambda a, b: a + b, lane_total, g_lane)

        flat = _arrays(g_batched)
        assert set(flat) == set(_arrays(lane_total)), 'gradient keys diverged'
        for k, v in flat.items():
            assert np.max(np.abs(v)) > 0.0, f'{k} is identically zero -- vacuous'
        _assert_trees_equal(g_batched, lane_total, rtol=2e-6, atol=1e-6,
                            msg='batched vs sum over independent lanes')

    def test_permuting_the_lanes_changes_the_vmapped_gradient(self):
        """Spec test 18, the negative control.

        Without this, the sum-over-lanes identity could hold for a driver that
        paired inputs with the wrong lane's targets, since a sum is symmetric
        under a *consistent* relabelling. Mis-pairing the two is not.
        """
        batch = 3
        xs, ys = _lane_data(batch)
        perm = jnp.asarray([1, 0, 2])

        straight = _vmap_learner(batch)
        g_straight = straight.etrace_grad(
            xs, ys, step_fn=lambda i, t: jnp.sum((straight(i) - t) ** 2))

        swapped = _vmap_learner(batch)
        g_swapped = swapped.etrace_grad(
            xs, ys[:, perm], step_fn=lambda i, t: jnp.sum((swapped(i) - t) ** 2))

        # measured 1.6e-01 relative
        oracle.assert_gradients_differ(_arrays(g_straight), _arrays(g_swapped),
                                       min_rel=1e-3)

    @pytest.mark.parametrize('batch', [3, K])  # B != k, and the silent B == k case
    def test_window_mode_is_refused_under_vmap(self, batch):
        """Spec test 19.

        ``compile(vmap=True)`` maps ``in_axes=0``, so a ``(k, B, ...)`` window
        slice would map *time* as the batch axis. At ``B != k`` that is a loud
        shape error; at ``B == k`` the shapes line up and it would train on
        transposed data. The ``B == k`` parametrization is the one that matters.
        """
        learner = _vmap_learner(batch)
        xs = jnp.zeros((T, batch, N_IN))

        with pytest.raises(ValueError, match='vmap'):
            learner.etrace_grad(xs, step_fn=lambda x: jnp.zeros(K),
                                chunk_size=K)
        with pytest.raises(ValueError, match='vmap'):
            learner.etrace_evolve(xs, chunk_size=K)

    @pytest.mark.parametrize('batch', [3, K])
    def test_chunk_size_one_is_admitted_under_vmap(self, batch):
        """Spec test 19, the other half -- the refusal must not overreach.

        ``chunk_size=1`` is the plain path, so it carries none of the axis
        collision that makes ``k >= 2`` unsafe. A guard written as
        ``if chunk_size is not None`` would refuse it, which is why both
        methods are exercised rather than just ``etrace_evolve``.
        """
        xs, ys = _lane_data(batch)
        learner = _vmap_learner(batch)
        learner.etrace_evolve(xs, chunk_size=1)

        grad_learner = _vmap_learner(batch)
        grads = grad_learner.etrace_grad(
            xs, ys, chunk_size=1,
            step_fn=lambda i, t: jnp.sum((grad_learner(i) - t) ** 2))
        for k, v in _arrays(grads).items():
            assert np.all(np.isfinite(v)), f'{k} is not finite'

    def test_the_vmap_return_value_is_still_a_brainstate_vmap(self):
        """Spec test 20 -- existing ``vmap=True`` users must be unaffected."""
        learner = _vmap_learner(3)
        assert isinstance(learner, brainstate.nn.Vmap)
        assert isinstance(learner, braintrace.ETraceVmap)
        assert isinstance(learner.module, braintrace.ETraceAlgorithm)


# ---------------------------------------------------------------------------
# 21--26. Surface
# ---------------------------------------------------------------------------

class TestTheReturnSurface:
    """``etrace_grad`` mirrors ``brainstate.transform.grad``'s return arity."""

    def test_all_four_return_value_and_has_aux_combinations(self):
        """Spec test 21."""
        xs, ys = _inputs(), _targets()

        def make(has_aux):
            learner = _learner()

            def step_fn(inp, tar):
                out = learner(inp)
                loss = _sq_error(out, tar)
                return (loss, out) if has_aux else loss

            return learner, step_fn

        learner, step_fn = make(False)
        grads = learner.etrace_grad(xs, ys, step_fn=step_fn)
        assert set(grads) == set(learner.param_states)

        learner, step_fn = make(False)
        grads, losses = learner.etrace_grad(xs, ys, step_fn=step_fn,
                                            return_value=True)
        assert losses.shape == (T,)

        learner, step_fn = make(True)
        grads, aux = learner.etrace_grad(xs, ys, step_fn=step_fn, has_aux=True)
        assert aux.shape[0] == T

        learner, step_fn = make(True)
        grads, losses, aux = learner.etrace_grad(
            xs, ys, step_fn=step_fn, has_aux=True, return_value=True)
        assert losses.shape == (T,)
        assert aux.shape[0] == T

    def test_aux_stacks_over_windows_in_window_mode(self):
        """Spec test 21, window half: the aux leading axis is ``T // k``."""
        learner = _learner('multi-step')

        def step_fn(x_win, y_win):
            out = learner(braintrace.MultiStepData(x_win))
            per_step = jnp.mean((out - y_win) ** 2,
                                axis=tuple(range(1, out.ndim)))
            return per_step, out

        grads, aux = learner.etrace_grad(
            _inputs(), _targets(), step_fn=step_fn, chunk_size=K, has_aux=True)
        assert aux.shape[0] == T // K

    @pytest.mark.parametrize('loss_output,expected_shape', [
        ('per_step', (T,)), ('masked', (T,)), ('scalar', ()),
    ])
    def test_loss_output_selects_what_comes_back(self, loss_output, expected_shape):
        """Spec test 22."""
        learner = _learner()
        _, losses = learner.etrace_grad(
            _inputs(), _targets(), step_fn=_plain_step(learner),
            loss_output=loss_output, return_value=True)
        assert jnp.shape(losses) == expected_shape

    def test_per_step_reports_the_real_loss_where_masked_reports_zero(self):
        """Spec test 22, the distinction that makes ``'per_step'`` the default.

        Monitoring a held-out span is the use case: the step is zero-weighted in
        the objective, but you still want to see what its loss was.
        """
        xs, ys = _inputs(), _targets()
        mask = jnp.asarray([1., 0., 1., 1., 1., 1.])

        a = _learner()
        _, raw = a.etrace_grad(xs, ys, step_fn=_plain_step(a), mask=mask,
                               loss_output='per_step', return_value=True)
        b = _learner()
        _, masked = b.etrace_grad(xs, ys, step_fn=_plain_step(b), mask=mask,
                                  loss_output='masked', return_value=True)

        assert float(raw[1]) > 0.0, 'a zero-weighted step still has a real loss'
        assert float(masked[1]) == 0.0
        np.testing.assert_allclose(np.asarray(raw)[[0, 2, 3, 4, 5]],
                                   np.asarray(masked)[[0, 2, 3, 4, 5]], rtol=1e-6)

    def test_scalar_loss_output_is_the_reduced_objective(self):
        """Spec test 22, continued: ``'scalar'`` removes the trailing ``.mean()``."""
        xs, ys = _inputs(), _targets()
        a = _learner()
        _, per_step = a.etrace_grad(xs, ys, step_fn=_plain_step(a),
                                    loss_output='per_step', return_value=True)
        b = _learner()
        _, scalar = b.etrace_grad(xs, ys, step_fn=_plain_step(b),
                                  loss_output='scalar', return_value=True)
        np.testing.assert_allclose(float(scalar), float(per_step.mean()),
                                   rtol=1e-6)

    def test_weights_can_be_restricted_to_a_subset(self):
        """Spec test 23 -- freezing the rest of the model.

        Three claims, because the key-set assertion alone is nearly free: the
        excluded key is absent, its *value* is untouched by the run, and the
        included key's gradient is the same one the unrestricted run produces.
        The last is what rules out a ``weights`` argument that quietly changed
        the differentiation set as well as the reported keys.

        Run at ``'multi-step'`` so the subset comparison covers the same
        sequence-window route used by the unrestricted control.
        """
        xs, ys = _inputs(), _targets()

        full = _learner('multi-step')
        g_full = full.etrace_grad(xs, ys, step_fn=_plain_step(full))
        assert set(g_full) == {('w',), ('win',)}
        assert np.max(np.abs(_arrays(g_full)['win|'])) > 0.0, (
            'win has no gradient even unrestricted, so excluding it is vacuous')

        learner = _learner('multi-step')
        excluded_before = np.asarray(learner.param_states[('win',)].value)
        subset = {k: v for k, v in learner.param_states.items() if k == ('w',)}
        grads = learner.etrace_grad(xs, ys, step_fn=_plain_step(learner),
                                    weights=subset)

        assert set(grads) == {('w',)}
        np.testing.assert_array_equal(
            np.asarray(learner.param_states[('win',)].value), excluded_before,
            err_msg='an excluded weight was modified by the run')
        _assert_trees_equal({('w',): grads[('w',)]}, {('w',): g_full[('w',)]},
                            rtol=1e-6, atol=1e-7,
                            msg='restricting the reported keys changed the gradient')

    def test_three_sequences_are_sliced_in_lockstep_and_passed_in_order(self):
        """Spec test 24 -- there is no distinguished ``targets`` argument.

        The body is traced once, so appending to a Python list records a single
        ``DynamicJaxprTracer`` and proves nothing about *values*: a driver that
        replayed step 0 six times, permuted the steps, or paired ``xs[t]`` with
        ``ys[t+1]`` would append exactly one tracer too. The slices therefore
        come back through ``aux``, stacked by the scan, and are compared
        against the inputs elementwise.
        """
        learner = _learner()
        xs, ys = _inputs(), _targets()
        zs = jnp.arange(T, dtype=jnp.float32) + 100.0

        def step_fn(inp, tar, tag):
            # Returning the slices makes the scan stack what each step received.
            return _sq_error(learner(inp), tar) + 0.0 * tag, (inp, tar, tag)

        _, (seen_x, seen_y, seen_z) = learner.etrace_grad(
            xs, ys, zs, step_fn=step_fn, has_aux=True)

        np.testing.assert_array_equal(np.asarray(seen_x), np.asarray(xs))
        np.testing.assert_array_equal(np.asarray(seen_y), np.asarray(ys))
        np.testing.assert_array_equal(np.asarray(seen_z), np.asarray(zs))

    def test_the_sequences_are_not_silently_reordered_or_transposed(self):
        """Spec test 24, the control for the identity check above.

        ``assert_array_equal`` against the inputs only bites if a *wrong*
        ordering would actually fail it, which is not obvious when every
        sequence has the same leading length. So the same comparison is run
        against a rolled copy and required to fail.
        """
        learner = _learner()
        xs, ys = _inputs(), _targets()

        def step_fn(inp, tar):
            return _sq_error(learner(inp), tar), inp

        _, seen_x = learner.etrace_grad(xs, ys, step_fn=step_fn, has_aux=True)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(np.asarray(seen_x),
                                          np.asarray(jnp.roll(xs, 1, axis=0)))

    @pytest.mark.parametrize('wrapper', [braintrace.SingleStepData,
                                         braintrace.MultiStepData])
    def test_a_wrapped_sequence_is_refused(self, wrapper):
        """Spec test 25.

        The wrappers say how one *slice* reaches the model, which is
        ``step_fn``'s decision. They are registered pytree nodes, so slicing one
        would decompose the wrapper rather than the data -- silently.
        """
        learner = _learner()
        with pytest.raises(TypeError, match='step_fn|SingleStepData|MultiStepData'):
            learner.etrace_grad(wrapper(_inputs()), step_fn=lambda x: 0.0)

    def test_step_fn_can_supervise_one_head_of_a_multi_head_model(self):
        """Spec test 26 -- the generality that motivates ``step_fn`` owning the call.

        Two *genuinely separate* heads reading the same recurrent state, not two
        terms computed from one output: the point of handing the model call to
        ``step_fn`` is that the driver never needs to know how many outputs
        there are or which of them is supervised. Supervising only the first
        head must leave the second head's weight with an exactly zero
        gradient -- that is the observable consequence, and finiteness alone
        would not show it.
        """

        class TwoHead(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(0), (N_REC, N_REC)))
                self.head_a = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(1), (N_REC, N_REC)))
                self.head_b = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(2), (N_REC, N_REC)))
                self.win = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(3), (N_IN, N_REC)))
                self.h = brainstate.HiddenState(jnp.zeros((1, N_REC)))

            def update(self, x):
                self.h.value = jax.nn.tanh(
                    x @ self.win.value + braintrace.matmul(self.h.value, self.w.value))
                return self.h.value @ self.head_a.value, self.h.value @ self.head_b.value

        learner = braintrace.compile(TwoHead(), 'D_RTRL', _inputs()[0],
                                     batch_size=1, vjp_method='multi-step')
        xs, ys = _inputs(), _targets()

        def step_fn(inp, tar):
            out_a, _out_b = learner(inp)      # head b is computed but unsupervised
            return _sq_error(out_a, tar)

        grads = learner.etrace_grad(xs, ys, step_fn=step_fn, reduction='sum')
        flat = _arrays(grads)
        for k, v in flat.items():
            assert np.all(np.isfinite(v)), f'{k} is not finite'
        assert np.max(np.abs(flat['head_a|'])) > 0.0, 'the supervised head got no gradient'
        np.testing.assert_array_equal(
            flat['head_b|'], np.zeros_like(flat['head_b|']),
            err_msg='an unsupervised head picked up a gradient')

    def test_step_fn_can_read_hidden_states_in_plain_mode(self):
        """Spec test 26, continued -- a firing-rate-style regularizer.

        Scoped to plain mode on purpose: after one ``MultiStepData`` call the
        hidden state holds the *final* step of the window, not a ``(k, ...)``
        history, so this does not generalize (declared limitation 2).
        """
        learner = _learner()
        hidden_key = list(learner.hidden_states)[0]

        def step_fn(inp, tar):
            out = learner(inp)
            h = learner.hidden_states[hidden_key].value
            return _sq_error(out, tar) + 0.01 * jnp.mean(h ** 2)

        grads = learner.etrace_grad(_inputs(), _targets(), step_fn=step_fn)
        for v in _arrays(grads).values():
            assert np.all(np.isfinite(v))


# ---------------------------------------------------------------------------
# 27--28. State lifecycle
# ---------------------------------------------------------------------------

class TestStateLifecycle:
    """Both methods are continuations, not sessions."""

    def test_evolve_then_grad_composes_into_one_trajectory(self):
        """Spec test 27.

        Read from the composition side rather than the masking side: the point
        here is that the second call *continues* from where the first left off,
        with no implicit reset.
        """
        xs, ys = _inputs(), _targets()
        split = 3

        composed = _learner()
        composed.etrace_evolve(xs[:split])
        g_composed = composed.etrace_grad(xs[split:], ys[split:],
                                          step_fn=_plain_step(composed),
                                          reduction='sum')

        whole = _learner()
        mask = jnp.concatenate([jnp.zeros(split), jnp.ones(T - split)])
        g_whole = whole.etrace_grad(xs, ys, step_fn=_plain_step(whole),
                                    mask=mask, reduction='sum')

        _assert_trees_equal(g_composed, g_whole, rtol=1e-6, atol=1e-7,
                            msg='composition')

    def test_repeated_calls_continue_rather_than_reset(self):
        """Spec test 28.

        ``running_index`` alone is a weak witness -- it would keep counting even
        if the hidden state and the eligibility trace were silently reset
        between calls. So the split run is also compared numerically against a
        single run over the concatenated sequence, which is the property users
        actually depend on when they call ``etrace_grad`` once per minibatch.
        """
        xs, ys = _inputs(), _targets()
        learner = _learner()
        assert learner.running_index.value == 0

        g_first = learner.etrace_grad(xs, ys, step_fn=_plain_step(learner),
                                      reduction='sum')
        assert learner.running_index.value == T

        g_second = learner.etrace_grad(xs, ys, step_fn=_plain_step(learner),
                                       reduction='sum')
        assert learner.running_index.value == 2 * T

        # The two calls must compose into one trajectory over `xs` twice.
        whole = _learner()
        g_whole = whole.etrace_grad(
            jnp.concatenate([xs, xs]), jnp.concatenate([ys, ys]),
            step_fn=_plain_step(whole), reduction='sum')
        g_split = jax.tree.map(lambda a, b: a + b, g_first, g_second)
        _assert_trees_equal(g_split, g_whole, rtol=1e-5, atol=1e-7,
                            msg='two calls vs one call over the concatenation')

        # ...and a genuinely reset learner must *not* match, or the above holds
        # for a driver that resets and the test says nothing.
        reset = _learner()
        g_reset = reset.etrace_grad(xs, ys, step_fn=_plain_step(reset),
                                    reduction='sum')
        oracle.assert_gradients_differ(_arrays(g_second), _arrays(g_reset))

    def test_running_index_advances_by_window_length(self):
        """A windowed run records completed timesteps, not update calls."""
        learner = _learner('multi-step')
        learner.etrace_grad(_inputs(), _targets(),
                            step_fn=_window_step(learner, K), chunk_size=K)
        assert learner.running_index.value == T


# ---------------------------------------------------------------------------
# 29--31. Algorithm interactions
# ---------------------------------------------------------------------------

class TestAlgorithmInteractions:
    """The driver must not launder a known limitation into a silent one."""

    def test_window_mode_uses_cumulative_trace_steps(self):
        """Repeated windowed runs accumulate their represented timesteps."""
        xs, ys = _inputs(), _targets()
        learner = _learner('multi-step', algorithm='pp_prop', decay_or_rank=0.9)
        step_fn = _window_step(learner, K)
        learner.etrace_grad(xs, ys, step_fn=step_fn, chunk_size=K)
        assert learner.running_index.value == T
        learner.etrace_grad(xs, ys, step_fn=step_fn, chunk_size=K)
        assert learner.running_index.value == 2 * T

    def test_the_driver_and_dni_agree_on_what_chunk_size_one_means(self):
        """Spec test 31 -- the F-35 correspondence, as an identity.

        ``dni._as_window`` resolves ``chunk_size == 1`` to the *unwrapped* array.
        If the driver ever diverged, a user who fit a synthesiser at the default
        ``chunk_size=1`` and drove at ``etrace_grad(chunk_size=1)`` would have
        fit the plain path and deployed a window -- an F-35 mismatch that
        nothing else checks.
        """
        seq = jnp.zeros((1, 1, N_IN))
        assert not isinstance(_as_window(seq, 1), braintrace.MultiStepData)
        assert isinstance(_as_window(jnp.zeros((K, 1, N_IN)), K),
                          braintrace.MultiStepData)

        seen = []
        learner = _learner('multi-step')

        def step_fn(x):
            seen.append(x)
            return _sq_error(learner(x), 0.0)

        xs = _inputs()
        learner.etrace_grad(xs, step_fn=step_fn, chunk_size=1)
        # Not merely "some other type": the slice must be the bare array, with
        # the step's own shape and no leading window axis.
        assert not isinstance(seen[0], braintrace.MultiStepData)
        assert not isinstance(seen[0], braintrace.SingleStepData)
        assert jnp.shape(seen[0]) == (N_IN,), (
            f'chunk_size=1 handed over {jnp.shape(seen[0])}, not seq[t]')

    def test_a_chunk_size_mismatch_degrades_a_fitted_synthesiser(self):
        """Spec test 30 -- F-35 through the driver's new surface.

        F-35: a synthesiser is valid only for the exact ``(loss_fn,
        chunk_size)`` pair it was fit at, and nothing checks that deployment
        matches. ``chunk_size`` is the surface the driver newly exposes, so the
        regression fits one synthesiser at ``chunk_size=1`` and another at
        ``chunk_size=K``, deploys *both* at ``K``, and requires them to differ:
        the mismatched one is not degraded DNI but noise shaped like a
        cotangent. Measured ``2.0e-02`` relative.

        ``reduction`` is deliberately *not* tested as an F-35 surface. It is a
        uniform rescale applied to the accumulated gradient after the scan and
        never enters the differentiated objective, so it cannot reach the
        synthesiser -- pinned as a regression below.
        """
        spec = om.tanh_rnn(N_IN, N_REC)
        xs = _inputs()

        def build(fit_chunk_size):
            learner = braintrace.compile(spec.factory(), 'dni', xs[0], batch_size=1)
            learner.attach_synthesizer(braintrace.SyntheticGradient(
                learner.group_signal_shapes(), scale=0.1, seed=0))
            if fit_chunk_size is not None:
                train_synthetic_gradient(learner, xs, chunk_size=fit_chunk_size,
                                         epochs=3, lr=0.05)
            return learner

        def drive(learner):
            return learner.etrace_grad(xs, step_fn=_window_step_sq(learner),
                                       chunk_size=K, reduction='sum')

        matched = build(K)
        g_matched = drive(matched)
        mismatched = build(1)
        g_mismatched = drive(mismatched)

        # The precondition: a fitted synthesiser must actually move the
        # gradient, or "matched vs mismatched" would compare two no-ops. A
        # freshly built ``SyntheticGradient`` is zero-initialised and predicts
        # exactly zero, which is the natural control. Measured 1.3e-01.
        untrained = braintrace.compile(spec.factory(), 'dni', xs[0], batch_size=1)
        untrained.attach_synthesizer(
            braintrace.SyntheticGradient(untrained.group_signal_shapes()))
        oracle.assert_gradients_differ(_arrays(drive(untrained)),
                                       _arrays(g_matched), min_rel=1e-3)

        oracle.assert_gradients_differ(_arrays(g_mismatched), _arrays(g_matched),
                                       min_rel=1e-3)

    def test_reduction_is_a_uniform_rescale_even_under_dni(self):
        """Spec test 30, continued -- the claim that ``reduction`` is *not* F-35.

        This started life as an F-35 test asserting that ``'mean'`` and
        ``'sum'`` diverge under a fitted synthesiser. They do not: measured
        ``|mean - sum/T| = 7.5e-09``, i.e. float32 round-off on the division.
        The driver divides once, after the scan, so the reduction is invisible
        to everything inside it -- including DNI's bootstrapped cotangent.

        Kept as a regression because the natural "fix" for a reduction-shaped
        mismatch would be to fold the scale into the differentiated objective,
        which *would* silently change what every synthesiser was fit against.
        """
        spec = om.tanh_rnn(N_IN, N_REC)
        xs = _inputs()

        def build():
            learner = braintrace.compile(spec.factory(), 'dni', xs[0], batch_size=1)
            learner.attach_synthesizer(braintrace.SyntheticGradient(
                learner.group_signal_shapes(), scale=0.1, seed=0))
            train_synthetic_gradient(learner, xs, chunk_size=K, epochs=3, lr=0.05)
            return learner

        summed = build()
        g_sum = summed.etrace_grad(xs, step_fn=_window_step_sq(summed),
                                   chunk_size=K, reduction='sum')
        meaned = build()
        g_mean = meaned.etrace_grad(xs, step_fn=_window_step_sq(meaned),
                                    chunk_size=K, reduction='mean')

        # The denominator is the total mask weight -- one entry per *step*, not
        # per window -- so it stays ``T`` however the sequence is chunked.
        _assert_trees_equal(g_mean, jax.tree.map(lambda v: v / T, g_sum),
                            rtol=1e-6, atol=1e-7,
                            msg='reduction under a fitted synthesiser')


def _window_step_sq(learner):
    def step_fn(x_win):
        out = learner(braintrace.MultiStepData(x_win))
        return (out ** 2).sum(axis=tuple(range(1, out.ndim)))
    return step_fn


# ---------------------------------------------------------------------------
# 32--36. Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_unit_carrying_weights_survive_the_whole_pipeline(self):
        """Spec test 32 (P6).

        The accumulator is built from parameter *values*, which is only sound
        because a gradient carries the parameter's unit. This exercises the
        accumulator, the mask multiply and the mean division together.

        Stripping the mantissa and checking finiteness -- the obvious way to
        write this -- passes for a dimensionless result, for zeros, and for an
        empty tree, i.e. for every way the units could actually be lost. So the
        assertions are on the unit itself: ``w`` is in mV, its gradient must
        also be in mV, and ``zeros_like(w) + grad`` (what the accumulator does
        on the first step) must not raise a dimension error.

        The model's input is ``(T, 1, n_in)``: the ETP op must see the same
        batch rank as the hidden state, or the trace state changes shape on the
        first ``update()`` and no scan-based driver -- this one or a
        hand-written block -- can carry it.
        """
        spec = om.unit_weight_rnn(N_IN, N_REC)
        model = spec.factory()
        rng = np.random.RandomState(0)
        xs = jnp.asarray(np.abs(rng.randn(T, 1, N_IN)).astype('float32'))
        learner = braintrace.compile(model, 'D_RTRL', xs[0], batch_size=1,
                                     vjp_method='multi-step')

        def step_fn(inp):
            return jnp.sum(u.get_mantissa(learner(inp)) ** 2)

        grads = learner.etrace_grad(
            xs, step_fn=step_fn,
            mask=jnp.asarray([1., 0., 1., 1., 0., 1.]), reduction='mean')

        assert set(grads) == {('w',)}, f'unexpected gradient keys: {set(grads)}'
        g = grads[('w',)]
        assert isinstance(g, u.Quantity), (
            f'the gradient of an mV parameter came back as {type(g).__name__}, '
            f'so the unit was stripped somewhere in the pipeline')
        assert g.unit == u.mV, f'expected mV, got {g.unit}'
        assert jnp.shape(u.get_mantissa(g)) == (N_IN, N_REC)
        mantissa = np.asarray(u.get_mantissa(g))
        assert np.all(np.isfinite(mantissa))
        assert np.max(np.abs(mantissa)) > 0.0, 'gradient is identically zero'

        # The accumulator's first step, which is where a stripped unit surfaces
        # as a brainunit dimension error rather than a wrong number.
        param = learner.param_states[('w',)].value
        assert isinstance(u.math.zeros_like(param) + g, u.Quantity)

    def test_no_sequences_is_refused(self):
        """Spec test 33 -- ``T`` is undefined with nothing to slice."""
        learner = _learner()
        with pytest.raises(ValueError, match='sequence'):
            learner.etrace_grad(step_fn=lambda: 0.0)

    def test_mismatched_sequence_lengths_are_refused(self):
        """Spec test 33, continued."""
        learner = _learner()
        with pytest.raises(ValueError, match='length|leading'):
            learner.etrace_grad(_inputs(t=T), _targets(t=T - 1),
                                step_fn=lambda x, y: 0.0)

    def test_an_empty_sequence_is_refused(self):
        """Spec test 33, continued."""
        learner = _learner()
        with pytest.raises(ValueError, match='empty|zero|T'):
            learner.etrace_grad(_inputs(t=0), step_fn=lambda x: 0.0)

    @pytest.mark.parametrize('empty', [{}, (), [], {'a': {}}])
    def test_an_empty_pytree_is_refused_with_a_real_error(self, empty):
        """Spec test 33, continued -- length-0 in the *pytree* sense, not the array sense.

        An empty container holds no leaf, so the length scan never runs and
        ``length`` stays ``None``. That used to fall through to an internal
        ``assert``, which is a stack trace about the driver's invariants rather
        than a message about the caller's argument.
        """
        learner = _learner()
        with pytest.raises(ValueError, match='empty pytree'):
            learner.etrace_grad(empty, step_fn=lambda x: 0.0)
        with pytest.raises(ValueError, match='empty pytree'):
            learner.etrace_evolve(empty)

    def test_a_bad_mask_shape_is_refused(self):
        """Spec test 33, continued."""
        learner = _learner()
        with pytest.raises(ValueError, match='mask'):
            learner.etrace_grad(_inputs(), step_fn=lambda x: 0.0,
                                mask=jnp.ones(T + 1))

    @pytest.mark.parametrize('kwargs', [
        {'reduction': 'median'}, {'loss_output': 'everything'},
    ])
    def test_an_illegal_enum_value_is_refused(self, kwargs):
        """Spec test 33, continued."""
        learner = _learner()
        with pytest.raises(ValueError, match='reduction|loss_output'):
            learner.etrace_grad(_inputs(), step_fn=lambda x: 0.0, **kwargs)

    def test_step_fn_is_required_for_etrace_grad(self):
        """Spec test 33, continued -- there is no default loss."""
        learner = _learner()
        with pytest.raises(TypeError, match='step_fn'):
            learner.etrace_grad(_inputs())

    def test_an_uncompiled_learner_is_refused_by_the_existing_guard(self):
        """Spec test 33, continued -- the driver adds no guard of its own.

        The spec claimed a ``RuntimeError`` here; the shipped guard raises
        ``ValueError`` ("The etrace algorithm has not been compiled"), and the
        driver deliberately does not pre-empt it -- one guard, one message, in
        the place that already owned it. Pinned so the spec and the code cannot
        drift apart again.
        """
        model = om.tanh_rnn(N_IN, N_REC).factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        uncompiled = braintrace.D_RTRL(model, vjp_method='single-step')

        with pytest.raises(ValueError, match='compile'):
            uncompiled.etrace_evolve(_inputs())
        with pytest.raises(ValueError, match='compile'):
            uncompiled.etrace_grad(
                _inputs(), step_fn=lambda x: jnp.sum(uncompiled(x) ** 2))

    def test_evolve_without_a_step_fn_drives_the_learner_directly(self):
        """Spec test 34.

        With no ``step_fn`` every sequence is by definition a model input, so
        the driver may wrap in ``MultiStepData`` itself -- which is what keeps
        the warm-up a one-liner.
        """
        plain = _learner()
        plain.etrace_evolve(_inputs())
        assert plain.running_index.value == T

        windowed = _learner('multi-step')
        windowed.etrace_evolve(_inputs(), chunk_size=K)
        assert windowed.running_index.value == T

    def test_evolve_with_a_custom_step_fn_does_not_wrap(self):
        """Spec test 34, continued -- supplying ``step_fn`` opts out of wrapping."""
        learner = _learner('multi-step')
        seen = []

        def step_fn(x):
            seen.append(type(x))
            return learner(braintrace.MultiStepData(x))

        learner.etrace_evolve(_inputs(), step_fn=step_fn, chunk_size=K)
        assert seen[0] is not braintrace.MultiStepData

    def test_return_outputs_controls_whether_anything_is_stacked(self):
        """Spec test 35 -- a long warm-up must not pay for outputs nobody reads."""
        quiet = _learner()
        assert quiet.etrace_evolve(_inputs()) is None

        loud = _learner()
        outs = loud.etrace_evolve(_inputs(), return_outputs=True)
        assert outs.shape[0] == T

        windowed = _learner('multi-step')
        outs = windowed.etrace_evolve(_inputs(), chunk_size=K,
                                      return_outputs=True)
        assert outs.shape[0] == T // K

    def test_a_mid_loop_failure_leaves_the_state_untouched(self):
        """Spec test 36, the failure row.

        The spec originally claimed a mid-loop failure "leaves partially
        advanced state", which is false: the body is traced into a functional
        ``scan``, so a step that raises at step 3 aborts before any state is
        written back and the learner is left exactly where it started. Measured
        by executing it: ``running_index`` stayed ``0`` and the hidden state was
        unchanged.

        That is the better guarantee, and users will build on it, so it is
        pinned rather than left implicit.
        """
        learner = _learner()
        hidden_key = list(learner.hidden_states)[0]
        before_index = int(learner.running_index.value)
        before_hidden = np.asarray(learner.hidden_states[hidden_key].value)

        class _Boom(RuntimeError):
            pass

        def exploding_step(inp, tar):
            out = learner(inp)
            raise _Boom('failure inside the loop body')

        with pytest.raises(_Boom):
            learner.etrace_grad(_inputs(), _targets(), step_fn=exploding_step)

        assert int(learner.running_index.value) == before_index
        np.testing.assert_array_equal(
            np.asarray(learner.hidden_states[hidden_key].value), before_hidden,
            err_msg='a failed run wrote state back')

        # ...and the learner is still usable afterwards.
        learner.etrace_grad(_inputs(), _targets(), step_fn=_plain_step(learner))
        assert int(learner.running_index.value) == before_index + T

    def test_both_methods_work_inside_an_outer_jit(self):
        """Spec test 36 -- neither method may call ``jit`` itself."""
        learner = _learner()

        @brainstate.transform.jit
        def train(xs, ys):
            learner.etrace_evolve(xs[:2])
            return learner.etrace_grad(xs[2:], ys[2:],
                                       step_fn=_plain_step(learner),
                                       loss_output='scalar', return_value=True)

        grads, loss = train(_inputs(), _targets())
        assert np.isfinite(float(loss))
        for v in _arrays(grads).values():
            assert np.all(np.isfinite(v))
