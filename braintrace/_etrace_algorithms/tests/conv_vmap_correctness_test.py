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

"""Conv / mixed ETP under ``brainstate.nn.Vmap(vmap_states='new')`` correctness.

Regression coverage for the eligibility-trace path through the *batched* online
executor wrapped by ``brainstate.nn.Vmap`` (the ``OnlineVmapTrainer`` flow used
by ``examples/004``). This path was previously uncovered — conv was exercised
only at the rule level (``_etrace_op/conv_test.py``) and the "conv" model in
``transform_correctness_test`` is actually a matmul — which let two regressions
through:

1. *Pure conv.* A conv forward forces a leading batch axis on its input, but
   under ``vmap_states='new'`` the hidden-state traces are per-lane and carry no
   batch axis, so the instantaneous, recurrent and solve terms saw a singleton
   batch on the input but none on the cotangent.

2. *Mixed batched + unbatched.* When a conv (batched primitive) is composed with
   a ``Linear`` that, compiled per-lane, dispatches to the *unbatched* ``etp_mv``
   (1-D input), the solve's trailing batch-sum was gated on a global "any
   relation batched" flag and so collapsed the *unbatched* Linear gradient's
   leading in-feature axis too — the ``examples/004`` layer4 failure.

3. *Norm in the transition.* ``conv -> LayerNorm -> IF`` makes ``dh/dy``
   non-diagonal; the all-ones jvp returns its row sums, exactly zero for the
   shift-invariant norm. A ``use_fast_variance=True`` norm leaves a float32
   residual instead, which under ``vmap_states='new'`` the recurrent trace and
   ``rsqrt(var+eps)`` amplify into an overflow that diverges from the eager
   reference — the ``examples/004`` ``loss=ln(10)`` stall.

**Oracle (exact, transform-invariance).** ``brainstate.nn.Vmap`` is a transform;
for parameters shared across lanes its grad sums the per-lane gradients. So the
gradient from the ``vmap_new_states`` + ``Vmap`` path on a batch of ``B`` samples
must equal the sum over ``b`` of the *eager, batch=1* gradient on sample ``b``.
The eager batch=1 path is independently healthy for conv (states are initialised
*with* a size-1 batch, so input and trace batch axes agree), which makes it a
trustworthy reference regardless of D_RTRL's approximation quality.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
import braintrace
import braintools
import brainpy.state

H = W = 6
C_IN = 2
C_OUT = 3
B = 2
N_STEP = 3
SEED = 0


@pytest.fixture(autouse=True)
def _dt():
    # Spiking-neuron dynamics need a simulation time step in the environment.
    #
    # These tests assert an *exact* transform-invariance identity (vmap grad ==
    # sum of eager grads) to a tight rtol. On GPU, conv/matmul default to TF32
    # (~1e-3 precision), and the vmap vs eager paths schedule those reduced-
    # precision kernels differently -> a ~1% disagreement that has nothing to do
    # with the etrace transform under test (the identity holds to ~1e-6 in true
    # float32, as it does on CPU). Force highest matmul precision so the tolerance
    # measures the transform, not the accelerator. Scoped per-test via the context
    # manager, so it never leaks into other test modules sharing the process.
    with jax.default_matmul_precision('highest'), brainstate.environ.context(dt=1.0):
        yield


def _make_net(neuron):
    """Fresh conv -> spiking-neuron net with deterministic weights.

    Pure conv -> neuron (no readout/flatten, which add their own batch handling)
    isolates the conv ETP path. ``neuron='IF'`` gives a single-state group
    (num_state == 1); ``neuron='ALIF'`` a coupled (V, a) group (num_state == 2),
    exercising the multi-state recurrent + solve branches.
    """
    brainstate.random.seed(SEED)  # identical conv weights on every build
    conv_inits = dict(w_init=braintools.init.XavierNormal(scale=5.0), b_init=None)
    surr = braintools.surrogate.Arctan()

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = braintrace.nn.Conv2d(
                (H, W, C_IN), C_OUT, kernel_size=3, padding=1, **conv_inits
            )
            if neuron == 'IF':
                self.cell = brainpy.state.IF(
                    self.conv.out_size, V_th=1.0, tau=2.0, spk_fun=surr,
                    V_initializer=braintools.init.ZeroInit(), R=1.,
                )
            else:  # 'ALIF' — dimensionless config (num_state == 2)
                self.cell = brainpy.state.ALIF(
                    self.conv.out_size, V_th=1.0, V_reset=0.0, V_rest=0.0, R=1.0,
                    tau=2.0, tau_a=20.0, beta=0.1, spk_fun=surr,
                    V_initializer=braintools.init.ZeroInit(),
                )

        def update(self, x):
            return self.cell(self.conv(x))

    return Net()


def _loss(out, target):
    # Sum (not mean) so vmap(batch) == sum_b eager(b) exactly.
    return ((out - target) ** 2).sum()


def _eager_grad_one(sample_seq, target, make_net):
    """Eager, batch=1 D_RTRL gradient accumulated over one sample's sequence."""
    net = make_net()
    brainstate.nn.init_all_states(net, batch_size=1)
    algo = braintrace.D_RTRL(net)
    with brainstate.environ.context(fit=True):
        algo.compile_graph(sample_seq[0][None])
    algo.init_etrace_state()
    weights = net.states().subset(brainstate.ParamState)
    grads = jax.tree.map(jnp.zeros_like, weights.to_dict_values())
    for t in range(N_STEP):
        def loss_fn(x):
            with brainstate.environ.context(fit=True):
                return _loss(algo.update(x), target[None])
        g, _ = brainstate.transform.grad(loss_fn, weights, return_value=True)(sample_seq[t][None])
        grads = jax.tree.map(lambda a, b: a + b, grads, g)
    return grads


def _vmap_grad(data, targets, make_net):
    """The ``OnlineVmapTrainer`` flow: vmap_new_states init + Vmap(vmap_states='new')."""
    net = make_net()
    model = braintrace.D_RTRL(net)

    @brainstate.transform.vmap_new_states(state_tag='new', axis_size=data.shape[1])
    def init():
        brainstate.nn.init_all_states(net)
        with brainstate.environ.context(fit=True):
            model.compile_graph(data[0, 0])

    init()
    vmodel = brainstate.nn.Vmap(model, vmap_states='new')
    weights = net.states().subset(brainstate.ParamState)

    def _grad(inp):
        with brainstate.environ.context(fit=True):
            return _loss(vmodel(inp), targets)

    def _step(prev, x):
        g = brainstate.transform.grad(_grad, weights)(x)
        return jax.tree.map(lambda a, b: a + b, prev, g), None

    grads = jax.tree.map(jnp.zeros_like, weights.to_dict_values())
    grads, _ = brainstate.transform.scan(_step, grads, data)
    return grads


N_HID = 7  # mixed-net Linear out features (!= flattened conv size, so a wrong
           # in-feature reduction is unmistakable in the gradient shape)


def _make_mixed_net():
    """conv -> IF -> flatten -> Linear -> IF: a *mixed* batched/unbatched model.

    Under ``vmap_states='new'`` the graph is compiled per-lane, so the conv stays
    a *batched* primitive (its parent layer forces a leading batch axis) while the
    flattened ``Linear`` input is 1-D and dispatches to the *unbatched* ``etp_mv``.
    The solve's trailing batch-sum must collapse only the conv gradient's batch
    axis; applying it to the unbatched ``Linear`` grad would reduce its leading
    in-feature axis ([flat, N_HID] -> [N_HID]) — the ``examples/004`` regression.
    """
    brainstate.random.seed(SEED)
    surr = braintools.surrogate.Arctan()
    if_param = dict(V_th=1.0, tau=2.0, spk_fun=surr,
                    V_initializer=braintools.init.ZeroInit(), R=1.)

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = braintrace.nn.Conv2d(
                (H, W, C_IN), C_OUT, kernel_size=3, padding=1,
                w_init=braintools.init.XavierNormal(scale=5.0), b_init=None,
            )
            self.cell1 = brainpy.state.IF(self.conv.out_size, **if_param)
            self._flat = int(np.prod(self.conv.out_size))
            self.fc = braintrace.nn.Linear(
                self._flat, N_HID, b_init=None,
                w_init=braintools.init.XavierNormal(scale=5.0),
            )
            self.cell2 = brainpy.state.IF((N_HID,), **if_param)

        def update(self, x):
            s1 = self.cell1(self.conv(x))
            flat = s1.reshape(s1.shape[:-3] + (self._flat,))
            return self.cell2(self.fc(flat))

    return Net()


def _make_conv_ln_net(use_fast_variance):
    """conv -> LayerNorm -> IF: a non-elementwise (mean-subtracting) transition.

    The param-dim trace reads ``dh/dy`` through the norm via an all-ones jvp; for
    a shift-invariant op its value is the Jacobian row sums = *exactly* zero (the
    upstream conv gets no eligibility gradient through the norm — a documented
    approximation, matching the eager path). That exactness is numerical: with
    ``use_fast_variance=True`` the ``E[x^2]-E[x]^2`` variance leaves a float32
    residual instead of zero, and under ``Vmap(vmap_states='new')`` the recurrent
    trace and the large ``rsqrt(var+eps)`` factor amplify it into an overflow that
    diverges from the eager reference (the ``examples/004`` ``loss=ln(10)`` stall).
    """
    brainstate.random.seed(SEED)
    conv_inits = dict(w_init=braintools.init.XavierNormal(scale=5.0), b_init=None)
    surr = braintools.surrogate.Arctan()

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = braintrace.nn.Conv2d(
                (H, W, C_IN), C_OUT, kernel_size=3, padding=1, **conv_inits
            )
            self.ln = brainstate.nn.LayerNorm(
                self.conv.out_size, use_fast_variance=use_fast_variance
            )
            self.cell = brainpy.state.IF(
                self.conv.out_size, V_th=1.0, tau=2.0, spk_fun=surr,
                V_initializer=braintools.init.ZeroInit(), R=1.,
            )

        def update(self, x):
            return self.cell(self.ln(self.conv(x)))

    return Net()


def _assert_grads_match(ref, got):
    ref_leaves = jax.tree.leaves(ref)
    got_leaves = jax.tree.leaves(got)
    assert len(ref_leaves) == len(got_leaves) and ref_leaves, 'no gradient leaves compared'
    for e, a in zip(ref_leaves, got_leaves):
        e, a = np.asarray(e), np.asarray(a)
        assert a.shape == e.shape, f'shape mismatch: got {a.shape} vs ref {e.shape}'
        # Relative tolerance: conv grads here are large-magnitude; float32 ~1e-6.
        np.testing.assert_allclose(a, e, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('neuron', ['IF', 'ALIF'], ids=['num_state1_IF', 'num_state2_ALIF'])
def test_conv_vmap_grad_equals_sum_of_eager_single_sample(neuron):
    """vmap_new_states+Vmap conv D_RTRL grad == sum over samples of eager batch=1 grad."""
    rng = np.random.RandomState(42)
    data = jnp.asarray(rng.rand(N_STEP, B, H, W, C_IN).astype('float32'))
    targets = jnp.asarray(rng.rand(B, H, W, C_OUT).astype('float32'))

    make_net = lambda: _make_net(neuron)
    ref = None
    for b in range(B):
        g = _eager_grad_one(data[:, b], targets[b], make_net)
        ref = g if ref is None else jax.tree.map(lambda a, c: a + c, ref, g)

    got = _vmap_grad(data, targets, make_net)
    _assert_grads_match(ref, got)


def test_mixed_conv_dense_vmap_grad_equals_sum_of_eager_single_sample():
    """Mixed batched(conv)+unbatched(dense-mv) model: vmap grad == sum of eager batch=1.

    Regression for the ``examples/004`` layer4 failure — the unbatched ``etp_mv``
    Linear gradient was being batch-summed because a *different* relation (conv)
    was batched, collapsing its in-feature axis.
    """
    rng = np.random.RandomState(42)
    data = jnp.asarray(rng.rand(N_STEP, B, H, W, C_IN).astype('float32'))
    targets = jnp.asarray(rng.rand(B, N_HID).astype('float32'))

    ref = None
    for b in range(B):
        g = _eager_grad_one(data[:, b], targets[b], _make_mixed_net)
        ref = g if ref is None else jax.tree.map(lambda a, c: a + c, ref, g)

    got = _vmap_grad(data, targets, _make_mixed_net)
    _assert_grads_match(ref, got)


def test_conv_layernorm_vmap_grad_matches_eager_and_stays_finite():
    """conv -> LayerNorm -> IF: vmap grad == sum of eager batch=1, and stays finite.

    Regression for the ``examples/004`` ``loss=ln(10)`` stall. A mean-subtracting
    norm makes ``dh/dy`` non-diagonal; the param-dim trace's all-ones jvp returns
    its row sums, which for shift-invariance are exactly zero, so the conv weight
    gets no eligibility gradient through the norm (both paths agree on ~0). With a
    numerically stable variance (``use_fast_variance=False``) that exact zero holds
    under ``Vmap(vmap_states='new')``; the transform-invariance oracle then makes
    vmap == sum-of-eager, and neither explodes.
    """
    rng = np.random.RandomState(42)
    data = jnp.asarray(rng.rand(N_STEP, B, H, W, C_IN).astype('float32'))
    targets = jnp.asarray(rng.rand(B, H, W, C_OUT).astype('float32'))

    make_net = lambda: _make_conv_ln_net(use_fast_variance=False)
    ref = None
    for b in range(B):
        g = _eager_grad_one(data[:, b], targets[b], make_net)
        ref = g if ref is None else jax.tree.map(lambda a, c: a + c, ref, g)

    got = _vmap_grad(data, targets, make_net)
    # No overflow/NaN in either path (the bug produced ~1e14 -> NaN under vmap).
    for leaf in jax.tree.leaves(got):
        assert np.all(np.isfinite(np.asarray(leaf))), 'vmap grad is non-finite'
    for leaf in jax.tree.leaves(ref):
        assert np.all(np.isfinite(np.asarray(leaf))), 'eager grad is non-finite'
    _assert_grads_match(ref, got)
