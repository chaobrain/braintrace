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

"""U6 — the projector ``nu^T J_f``, pinned per primitive against ``jax.vjp``.

The engine claims that ``nu^T J_f`` needs no new per-primitive kernel: it is
``solve_rule(nu, instant_rule(x, df, weights))``, the same composition the
framework already uses for the instantaneous part of a gradient. A claim of that
shape can only be tested per primitive, against an *independent* reference —
finite gradients do not test a projector, and neither does agreement with another
braintrace path.

The reference is ``jax.vjp`` of each model's own drive map. Every fixture here is
a leaky integrator ``h <- LEAK * h + drive(x, w)``, so ``d h / d drive == 1`` and
the ``y -> hidden`` tail Jacobian ``df`` is all ones. The projector must then be
exactly ``vjp(lambda w: drive(x, w), w)(nu)``.

How ``proj`` is recovered from the public carrier: after one step from a zero
carrier, ``s_tilde == rho1 * nu`` and ``theta_tilde == proj / rho1``, so with a
*known* ``nu`` (a tabulated draw of ±1, hence ``||nu|| == sqrt(size)``),

    proj == theta_tilde * ||s_tilde|| / ||nu||

with no access to a private normaliser. The recovery itself is checked against a
hand-computed dense case before being trusted for the rest.
"""

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._testing.oracle_test import _FAMILIES, _xs_for

LEAK = 0.9


class _TabulatedUORO(braintrace.UORO):
    """UORO whose ``nu`` is a fixed array (see ``uoro_test`` for why a table)."""

    def __init__(self, model, nu, **kwargs):
        super().__init__(model, **kwargs)
        self._nu = jnp.asarray(nu)

    def _draw_projection(self, key, step, group_index, shape, dtype):
        return jnp.reshape(self._nu, shape).astype(dtype)


def _one_step_projection(factory, x, nu_sign=1.0):
    """Run one step and recover ``{path: proj}`` from the public carrier.

    Returns ``(proj, nu, model)`` where ``nu`` is the draw that was actually
    used, shaped like the group's hidden factor.
    """
    model = factory()
    brainstate.nn.init_all_states(model, batch_size=1)

    probe = braintrace.UORO(model, vjp_method='multi-step')
    probe.compile_graph(braintrace.MultiStepData(x[None]))
    probe.init_etrace_state()
    shape = probe._get_etrace_data()['s_tilde'][0].shape

    # Deterministic ±1 pattern: alternate signs so a projector that ignores nu's
    # sign structure (or contracts the wrong axis) cannot pass by symmetry.
    flat = np.where(np.arange(int(np.prod(shape))) % 2 == 0, 1.0, -1.0)
    nu = nu_sign * flat.reshape(shape)

    model = factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = _TabulatedUORO(model, nu, vjp_method='multi-step')
    algo.compile_graph(braintrace.MultiStepData(x[None]))
    algo.init_etrace_state()
    algo(braintrace.MultiStepData(x[None]))

    data = algo._get_etrace_data()
    s = np.asarray(u.get_mantissa(data['s_tilde'][0]))
    rho1 = float(np.linalg.norm(s) / np.linalg.norm(nu))
    proj = {}
    for (gi, path), th in data['theta_tilde'].items():
        assert gi == 0, 'these fixtures have exactly one hidden group'
        proj[path] = jax.tree.map(lambda a: a * rho1, th)
    return proj, nu, model


def _vjp_reference(factory, x, nu, drive_fn, param_paths):
    """``vjp(lambda params: drive(x, params), params)(nu)`` — independent."""
    model = factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    params = model.states(brainstate.ParamState)
    vals = {p: params[p].value for p in param_paths}

    def f(vs):
        return drive_fn(model, x, vs)

    out, pullback = jax.vjp(f, vals)
    # nu carries the trailing state axis; the drive does not.
    ct = jnp.reshape(jnp.asarray(nu), jnp.shape(out))
    return pullback(ct)[0]


def _assert_close(got, want, label, *, atol=1e-5, rtol=1e-4):
    got_l = jax.tree.leaves(jax.tree.map(lambda a: np.asarray(u.get_mantissa(a)), got))
    want_l = jax.tree.leaves(jax.tree.map(lambda a: np.asarray(u.get_mantissa(a)), want))
    assert len(got_l) == len(want_l), f'{label}: leaf count {len(got_l)} != {len(want_l)}'
    for g, w in zip(got_l, want_l):
        assert g.shape == w.shape, f'{label}: shape {g.shape} != {w.shape}'
        np.testing.assert_allclose(g, w, atol=atol, rtol=rtol,
                                   err_msg=f'{label}: projector != jax.vjp')


# ---------------------------------------------------------------------------
# The recovery recipe itself, pinned by hand before it is trusted.
# ---------------------------------------------------------------------------

def test_the_recovery_recipe_matches_a_hand_computed_dense_projection():
    factory, seed = _FAMILIES['dense_mm']
    x = _xs_for('dense_mm', 1, seed)[0]
    proj, nu, model = _one_step_projection(factory, x)

    # dense mm drive == x @ w, df == 1 (the tail is a plain leak), so
    # nu^T J_f == outer(x, nu).
    want = np.einsum('bi,bo->io', np.asarray(x), np.asarray(nu)[..., 0])
    _assert_close(proj[('w',)], want, 'dense_mm hand-computed')


# ---------------------------------------------------------------------------
# The per-primitive sweep.
# ---------------------------------------------------------------------------

_DRIVES = {
    'dense_mm': lambda m, x, v: braintrace.matmul(
        x, v[('w',)], bias=v[('b',)]),
    'dense_mv': lambda m, x, v: braintrace.matmul(x, v[('w',)]),
    'elemwise': lambda m, x, v: x * braintrace.element_wise(v[('w',)]),
    'conv_default': lambda m, x, v: braintrace.conv(
        x, v[('k',)], strides=(1,), padding='SAME'),
    'conv_nwc_bias': lambda m, x, v: braintrace.conv(
        x, v[('k',)], v[('b',)], strides=(1,), padding='SAME',
        dimension_numbers=('NWC', 'WIO', 'NWC')),
    'sparse_unbatched': None,   # filled in below (needs the CSR)
    'sparse_batched': None,
    'lora': lambda m, x, v: braintrace.lora_matmul(
        x, v[('B',)], v[('A',)], alpha=2.0, bias=v[('bias',)]),
}

_PARAMS = {
    'dense_mm': [('w',), ('b',)],
    'dense_mv': [('w',)],
    'elemwise': [('w',)],
    'conv_default': [('k',)],
    'conv_nwc_bias': [('k',), ('b',)],
    'sparse_unbatched': [('w',)],
    'sparse_batched': [('w',)],
    'lora': [('B',), ('A',), ('bias',)],
}


def _sparse_drive(m, x, v):
    from braintrace._testing.oracle_test import _sparse_csr
    csr, _ = _sparse_csr()
    return braintrace.sparse_matmul(x, v[('w',)], sparse_mat=csr)


_DRIVES['sparse_unbatched'] = _sparse_drive
_DRIVES['sparse_batched'] = _sparse_drive


@pytest.mark.parametrize('name', sorted(_DRIVES))
def test_the_projector_equals_jax_vjp_per_primitive(name):
    factory, seed = _FAMILIES[name]
    x = _xs_for(name, 1, seed)[0]
    proj, nu, _ = _one_step_projection(factory, x)
    want = _vjp_reference(factory, x, nu, _DRIVES[name], _PARAMS[name])

    assert set(proj) == set(want), (
        f'{name}: projector covers {sorted(proj)}, vjp covers {sorted(want)}')
    for path in want:
        _assert_close(proj[path], want[path], f'{name} {path}')


def test_the_conv_bias_projection_is_the_spatial_reduction():
    # Roadmap lesson 7: a conv bias is shared across every spatial position, so
    # its projection is the *sum* of nu over the spatial axis -- not a per-
    # position array reshaped. Pinned separately because a projector that
    # silently kept the spatial axis would still have the right leaf count.
    factory, seed = _FAMILIES['conv_nwc_bias']
    x = _xs_for('conv_nwc_bias', 1, seed)[0]
    proj, nu, model = _one_step_projection(factory, x)
    b = np.asarray(model.states(brainstate.ParamState)[('b',)].value)
    got = np.asarray(u.get_mantissa(jax.tree.leaves(proj[('b',)])[0]))
    assert got.shape == b.shape
    # NWC: nu is (1, length, out_ch, 1); the bias reduces N and W.
    want = np.asarray(nu)[..., 0].sum(axis=(0, 1))
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# The documented sharp edges.
# ---------------------------------------------------------------------------

def _einsum_shared_axis_factory():
    """``einsum('bhd,hde->bhe')`` — ``h`` is shared between x and the weight.

    ``_einsum_dt_to_t`` sums the signal over shared letters *before* multiplying
    by a trace already summed over them, so ``sum_t nu_t x_t`` and
    ``(sum_t nu_t)(sum_t x_t)`` are conflated. That is einsum's documented regime
    restriction, inherited unchanged by the projector: the test records the
    inherited behaviour rather than asserting an exactness the rule never had.
    """
    brainstate.random.seed(31)
    nh, nd, ne = 2, 3, 4
    w0 = 0.1 * brainstate.random.randn(nh, nd, ne)

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(w0)
            self.h = brainstate.HiddenState(jnp.zeros((1, nh, ne)))

        def update(self, x):
            drive = braintrace.einsum('bhd,hde->bhe', x, self.w.value)
            self.h.value = LEAK * self.h.value + drive
            return self.h.value

    return Net()


def test_a_shared_axis_einsum_projection_is_finite_and_parameter_shaped():
    # Not an exactness pin: the shared-axis regime is documented as approximate
    # in ``einsum.py`` and the projector inherits it. What must hold is that the
    # composition stays well-formed -- parameter-shaped, finite, and sensitive to
    # nu's sign.
    brainstate.random.seed(41)
    x = 0.3 * brainstate.random.randn(1, 2, 3)
    proj_p, nu_p, model = _one_step_projection(_einsum_shared_axis_factory, x)
    proj_m, nu_m, _ = _one_step_projection(
        _einsum_shared_axis_factory, x, nu_sign=-1.0)

    w = np.asarray(model.states(brainstate.ParamState)[('w',)].value)
    got_p = np.asarray(u.get_mantissa(jax.tree.leaves(proj_p[('w',)])[0]))
    got_m = np.asarray(u.get_mantissa(jax.tree.leaves(proj_m[('w',)])[0]))
    assert got_p.shape == w.shape
    assert bool(np.all(np.isfinite(got_p)))
    # Odd in nu: flipping every sign flips the projection.
    np.testing.assert_allclose(got_m, -got_p, atol=1e-5, rtol=1e-4)


def _weight_fn_factory():
    """A dense relation with a ``weight_fn`` transform.

    ``fp.applicable`` is false here (the closed-form kernels drop ``f'(W)``), so
    this fixture forces the legacy nested-vmap composition on both halves of the
    projector — the branch the fast path would otherwise hide.
    """
    brainstate.random.seed(32)
    n_in, n_out = 3, 4
    w0 = 0.1 * brainstate.random.randn(n_in, n_out)

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(w0)
            self.h = brainstate.HiddenState(jnp.zeros((1, n_out)))

        def update(self, x):
            drive = braintrace.matmul(x, self.w.value, weight_fn=jnp.tanh)
            self.h.value = LEAK * self.h.value + drive
            return self.h.value

    return Net()


def test_a_transformed_weight_projection_chains_through_the_transform():
    brainstate.random.seed(42)
    x = 0.3 * brainstate.random.randn(1, 3)
    proj, nu, model = _one_step_projection(_weight_fn_factory, x)
    want = _vjp_reference(
        _weight_fn_factory, x, nu,
        lambda m, xx, v: braintrace.matmul(xx, v[('w',)], weight_fn=jnp.tanh),
        [('w',)])
    _assert_close(proj[('w',)], want[('w',)], 'weight_fn dense')


def test_a_tied_parameter_accumulates_both_relations():
    # One ParamState, two ETP call sites: the projection must be the *sum* of the
    # two vjps. A projector keyed per relation without accumulation would return
    # one of them, which is the same shape and therefore invisible to a shape
    # check.
    from braintrace._testing import oracle_models as om
    spec = om.tied_weight_rnn(n_rec=4)
    x = jnp.asarray(np.random.RandomState(0).randn(1, 4).astype('float32')) * 0.3

    proj, nu, model = _one_step_projection(spec.factory, x)

    ref_model = spec.factory()
    brainstate.nn.init_all_states(ref_model, batch_size=1)
    h0 = np.asarray(ref_model.states(brainstate.HiddenState)[('h',)].value)
    w_val = ref_model.states(brainstate.ParamState)[('w',)].value

    def drive(w):
        return (braintrace.matmul(jnp.reshape(x, (1, -1)), w)
                + braintrace.matmul(jnp.asarray(h0), w))

    # The tail here is tanh, not a leak, so df is the tanh derivative.
    pre = np.asarray(drive(w_val))
    df = 1.0 - np.tanh(pre) ** 2
    ct = jnp.asarray(np.asarray(nu)[..., 0] * df)
    want = jax.vjp(drive, w_val)[1](ct)[0]
    _assert_close(proj[('w',)], want, 'tied weight')


def test_the_projector_is_linear_in_nu():
    # nu^T J_f is linear in nu by construction. Scaling the draw must scale the
    # projection exactly -- the cheapest check that no norm has leaked into the
    # projector itself (the norms belong in rho, not in proj).
    factory, seed = _FAMILIES['dense_mm']
    x = _xs_for('dense_mm', 1, seed)[0]
    plus, nu, _ = _one_step_projection(factory, x)
    minus, nu_m, _ = _one_step_projection(factory, x, nu_sign=-1.0)
    a = np.asarray(u.get_mantissa(jax.tree.leaves(plus[('w',)])[0]))
    b = np.asarray(u.get_mantissa(jax.tree.leaves(minus[('w',)])[0]))
    np.testing.assert_allclose(b, -a, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Factor bookkeeping.
# ---------------------------------------------------------------------------

def test_one_hidden_factor_per_group_and_one_parameter_factor_per_group_path():
    # ``two_island_rnn`` is the *only* multi-group fixture at coupled scope; see
    # the test below for why no pre-existing spec can carry this assertion.
    from braintrace._testing import oracle_models as om
    spec = om.two_island_rnn(n_in=3, n_rec=3)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = braintrace.UORO(model, vjp_method='multi-step')
    algo.compile_graph(braintrace.MultiStepData(spec.make_inputs(2, 3)))
    algo.init_etrace_state()

    data = algo._get_etrace_data()
    groups = algo.graph.hidden_groups
    assert len(groups) == 2, 'the fixture must stay disconnected'
    assert [g.hidden_paths for g in groups] == [[('ha',)], [('hb',)]]
    assert len(data['s_tilde']) == len(groups) == 2
    assert set(data['theta_tilde']) == {(0, ('wa',)), (1, ('wb',))}
    # One hidden factor per group, shaped like that group's own hidden slab
    # ``(*varshape, num_state)`` -- not like the concatenation of both islands.
    for gi, group in enumerate(groups):
        assert group.varshape == (1, 3) and group.num_state == 1
        assert data['s_tilde'][gi].shape == (1, 3, 1), group.hidden_paths


def test_coupled_scope_merges_the_stacked_layers_into_one_group():
    """Why ``two_island_rnn`` exists: grouping follows the transition.

    Under ``diagonal`` the two layers of ``stacked_tanh_rnn`` are separate
    groups; under ``coupled`` -- which UORO requires -- they are one, because the
    inter-layer projection puts ``h2`` downstream of ``h1``. So on this fixture
    the per-group and per-(group, path) clauses above would be satisfied by a
    single group holding both paths, and the test would prove nothing.
    """
    from braintrace._testing import oracle_models as om
    spec = om.stacked_tanh_rnn(n_in=4, n_rec=4)
    xs = spec.make_inputs(2, 4)

    def groups_of(algo_fn):
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = algo_fn(model)
        algo.compile_graph(braintrace.MultiStepData(xs))
        return [g.hidden_paths for g in algo.graph.hidden_groups]

    diagonal = groups_of(lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
    coupled = groups_of(lambda m: braintrace.UORO(m, vjp_method='multi-step'))
    assert diagonal == [[('h1',)], [('h2',)]]
    assert coupled == [[('h1',), ('h2',)]]


def test_a_tied_weight_shares_one_hidden_factor_and_one_parameter_factor():
    from braintrace._testing import oracle_models as om
    spec = om.tied_weight_rnn(n_rec=4)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = braintrace.UORO(model, vjp_method='multi-step')
    x = jnp.ones((4,))
    algo.compile_graph(braintrace.MultiStepData(x[None]))
    algo.init_etrace_state()

    data = algo._get_etrace_data()
    assert len(algo.graph.hidden_param_op_relations) == 2, (
        'the fixture must have two relations for this test to mean anything')
    assert len(data['s_tilde']) == 1
    # Two relations, one group, one ParamState path -> ONE parameter factor.
    # The rank-1 carrier is per (group, parameter), not per relation, so the two
    # relations' projections are summed into it and no root-sum-of-squares
    # approximation of the norm is needed.
    assert set(data['theta_tilde']) == {(0, ('w',))}
