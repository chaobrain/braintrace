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

"""Random graphs for the in-step read cuts of the hidden perturbation (#239).

The review of PR #247 fuzzed the planner with random models of 2 to 4 hidden
states. The input terms are position-diagonal, so at one step from the
initial state every single-step algorithm must equal BPTT unless a read is
double counted. The reads between hidden states are drawn from elementwise
terms and from terms that mix positions without a ``dot_general`` at top
level (``roll``, ``v - mean(v)``, a dense dot inside a ``custom_jvp`` or a
``checkpoint``). On 003943c, 52 of 220 seeds declared ``new_read_exact=True``
and were off BPTT, every one through such a read. Here those seeds must
report the read, or equal BPTT. The generator is the review's, kept
verbatim so the seeds draw the same models.
"""

import copy
import random
import warnings

import brainstate
import brainunit as u
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

_N = 3
_KINDS = ['tanh', 'sin', 'sq', 'diag', 'cond', 'cjvp', 'dense', 'roll', 'mean', 'cjdense', 'ckpt']
_DENSE = np.linspace(0.1, 0.9, _N * _N).reshape(_N, _N) / 2


@jax.custom_jvp
def _surrogate(v):
    return jnp.tanh(v)


@_surrogate.defjvp
def _surrogate_jvp(primals, tangents):
    (v,), (dv,) = primals, tangents
    return jnp.tanh(v), dv * 1.5 / (1 + v ** 2)


def _dot(v):
    return v @ jnp.asarray(_DENSE, v.dtype)


@jax.custom_jvp
def _custom_jvp_dense(v):
    return _dot(v)


@_custom_jvp_dense.defjvp
def _custom_jvp_dense_jvp(primals, tangents):
    (v,), (dv,) = primals, tangents
    return _dot(v), _dot(dv)


@jax.custom_vjp
def _custom_vjp_tanh(v):
    return jnp.tanh(v)


_custom_vjp_tanh.defvjp(lambda v: (jnp.tanh(v), v), lambda v, g: (g * (1 - jnp.tanh(v) ** 2),))


def _term(kind, v, x):
    if kind == 'tanh':
        return jnp.tanh(v)
    if kind == 'sin':
        return jnp.sin(v)
    if kind == 'sq':
        return 0.3 * v * v
    if kind == 'diag':  # splits hidden groups, stays diagonal
        return v @ jnp.asarray(0.6 * np.eye(_N), v.dtype)
    if kind == 'cond':
        return jax.lax.cond(jnp.sum(x) > -100.0, jnp.tanh, jnp.sin, v)
    if kind == 'cjvp':
        return _surrogate(v)
    if kind == 'dense':
        return _dot(v)
    if kind == 'roll':
        return jnp.roll(v, 1, axis=-1)
    if kind == 'mean':
        return v - jnp.mean(v, axis=-1, keepdims=True)
    if kind == 'cjdense':
        return _custom_jvp_dense(v)
    if kind == 'ckpt':
        return jax.checkpoint(_dot)(v)
    # not drawn by _spec (adding them to _KINDS would redraw every seed): kinds
    # for _variant. A custom_vjp the walker proves elementwise and the planner
    # never proves; a call holding a roll, which the walker leaves out (#249).
    if kind == 'cvjp':
        return _custom_vjp_tanh(v)
    if kind == 'ckptroll':
        return jax.checkpoint(lambda a: jnp.roll(a, 1, axis=-1))(v)
    raise ValueError(kind)


def _spec(seed):
    r = random.Random(seed)
    nh = r.randint(2, 4)
    nw = r.randint(1, 3)
    spec = {'nh': nh, 'nw': nw, 'upd': [], 'out': [], 'st': None}
    used_w = set()
    shared = None
    if r.random() < 0.5:
        shared = (r.randrange(nh - 1), r.choice(_KINDS))  # an intermediate from a new h_j
    spec['shared'] = shared
    for i in range(nh):
        terms = []
        if r.random() < 0.8 or i == 0:
            w = r.randrange(nw)
            used_w.add(w)
            terms.append(('y', w, r.choice(['tanh', 'sin', 'id', 'diag'])))
        for j in range(i):
            if r.random() < 0.55:
                terms.append(('new', j, r.choice(_KINDS)))
        if r.random() < 0.3:
            terms.append(('old', r.randrange(nh), r.choice(_KINDS)))
        if shared is not None and shared[0] < i and r.random() < 0.5:
            terms.append(('shared',))
        spec['upd'].append((r.choice([0.0, 0.5, 0.9]), terms))
    if r.random() < 0.35:
        spec['st'] = (r.randrange(nh), r.choice(_KINDS))  # a ShortTermState reads a new h_j
    outs = [('new', j, r.choice(_KINDS + ['id'])) for j in range(nh) if r.random() < 0.6] or [('new', nh - 1, 'id')]
    if shared is not None and r.random() < 0.5:
        outs.append(('shared',))
    if r.random() < 0.3:
        w = r.randrange(nw)
        used_w.add(w)
        outs.append(('y', w, 'tanh'))
    spec['out'] = outs
    for w in range(nw):
        if w not in used_w:
            spec['upd'][r.randrange(nh)][1].append(('y', w, 'sin'))
    return spec


def _model(spec):
    nh, nw = spec['nh'], spec['nw']

    class Model(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.Ws = [braintrace.nn.Linear(3, _N) for _ in range(nw)]
            for k, w in enumerate(self.Ws):
                setattr(self, f'W{k}', w)

        def init_state(self, batch_size=None, **kwargs):
            for i in range(nh):
                setattr(self, f'h{i}', brainstate.HiddenState(0.3 * (i + 1) * jnp.ones((batch_size, _N))))
            if spec['st'] is not None:
                self.r = brainstate.ShortTermState(jnp.zeros((batch_size, _N)))

        def update(self, x):
            ys = [w(x) for w in self.Ws]
            old = [getattr(self, f'h{i}').value for i in range(nh)]
            new = {}
            sh = None
            for i, (decay, terms) in enumerate(spec['upd']):
                if spec['shared'] is not None and spec['shared'][0] in new and sh is None:
                    sh = _term(spec['shared'][1], new[spec['shared'][0]], x)
                v = decay * old[i]
                for t in terms:
                    if t[0] == 'y':
                        yv = ys[t[1]]
                        v = v + (yv if t[2] == 'id' else _term(t[2], yv, x))
                    elif t[0] == 'new':
                        v = v + _term(t[2], new[t[1]], x)
                    elif t[0] == 'old':
                        v = v + 0.5 * _term(t[2], old[t[1]], x)
                    elif t[0] == 'shared':
                        v = v + sh
                getattr(self, f'h{i}').value = v
                new[i] = getattr(self, f'h{i}').value
                if spec['st'] is not None and spec['st'][0] == i:
                    self.r.value = _term(spec['st'][1], new[i], x)
            o = 0.0
            for t in spec['out']:
                if t[0] == 'new':
                    o = o + (new[t[1]] if t[2] == 'id' else _term(t[2], new[t[1]], x))
                elif t[0] == 'shared':
                    o = o + sh
                elif t[0] == 'y':
                    o = o + _term(t[2], ys[t[1]], x)
            if spec['st'] is not None:
                o = o + 0.5 * self.r.value
            return o

    def make():
        with brainstate.random.seed_context(1):
            return Model()

    return make


def _inputs(steps):
    with brainstate.random.seed_context(7):
        return (brainstate.random.rand(3, 1, 3) + 0.1)[:steps]


def _graph(make):
    model = make()
    brainstate.nn.init_all_states(model, batch_size=1)
    learner = braintrace.D_RTRL(model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        learner.compile_graph(_inputs(1)[0])
    return learner.graph


def _flat(tree):
    return {jax.tree_util.keystr(k): np.asarray(u.get_mantissa(v), dtype=np.float64)
            for k, v in jax.tree_util.tree_flatten_with_path(tree, is_leaf=u.math.is_quantity)[0]}


def _worst(got, ref):
    got, ref = _flat(got), _flat(ref)
    return max((float(np.linalg.norm(got[k] - r) / np.linalg.norm(r)) for k, r in ref.items()
                if np.linalg.norm(r) > 0), default=0.0)


# seeds with new_read_exact=True and single-step D-RTRL off BPTT at T = 1 on 003943c.
# Since #249 the hidden-group walker leaves a call whose body is not proven to keep
# positions out of the transition, as a top-level dense dot: the 'ckpt' and 'cjdense'
# reads are group boundaries. 36 of the seeds below change graph; eight (0, 70, 71, 83,
# 97, 114, 174, 187) move from a reported conflict (a read through that call) to
# new_read_exact=True, since the read now crosses groups and no trace carries it, and
# test_a_seed_that_double_counted_is_reported_or_exact finds each equal to BPTT at T = 1.
_DOUBLE_COUNTED_ON_003943C = (
    0, 21, 24, 30, 33, 34, 37, 43, 47, 48, 50, 54, 56, 60, 62, 63, 65, 68, 70, 71, 83, 84, 85, 86,
    97, 98, 99, 106, 107, 112, 114, 120, 123, 131, 132, 143, 145, 160, 163, 174, 181, 185, 187,
    191, 193, 195, 199, 200, 201, 207, 210, 211,
)
# seeds with one hidden group, new_read_exact=True, forced multi-step exact at T = 3 and
# single-step not, on 003943c (seed 35 was too, and still declares no double count:
# test_a_dense_old_read_in_a_call_is_not_a_new_read; seed 114 was too, and is one of the
# eight above that the walker now splits at a call, #249)
_OFF_AT_THREE_STEPS_ON_003943C = (21, 115, 123, 136, 143, 201)


def _variant(seed, new=None, old=None):
    """The model of ``seed`` with its reads of new values of the kinds in
    ``new`` redrawn as ``new[kind]``, and every read of an old value as
    ``old``."""
    spec = copy.deepcopy(_spec(seed))
    spec['upd'] = [(decay, [(t[0], t[1], new.get(t[2], t[2])) if t[0] == 'new' and new else
                            (t[0], t[1], old) if t[0] == 'old' and old else t for t in terms])
                   for decay, terms in spec['upd']]
    return _model(spec)


def _off_bptt(make, steps):
    """How far single-step and multi-step D-RTRL are off BPTT after ``steps``."""
    xs = _inputs(steps)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = bptt_param_gradients(make, xs)
        single = online_param_gradients_singlestep_naive(make, xs, algo_factory=braintrace.D_RTRL)
        multi = chunked_online_param_gradients(
            make, xs, algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'), chunk_size=1)
    return _worst(single, ref), _worst(multi, ref)


@pytest.mark.parametrize('seed', _DOUBLE_COUNTED_ON_003943C)
def test_a_seed_that_double_counted_is_reported_or_exact(seed):
    """Each seed now reports its reads, or declares no double count and
    equals BPTT at T = 1."""
    make = _model(_spec(seed))
    with brainstate.environ.context(precision=64):
        hp = _graph(make).hidden_perturb
        if hp.new_read_exact is False:
            assert hp.new_read_cuts == () and hp.new_read_conflicts
            return
        xs = _inputs(1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            single = online_param_gradients_singlestep_naive(make, xs, algo_factory=braintrace.D_RTRL)
            ref = bptt_param_gradients(make, xs)
    assert _worst(single, ref) < 1e-9


@pytest.mark.parametrize('seed', _OFF_AT_THREE_STEPS_ON_003943C)
def test_a_seed_off_multi_step_on_003943c_is_reported(seed):
    """Each seed declared no double count on 003943c; now it reports a read
    it cut there."""
    with brainstate.environ.context(precision=64):
        hp = _graph(_model(_spec(seed))).hidden_perturb
    assert hp.new_read_exact is False and hp.new_read_cuts == () and hp.new_read_conflicts


# The 'ckpt' and 'cjdense' reads are boundaries of the walker now (#249), so they no
# longer reach the planner's refusal of a call it cannot prove. These variants redraw
# them, and the surrogate 'cjvp' reads, as a custom_vjp the walker proves elementwise but
# the planner never proves ('cvjp'), and a checkpoint holding a roll ('ckptroll').
# On the seeds of the first tuple a read goes through the custom_vjp in the group.
_UNPROVEN_CALL_SEEDS = (24, 70, 98, 163, 211)
_CALL_VARIANT = {'ckpt': 'cvjp', 'cjdense': 'ckptroll', 'cjvp': 'cvjp'}


@pytest.mark.parametrize('seed', _UNPROVEN_CALL_SEEDS + (0, 56, 114, 132))
def test_a_read_through_a_call_the_planner_cannot_prove_is_reported_or_exact(seed):
    make = _variant(seed, new=_CALL_VARIANT)
    with brainstate.environ.context(precision=64):
        hp = _graph(make).hidden_perturb
        if seed in _UNPROVEN_CALL_SEEDS:
            assert any('custom_vjp_call' in c for c in hp.new_read_conflicts)
        if hp.new_read_exact is False:
            assert hp.new_read_cuts == () and hp.new_read_conflicts
            return
        xs = _inputs(1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            single = online_param_gradients_singlestep_naive(make, xs, algo_factory=braintrace.D_RTRL)
            ref = bptt_param_gradients(make, xs)
    assert _worst(single, ref) < 1e-9


# one hidden group, cut reads, new_read_exact=True and every read of an old value
# elementwise: seeds 35 and 136 with their old reads made tanh (and 136's cond reads, not
# proven to keep positions, made tanh so that they are cut)
_EXACT_ONE_GROUP = {'35-old-tanh': (35, None, 'tanh'), '136-cond-old-tanh': (136, {'cond': 'tanh'}, 'tanh')}


@pytest.mark.parametrize('name', sorted(_EXACT_ONE_GROUP))
def test_a_declared_exact_one_group_graph_matches_multi_step(name):
    """With every read of an old value elementwise, D-RTRL's transition holds
    the whole recurrence, so with the carried reads cut single-step D-RTRL
    equals multi-step and BPTT at T = 3."""
    make = _variant(*_EXACT_ONE_GROUP[name])
    with brainstate.environ.context(precision=64):
        graph = _graph(make)
        hp = graph.hidden_perturb
        assert hp.new_read_exact is True and hp.new_read_cuts and len(graph.hidden_groups) == 1
        single, multi = _off_bptt(make, 3)
    assert multi < 1e-12 and single < 1e-12


def test_a_position_coupling_old_read_is_a_known_approximation():
    """Seed 136 with its cond reads made tanh: one group, three cut reads and
    new_read_exact=True, and ``h3`` reads the old ``h2`` through a dense dot.
    D-RTRL's transition is diagonal in positions, so it approximates that
    read: single-step D-RTRL equals BPTT at T = 1 but not at T = 3, where
    multi-step does. That is consistent with ``new_read_exact``, which says
    only that no in-step read of a new value is double counted, not that the
    single-step gradient is exact. With the old read elementwise single-step
    matches (``136-cond-old-tanh`` above)."""
    make = _variant(136, new={'cond': 'tanh'})
    with brainstate.environ.context(precision=64):
        graph = _graph(make)
        hp = graph.hidden_perturb
        assert hp.new_read_exact is True and len(hp.new_read_cuts) == 3 and len(graph.hidden_groups) == 1
        assert _off_bptt(make, 1)[0] < 1e-12
        single, multi = _off_bptt(make, 3)
    assert multi < 1e-12
    assert single > 0.1  # 0.198 at the time of writing


def test_a_dense_old_read_in_a_call_is_not_a_new_read():
    """Seed 35: ``h3`` reads the old ``h1`` through a dense dot inside a
    ``custom_jvp``. That is not an in-step read of a new value, so the
    planner does not report it, and the elementwise cuts stay exact at T = 1.
    At T = 3 single-step D-RTRL is off BPTT, as it is with a plain dense dot
    there: D-RTRL's diagonal transition approximates a read of an old value
    that couples positions (see
    ``test_a_position_coupling_old_read_is_a_known_approximation``); with the
    old read elementwise it matches (``35-old-tanh``)."""
    make = _model(_spec(35))
    with brainstate.environ.context(precision=64):
        graph = _graph(make)
        assert graph.hidden_perturb.new_read_exact is True
        xs = _inputs(1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            single = online_param_gradients_singlestep_naive(make, xs, algo_factory=braintrace.D_RTRL)
            ref = bptt_param_gradients(make, xs)
    assert _worst(single, ref) < 1e-12
