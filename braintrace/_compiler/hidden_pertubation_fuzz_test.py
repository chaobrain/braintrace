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

import random
import warnings

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._compiler.diagnostics import DiagnosticKind
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


# seeds with new_read_exact=True and single-step D-RTRL off BPTT at T = 1 on 003943c
_DOUBLE_COUNTED_ON_003943C = (
    0, 21, 24, 30, 33, 34, 37, 43, 47, 48, 50, 54, 56, 60, 62, 63, 65, 68, 70, 71, 83, 84, 85, 86,
    97, 98, 99, 106, 107, 112, 114, 120, 123, 131, 132, 143, 145, 160, 163, 174, 181, 185, 187,
    191, 193, 195, 199, 200, 201, 207, 210, 211,
)
# seeds with one hidden group, new_read_exact=True, forced multi-step exact at T = 3 and
# single-step not, on 003943c (35: see test_a_dense_old_read_is_reported_not_cut)
_OFF_AT_THREE_STEPS_ON_003943C = (21, 114, 115, 123, 136, 143, 201)


@pytest.mark.parametrize('seed', _DOUBLE_COUNTED_ON_003943C)
def test_a_seed_that_double_counted_is_reported_or_exact(seed):
    """Each seed now reports its reads, or declares no double count and
    equals BPTT at T = 1. The second happens where a dense dot inside a call
    now splits the hidden groups (the walker leaves the call out), so the
    read that mixed positions is no longer within one group."""
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
def test_a_declared_exact_one_group_graph_matches_multi_step(seed):
    make = _model(_spec(seed))
    with brainstate.environ.context(precision=64):
        graph = _graph(make)
        hp = graph.hidden_perturb
        if not hp.new_read_exact or len(graph.hidden_groups) != 1:
            return  # reported, or several groups (not a claim about multi-step)
        xs = _inputs(3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            single = online_param_gradients_singlestep_naive(make, xs, algo_factory=braintrace.D_RTRL)
            multi = chunked_online_param_gradients(
                make, xs, algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'), chunk_size=1)
            ref = bptt_param_gradients(make, xs)
    if _worst(multi, ref) < 1e-12:
        assert _worst(single, ref) < 1e-9


def test_a_dense_old_read_is_reported_not_cut():
    """Seed 35: ``h3`` reads the old ``h1`` through a dense dot inside a
    ``custom_jvp``. The transition leaves the call out and reports it
    (``CONTROL_FLOW_RECURRENT_MIXING``), so single-step D-RTRL is off BPTT at
    T = 3 by its diagonal approximation; the elementwise cuts stay exact at
    T = 1."""
    make = _model(_spec(35))
    with brainstate.environ.context(precision=64):
        graph = _graph(make)
        assert graph.hidden_perturb.new_read_exact is True
        assert any(r.kind is DiagnosticKind.CONTROL_FLOW_RECURRENT_MIXING for r in graph.diagnostics)
        xs = _inputs(1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            single = online_param_gradients_singlestep_naive(make, xs, algo_factory=braintrace.D_RTRL)
            ref = bptt_param_gradients(make, xs)
    assert _worst(single, ref) < 1e-12
