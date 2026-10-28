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

"""Tests of the online-credit check (#199).

The acceptance invariant: for the algorithm in use, the ``ONLINE_GRADIENT_ZERO``
warning fires for a weight iff its online gradient is exactly zero while its
BPTT gradient is not. It is checked numerically over model families that
cover the ways credit can reach the loss, under six online algorithms.
"""

import warnings

import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace import DiagnosticKind, DiagnosticLevel
from braintrace._compiler.online_credit import OnlineCreditGraph

N = 4
_STEPS = 6


# ---------------------------------------------------------------------------
# model families
# ---------------------------------------------------------------------------

def _w(key, shape):
    return brainstate.ParamState(0.8 * jax.random.normal(jax.random.PRNGKey(key), shape))


def _delay_net(kind='short', direct=False):
    """x -> h1 -> line -> h2 -> out, the line one step long (the #199 model).

    ``kind`` holds the line in a ``'short'`` (``ShortTermState``) or a
    ``'hidden'`` (``HiddenState``); ``direct`` also adds layer 1's output to
    the model output in the same step.
    """

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = _w(0, (3, N))
            self.w2 = _w(1, (N, N))
            self.h1 = brainstate.HiddenState(jnp.zeros((1, N)))
            self.h2 = brainstate.HiddenState(jnp.zeros((1, N)))
            cls = brainstate.ShortTermState if kind == 'short' else brainstate.HiddenState
            self.ring = cls(jnp.zeros((1, N)))

        def update(self, x):
            self.h1.value = 0.9 * self.h1.value + braintrace.matmul(x, self.w1.value)
            y1 = jnp.tanh(self.h1.value)
            inp = self.ring.value  # written one step ago
            self.ring.value = y1
            self.h2.value = 0.9 * self.h2.value + braintrace.matmul(inp, self.w2.value)
            out = jnp.tanh(self.h2.value)
            return out + 0.1 * y1 if direct else out

    return Net


def _ring_net(length, impl='roll', order='read_first'):
    """The line as a ring of ``length`` slots (rolled, or indexed by a pointer)."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = _w(0, (3, N))
            self.w2 = _w(1, (N, N))
            self.h1 = brainstate.HiddenState(jnp.zeros((1, N)))
            self.h2 = brainstate.HiddenState(jnp.zeros((1, N)))
            self.ring = brainstate.ShortTermState(jnp.zeros((length, 1, N)))
            self.ptr = brainstate.ShortTermState(jnp.array(0, dtype=jnp.int32))

        def update(self, x):
            self.h1.value = 0.9 * self.h1.value + braintrace.matmul(x, self.w1.value)
            y1 = jnp.tanh(self.h1.value)
            if impl == 'roll':
                if order == 'read_first':
                    inp = self.ring.value[length - 1]
                    self.ring.value = jnp.roll(self.ring.value, 1, axis=0).at[0].set(y1)
                else:
                    self.ring.value = jnp.roll(self.ring.value, 1, axis=0).at[0].set(y1)
                    inp = self.ring.value[length - 1]
            else:
                p = self.ptr.value
                inp = jax.lax.dynamic_index_in_dim(self.ring.value, p, 0, keepdims=False)
                self.ring.value = jax.lax.dynamic_update_index_in_dim(self.ring.value, y1, p, 0)
                self.ptr.value = (p + 1) % length
            self.h2.value = 0.9 * self.h2.value + braintrace.matmul(inp, self.w2.value)
            return jnp.tanh(self.h2.value)

    return Net


def _scan_buffer_net(length=3):
    """The line shifted by a ``scan`` each step (read first)."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = _w(0, (3, N))
            self.w2 = _w(1, (N, N))
            self.h1 = brainstate.HiddenState(jnp.zeros((1, N)))
            self.h2 = brainstate.HiddenState(jnp.zeros((1, N)))
            self.ring = brainstate.ShortTermState(jnp.zeros((length, 1, N)))

        def update(self, x):
            self.h1.value = 0.9 * self.h1.value + braintrace.matmul(x, self.w1.value)
            inp = self.ring.value[length - 1]
            _, shifted = jax.lax.scan(lambda c, slot: (slot, c), jnp.tanh(self.h1.value), self.ring.value)
            self.ring.value = shifted
            self.h2.value = 0.9 * self.h2.value + braintrace.matmul(inp, self.w2.value)
            return jnp.tanh(self.h2.value)

    return Net


def _series_net():
    """Two lines in series: x -> h1 -> r1 -> h2 -> r2 -> h3 -> out."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = _w(0, (3, N))
            self.w2 = _w(1, (N, N))
            self.w3 = _w(2, (N, N))
            self.h1 = brainstate.HiddenState(jnp.zeros((1, N)))
            self.h2 = brainstate.HiddenState(jnp.zeros((1, N)))
            self.h3 = brainstate.HiddenState(jnp.zeros((1, N)))
            self.r1 = brainstate.ShortTermState(jnp.zeros((1, N)))
            self.r2 = brainstate.ShortTermState(jnp.zeros((1, N)))

        def update(self, x):
            self.h1.value = 0.9 * self.h1.value + braintrace.matmul(x, self.w1.value)
            i2 = self.r1.value
            self.r1.value = jnp.tanh(self.h1.value)
            self.h2.value = 0.9 * self.h2.value + braintrace.matmul(i2, self.w2.value)
            i3 = self.r2.value
            self.r2.value = jnp.tanh(self.h2.value)
            self.h3.value = 0.9 * self.h3.value + braintrace.matmul(i3, self.w3.value)
            return jnp.tanh(self.h3.value)

    return Net


def _rate_net():
    """The #199 model plus a rate trace of layer 1 that the loss reads."""
    base = _delay_net('short')

    class Net(base):
        def __init__(self):
            super().__init__()
            self.rate = brainstate.ShortTermState(jnp.zeros((1, N)))

        def update(self, x):
            out = super().update(x)
            self.rate.value = 0.5 * self.rate.value + jnp.tanh(self.h1.value)
            return out

    return Net


def _logged_net():
    """Layer 1 is copied into a monitor nobody reads back; layer 2 is independent."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = _w(0, (3, N))
            self.w2 = _w(1, (3, N))
            self.h1 = brainstate.HiddenState(jnp.zeros((1, N)))
            self.h2 = brainstate.HiddenState(jnp.zeros((1, N)))
            self.log = brainstate.ShortTermState(jnp.zeros((1, N)))

        def update(self, x):
            self.h1.value = 0.9 * self.h1.value + braintrace.matmul(x, self.w1.value)
            self.log.value = self.h1.value
            self.h2.value = 0.9 * self.h2.value + braintrace.matmul(x, self.w2.value)
            return jnp.tanh(self.h2.value)

    return Net


def _ref_net():
    """A same-step path through a ``jax`` reference, plus a reached line."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = _w(0, (3, N))
            self.h1 = brainstate.HiddenState(jnp.zeros((1, N)))
            self.ring = brainstate.ShortTermState(jnp.zeros((1, N)))

        def update(self, x):
            self.h1.value = 0.9 * self.h1.value + braintrace.matmul(x, self.w1.value)
            self.ring.value = self.h1.value
            r = jax.new_ref(jnp.zeros_like(self.h1.value))
            r[...] = self.h1.value
            return r[...]

    return Net


def _sse(model, out):
    return (out ** 2).sum()


def _sse_and_rate(model, out):
    return (out ** 2).sum() + (model.rate.value ** 2).sum()


_FAMILIES = {
    'issue': (_delay_net('short'), _sse),
    'ring-2': (_ring_net(2), _sse),
    'ring-5': (_ring_net(5), _sse),
    'ring-ptr-2': (_ring_net(2, impl='ptr'), _sse),
    'scan-buffer': (_scan_buffer_net(3), _sse),
    'series': (_series_net(), _sse),
    'hidden-line': (_delay_net('hidden'), _sse),
    'loss-reads-state': (_rate_net(), _sse_and_rate),
    'direct-and-line': (_delay_net('short', direct=True), _sse),
    'logged-only': (_logged_net(), _sse),
}
if hasattr(jax, 'new_ref'):
    _FAMILIES['new-ref'] = (_ref_net(), _sse)

# (factory, given one step per call as MultiStepData)
_ALGORITHMS = {
    'd_rtrl': (lambda m: braintrace.D_RTRL(m), False),
    'd_rtrl-multi': (lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'), True),
    'pp_prop': (lambda m: braintrace.pp_prop(m, decay_or_rank=0.9), False),
    'eprop': (lambda m: braintrace.EProp(m), False),
    'ostl': (lambda m: braintrace.OSTLRecurrent(m), False),
    'snap': (lambda m: braintrace.SnAp(m, n=2), False),
}


def _xs():
    return jax.random.uniform(jax.random.PRNGKey(7), (_STEPS, 1, 3))


def _online(make, loss, algorithm, multi, xs):
    """Summed per-step online gradients, the learner and its warnings."""
    model = make()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = algorithm(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        algo.compile_graph(xs[0])
        algo.init_etrace_state()
        params = model.states(brainstate.ParamState)

        def step(x):
            x = braintrace.MultiStepData(x[None]) if multi else x
            return brainstate.transform.grad(lambda x: loss(model, algo(x)), params)(x)

        grads = brainstate.transform.for_loop(step, xs)
    total = {k: v.sum(0) for k, v in grads.items()}
    return total, algo, caught


def _bptt(make, loss, xs):
    model = make()
    brainstate.nn.init_all_states(model, batch_size=1)
    params = model.states(brainstate.ParamState)

    def total():
        return brainstate.transform.for_loop(lambda x: loss(model, model(x)), xs).sum()

    return brainstate.transform.grad(total, params)()


def _records(algo, kind=DiagnosticKind.ONLINE_GRADIENT_ZERO, stage=None):
    return [r for r in algo.report.diagnostics
            if r.kind is kind and (stage is None or r.context['stage'] == stage)]


def _max(g):
    return float(jnp.abs(g).max())


# ---------------------------------------------------------------------------
# the acceptance invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('algorithm', list(_ALGORITHMS))
@pytest.mark.parametrize('family', list(_FAMILIES))
def test_warning_iff_online_gradient_is_zero_and_bptt_is_not(family, algorithm):
    make, loss = _FAMILIES[family]
    factory, multi = _ALGORITHMS[algorithm]
    xs = _xs()
    with brainstate.environ.context(precision=64):
        online, algo, caught = _online(make, loss, factory, multi, xs)
        bptt = _bptt(make, loss, xs)
    warned = {r.weight_path for r in _records(algo, stage='gradient')}
    for r in _records(algo, stage='gradient'):
        assert r.level is DiagnosticLevel.WARNING
        assert any(r.message == str(w.message) for w in caught)
    expected = {p for p in bptt if _max(online[p]) == 0.0 and _max(bptt[p]) > 0.0}
    assert warned == expected, {p: (_max(online[p]), _max(bptt[p])) for p in bptt}


def test_the_invariant_is_not_vacuous():
    # the issue model is reported under the default algorithm, and its zero is real
    make, loss = _FAMILIES['issue']
    with brainstate.environ.context(precision=64):
        online, algo, _ = _online(make, loss, _ALGORITHMS['d_rtrl'][0], False, _xs())
        bptt = _bptt(make, loss, _xs())
    assert [r.weight_path for r in _records(algo, stage='gradient')] == [('w1',)]
    assert _max(online[('w1',)]) == 0.0 and _max(bptt[('w1',)]) > 0.0
    assert _max(online[('w2',)]) > 0.0


@pytest.mark.parametrize('algorithm', ['d_rtrl-multi', 'ostl', 'snap'])
def test_the_named_remedies_credit_the_weight(algorithm):
    # the message for a ShortTermState line under a diagonal single-step algorithm
    # names these algorithms with a HiddenState line; each gives a non-zero gradient
    factory, multi = _ALGORITHMS[algorithm]
    with brainstate.environ.context(precision=64):
        online, algo, _ = _online(_delay_net('hidden'), _sse, factory, multi, _xs())
    assert _max(online[('w1',)]) > 0.0
    assert _records(algo) == []


@pytest.mark.xfail(strict=True, reason='known false negative (#216): a ring written and '
                                       'then read back in the same step is one array')
def test_write_first_ring_is_reported():
    make = _ring_net(2, order='write_first')
    with brainstate.environ.context(precision=64):
        online, algo, _ = _online(make, _sse, _ALGORITHMS['d_rtrl'][0], False, _xs())
        bptt = _bptt(make, _sse, _xs())
    assert _max(online[('w1',)]) == 0.0 and _max(bptt[('w1',)]) > 0.0  # the zero is real
    assert [r.weight_path for r in _records(algo, stage='gradient')] == [('w1',)]


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def _compiled(make, algorithm='d_rtrl'):
    model = make()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = _ALGORITHMS[algorithm][0](model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        algo.compile_graph(jnp.zeros((1, 3)))
    return algo, caught


class TestRecords:

    def test_compile_record_is_info_and_assumes_the_outputs(self):
        algo, caught = _compiled(_delay_net('short'))
        (rec,) = _records(algo)
        assert rec.level is DiagnosticLevel.INFO
        assert rec.context == {'through': (('ring',),), 'stage': 'compile'}
        assert rec.weight_path == ('w1',) and ('h1',) in rec.hidden_paths
        assert not any('online gradient' in str(w.message) for w in caught)
        for part in ('If the loss reads only the model outputs', "('ring',) (ShortTermState)",
                     'HiddenState', 'OSTLRecurrent', 'BPTT'):
            assert part in rec.message

    def test_message_follows_the_credit_model(self):
        multi, _ = _compiled(_delay_net('short'), 'd_rtrl-multi')
        (rec,) = _records(multi)
        assert 'given one step per call' in rec.message
        assert 'MultiStepData of two or more steps' in rec.message
        assert 'OSTLRecurrent' not in rec.message
        coupled, _ = _compiled(_delay_net('short'), 'ostl')
        (rec,) = _records(coupled)
        assert 'couples it into' in rec.message
        hidden, _ = _compiled(_delay_net('hidden'))
        (rec,) = _records(hidden)
        assert rec.context['through'] == (('ring',),)
        assert 'HiddenState)' in rec.message and 'old value' in rec.message
        assert 'hold the state' not in rec.message

    def test_loss_that_reads_a_state_is_not_warned(self):
        # compile time assumes the outputs only; the gradient sees the rate read
        with brainstate.environ.context(precision=64):
            online, algo, caught = _online(_rate_net(), _sse_and_rate,
                                           _ALGORITHMS['d_rtrl'][0], False, _xs())
        assert [r.weight_path for r in _records(algo, stage='compile')] == [('w1',)]
        assert _records(algo, stage='gradient') == []
        assert not any('online gradient' in str(w.message) for w in caught)
        assert _max(online[('w1',)]) > 0.0

    def test_partial_path_is_an_info_record(self):
        algo, caught = _compiled(_delay_net('short', direct=True))
        assert _records(algo) == []
        (rec,) = _records(algo, DiagnosticKind.ONLINE_GRADIENT_PARTIAL)
        assert rec.level is DiagnosticLevel.INFO
        assert rec.weight_path == ('w1',) and rec.context['through'] == (('ring',),)
        assert 'cosine -0.92' in rec.message
        assert not any('online gradient' in str(w.message) for w in caught)

    def test_gradient_warning_is_emitted_once_per_weight(self):
        model = _delay_net('short')()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model)
        algo.compile_graph(jnp.zeros((1, 3)))
        params = model.states(brainstate.ParamState)
        with pytest.warns(UserWarning, match=r"online gradient of weight \('w1',\)"):
            brainstate.transform.grad(lambda x: (algo(x) ** 2).sum(), params)(jnp.ones((1, 3)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            brainstate.transform.grad(lambda x: (algo(x) ** 3).sum(), params)(jnp.ones((1, 3)))
        assert not any('online gradient' in str(w.message) for w in caught)
        assert len(_records(algo, stage='gradient')) == 1

    def test_multi_step_chunks_are_not_warned(self):
        # within a call of two or more steps, the multi-step VJP credits the line
        model = _delay_net('short')()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model, vjp_method='multi-step')
        algo.compile_graph(jnp.zeros((1, 3)))
        params = model.states(brainstate.ParamState)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            g = brainstate.transform.grad(
                lambda xs: (algo(braintrace.MultiStepData(xs)) ** 2).sum(), params)(_xs()[:4])
        assert not any('online gradient' in str(w.message) for w in caught)
        assert _max(g[('w1',)]) > 0.0

    def test_other_learning_signals_are_not_checked(self):
        model = _delay_net('short')()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.ThreeFactor(model)
        algo.compile_graph(jnp.zeros((1, 3)))
        assert _records(algo) == []

    def test_random_projection_traces_are_not_checked(self):
        model = _delay_net('short')()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, projection_key=0)
        algo.compile_graph(jnp.zeros((1, 3)))
        algo.init_etrace_state()
        params = model.states(brainstate.ParamState)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            brainstate.transform.grad(lambda x: (algo(x) ** 2).sum(), params)(jnp.ones((1, 3)))
        assert _records(algo) == []
        assert not any('online gradient' in str(w.message) for w in caught)

    def test_compiling_again_adds_no_record(self):
        algo, _ = _compiled(_delay_net('short'))
        algo.compile_graph(jnp.zeros((1, 3)))
        assert len(_records(algo)) == 1

    def test_unmatched_cotangents_are_not_judged(self):
        # the backward rule does not guess when it cannot map what the loss read
        algo, _ = _compiled(_delay_net('short'))
        x = jnp.zeros((1, 3))
        algo._check_online_credit((x,), ((jnp.ones((1, N)), jnp.ones((1, N))), {}, {}, None))
        algo._check_online_credit((x,), (jnp.ones((1, N)), {}, {('elsewhere',): jnp.ones(3)}, None))
        assert _records(algo, stage='gradient') == []
        algo._check_online_credit((x,), (jnp.ones((1, N)), {}, {}, None))
        assert [r.weight_path for r in _records(algo, stage='gradient')] == [('w1',)]

    def test_logged_only_weight_is_not_reported(self):
        algo, _ = _compiled(_logged_net())
        assert _records(algo) == [] and _records(algo, DiagnosticKind.ONLINE_GRADIENT_PARTIAL) == []


# ---------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------

def _graph(make, x=None):
    model = make()
    brainstate.nn.init_all_states(model, batch_size=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = braintrace.compile_etrace_graph(model, jnp.zeros((1, 3)) if x is None else x)
    return OnlineCreditGraph(graph)


class TestAnalysis:

    def test_series_names_the_first_line(self):
        found = {f.weight_path: f.through for f in _graph(_series_net()).find(False)}
        assert found == {('w1',): (('r1',),), ('w2',): (('r2',),)}

    def test_every_state_sharing_a_value_is_named(self):
        class Net(_delay_net('short')):
            def __init__(self):
                super().__init__()
                self.copy = brainstate.ShortTermState(jnp.zeros((1, N)))

            def update(self, x):
                inp = self.copy.value
                out = super().update(x)
                self.copy.value = self.ring.value
                return out + 0.5 * braintrace.matmul(inp, self.w2.value)

        (finding,) = _graph(Net).find(False)
        assert set(finding.through) == {('ring',), ('copy',)}

    def test_several_relations_of_one_weight(self):
        class Net(_delay_net('short')):
            def __init__(self):
                super().__init__()
                self.h3 = brainstate.HiddenState(jnp.zeros((1, N)))
                self.r3 = brainstate.ShortTermState(jnp.zeros((1, N)))

            def update(self, x):
                out = super().update(x)
                inp = self.r3.value
                self.h3.value = 0.9 * self.h3.value + braintrace.matmul(x, self.w1.value)
                self.r3.value = jnp.tanh(self.h3.value)
                return out + inp

        (finding,) = _graph(Net).find(False)
        assert finding.weight_path == ('w1',) and finding.zero
        assert set(finding.through) == {('ring',), ('r3',)}

    def test_used_outputs_decide(self):
        credit = _graph(_delay_net('short'))
        assert [f.zero for f in credit.find(False)] == [True]
        assert credit.find(False, used=[]) == []
        # a loss that reads the line's new value gives w1 a same-step path; the
        # credit the line carries into later steps is still missed
        used = list(credit.model_outvars) + list(credit.state_outvars[('ring',)])
        assert [f.zero for f in credit.find(False, used=used)] == [False]
        assert credit.find(False, used=used, partial=False) == []

    def test_old_hidden_values_count_under_a_multi_step_vjp(self):
        credit = _graph(_delay_net('hidden'))
        assert [f.weight_path for f in credit.find(False)] == [('w1',)]
        assert credit.find(True) == []

    def test_descended_scan_weights_are_skipped(self):
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = _w(0, (3, N))
                self.h1 = brainstate.HiddenState(jnp.zeros((1, N)))
                self.ring = brainstate.ShortTermState(jnp.zeros((1, N)))
                self.w2 = _w(1, (N, N))
                self.h2 = brainstate.HiddenState(jnp.zeros((1, N)))

            def update(self, x):
                def sub(h, _):
                    return 0.9 * h + braintrace.matmul(x, self.w1.value), None

                self.h1.value = jax.lax.scan(sub, self.h1.value, None, length=20)[0]
                inp = self.ring.value
                self.ring.value = jnp.tanh(self.h1.value)
                self.h2.value = 0.9 * self.h2.value + braintrace.matmul(inp, self.w2.value)
                return jnp.tanh(self.h2.value)

        credit = _graph(Net)
        assert ('w1',) not in credit.weights
        assert credit.find(False) == []

    @pytest.mark.skipif(not hasattr(jax, 'new_ref'), reason='jax without references')
    def test_a_reference_carries_the_value(self):
        assert _graph(_ref_net()).find(False) == []
