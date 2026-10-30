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

"""The same-step output-bypass rewrite of the perturbed step (issue #185)."""

import warnings

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import braintrace
from braintrace._compiler.diagnostics import DiagnosticKind, DiagnosticLevel
from braintrace._compiler.output_bypass import add_output_bypass

N_IN, N = 3, 4
_sg = braintools.surrogate.Sigmoid()


class _LIF(brainstate.nn.Module):
    """``V_pre = 0.9 V + x @ w``; the output is read from ``V_pre`` or ``V``."""

    def __init__(self, out='pre_reset'):
        super().__init__()
        self.out = out
        self.w = brainstate.ParamState(brainstate.random.randn(N_IN, N))

    def init_state(self, *args, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(N))

    def update(self, x):
        V_pre = 0.9 * self.V.value + braintrace.matmul(x, self.w.value)
        spk = _sg(V_pre - 1.0)
        self.V.value = V_pre - jax.lax.stop_gradient(spk)
        if self.out == 'pre_reset':
            return spk
        if self.out == 'overflow':
            return jnp.exp(1e3 * V_pre)
        if self.out == 'while':
            return jax.lax.while_loop(lambda c: c[0] < 3, lambda c: (c[0] + 1, c[1] * 0.5), (0, V_pre))[1]
        return _sg(self.V.value - 0.5)


class _Readout(brainstate.nn.Module):
    """A plain readout ``tanh(V) @ w_out`` applied inside the step."""

    def __init__(self):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(N_IN, N))
        self.w_out = brainstate.ParamState(brainstate.random.randn(N, 2))

    def init_state(self, *args, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(N))

    def update(self, x):
        self.V.value = 0.9 * self.V.value + braintrace.matmul(x, self.w.value)
        return jnp.tanh(self.V.value) @ self.w_out.value


def _graph(model, x=None, **kwargs):
    brainstate.nn.init_all_states(model)
    x = jnp.linspace(0.5, 1.5, N_IN) if x is None else x
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return braintrace.compile_etrace_graph(model, x, **kwargs), x


def _kinds(graph, level=None):
    return [r.kind for r in graph.diagnostics if level is None or r.level == level]


class TestDetection:
    def test_pre_reset_output_has_param_and_hidden_bypass(self):
        brainstate.random.seed(0)
        graph, _ = _graph(_LIF('pre_reset'))
        hp = graph.hidden_perturb
        assert list(hp.bypass_param_paths) == [('w',)]
        assert list(hp.bypass_hidden_paths) == [('V',)]
        assert len(hp.bypass_vars) == 1 and len(hp.bypass_hidden_vars) == 1
        assert DiagnosticKind.OUTPUT_BYPASSES_HIDDEN in _kinds(graph, DiagnosticLevel.INFO)

    def test_output_from_new_hidden_has_no_bypass(self):
        brainstate.random.seed(0)
        graph, _ = _graph(_LIF('hidden'))
        ref, _ = _graph(_LIF('hidden'), include_output_bypass=False)
        hp = graph.hidden_perturb
        assert list(hp.bypass_param_paths) == [] and list(hp.bypass_hidden_paths) == []
        # no bypass found -> the perturbed jaxpr is not touched at all
        assert len(hp.perturb_jaxpr.jaxpr.eqns) == len(ref.hidden_perturb.perturb_jaxpr.jaxpr.eqns)
        assert DiagnosticKind.OUTPUT_BYPASSES_HIDDEN not in _kinds(graph)

    def test_plain_readout_is_param_bypass_only(self):
        brainstate.random.seed(0)
        graph, _ = _graph(_Readout())
        hp = graph.hidden_perturb
        assert list(hp.bypass_param_paths) == [('w_out',)]
        assert list(hp.bypass_hidden_paths) == []

    def test_disabled(self):
        brainstate.random.seed(0)
        graph, _ = _graph(_LIF('pre_reset'), include_output_bypass=False)
        hp = graph.hidden_perturb
        assert list(hp.bypass_param_paths) == [] and list(hp.bypass_hidden_paths) == []
        assert len(hp.perturb_jaxpr.jaxpr.invars) == (
            len(graph.module_info.closed_jaxpr.jaxpr.invars) + len(hp.perturb_vars))

    def test_while_on_bypass_path_is_dropped_with_warning(self):
        brainstate.random.seed(0)
        graph, _ = _graph(_LIF('while'))
        hp = graph.hidden_perturb
        assert list(hp.bypass_param_paths) == [] and list(hp.bypass_hidden_paths) == []
        dropped = [r for r in graph.diagnostics if r.kind == DiagnosticKind.OUTPUT_BYPASS_DROPPED]
        assert len(dropped) == 1 and dropped[0].level == DiagnosticLevel.WARNING
        assert set(dropped[0].context['sources']) == {('w',), ('V',)}


class TestEvaluation:
    def _eval(self, out, x=None):
        brainstate.random.seed(0)
        model = _LIF(out)
        graph, x = _graph(model, x)
        hp = graph.hidden_perturb
        inputs = jax.tree.leaves((x, [st.value for st in graph.module_info.compiled_model_states]))
        w, V = model.w.value, model.V.value
        n_out = graph.module_info.num_var_out

        def run(w_copy, v_copy):
            outs = hp.eval_jaxpr(inputs, hp.init_perturb_data(), [w_copy], [v_copy])
            return outs[:n_out]

        return graph, run, w, V, x

    def test_forward_values_unchanged(self):
        graph, run, w, V, x = self._eval('pre_reset')
        ref = graph.module_info.closed_jaxpr
        inputs = jax.tree.leaves((x, [st.value for st in graph.module_info.compiled_model_states]))
        expected = jax.core.eval_jaxpr(ref.jaxpr, ref.consts, *inputs)[:graph.module_info.num_var_out]
        for a, b in zip(run(w, V), expected):
            npt.assert_array_equal(a, b)

    def test_copy_gradients_are_the_bypass_partials(self):
        _, run, w, V, x = self._eval('pre_reset')
        V = V + 0.3  # a non-trivial previous value
        gw, gv = jax.grad(lambda a, b: run(a, b)[0].sum(), argnums=(0, 1))(w, V)
        # spk = sg(0.9 V + x @ w - 1): the whole output is a bypass
        d = jax.vmap(jax.grad(lambda z: _sg(z - 1.0)))(0.9 * V + x @ w)
        npt.assert_allclose(gw, jnp.outer(x, d), rtol=1e-6)
        npt.assert_allclose(gv, 0.9 * d, rtol=1e-6)

    def test_overflowing_output_stays_inf(self):
        # o + (d - sg(d)) would be inf - inf = nan at an overflowing slot
        graph, run, w, V, x = self._eval('overflow', x=jnp.full((N_IN,), 3.0))
        out = run(w, V)[0]
        assert bool(jnp.any(jnp.isposinf(out))) and not bool(jnp.any(jnp.isnan(out)))
        ref = graph.module_info.closed_jaxpr
        inputs = jax.tree.leaves((x, [st.value for st in graph.module_info.compiled_model_states]))
        npt.assert_array_equal(out, jax.core.eval_jaxpr(ref.jaxpr, ref.consts, *inputs)[0])

    def test_wrong_number_of_bypass_values_raises(self):
        graph, run, w, V, x = self._eval('pre_reset')
        hp = graph.hidden_perturb
        inputs = jax.tree.leaves((x, [st.value for st in graph.module_info.compiled_model_states]))
        with pytest.raises(ValueError, match='parameters'):
            hp.eval_jaxpr(inputs, hp.init_perturb_data(), [], [V])
        with pytest.raises(ValueError, match='hidden states'):
            hp.eval_jaxpr(inputs, hp.init_perturb_data(), [w], [])
        with pytest.raises(ValueError, match='hidden states'):
            hp.bypass_hidden_data_to_hidden_group_data([], graph.hidden_groups)


class TestRewrite:
    """``add_output_bypass`` on hand-written jaxprs."""

    @staticmethod
    def _rewrite(f, *args, sources, cut=(), merge=None):
        closed = jax.make_jaxpr(f)(*args)
        jaxpr = closed.jaxpr
        src = {k: [jaxpr.invars[i] for i in idx] for k, idx in sources.items()}
        cut_vars = {jaxpr.outvars[i] for i in cut}
        merge = range(len(jaxpr.outvars)) if merge is None else merge
        return jaxpr, add_output_bypass(
            jaxpr.eqns, jaxpr.outvars, sources=src, cut_vars=cut_vars, merge_positions=list(merge))

    def test_stop_gradient_cuts_the_cone(self):
        _, res = self._rewrite(lambda w, x: jax.lax.stop_gradient(w * x), 1.0, 2.0, sources={'w': [0]})
        assert res.bypass_keys == [] and res.bypass_vars == []

    def test_integer_outputs_are_not_merged(self):
        _, res = self._rewrite(lambda w: (w > 0).astype(jnp.int32), 1.0, sources={'w': [0]})
        assert res.bypass_keys == []

    def test_cone_stops_at_cut_vars(self):
        # h = w * x is a hidden slot; y = h * 2 reaches the output only through it
        def f(w, x):
            h = w * x
            return h, h * 2.0

        _, res = self._rewrite(f, 1.0, 2.0, sources={'w': [0]}, cut=(0,), merge=(1,))
        assert res.bypass_keys == []

    def test_only_listed_positions_are_merged(self):
        jaxpr, res = self._rewrite(lambda w, x: (w * x, w + x), 1.0, 2.0, sources={'w': [0]}, merge=(1,))
        assert res.bypass_keys == ['w']
        assert res.outvars[0] is jaxpr.outvars[0]
        assert res.outvars[1] is not jaxpr.outvars[1]


def _eqn_of(fn, *args):
    """The single top-level equation of *fn*, ignoring the
    ``convert_element_type`` casts that tracing may add around it."""
    (eqn,) = [e for e in jax.make_jaxpr(fn)(*args).jaxpr.eqns
              if e.primitive.name not in ('convert_element_type',)]
    return eqn


def _settle(h):
    return jax.lax.while_loop(lambda s: s[0] < 2, lambda s: (s[0] + 1, jnp.tanh(s[1])), (0, h))[1]


class TestBlocking:
    """``_is_blocking`` inspects one equation without building a ``Jaxpr``
    around it (#197): a ``while`` anywhere inside it, or an effect, blocks
    the bypass."""

    @pytest.mark.parametrize('fn, blocking', [
        (lambda h: jnp.tanh(h), False),
        (lambda h: jax.lax.scan(lambda c, _: (jnp.tanh(c), None), h, None, length=3)[0], False),
        (_settle, True),
        (lambda h: jax.lax.scan(lambda c, _: (_settle(c), None), h, None, length=3)[0], True),
        (lambda h: jax.lax.cond(True, _settle, jnp.tanh, h), True),
        (lambda h: jax.jit(_settle)(h), True),
    ], ids=['elementwise', 'scan', 'while', 'while-in-scan', 'while-in-cond', 'while-in-jit'])
    def test_a_while_anywhere_inside_blocks(self, fn, blocking):
        from braintrace._compiler.output_bypass import _is_blocking
        eqn = _eqn_of(fn, jnp.ones(3))
        assert _is_blocking(eqn) is blocking

    def test_an_effectful_equation_blocks(self):
        from braintrace._compiler.output_bypass import _is_blocking

        def f(h):
            jax.debug.print('{}', h)
            return h

        eqns = jax.make_jaxpr(f)(jnp.ones(3)).jaxpr.eqns
        (eqn,) = [e for e in eqns if e.effects]
        assert _is_blocking(eqn)

    def test_no_equation_holds_no_while(self):
        from braintrace._compiler.hidden_group import _eqns_contain_while
        assert not _eqns_contain_while([])

    def test_compiling_and_training_warns_no_missing_debug_info(self):
        # the issue model: _is_blocking used to wrap every equation in a
        # Jaxpr without DebugInfo, one jax DeprecationWarning each
        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(brainstate.random.randn(3, 4))

            def init_state(self, *args, **kwargs):
                self.V = brainstate.HiddenState(jnp.zeros(4))

            def update(self, x):
                self.V.value = 0.9 * self.V.value + braintrace.matmul(x, self.w.value)
                return jnp.tanh(self.V.value)

        brainstate.random.seed(0)
        xs = brainstate.random.rand(10, 3)
        ys = jnp.full((10, 4), 0.3)
        net = Cell()
        brainstate.nn.init_all_states(net)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner = braintrace.compile(net, 'd_rtrl', xs[0])
            learner.etrace_grad(xs, ys, step_fn=lambda x, y: jnp.mean((learner(x) - y) ** 2))
        assert not [w for w in caught if 'DebugInfo' in str(w.message)]

