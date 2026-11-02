# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import warnings
from pprint import pprint

import brainstate
import pytest
import brainunit as u

import braintrace
from braintrace._compiler.hidden_pertubation import add_hidden_perturbation_in_module
from braintrace._testing.models import (
    IF_Delta_Dense_Layer,
    LIF_ExpCo_Dense_Layer,
    ALIF_ExpCo_Dense_Layer,
    LIF_ExpCu_Dense_Layer,
    LIF_STDExpCu_Dense_Layer,
    LIF_STPExpCu_Dense_Layer,
    ALIF_ExpCu_Dense_Layer,
    ALIF_Delta_Dense_Layer,
    ALIF_STDExpCu_Dense_Layer,
    ALIF_STPExpCu_Dense_Layer,
)


class TestFindHiddenGroupsFromModule:
    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.LRUCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_rnn_one_layer(self, cls):
        n_in = 3
        n_out = 4

        gru = cls(n_in, n_out)
        brainstate.nn.init_all_states(gru)
        states = brainstate.graph.states(gru, brainstate.HiddenState)

        input = brainstate.random.rand(n_in)
        hidden_perturb = add_hidden_perturbation_in_module(gru, input)

        print()
        pprint(hidden_perturb)
        assert len(states) == len(hidden_perturb.init_perturb_data())

    @pytest.mark.parametrize(
        'cls',
        [
            IF_Delta_Dense_Layer,
            LIF_ExpCo_Dense_Layer,
            ALIF_ExpCo_Dense_Layer,
            LIF_ExpCu_Dense_Layer,
            LIF_STDExpCu_Dense_Layer,
            LIF_STPExpCu_Dense_Layer,
            ALIF_ExpCu_Dense_Layer,
            ALIF_Delta_Dense_Layer,
            ALIF_STDExpCu_Dense_Layer,
            ALIF_STPExpCu_Dense_Layer,
        ]
    )
    def test_snn_single_layer(self, cls):
        n_in = 3
        n_out = 4
        input = brainstate.random.rand(n_in)

        print(cls)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer = cls(n_in, n_out)
            brainstate.nn.init_all_states(layer)
            states = brainstate.graph.states(layer, brainstate.HiddenState)
            hidden_perturb = add_hidden_perturbation_in_module(layer, input)

        print()
        perturb = hidden_perturb.init_perturb_data()
        assert len(states) == len(perturb)

    @pytest.mark.parametrize(
        'cls',
        [
            IF_Delta_Dense_Layer,
            LIF_ExpCo_Dense_Layer,
            ALIF_ExpCo_Dense_Layer,
            LIF_ExpCu_Dense_Layer,
            LIF_STDExpCu_Dense_Layer,
            LIF_STPExpCu_Dense_Layer,
            ALIF_ExpCu_Dense_Layer,
            ALIF_Delta_Dense_Layer,
            ALIF_STDExpCu_Dense_Layer,
            ALIF_STPExpCu_Dense_Layer,
        ]
    )
    def test_snn_two_layers(self, cls):
        n_in = 3
        n_out = 4
        input = brainstate.random.rand(n_in)

        print()
        print(cls)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer = brainstate.nn.Sequential(cls(n_in, n_out), cls(n_out, n_out))
            brainstate.nn.init_all_states(layer)
            states = brainstate.graph.states(layer, brainstate.HiddenState)
            hidden_perturb = add_hidden_perturbation_in_module(layer, input)

        perturb = hidden_perturb.init_perturb_data()
        assert len(states) == len(perturb)


class TestPerturbationRobustness:
    """Task-6 contracts: multi-output equations, read-only hidden states,
    and jaxpr-effects preservation."""

    def test_multi_output_eqn_both_hidden_perturbed(self):
        # One `sort` equation with TWO outvars, both of which are hidden
        # outvars. Both must receive their perturbation, regardless of the
        # position of the hidden var in the equation's outvar list.
        import jax
        import jax.numpy as jnp
        import numpy as np
        from braintrace._compiler.hidden_pertubation import (
            JaxprEvalForHiddenPerturbation,
        )

        def f(a, b):
            return jax.lax.sort((a, b), num_keys=1)

        a = jnp.asarray([3.0, 1.0, 2.0])
        b = jnp.asarray([10.0, 20.0, 30.0])
        closed = jax.make_jaxpr(f)(a, b)
        (eqn,) = closed.jaxpr.eqns
        assert len(eqn.outvars) == 2, 'expected a single multi-output eqn'
        h1, h2 = closed.jaxpr.outvars
        a_in, b_in = closed.jaxpr.invars

        perturb = JaxprEvalForHiddenPerturbation(
            closed_jaxpr=closed,
            hidden_outvar_to_invar={h1: a_in, h2: b_in},
            weight_invars=set(),
            invar_to_hidden_path={a_in: ('h1',), b_in: ('h2',)},
            outvar_to_hidden_path={h1: ('h1',), h2: ('h2',)},
            path_to_state={('h1',): None, ('h2',): None},
        ).compile()

        assert len(perturb.perturb_vars) == 2
        p1 = jnp.full(3, 0.5)
        p2 = jnp.full(3, -2.0)
        base1, base2 = f(a, b)
        out1, out2 = perturb.eval_jaxpr([a, b], [p1, p2])
        np.testing.assert_allclose(out1, base1 + p1)
        np.testing.assert_allclose(out2, base2 + p2)

    def test_read_only_hidden_state_compiles_and_shifts(self):
        # ``h_ro`` is read but never written: its jaxpr outvar is its invar
        # and no equation produces it. The perturbation pass must synthesize
        # the passthrough ``h_ro^t = h_ro^{t-1} + p`` instead of raising.
        import jax
        import jax.numpy as jnp
        import numpy as np
        from braintrace._compiler.hidden_pertubation import (
            add_hidden_perturbation_from_minfo,
        )

        class _ReadOnlyHiddenRNN(brainstate.nn.Module):
            def __init__(self, n):
                super().__init__()
                self.h_ro = brainstate.HiddenState(jnp.ones(n))
                self.h = brainstate.HiddenState(jnp.zeros(n))

            def init_state(self, *args, **kwargs):
                pass

            def update(self, x):
                self.h.value = jnp.tanh(x + 0.5 * self.h.value + self.h_ro.value)
                return self.h.value

        model = _ReadOnlyHiddenRNN(4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(4)
        minfo = braintrace.extract_module_info(model, inp)

        perturb = add_hidden_perturbation_from_minfo(minfo)

        assert set(perturb.perturb_hidden_paths) == {('h_ro',), ('h',)}
        idx_ro = list(perturb.perturb_hidden_paths).index(('h_ro',))
        flat_inputs = jax.tree.leaves(
            ((inp,), [st.value for st in minfo.compiled_model_states])
        )
        zeros = perturb.init_perturb_data()
        base = perturb.eval_jaxpr(flat_inputs, zeros)
        pdata = list(zeros)
        pdata[idx_ro] = jnp.full(4, 0.3)
        shifted = perturb.eval_jaxpr(flat_inputs, pdata)

        ro_outvar = minfo.hidden_path_to_outvar[('h_ro',)]
        slot = list(minfo.jaxpr.outvars).index(ro_outvar)
        np.testing.assert_allclose(
            np.asarray(shifted[slot]), np.asarray(base[slot]) + 0.3, rtol=1e-6,
        )
        for i, (s, b) in enumerate(zip(shifted, base)):
            if i != slot:
                np.testing.assert_allclose(np.asarray(s), np.asarray(b))

    def test_effects_preserved(self):
        from braintrace._compiler.hidden_pertubation import (
            add_hidden_perturbation_from_minfo,
            JaxprEvalForHiddenPerturbation,
        )
        import jax
        import jax.numpy as jnp

        # Effectful jaxpr built directly (debug print inside): the perturbed
        # jaxpr must carry the source jaxpr's effect set.
        def f(a):
            jax.debug.print('x={x}', x=a.sum())
            return a * 2.0

        closed = jax.make_jaxpr(f)(jnp.ones(3))
        assert closed.jaxpr.effects, 'fixture must carry at least one effect'
        h = closed.jaxpr.outvars[0]
        a_in = closed.jaxpr.invars[0]

        perturb = JaxprEvalForHiddenPerturbation(
            closed_jaxpr=closed,
            hidden_outvar_to_invar={h: a_in},
            weight_invars=set(),
            invar_to_hidden_path={a_in: ('h',)},
            outvar_to_hidden_path={h: ('h',)},
            path_to_state={('h',): None},
        ).compile()

        assert perturb.perturb_jaxpr.jaxpr.effects == closed.jaxpr.effects


class TestWhileDetach:
    """Phase 3: the perturbation pass detaches (``stop_gradient``) every Var
    input of a hidden-producing ``while`` equation, so the perturbed jaxpr
    stays reverse-mode traceable (JAX cannot transpose ``while``). Temporal
    credit through the loop flows via the forward-computed hidden-to-hidden
    Jacobian instead; the perturbation add remains outside the detach."""

    def _make_while_fixture(self, n=4, k=3):
        import jax
        import jax.numpy as jnp

        def f(h_prev, x):
            def body(s):
                i, h = s
                return i + 1, h + 0.5 * jnp.tanh(x - h)

            _, h_new = jax.lax.while_loop(lambda s: s[0] < k, body, (0, h_prev))
            return h_new, jnp.sum(h_new)

        h0 = jnp.zeros(n)
        x = jnp.linspace(0.1, 0.4, n)
        closed = jax.make_jaxpr(f)(h0, x)
        h_in = closed.jaxpr.invars[0]
        h_out = closed.jaxpr.outvars[0]
        return closed, h_in, h_out, h0, x

    def _perturb(self, closed, h_in, h_out):
        from braintrace._compiler.hidden_pertubation import (
            JaxprEvalForHiddenPerturbation,
        )
        return JaxprEvalForHiddenPerturbation(
            closed_jaxpr=closed,
            hidden_outvar_to_invar={h_out: h_in},
            weight_invars=set(),
            invar_to_hidden_path={h_in: ('h',)},
            outvar_to_hidden_path={h_out: ('h',)},
            path_to_state={('h',): None},
        ).compile()

    def test_structure_stop_gradient_per_var_invar(self):
        from braintrace._compatible_imports import Var

        closed, h_in, h_out, _h0, _x = self._make_while_fixture()
        (while_eqn,) = [e for e in closed.jaxpr.eqns if e.primitive.name == 'while']
        n_var_invars = sum(isinstance(v, Var) for v in while_eqn.invars)

        perturb = self._perturb(closed, h_in, h_out)
        eqns = perturb.perturb_jaxpr.jaxpr.eqns
        names = [e.primitive.name for e in eqns]

        # one stop_gradient per Var invar, all before the while
        assert names.count('stop_gradient') == n_var_invars
        sg_outs = {
            e.outvars[0] for e in eqns if e.primitive.name == 'stop_gradient'
        }
        (new_while,) = [e for e in eqns if e.primitive.name == 'while']
        var_invars = [v for v in new_while.invars if isinstance(v, Var)]
        assert all(v in sg_outs for v in var_invars), (
            'the while must consume ONLY detached copies of its Var inputs'
        )

        # the hidden slot is redirected to a fresh var, and an add re-defines
        # the ORIGINAL hidden outvar (so downstream eqns keep their reference)
        assert h_out not in new_while.outvars
        (add_eqn,) = [
            e for e in eqns
            if e.primitive.name == 'add' and e.outvars[0] is h_out
        ]
        assert add_eqn.invars[0] in set(new_while.outvars)
        assert add_eqn.invars[1] in set(perturb.perturb_vars)

        # downstream consumer (reduce_sum) still consumes the original var
        (sum_eqn,) = [e for e in eqns if e.primitive.name == 'reduce_sum']
        assert h_out in sum_eqn.invars

    def test_perturbed_jaxpr_is_reverse_traceable(self):
        import jax
        import jax.numpy as jnp
        import numpy as np

        closed, h_in, h_out, h0, x = self._make_while_fixture()
        perturb = self._perturb(closed, h_in, h_out)

        def eval_loss(h0_val, p_val):
            outs = perturb.eval_jaxpr([h0_val, x], [p_val])
            return outs[1]

        p0 = jnp.zeros_like(h0)
        _, f_vjp = jax.vjp(eval_loss, h0, p0)
        # the load-bearing property: tracing f_vjp must not hit
        # "Reverse-mode differentiation does not work for lax.while_loop"
        jax.make_jaxpr(f_vjp)(jnp.ones(()))

        dh0, dp = f_vjp(jnp.ones(()))
        # L = sum(while(...) + p): dL/dp is exactly ones
        np.testing.assert_allclose(np.asarray(dp), np.ones_like(h0), rtol=1e-6)
        # the loop inputs are detached, so the reverse signal into h0 is zero
        # (temporal credit flows via the forward-computed D^t instead)
        np.testing.assert_allclose(np.asarray(dh0), np.zeros_like(h0))

    def test_perturbed_values_unchanged_by_detach(self):
        import jax
        import jax.numpy as jnp
        import numpy as np

        closed, h_in, h_out, h0, x = self._make_while_fixture()
        perturb = self._perturb(closed, h_in, h_out)

        base_h, base_loss = jax.core.eval_jaxpr(
            closed.jaxpr, closed.consts, h0, x
        )
        p = jnp.full_like(h0, 0.25)
        out_h, out_loss = perturb.eval_jaxpr([h0, x], [p])
        np.testing.assert_allclose(np.asarray(out_h), np.asarray(base_h) + 0.25, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(out_loss), np.asarray(base_loss) + 0.25 * h0.size, rtol=1e-6
        )

    @staticmethod
    def _perturb_fn(f, h0, x):
        import jax
        from braintrace._compiler.hidden_pertubation import (
            JaxprEvalForHiddenPerturbation,
        )
        closed = jax.make_jaxpr(f)(h0, x)
        h_in = closed.jaxpr.invars[0]
        h_out = closed.jaxpr.outvars[0]
        perturb = JaxprEvalForHiddenPerturbation(
            closed_jaxpr=closed,
            hidden_outvar_to_invar={h_out: h_in},
            weight_invars=set(),
            invar_to_hidden_path={h_in: ('h',)},
            outvar_to_hidden_path={h_out: ('h',)},
            path_to_state={('h',): None},
        ).compile()
        return closed, perturb

    @staticmethod
    def _halve_twice(v):
        import jax
        return jax.lax.while_loop(
            lambda s: s[0] < 2, lambda s: (s[0] + 1, s[1] * 0.5), (0, v))[1]

    @pytest.mark.parametrize('mode', ['indirect', 'input_fed'])
    def test_while_reaching_hidden_indirectly_is_detached(self, mode):
        """A loop whose result reaches the hidden state through further
        equations is detached too, so the perturbed VJP is traceable (#186)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from braintrace import DiagnosticKind
        from braintrace._compiler.diagnostics import diagnostic_context

        def f(h_prev, x):
            if mode == 'indirect':
                h_new = self._halve_twice(h_prev) + x      # while(h) + I
            else:
                h_new = jnp.tanh(h_prev + self._halve_twice(x))
            return h_new, h_new.sum()

        h0, x = jnp.ones(3), jnp.full(3, 2.0)
        with diagnostic_context() as reporter, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            closed, perturb = self._perturb_fn(f, h0, x)
        eqns = perturb.perturb_jaxpr.jaxpr.eqns
        (orig_while,) = [e for e in closed.jaxpr.eqns if e.primitive.name == 'while']
        (new_while,) = [e for e in eqns if e.primitive.name == 'while']
        assert list(new_while.invars) != list(orig_while.invars)
        detaches = [r for r in reporter.records()
                    if r.kind == DiagnosticKind.CONTROL_FLOW_OPAQUE_FWD
                    and r.context.get('site') == 'perturbation']
        assert len(detaches) == 1

        def loss(h, p):
            return perturb.eval_jaxpr([h, x], [p])[1]

        _, f_vjp = jax.vjp(loss, h0, jnp.zeros_like(h0))
        dh0, dp = f_vjp(jnp.ones(()))
        np.testing.assert_allclose(np.asarray(dp), np.ones(3), rtol=1e-6)
        # values are unchanged by the detach
        base = jax.core.eval_jaxpr(closed.jaxpr, closed.consts, h0, x)
        out = perturb.eval_jaxpr([h0, x], [jnp.zeros(3)])
        np.testing.assert_allclose(np.asarray(out[0]), np.asarray(base[0]), rtol=1e-6)

    def test_while_not_reaching_hidden_untouched(self):
        """A loop on the readout only reaches no hidden state: it is kept as
        is (detaching it would silently zero dL/dh)."""
        import jax.numpy as jnp

        def f(h_prev, x):
            h_new = jnp.tanh(h_prev + x)
            return h_new, self._halve_twice(h_new).sum()

        closed, perturb = self._perturb_fn(f, jnp.zeros(3), jnp.ones(3))
        eqns = perturb.perturb_jaxpr.jaxpr.eqns
        assert 'stop_gradient' not in [e.primitive.name for e in eqns]
        (orig_while,) = [e for e in closed.jaxpr.eqns if e.primitive.name == 'while']
        (new_while,) = [e for e in eqns if e.primitive.name == 'while']
        assert list(new_while.invars) == list(orig_while.invars)

    def test_while_hidden_error_policy_raises(self):
        import jax.numpy as jnp
        from braintrace._compiler.hidden_pertubation import (
            add_hidden_perturbation_from_minfo,
        )

        class _WhileCell(brainstate.nn.Module):
            def __init__(self, n, k=3):
                super().__init__()
                self.k = k
                self.h = brainstate.HiddenState(jnp.zeros(n))

            def init_state(self, *args, **kwargs):
                self.h.value = jnp.zeros_like(self.h.value)

            def update(self, x):
                import jax

                def body(s):
                    i, h = s
                    return i + 1, h + 0.5 * jnp.tanh(x - h)

                _, h_new = jax.lax.while_loop(
                    lambda s: s[0] < self.k, body, (0, self.h.value)
                )
                self.h.value = h_new
                return h_new

        cell = _WhileCell(4)
        brainstate.nn.init_all_states(cell)
        x = brainstate.random.rand(4)
        policy = braintrace.ControlFlowPolicy(while_hidden='error')
        minfo = braintrace.extract_module_info(cell, x, control_flow=policy)
        with pytest.raises(NotImplementedError):
            add_hidden_perturbation_from_minfo(minfo)


# ---------------------------------------------------------------------------
# E-01: perturbation data -> hidden-group data must raise, not assert
#
# `perturb_data_to_hidden_group_data` maps perturbation values onto paths
# positionally, so a length disagreement re-attributes every value past the
# mismatch. The guard used to be an `assert`, which `python -O` strips, and a
# group asking for an unperturbed path used to surface as a bare `KeyError`.
# ---------------------------------------------------------------------------

class TestPerturbDataToHiddenGroupDataGuards:

    @staticmethod
    def _perturbation():
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        x = brainstate.random.randn(3)
        perturb = add_hidden_perturbation_in_module(gru, x)
        groups, _ = braintrace.find_hidden_groups_from_module(gru, x)
        return perturb, list(groups)

    def test_wrong_length_raises_value_error_not_assertion_error(self):
        import jax.numpy as jnp

        perturb, groups = self._perturbation()
        data = perturb.init_perturb_data()
        too_long = list(data) + [jnp.zeros_like(data[0])]

        with pytest.raises(ValueError) as exc:
            perturb.perturb_data_to_hidden_group_data(too_long, groups)
        message = str(exc.value)
        assert 'perturb data' in message
        assert str(len(too_long)) in message

    def test_wrong_length_short_raises(self):
        perturb, groups = self._perturbation()
        with pytest.raises(ValueError):
            perturb.perturb_data_to_hidden_group_data([], groups)

    def test_unperturbed_path_is_named(self):
        perturb, groups = self._perturbation()
        data = perturb.init_perturb_data()
        # Same cardinality, but the perturbed path no longer matches what the
        # group asks for -- previously a bare `KeyError`.
        renamed = perturb._replace(
            perturb_hidden_paths=[('not', 'the', 'group', 'path')]
            * len(perturb.perturb_hidden_paths)
        )

        with pytest.raises(ValueError) as exc:
            renamed.perturb_data_to_hidden_group_data(data, groups)
        message = str(exc.value)
        assert str(groups[0].hidden_paths[0]) in message
        assert 'not' in message

    def test_well_formed_data_still_maps(self):
        perturb, groups = self._perturbation()
        out = perturb.perturb_data_to_hidden_group_data(perturb.init_perturb_data(), groups)
        assert len(out) == len(groups)
        for arr, group in zip(out, groups):
            assert arr.shape == (*group.varshape, group.num_state)


# =============================================================================
# In-step reads of new hidden values (#239)
# =============================================================================
#
# Under ``vjp_method='single-step'`` the learning signal of ``h_k`` is read off
# its perturbation. When the update of ``h_j`` reads the new ``h_k`` and every
# trace paired with ``h_k``'s signal carries that read, the read is cut: it
# sees the unperturbed value, so the path ``W -> h_k -> h_j`` is counted once
# (in ``s_j e_j``), not twice.

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

from braintrace._compatible_imports import Var
from braintrace._compiler.diagnostics import DiagnosticKind, DiagnosticLevel
from braintrace._compiler.hidden_pertubation import (
    NewReadPolicy,
    add_hidden_perturbation_from_minfo,
    plan_new_read_cuts,
)
from braintrace._compiler.position_graph import eqn_preserves_positions
from braintrace._testing.oracle import (
    bptt_param_gradients,
    chunked_online_param_gradients,
    online_param_gradients_singlestep_naive,
)

_N = 4
_MIX = np.linspace(0.5, 1.5, _N * 3).reshape(_N, 3)


def _zeros(batch_size, n=_N):
    return jnp.zeros((n,) if batch_size is None else (batch_size, n))


class _SameGroup(brainstate.nn.Module):
    """g -> V in one group: the read is cut."""

    def __init__(self):
        super().__init__()
        self.W = braintrace.nn.Linear(3, _N)

    def init_state(self, batch_size=None, **kwargs):
        self.g = brainstate.HiddenState(_zeros(batch_size))
        self.V = brainstate.HiddenState(_zeros(batch_size))

    def update(self, x):
        self.g.value = 0.8 * self.g.value + self.W(x)
        self.V.value = 0.9 * self.V.value + jnp.tanh(self.g.value)
        return self.V.value


class _Shared(_SameGroup):
    """The read g -> V also feeds the output: its equations are emitted twice."""

    def update(self, x):
        self.g.value = 0.8 * self.g.value + self.W(x)
        i = jnp.tanh(self.g.value) * 2.0
        self.V.value = 0.9 * self.V.value + i
        return self.V.value + 0.5 * i


class _ReturnsG(_SameGroup):
    """The output reads the new g directly and through V."""

    def update(self, x):
        self.g.value = 0.8 * self.g.value + self.W(x)
        self.V.value = 0.9 * self.V.value + jnp.sin(self.g.value)
        return jnp.concatenate([self.V.value, self.g.value], axis=-1)


class _Chain(_SameGroup):
    """g -> V -> a in one group: two cut reads in a row."""

    def init_state(self, batch_size=None, **kwargs):
        super().init_state(batch_size)
        self.a = brainstate.HiddenState(_zeros(batch_size))

    def update(self, x):
        self.g.value = 0.8 * self.g.value + self.W(x)
        self.V.value = 0.9 * self.V.value + jnp.tanh(self.g.value)
        self.a.value = 0.7 * self.a.value + self.V.value ** 2
        return self.a.value - self.V.value


class _ToShortTerm(_SameGroup):
    """A non-hidden state reads the new g too: it is in the perturbed world."""

    def init_state(self, batch_size=None, **kwargs):
        super().init_state(batch_size)
        self.r = brainstate.ShortTermState(_zeros(batch_size))

    def update(self, x):
        self.g.value = 0.8 * self.g.value + self.W(x)
        self.V.value = 0.9 * self.V.value + jnp.tanh(self.g.value)
        self.r.value = self.g.value * 2.0
        return self.V.value + self.r.value


class _Stacked(brainstate.nn.Module):
    """Layer 2 reads layer 1's new state through an ETP op: kept."""

    def __init__(self):
        super().__init__()
        self.W1 = braintrace.nn.Linear(3, _N)
        self.W2 = braintrace.nn.Linear(_N, _N)

    def init_state(self, batch_size=None, **kwargs):
        self.h1 = brainstate.HiddenState(_zeros(batch_size))
        self.h2 = brainstate.HiddenState(_zeros(batch_size))

    def update(self, x):
        self.h1.value = 0.8 * self.h1.value + jnp.tanh(self.W1(x))
        self.h2.value = 0.9 * self.h2.value + jnp.tanh(self.W2(self.h1.value))
        return self.h2.value


class _CrossGroup(_SameGroup):
    """V (size 3) reads the new g through a constant matrix; no relation
    traces V's group: kept."""

    def init_state(self, batch_size=None, **kwargs):
        self.g = brainstate.HiddenState(_zeros(batch_size))
        self.V = brainstate.HiddenState(_zeros(batch_size, 3))

    def update(self, x):
        self.g.value = 0.8 * self.g.value + self.W(x)
        self.V.value = 0.9 * self.V.value + jnp.tanh(self.g.value) @ jnp.asarray(_MIX, self.g.value.dtype)
        return self.V.value


class _Disagree(brainstate.nn.Module):
    """W1 and W2 feed g; only W1 also feeds V; V reads the new g through a
    constant mixing (V's own group). The relations disagree on g -> V."""

    def __init__(self):
        super().__init__()
        self.W1 = braintrace.nn.Linear(3, _N)
        self.W2 = braintrace.nn.Linear(3, _N)
        self.U = braintrace.nn.Linear(3, _N)

    def init_state(self, batch_size=None, **kwargs):
        self.g = brainstate.HiddenState(_zeros(batch_size))
        self.V = brainstate.HiddenState(_zeros(batch_size))

    def update(self, x):
        y1 = self.W1(x)
        self.g.value = 0.8 * self.g.value + y1 + self.W2(x)
        mix = jnp.asarray(np.linspace(0.2, 1.0, _N * _N).reshape(_N, _N), self.g.value.dtype)
        self.V.value = 0.9 * self.V.value + self.U(x) * jnp.tanh(y1) + jnp.tanh(self.g.value) @ mix
        return self.V.value


class _DirectAndEtp(_SameGroup):
    """V reads the new g directly and through a second ETP op."""

    def __init__(self):
        super().__init__()
        self.W2 = braintrace.nn.Linear(_N, _N)

    def update(self, x):
        self.g.value = 0.8 * self.g.value + self.W(x)
        self.V.value = 0.9 * self.V.value + jnp.tanh(self.g.value) + self.W2(self.g.value)
        return self.V.value


class _ConflictPlusCut(brainstate.nn.Module):
    """A conflict (h2 reads h1 directly and through an ETP op) next to an
    independent read the policy would cut (g -> V): no read is cut."""

    def __init__(self):
        super().__init__()
        self.W1 = braintrace.nn.Linear(3, _N)
        self.W2 = braintrace.nn.Linear(_N, _N)
        self.Wg = braintrace.nn.Linear(3, _N)

    def init_state(self, batch_size=None, **kwargs):
        for name in ('h1', 'h2', 'g', 'V'):
            setattr(self, name, brainstate.HiddenState(_zeros(batch_size)))

    def update(self, x):
        self.h1.value = 0.8 * self.h1.value + jnp.tanh(self.W1(x))
        self.h2.value = 0.9 * self.h2.value + jnp.tanh(self.W2(self.h1.value)) + 0.3 * jnp.sin(self.h1.value)
        self.g.value = 0.8 * self.g.value + self.Wg(x)
        self.V.value = 0.9 * self.V.value + jnp.tanh(self.g.value)
        return self.h2.value + self.V.value


class _CutSourceEtpReader(brainstate.nn.Module):
    """g1 -> V1 is cut; a stacked layer also reads the new g1 through an ETP
    op, and that read must see the perturbed g1."""

    def __init__(self):
        super().__init__()
        self.W1 = braintrace.nn.Linear(3, _N)
        self.W2 = braintrace.nn.Linear(_N, _N)

    def init_state(self, batch_size=None, **kwargs):
        for name in ('g1', 'V1', 'h2'):
            setattr(self, name, brainstate.HiddenState(_zeros(batch_size)))

    def update(self, x):
        self.g1.value = 0.8 * self.g1.value + self.W1(x)
        self.V1.value = 0.9 * self.V1.value + jnp.tanh(self.g1.value)
        self.h2.value = 0.9 * self.h2.value + jnp.tanh(self.W2(self.g1.value))
        return self.V1.value + self.h2.value


# The reads a trace does not carry across steps (review of #247): the
# relations would cut them, but the transition of the reader's group does not
# hold the read, so a cut would drop the history term ``h_k^{t-1} -> h_k^t ->
# h_j^t`` from every signal. Each must be reported, not cut.
_MIX44 = np.linspace(0.2, 1.0, _N * _N).reshape(_N, _N)


class _TwoStates(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = braintrace.nn.Linear(3, _N)

    def init_state(self, batch_size=None, **kwargs):
        self.g = brainstate.HiddenState(_zeros(batch_size))
        self.V = brainstate.HiddenState(_zeros(batch_size))


class _NoRecG(_TwoStates):
    """g has no recurrence: g and V fall in two groups, V reads the new g."""

    def update(self, x):
        y = self.W(x)
        self.g.value = y * 1.0
        self.V.value = 0.9 * self.V.value + jnp.tanh(self.g.value) + jnp.sin(y)
        return self.V.value


class _DiagMix(_TwoStates):
    """V reads the new g through 0.7 I (a dot_general): two groups."""

    def update(self, x):
        y = self.W(x)
        self.g.value = 0.8 * self.g.value + y
        mix = jnp.asarray(0.7 * np.eye(_N), self.g.value.dtype)
        self.V.value = 0.9 * self.V.value + jnp.sin(y) + jnp.tanh(self.g.value) @ mix
        return self.V.value


class _MixG(_TwoStates):
    """g recurs through a constant mix (its own group); V reads it elementwise."""

    def update(self, x):
        y = self.W(x)
        mix = jnp.asarray(_MIX44 / 3, self.g.value.dtype)
        self.g.value = self.g.value @ mix + y
        self.V.value = 0.9 * self.V.value + jnp.sin(y) + jnp.tanh(self.g.value)
        return self.V.value


class _RecViaEtp(_TwoStates):
    """g recurs through an ETP matrix: the relations disagree on g -> V."""

    def __init__(self):
        super().__init__()
        self.R = braintrace.nn.Linear(_N, _N)

    def update(self, x):
        y = self.W(x)
        self.g.value = 0.3 * self.R(self.g.value) + y
        self.V.value = 0.9 * self.V.value + jnp.sin(y) + jnp.tanh(self.g.value)
        return self.V.value


class _Real(brainstate.nn.Module):
    """One group: h1 and h2 read the new h0 elementwise, h2 reads the new h1
    through M = 0.6 I (a dot_general the diagonal transition leaves out)."""

    M = 0.6 * np.eye(3)

    def __init__(self):
        super().__init__()
        self.W = braintrace.nn.Linear(3, 3)

    def init_state(self, batch_size=None, **kwargs):
        for i in range(3):
            setattr(self, f'h{i}', brainstate.HiddenState(0.3 * (i + 1) * jnp.ones((batch_size, 3))))

    def read_h1(self, h1):
        return h1 @ jnp.asarray(self.M, h1.dtype)

    def update(self, x):
        self.h0.value = 0.8 * self.h0.value + jnp.tanh(self.W(x))
        self.h1.value = 0.5 * self.h1.value + jnp.tanh(self.h0.value)
        self.h2.value = 0.5 * self.h2.value + self.read_h1(self.h1.value) + 0.1 * jnp.tanh(self.h0.value)
        return self.h2.value


class _RealDense(_Real):
    """The same with a dense M."""

    M = np.linspace(0.1, 0.9, 9).reshape(3, 3) / 2


class _RealMul(_Real):
    """The same read written elementwise: every read is carried and cut."""

    def read_h1(self, h1):
        return 0.6 * h1


_ALGORITHMS = {'D_RTRL': braintrace.D_RTRL, 'EProp': braintrace.EProp, 'OSTLRecurrent': braintrace.OSTLRecurrent}
_P3 = ('D_RTRL', 'EProp')
_P7 = ('D_RTRL', 'EProp', 'OSTLRecurrent')
_ALL_THREE = [(('h0',), ('h1',)), (('h0',), ('h2',)), (('h1',), ('h2',))]
# name: (model, algorithms, the declared cuts, or None for a reported conflict)
_CARRY_CASES = {
    'no-rec-g': (_NoRecG, _P3, None),
    'diag-mix': (_DiagMix, _P3, None),
    'mix-g': (_MixG, _P3, None),
    'rec-via-etp': (_RecViaEtp, _P3, None),
    'real': (_Real, _P7, None),
    'real-dense': (_RealDense, _P7, None),
    'real-mul': (_RealMul, _P7, _ALL_THREE),
}

_CUT_MODELS = {
    'same-group': (_SameGroup, [(('g',), ('V',))]),
    'shared': (_Shared, [(('g',), ('V',))]),
    'returns-g': (_ReturnsG, [(('g',), ('V',))]),
    'chain': (_Chain, [(('g',), ('V',)), (('V',), ('a',))]),
    'short-term': (_ToShortTerm, [(('g',), ('V',))]),
}
_KEEP_MODELS = {'stacked': _Stacked, 'cross-group': _CrossGroup}


def _make(cls):
    def make():
        with brainstate.random.seed_context(1):
            return cls()
    return make


def _graph(cls, algo=braintrace.D_RTRL, **kwargs):
    model = _make(cls)()
    brainstate.nn.init_all_states(model, batch_size=1)
    learner = algo(model, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        learner.compile_graph(jnp.ones((1, 3)))
    return learner.graph


def _records(graph, kind):
    return [r for r in graph.diagnostics if r.kind is kind]


def _flat(tree):
    return {jax.tree_util.keystr(k): np.asarray(u.get_mantissa(v), dtype=np.float64)
            for k, v in jax.tree_util.tree_flatten_with_path(tree, is_leaf=u.math.is_quantity)[0]}


def _worst_relative_error(cls, steps, algo_factory=braintrace.D_RTRL, precision=64):
    with brainstate.environ.context(precision=precision):
        with brainstate.random.seed_context(2):
            xs = brainstate.random.randn(steps, 1, 3)
        ref = _flat(bptt_param_gradients(_make(cls), xs))
        got = _flat(online_param_gradients_singlestep_naive(_make(cls), xs, algo_factory=algo_factory))
    return max(float(np.linalg.norm(got[k] - r) / np.linalg.norm(r)) for k, r in ref.items()
               if np.linalg.norm(r) > 0)


class _Everything:
    def __contains__(self, item):
        return True


class _Group:
    """A group stub; its transition computes every value unless ``carried``
    names the ones it does."""

    def __init__(self, index, carried=None, varshape=(1, _N)):
        self.index, self.varshape = index, varshape
        self._carried = _Everything() if carried is None else frozenset(carried)

    def carried_vars(self):
        return self._carried


class _Relation:
    def __init__(self, *indices):
        self.hidden_groups = [_Group(i) for i in indices]


class TestNewReadPolicy:
    """The cut rule: cut iff every relation that traces h_k's group also
    traces h_j's group."""

    groups = {('a',): _Group(0), ('b',): _Group(0), ('c',): _Group(1), ('d',): _Group(2)}

    def test_a_read_within_a_traced_group_is_cut(self):
        policy = NewReadPolicy(self.groups, [_Relation(0)])
        assert policy.edge(('a',), ('b',)) == 'cut'

    def test_a_read_across_groups_that_every_relation_traces_is_cut(self):
        policy = NewReadPolicy(self.groups, [_Relation(0, 1), _Relation(0, 1, 2)])
        assert policy.edge(('a',), ('c',)) == 'cut'

    def test_a_read_no_relation_of_the_source_traces_is_kept(self):
        policy = NewReadPolicy(self.groups, [_Relation(0), _Relation(1)])
        assert policy.edge(('a',), ('c',)) == 'keep'

    def test_relations_that_disagree_conflict(self):
        policy = NewReadPolicy(self.groups, [_Relation(0, 1), _Relation(0)])
        assert policy.edge(('a',), ('c',)) == 'conflict'

    def test_an_untraced_or_ungrouped_state_is_kept(self):
        policy = NewReadPolicy(self.groups, [_Relation(1)])
        assert policy.edge(('a',), ('b',)) == 'keep'      # no relation traces group 0
        assert policy.edge(('x',), ('c',)) == 'keep'      # x is in no group (#240)
        assert policy.edge(('c',), ('x',)) == 'keep'
        assert policy.group(('x',)) is None

    def test_traced_is_the_union_of_the_source_s_relations(self):
        policy = NewReadPolicy(self.groups, [_Relation(0, 1), _Relation(0, 2), _Relation(1)])
        assert policy.traced(('a',)) == {0, 1, 2}
        assert policy.traced(('c',)) == {0, 1}
        assert policy.traced(('x',)) == frozenset()


def _read_of(fn):
    """The equations of ``fn(g, v) -> new v`` that read the new ``g``, and the
    values on that read."""
    closed = jax.make_jaxpr(fn)(jnp.ones((1, _N)), jnp.ones((1, _N)))
    read_vars = {closed.jaxpr.invars[0]}
    eqns = []
    for eqn in closed.jaxpr.eqns:
        if any(isinstance(v, Var) and v in read_vars for v in eqn.invars):
            eqns.append(eqn)
            read_vars.update(eqn.outvars)
    return eqns, read_vars


_DENSE_NN = np.linspace(0.1, 0.9, _N * _N).reshape(_N, _N) / 2
_PERM = np.array([2, 0, 3, 1])


def _mix(a):
    return a @ jnp.asarray(_DENSE_NN, a.dtype)


@jax.custom_jvp
def _cj_mix(a):
    return _mix(a)


@_cj_mix.defjvp
def _cj_mix_jvp(primals, tangents):
    return _mix(primals[0]), _mix(tangents[0])


@jax.custom_jvp
def _cj_tanh(a):
    return jnp.tanh(a)


@_cj_tanh.defjvp
def _cj_tanh_jvp(primals, tangents):
    return jnp.tanh(primals[0]), tangents[0] * 1.5 / (1 + primals[0] ** 2)


@jax.custom_jvp
def _cj_tanh_mixing_rule(a):
    """An elementwise primal whose derivative rule mixes positions."""
    return jnp.tanh(a)


@_cj_tanh_mixing_rule.defjvp
def _cj_tanh_mixing_rule_jvp(primals, tangents):
    return jnp.tanh(primals[0]), _mix(tangents[0])


@jax.custom_vjp
def _cv_tanh(a):
    return jnp.tanh(a)


_cv_tanh.defvjp(lambda a: (jnp.tanh(a), a), lambda a, g: (g * (1 - jnp.tanh(a) ** 2),))


def _scan_tanh(a):
    out, _ = jax.lax.scan(lambda c, _: (jnp.tanh(c), None), a, None, length=1)
    return out


# reads of the new g that do not keep positions, or are not proven to (the
# review's p18 and p19 probes): a cut would not be carried
_UNPROVEN_READS = {
    'roll': lambda a: jnp.roll(a, 1, axis=-1),
    'flip': lambda a: jnp.flip(a, axis=-1),
    'cumsum': lambda a: 0.5 * jnp.cumsum(a, axis=-1),
    'mean-field': lambda a: a - jnp.mean(a, axis=-1, keepdims=True),
    'gather': lambda a: a[..., _PERM],
    # indices of the full position shape: only the allowlist rejects it
    'gather-full-index': lambda a: a[jnp.zeros(a.shape, jnp.int32),
                                     jnp.broadcast_to(jnp.asarray(_PERM, jnp.int32), a.shape)],
    'sum-broadcast': lambda a: 0.3 * jnp.sum(a, axis=-1, keepdims=True) * jnp.ones_like(a),
    'pad-shift': lambda a: jnp.pad(a, ((0, 0), (1, 0)))[..., :_N],
    'sort': lambda a: jnp.sort(a, axis=-1),
    'reshape': lambda a: jnp.tanh(a.reshape(_N, 1)).reshape(1, _N),
    'diagonal-dot': lambda a: a @ jnp.asarray(0.6 * np.eye(_N), a.dtype),
    'dense-dot': _mix,
    'custom_jvp-dense': _cj_mix,
    'custom_jvp-mixing-rule': _cj_tanh_mixing_rule,
    'checkpoint-dense': jax.checkpoint(_mix),
    'custom_vjp': _cv_tanh,
    'cond': lambda a: jax.lax.cond(True, jnp.tanh, jnp.sin, a),
    'scan': _scan_tanh,
    # widens the new g into the slots of a HiddenGroupState: slot l at
    # position i reads position l (in reads-into-slots, only the allowlist
    # rejects it)
    'broadcast-transposed': lambda a: jax.lax.broadcast_in_dim(a, a.shape + a.shape[-1:], (0, a.ndim)),
}
# the reads above that widen the read value into the slots of a group state
_INTO_SLOTS = ('broadcast-transposed',)
_PROVEN_READS = {
    'elementwise': lambda a: 0.6 * jnp.tanh(a) + jnp.where(a > 0, a, 0.1 * a),
    'custom_jvp-elementwise': _cj_tanh,
    'checkpoint-elementwise': jax.checkpoint(lambda a: jnp.exp(-a) * jnp.sin(a)),
    'nested-calls': jax.checkpoint(lambda a: _cj_tanh(a) * 2.0),
}


class _ReadsH1(brainstate.nn.Module):
    """One group over ``_N`` positions (h2 also reads the new h0): h1 and h2
    read the new h0 elementwise, h2 reads the new h1 through ``read``."""

    read = staticmethod(lambda a: 0.6 * a)

    def __init__(self):
        super().__init__()
        self.W = braintrace.nn.Linear(3, _N)

    def init_state(self, batch_size=None, **kwargs):
        for i in range(3):
            setattr(self, f'h{i}', brainstate.HiddenState(0.3 * (i + 1) * jnp.ones((batch_size, _N))))

    def update(self, x):
        self.h0.value = 0.8 * self.h0.value + jnp.tanh(self.W(x))
        self.h1.value = 0.5 * self.h1.value + jnp.tanh(self.h0.value)
        self.h2.value = 0.5 * self.h2.value + type(self).read(self.h1.value) + 0.1 * jnp.tanh(self.h0.value)
        return self.h2.value


def _reads_h1(name, read):
    return type(f'_ReadsH1_{name}', (_ReadsH1,), {'read': staticmethod(read)})


class _CrossGroupAdd(_TwoStates):
    """g recurs through a constant mix (its own group); V adds the new g, so
    every equation on the read (an ``add``) is in V's transition. Only the
    group condition keeps the read (review of 003943c, p24)."""

    def update(self, x):
        y = self.W(x)
        self.g.value = self.g.value @ jnp.asarray(_MIX44 / 3, y.dtype) + y
        self.V.value = 0.9 * self.V.value + jnp.sin(y) + self.g.value
        return self.V.value


class _NoRecGAdd(_TwoStates):
    """The same with a g that has no recurrence."""

    def update(self, x):
        y = self.W(x)
        self.g.value = y * 1.0
        self.V.value = 0.9 * self.V.value + jnp.sin(y) + self.g.value
        return self.V.value


class _ReadsIntoSlots(brainstate.nn.Module):
    """One group: ``st``, a HiddenGroupState of ``_N`` slots, reads the new h0
    through ``read`` (the second fix-check's p28)."""

    read = None

    def __init__(self):
        super().__init__()
        self.W = braintrace.nn.Linear(3, _N)

    def init_state(self, batch_size=None, **kwargs):
        self.h0 = brainstate.HiddenState(0.3 * jnp.ones((batch_size, _N)))
        self.st = brainstate.HiddenGroupState(0.6 * jnp.ones((batch_size, _N, _N)))

    def update(self, x):
        self.h0.value = 0.8 * self.h0.value + jnp.tanh(self.W(x))
        self.st.value = 0.5 * self.st.value + type(self).read(self.h0.value)
        return self.st.value.sum(-1)


def _reads_into_slots(name, read):
    return type(f'_ReadsIntoSlots_{name}', (_ReadsIntoSlots,), {'read': staticmethod(read)})


_CARRY_CASES.update({f'reads-h1-{n}': (_reads_h1(n, r), _P7, None)
                     for n, r in _UNPROVEN_READS.items() if n not in _INTO_SLOTS})
_CARRY_CASES.update({f'reads-into-slots-{n}': (_reads_into_slots(n, _UNPROVEN_READS[n]), _P7, None) for n in _INTO_SLOTS})
_CARRY_CASES.update({f'reads-h1-{n}': (_reads_h1(n, r), _P7, _ALL_THREE) for n, r in _PROVEN_READS.items()})
_CARRY_CASES.update({'cross-group-add': (_CrossGroupAdd, _P3, None), 'no-rec-g-add': (_NoRecGAdd, _P3, None)})


class TestWhyNotCarried:
    """A read the relations would cut is cut only if the group transition
    provably carries it across steps: one group, and every equation on the
    read computing a value of ``HiddenGroup.carried_vars`` and proven to
    keep positions (an allow-listed elementwise primitive on the full
    position shape, or a call made of them)."""

    def _policy(self, carried=None):
        g0 = _Group(0, carried=carried)
        return NewReadPolicy({('g',): g0, ('V',): g0, ('c',): _Group(1)}, [_Relation(0, 1)])

    @pytest.mark.parametrize('name', sorted(_PROVEN_READS))
    def test_a_read_through_elementwise_equations_is_carried(self, name):
        read = _PROVEN_READS[name]
        eqns, read_vars = _read_of(lambda g, v: 0.9 * v + read(g))
        assert self._policy().why_not_carried(('g',), ('V',), eqns, read_vars) is None

    def test_an_empty_read_is_carried(self):
        assert self._policy().why_not_carried(('g',), ('V',), [], set()) is None

    def test_a_read_across_groups_is_not_carried(self):
        policy = self._policy()
        assert policy.edge(('g',), ('c',)) == 'cut'
        eqns, read_vars = _read_of(lambda g, v: 0.9 * v + jnp.tanh(g))
        why = policy.why_not_carried(('g',), ('c',), eqns, read_vars)
        assert why == "the read ('g',) -> ('c',) crosses hidden groups, whose traces do not carry it across steps"
        assert 'crosses hidden groups' in policy.why_not_carried(('x',), ('V',), [], set())

    def test_a_read_through_an_equation_the_transition_leaves_out_is_not_carried(self):
        eqns, read_vars = _read_of(lambda g, v: 0.9 * v + jnp.tanh(g))
        tanh, add = eqns
        why = self._policy(carried=add.outvars).why_not_carried(('g',), ('V',), eqns, read_vars)
        assert why == ("the read ('g',) -> ('V',) goes through tanh, which the transition "
                       "of their group leaves out or holds fixed")

    @pytest.mark.parametrize('name', sorted(_UNPROVEN_READS))
    def test_a_read_not_proven_to_keep_positions_is_not_carried(self, name):
        read = _UNPROVEN_READS[name]
        eqns, read_vars = _read_of(lambda g, v: 0.9 * v + read(g))
        why = self._policy().why_not_carried(('g',), ('V',), eqns, read_vars)
        assert why is not None and why.startswith("the read ('g',) -> ('V',) goes through ")
        assert why.endswith('which is not proven to keep positions '
                            '(only elementwise equations on the full position shape are)')

    def test_an_elementwise_read_on_a_partial_shape_is_not_carried(self):
        """The group's positions are (1, 2 * _N): a read of the first half only
        is elementwise on a value that is not the full position shape."""
        g0 = _Group(0, varshape=(1, 2 * _N))
        policy = NewReadPolicy({('g',): g0, ('V',): g0}, [_Relation(0)])
        eqns, read_vars = _read_of(lambda g, v: 0.9 * v + jnp.tanh(g))
        assert 'not proven to keep positions' in policy.why_not_carried(('g',), ('V',), eqns, read_vars)

    def test_a_custom_jvp_constant_on_the_read_is_not_proven(self):
        """A custom_jvp has no derivative in a value it closes over."""
        def fn(g, v):
            @jax.custom_jvp
            def scaled(a):
                return a * g

            scaled.defjvp(lambda p, t: (scaled(p[0]), t[0] * g))
            return scaled(v)

        eqns, read_vars = _read_of(fn)
        (call,) = [e for e in eqns if e.primitive.name == 'custom_jvp_call']
        assert call.params['num_consts'] >= 1
        assert not eqn_preserves_positions(call, (1, _N), [0])


class _Rule:
    """A planner policy from fixed decisions (for hand-built jaxprs)."""

    def __init__(self, edges, groups=None, traced=(), uncarried=()):
        self.edges, self.groups, self._traced = edges, groups or {}, frozenset(traced)
        self.uncarried, self.asked = frozenset(uncarried), []

    def edge(self, path_k, path_j):
        return self.edges.get((path_k, path_j), 'keep')

    def why_not_carried(self, path_k, path_j, read_eqns, read_vars):
        self.asked.append((path_k, path_j, [str(e.primitive) for e in read_eqns]))
        return f'not carried: {path_k} -> {path_j}' if (path_k, path_j) in self.uncarried else None

    def group(self, path):
        return self.groups.get(path)

    def traced(self, path):
        return self._traced


def _plan(fn, n_hidden, rule, n_args=3, shape=(1, _N)):
    """Plan ``fn(*hidden_old, x, w) -> (*hidden_new, out)``, hidden paths named
    'h0', 'h1', ..."""
    args = [jnp.ones(shape)] * n_hidden + [jnp.ones((1, 3)), jnp.ones((3, _N))][:n_args - n_hidden + 1]
    closed = jax.make_jaxpr(fn)(*args)
    hidden = closed.jaxpr.outvars[:n_hidden]
    paths = {v: (f'h{i}',) for i, v in enumerate(hidden)}
    return plan_new_read_cuts(closed.jaxpr, hidden, paths, [n_hidden], rule)


class TestPlanNewReadCuts:
    """The planner on hand-built jaxprs: each row of the #239 decision table,
    and every conflict."""

    cut01 = {(('h0',), ('h1',)): 'cut'}

    def test_cut_read(self):
        def fn(h0, h1, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            return n0, 0.9 * h1 + jnp.tanh(n0), h1

        plan = _plan(fn, 2, _Rule(self.cut01))
        assert plan.cuts == ((('h0',), ('h1',)),) and plan.conflicts == ()
        assert len(plan.sources) == 1 and plan.state_eqns and not plan.output_eqns

    def test_a_read_that_also_feeds_the_output_is_emitted_in_both_worlds(self):
        def fn(h0, h1, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            t = jnp.tanh(n0)
            return n0, 0.9 * h1 + t, t

        plan = _plan(fn, 2, _Rule(self.cut01))
        assert plan.conflicts == ()
        assert plan.state_eqns & plan.output_eqns  # tanh serves both

    def test_kept_read_has_no_plan(self):
        def fn(h0, h1, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            return n0, 0.9 * h1 + jnp.tanh(n0), h1

        plan = _plan(fn, 2, _Rule({}))
        assert plan.cuts == () and plan.sources == {} and plan.conflicts == ()

    def test_a_read_only_through_an_etp_op_is_kept_without_asking(self):
        def fn(h0, h1, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            return n0, 0.9 * h1 + braintrace.matmul(n0[:, :3], w), h1

        plan = _plan(fn, 2, _Rule(self.cut01))
        assert plan.cuts == () and plan.conflicts == ()

    def test_a_read_inside_one_equation_is_not_a_read(self):
        def fn(h0, h1, x, w):
            def both(a, b):
                n = 0.9 * a + braintrace.matmul(x, w)
                return n, 0.9 * b + jnp.tanh(n)
            return (*jax.jit(both)(h0, h1), h1)

        plan = _plan(fn, 2, _Rule(self.cut01))
        assert plan.cuts == () and plan.conflicts == ()

    def test_a_read_only_state_is_not_a_source(self):
        def fn(h0, h1, x, w):
            return h0, 0.9 * h1 + jnp.tanh(h0), h1   # h0 is read, never written

        plan = _plan(fn, 2, _Rule(self.cut01))
        assert plan.cuts == () and plan.sources == {}

    def test_disagreeing_relations_conflict(self):
        def fn(h0, h1, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            return n0, 0.9 * h1 + jnp.tanh(n0), h1

        plan = _plan(fn, 2, _Rule({(('h0',), ('h1',)): 'conflict'}))
        assert plan.conflicts == ("the relations disagree on the read ('h0',) -> ('h1',)",)

    def test_a_cut_read_also_made_through_an_etp_op_conflicts(self):
        def fn(h0, h1, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            return n0, 0.9 * h1 + jnp.tanh(n0) + braintrace.matmul(n0[:, :3], w), h1

        plan = _plan(fn, 2, _Rule(self.cut01))
        assert plan.conflicts == ("('h1',) reads ('h0',) both directly and through an ETP op",)

    def test_a_kept_read_that_reaches_a_traced_group_conflicts(self):
        # h0 -> h1 is kept, but h1 -> h2 reaches a group h0's relations trace:
        # their traces carry h0 -> h1 -> h2, the kept read counts it again
        def fn(h0, h1, h2, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            n1 = 0.9 * h1 + jnp.tanh(n0)
            return n0, n1, 0.9 * h2 + jnp.tanh(n1), h2

        rule = _Rule({}, groups={('h0',): 0, ('h1',): 1, ('h2',): 0}, traced={0})
        plan = _plan(fn, 3, rule, n_args=4)
        assert any('kept read' in c for c in plan.conflicts), plan.conflicts

    def test_a_state_with_a_cut_and_a_kept_read_of_a_cut_source_conflicts(self):
        def fn(h0, h1, h2, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            n1 = 0.9 * h1 + jnp.tanh(n0)
            return n0, n1, 0.9 * h2 + jnp.tanh(n1) + jnp.sin(n0), h2

        rule = _Rule({(('h0',), ('h1',)): 'cut', (('h1',), ('h2',)): 'cut'})  # h0 -> h2 kept
        plan = _plan(fn, 3, rule, n_args=4)
        assert "('h2',) has a cut read and a kept read of ('h0',)" in plan.conflicts

    @pytest.mark.parametrize('kind', ['while', 'effects'])
    def test_a_shared_equation_that_cannot_be_emitted_twice_conflicts(self, kind):
        def fn(h0, h1, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            t = jnp.tanh(n0)
            if kind == 'while':
                t = jax.lax.while_loop(lambda c: c[1] < 3, lambda c: (c[0] * 0.5, c[1] + 1), (t, 0))[0]
            else:
                t = io_callback(lambda a: a, jax.ShapeDtypeStruct(t.shape, t.dtype), t)
            return n0, 0.9 * h1 + t, t

        plan = _plan(fn, 2, _Rule(self.cut01))
        assert any('cannot be emitted twice' in c for c in plan.conflicts), plan.conflicts

    def test_one_equation_with_a_cut_and_a_kept_read_conflicts(self):
        def fn(h0, h1, h2, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            n1, n2 = jax.jit(lambda a, b, c: (0.9 * b + jnp.tanh(a), 0.9 * c + jnp.sin(a)))(n0, h1, h2)
            return n0, n1, n2, h2

        plan = _plan(fn, 3, _Rule(self.cut01), n_args=4)  # h0 -> h2 is kept
        assert any("kept read of ('h2',)" in c for c in plan.conflicts), plan.conflicts

    def test_two_cut_reads_in_a_row(self):
        # g -> V -> a: each read is cut, each source keeps its own cut read
        def fn(h0, h1, h2, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            n1 = 0.9 * h1 + jnp.tanh(n0)
            return n0, n1, 0.9 * h2 + jnp.sin(n1), h2

        rule = _Rule({(('h0',), ('h1',)): 'cut', (('h1',), ('h2',)): 'cut'})
        plan = _plan(fn, 3, rule, n_args=4)
        assert plan.cuts == ((('h0',), ('h1',)), (('h1',), ('h2',))) and plan.conflicts == ()
        assert len(plan.sources) == 2 and not plan.output_eqns

    def test_a_non_hidden_output_of_the_source_is_in_the_perturbed_world(self):
        # the model output (a merge position) reads the new h0 through mul
        def fn(h0, h1, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            return n0, 0.9 * h1 + jnp.tanh(n0), 2.0 * n0

        closed = jax.make_jaxpr(fn)(jnp.ones((1, _N)), jnp.ones((1, _N)), jnp.ones((1, 3)), jnp.ones((3, _N)))
        hidden = closed.jaxpr.outvars[:2]
        plan = plan_new_read_cuts(closed.jaxpr, hidden, {v: (f'h{i}',) for i, v in enumerate(hidden)},
                                  [2], _Rule(self.cut01))
        (out_eqn,) = [i for i, e in enumerate(closed.jaxpr.eqns) if closed.jaxpr.outvars[2] in e.outvars]
        assert plan.cuts == ((('h0',), ('h1',)),) and plan.conflicts == ()
        assert out_eqn in plan.output_eqns and out_eqn not in plan.state_eqns

    def test_one_source_with_two_cut_reads_cuts_both(self):
        # h0 -> h1 and h0 -> h2: every cut read of one source is cut
        def fn(h0, h1, h2, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            return n0, 0.9 * h1 + jnp.tanh(n0), 0.9 * h2 + jnp.sin(n0), h2

        rule = _Rule({(('h0',), ('h1',)): 'cut', (('h0',), ('h2',)): 'cut'})
        plan = _plan(fn, 3, rule, n_args=4)
        assert plan.cuts == ((('h0',), ('h1',)), (('h0',), ('h2',))) and plan.conflicts == ()
        assert len(plan.sources) == 1

    def test_the_policy_is_asked_about_the_equations_on_the_read(self):
        def fn(h0, h1, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            return n0, 0.9 * h1 + jnp.tanh(n0), h1

        rule = _Rule(self.cut01)
        _plan(fn, 2, rule)
        assert rule.asked == [(('h0',), ('h1',), ['tanh', 'add'])]

    def test_a_path_through_stop_gradient_is_no_part_of_a_read(self):
        # h1 = pre - stop_gradient(f(pre)), pre = 0.9 h1 + n0: the detached
        # branch carries no derivative, in a trace or a signal
        def fn(h0, h1, h2, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            pre = 0.9 * h1 + n0
            return n0, pre - jax.lax.stop_gradient(jnp.tanh(pre - 1.0)), 0.9 * h2 + jax.lax.stop_gradient(n0), pre

        rule = _Rule({(('h0',), ('h1',)): 'cut', (('h0',), ('h2',)): 'cut'})
        plan = _plan(fn, 3, rule, n_args=4)
        assert plan.cuts == ((('h0',), ('h1',)),) and plan.conflicts == ()   # h2 reads h0 only detached
        assert rule.asked == [(('h0',), ('h1',), ['add', 'sub'])]

    def test_a_cut_read_the_transition_does_not_carry_conflicts(self):
        def fn(h0, h1, h2, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            return n0, 0.9 * h1 + jnp.tanh(n0), 0.9 * h2 + jnp.sin(n0), h2

        rule = _Rule({(('h0',), ('h1',)): 'cut', (('h0',), ('h2',)): 'cut'},
                     uncarried={(('h0',), ('h2',))})
        plan = _plan(fn, 3, rule, n_args=4)
        assert plan.conflicts == ("not carried: ('h0',) -> ('h2',)",)

    def test_a_reader_in_no_group_is_kept_and_reads_the_perturbed_value(self):
        # #240's latch: h2 (in no group) copies the new h0; h1 (h0's group) is cut
        def fn(h0, h1, h2, x, w):
            n0 = 0.9 * h0 + braintrace.matmul(x, w)
            return n0, 0.9 * h1 + jnp.tanh(n0), jnp.sin(n0), h2

        policy = NewReadPolicy({('h0',): _Group(0), ('h1',): _Group(0)}, [_Relation(0)])
        plan = _plan(fn, 3, policy, n_args=4)
        assert plan.cuts == ((('h0',), ('h1',)),) and plan.conflicts == ()
        assert plan.output_eqns  # the eqn that feeds h2 reads the perturbed h0


class TestNewReadCutsOnGraphs:
    """The plan applied by ``compile_etrace_graph`` (single-step D-RTRL)."""

    @pytest.mark.parametrize('name', list(_CUT_MODELS))
    def test_cut_graph_declares_its_cuts(self, name):
        cls, cuts = _CUT_MODELS[name]
        graph = _graph(cls)
        hp = graph.hidden_perturb
        assert list(hp.new_read_cuts) == cuts
        assert hp.new_read_conflicts == () and hp.new_read_exact is True
        (record,) = _records(graph, DiagnosticKind.HIDDEN_NEW_READ_CUT)
        assert record.level is DiagnosticLevel.INFO and list(record.context['cuts']) == cuts
        assert _records(graph, DiagnosticKind.HIDDEN_NEW_READ_DOUBLE_COUNT) == []

    @pytest.mark.parametrize('name', list(_CUT_MODELS))
    @pytest.mark.parametrize('steps', [1, 6])
    def test_cut_graph_is_exact(self, name, steps):
        """Every cut model has a diagonal, weight-free recurrence: single-step
        D-RTRL is exact there, at every window."""
        assert _worst_relative_error(_CUT_MODELS[name][0], steps) < 1e-12

    @pytest.mark.parametrize('name', list(_KEEP_MODELS))
    def test_a_kept_read_leaves_the_perturbation_unchanged(self, name):
        graph = _graph(_KEEP_MODELS[name])
        hp = graph.hidden_perturb
        assert hp.new_read_cuts == () and hp.new_read_exact is True
        assert not [r for r in graph.diagnostics if r.kind.value.startswith('hidden_new_read')]
        today = add_hidden_perturbation_from_minfo(graph.module_info)
        assert str(hp.perturb_jaxpr) == str(today.perturb_jaxpr)

    @pytest.mark.parametrize('name', list(_KEEP_MODELS))
    def test_a_kept_read_is_exact_at_one_step(self, name):
        assert _worst_relative_error(_KEEP_MODELS[name], 1) < 1e-12

    # _ConflictPlusCut and _Real also hold a read the policy would cut: the
    # plan of a graph with a conflict must not be applied at all
    @pytest.mark.parametrize('cls', [_Disagree, _DirectAndEtp, _ConflictPlusCut, _Real])
    def test_a_conflict_keeps_today_s_signal_and_records_an_error(self, cls):
        with pytest.warns(UserWarning, match='cannot be separated'):
            model = _make(cls)()
            brainstate.nn.init_all_states(model, batch_size=1)
            learner = braintrace.D_RTRL(model)
            learner.compile_graph(jnp.ones((1, 3)))
        graph = learner.graph
        hp = graph.hidden_perturb
        assert hp.new_read_cuts == () and hp.new_read_conflicts and hp.new_read_exact is False
        (record,) = _records(graph, DiagnosticKind.HIDDEN_NEW_READ_DOUBLE_COUNT)
        assert record.level is DiagnosticLevel.ERROR
        assert tuple(record.context['conflicts']) == tuple(hp.new_read_conflicts)
        today = add_hidden_perturbation_from_minfo(graph.module_info)
        assert str(hp.perturb_jaxpr) == str(today.perturb_jaxpr)

    def test_multi_step_builds_no_perturbation(self):
        graph = _graph(_SameGroup, vjp_method='multi-step')
        assert graph.hidden_perturb is None
        assert not [r for r in graph.diagnostics if r.kind.value.startswith('hidden_new_read')]

    def test_without_relations_nothing_is_planned(self):
        model = _make(_SameGroup)()
        brainstate.nn.init_all_states(model)
        hp = add_hidden_perturbation_in_module(model, jnp.ones(3))
        assert hp.new_read_cuts == () and hp.new_read_conflicts == () and hp.new_read_exact is False

    def test_the_cut_holds_the_read_fixed(self):
        """dL/dp_g: the partial derivative with V's read of the new g held
        fixed. Here the loss reads only V, so it is zero; today it is the
        total derivative through V."""
        graph = _graph(_SameGroup)
        hp = graph.hidden_perturb
        today = add_hidden_perturbation_from_minfo(graph.module_info)
        minfo = graph.module_info
        args = (jnp.ones((1, 3)),)
        old = [st.value for st in minfo.compiled_model_states]
        inputs = jax.tree.leaves((args, old))
        out_index = 0  # the model output (V)

        def signal(perturbation):
            def loss(perturb):
                return jnp.sum(perturbation.eval_jaxpr(inputs, perturb)[out_index] ** 2)
            return jax.grad(loss)(perturbation.init_perturb_data())

        cut, total = signal(hp), signal(today)
        g = list(hp.perturb_hidden_paths).index(('g',))
        assert float(jnp.abs(cut[g]).max()) == 0.0
        assert float(jnp.abs(total[g]).max()) > 0.0

    @pytest.mark.parametrize('name', ['same-group', 'chain'])
    def test_float32_is_exact_against_float32_bptt(self, name):
        assert _worst_relative_error(_CUT_MODELS[name][0], 1, precision=32) < 1e-6

    def test_a_cut_source_s_own_slot_is_perturbed(self):
        """The new value of a cut source leaves the step perturbed: a loss on
        it gets ``dL/dp`` through the slot (the cut reads alone see the
        unperturbed value)."""
        graph = _graph(_SameGroup)
        hp, minfo = graph.hidden_perturb, graph.module_info
        (g_out,) = [v for v, path in minfo.outvar_to_hidden_path.items() if path == ('g',)]
        slot = list(minfo.jaxpr.outvars).index(g_out)
        inputs = jax.tree.leaves(((jnp.ones((1, 3)),), [st.value for st in minfo.compiled_model_states]))
        grads = jax.grad(lambda p: jnp.sum(hp.eval_jaxpr(inputs, p)[slot]))(hp.init_perturb_data())
        g = list(hp.perturb_hidden_paths).index(('g',))
        np.testing.assert_array_equal(np.asarray(grads[g]), np.ones_like(np.asarray(grads[g])))

    def test_a_loss_on_the_new_hidden_values_is_exact_at_one_step(self):
        """A loss that also reads the new hidden values, through the state
        slots, at one step from a zero state."""
        def step_loss(model, out):
            return (out ** 2).sum() + 0.7 * (model.g.value ** 2).sum() + 0.3 * jnp.sin(model.V.value).sum()

        with brainstate.environ.context(precision=64):
            x = jnp.asarray([[0.3, 0.7, 0.2]])
            ref_model = _make(_SameGroup)()
            brainstate.nn.init_all_states(ref_model, batch_size=1)
            ref_params = ref_model.states(brainstate.ParamState)
            ref = brainstate.transform.grad(lambda: step_loss(ref_model, ref_model(x)), ref_params)()

            model = _make(_SameGroup)()
            brainstate.nn.init_all_states(model, batch_size=1)
            learner = braintrace.D_RTRL(model)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                learner.compile_graph(x)
            learner.init_etrace_state()
            params = model.states(brainstate.ParamState)
            got = brainstate.transform.grad(lambda xx: step_loss(model, learner(xx)), params)(x)
        assert list(learner.graph.hidden_perturb.new_read_cuts) == [(('g',), ('V',))]
        ref, got = _flat(ref), _flat(got)
        for key, r in ref.items():
            assert np.linalg.norm(got[key] - r) <= 1e-12 * np.linalg.norm(r), key

    @pytest.mark.parametrize('algo', [braintrace.D_RTRL, braintrace.EProp])
    def test_an_etp_read_of_a_cut_source_sees_the_perturbed_value(self, algo):
        """A stacked layer reads the cut source g1 through an ETP op: that
        read is kept, in the perturbed world."""
        hp = _graph(_CutSourceEtpReader, algo).hidden_perturb
        assert list(hp.new_read_cuts) == [(('g1',), ('V1',))] and hp.new_read_exact is True
        assert _worst_relative_error(_CutSourceEtpReader, 1, algo_factory=algo) < 1e-12


def _compile_recording(cls, algo):
    model = _make(cls)()
    brainstate.nn.init_all_states(model, batch_size=1)
    learner = algo(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        learner.compile_graph(jnp.ones((1, 3)))
    return learner.graph, [w for w in caught if 'cannot be separated' in str(w.message)]


def _single_vs_multi(cls, steps, algo):
    """Single-step against forced multi-step (one step per chunk), f64."""
    with brainstate.environ.context(precision=64):
        with brainstate.random.seed_context(7):
            xs = brainstate.random.rand(steps, 1, 3) + 0.1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            single = _flat(online_param_gradients_singlestep_naive(_make(cls), xs, algo_factory=algo))
            multi = _flat(chunked_online_param_gradients(
                _make(cls), xs, algo_factory=lambda m: algo(m, vjp_method='multi-step'), chunk_size=1))
    return max(float(np.linalg.norm(single[k] - r) / np.linalg.norm(r)) for k, r in multi.items()
               if np.linalg.norm(r) > 0)


class TestOnlyCarriedReadsAreCut:
    """A read is cut only when the group transition carries it across steps
    (review of #247): otherwise the cut would drop ``h_k^{t-1} -> h_k^t ->
    h_j^t`` from every signal. Each case either declares no double count and
    matches multi-step at T in {1, 2, 6}, or is reported."""

    @pytest.mark.parametrize('name, algo', [(n, a) for n, case in _CARRY_CASES.items() for a in case[1]])
    def test_a_read_is_cut_only_when_the_transition_carries_it(self, name, algo):
        cls, _, cuts = _CARRY_CASES[name]
        graph, warned = _compile_recording(cls, _ALGORITHMS[algo])
        hp = graph.hidden_perturb
        if cuts is None:
            assert hp.new_read_exact is False and hp.new_read_cuts == () and hp.new_read_conflicts
            assert len(warned) == 1 and issubclass(warned[0].category, UserWarning)
            (record,) = _records(graph, DiagnosticKind.HIDDEN_NEW_READ_DOUBLE_COUNT)
            assert record.level is DiagnosticLevel.ERROR
            today = add_hidden_perturbation_from_minfo(graph.module_info)
            assert str(hp.perturb_jaxpr) == str(today.perturb_jaxpr)
        else:
            assert hp.new_read_exact is True and list(hp.new_read_cuts) == cuts and warned == []

    @pytest.mark.parametrize('steps', [1, 2, 6])
    @pytest.mark.parametrize('name, algo', [(n, a) for n, case in _CARRY_CASES.items() if case[2] for a in case[1]])
    def test_a_carried_cut_matches_multi_step(self, name, algo, steps):
        assert _single_vs_multi(_CARRY_CASES[name][0], steps, _ALGORITHMS[algo]) < 1e-12
