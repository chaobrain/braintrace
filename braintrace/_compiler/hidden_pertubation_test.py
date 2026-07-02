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

from pprint import pprint

import brainstate
import pytest
import brainunit as u

import braintrace
from braintrace._compiler.hidden_pertubation import add_hidden_perturbation_in_module
from braintrace._etrace_model_test import (
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

    def test_non_hidden_while_untouched(self):
        import jax
        import jax.numpy as jnp
        from braintrace._compiler.hidden_pertubation import (
            JaxprEvalForHiddenPerturbation,
        )

        def f(h_prev, x):
            # the while computes an auxiliary value; the hidden state is
            # produced by a plain tanh equation
            def body(s):
                i, v = s
                return i + 1, v * 0.5

            _, aux = jax.lax.while_loop(lambda s: s[0] < 2, body, (0, x))
            h_new = jnp.tanh(h_prev + aux)
            return h_new

        h0 = jnp.zeros(3)
        x = jnp.ones(3)
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

        eqns = perturb.perturb_jaxpr.jaxpr.eqns
        names = [e.primitive.name for e in eqns]
        assert 'stop_gradient' not in names
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
