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

"""Acceptance test for ``ETPPrimitiveSpec`` plug-in registration.

The goal of this test is the *extensibility contract*: a third-party user
can define a new ETP primitive purely through
:class:`braintrace.ETPPrimitiveSpec` and have the compiler pick it up with
**no edits to the compiler**.  A passing test therefore demonstrates
Principle 1 (primitive-type identity dispatch) and the plugin interface
promised by the rewrite.

The primitive registered here — ``etp_rtest_mv`` — is a stand-alone
unbatched matmul variant registered only inside this module.  It has the
same semantics as ``etp_mv_p`` but a different *identity*, so the
compiler must recognise it via the spec registry rather than any hard-coded
check.
"""

import brainstate
import jax
import jax.numpy as jnp
import saiunit as u

import braintrace
from braintrace import (
    CompilationRecord,
    DiagnosticKind,
    ETPPrimitiveSpec,
    HiddenParamOpRelation,
    compile_etrace_graph,
    get_primitive_spec,
    register_primitive_spec,
)


# ---------------------------------------------------------------------------
# Define a new ETP primitive purely through the public spec API
# ---------------------------------------------------------------------------

def _rtest_mv_impl(x, w):
    return x @ w


def _rtest_mv_yw_to_w(hidden_dim, trace):
    return trace * jnp.expand_dims(hidden_dim, axis=0)


def _rtest_mv_xy_to_dw(x, hidden_dim, w):
    _, vjp_fn = jax.vjp(lambda w_: u.get_mantissa(x @ w_), w)
    return u.get_mantissa(vjp_fn(hidden_dim)[0])


def _rtest_mv_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    w_shape = weight_var.aval.shape
    return jnp.zeros((*w_shape, num_hidden_state))


def _rtest_mv_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros(
        (*y_var.aval.shape, num_hidden_state),
        dtype=y_var.aval.dtype,
    )


etp_rtest_mv_p = register_primitive_spec(ETPPrimitiveSpec(
    name='etp_rtest_mv',
    impl=_rtest_mv_impl,
    yw_to_w=_rtest_mv_yw_to_w,
    xy_to_dw=_rtest_mv_xy_to_dw,
    init_drtrl=_rtest_mv_init_drtrl,
    init_pp=_rtest_mv_init_pp,
    trainable_invars_fn=lambda params: {'weight': 1},
    x_invar_index=0,
    batched=False,
))


def rtest_matmul(x, weight):
    return etp_rtest_mv_p.bind(x, weight)


# ---------------------------------------------------------------------------
# Minimal recurrent model that uses the new primitive
# ---------------------------------------------------------------------------

class _RNNWithPluginPrimitive(brainstate.nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w_rec = brainstate.ParamState(
            brainstate.random.randn(n_in + n_out, n_out) * 0.1
        )
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value])
        self.h.value = jnp.tanh(rtest_matmul(xh, self.w_rec.value))
        return self.h.value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpecRegistry:
    """The spec is queryable for every primitive registered through it."""

    def test_all_builtin_primitives_have_spec(self):
        from braintrace._etrace_op import (
            etp_mm_p, etp_mv_p, etp_elemwise_p, etp_conv_p,
            etp_sp_mm_p, etp_sp_mv_p, etp_lora_mm_p, etp_lora_mv_p,
        )
        for prim in (
            etp_mm_p, etp_mv_p, etp_elemwise_p, etp_conv_p,
            etp_sp_mm_p, etp_sp_mv_p, etp_lora_mm_p, etp_lora_mv_p,
        ):
            spec = get_primitive_spec(prim)
            assert spec is not None, f'No spec registered for {prim}'
            assert spec.name == prim.name

    def test_elemwise_spec_is_gradient_enabled_and_x_is_none(self):
        """Principle 2: only ``etp_elemwise_p``-class primitives are
        gradient-enabled traversable on the tail."""
        from braintrace._etrace_op import etp_elemwise_p
        spec = get_primitive_spec(etp_elemwise_p)
        assert spec.gradient_enabled is True
        assert spec.x_invar_index is None

    def test_mm_mv_conv_lora_are_not_gradient_enabled(self):
        from braintrace._etrace_op import (
            etp_mm_p, etp_mv_p, etp_conv_p,
            etp_sp_mm_p, etp_sp_mv_p,
            etp_lora_mm_p, etp_lora_mv_p,
        )
        for prim in (
            etp_mm_p, etp_mv_p, etp_conv_p,
            etp_sp_mm_p, etp_sp_mv_p,
            etp_lora_mm_p, etp_lora_mv_p,
        ):
            spec = get_primitive_spec(prim)
            assert spec.gradient_enabled is False, (
                f'{prim.name} must not be gradient-enabled or the W->W->h '
                f'tail-boundary rule would silently allow double counting'
            )


class TestPluginPrimitiveEndToEnd:
    """A brand-new primitive registered through :class:`ETPPrimitiveSpec`
    participates in compilation with no compiler edits."""

    def test_plugin_primitive_is_discovered(self):
        model = _RNNWithPluginPrimitive(n_in=3, n_out=4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = compile_etrace_graph(model, inp, include_hidden_perturb=False)

        relations = graph.hidden_param_op_relations
        assert len(relations) == 1, (
            f'Expected exactly one relation (the plugin primitive); '
            f'got {len(relations)}: {[r.primitive for r in relations]}'
        )

        rel = relations[0]
        assert isinstance(rel, HiddenParamOpRelation)
        # Type-identity dispatch: the relation carries the exact primitive
        # instance we registered, not a name-matched copy.
        assert rel.primitive is etp_rtest_mv_p

    def test_plugin_primitive_records_inclusion(self):
        model = _RNNWithPluginPrimitive(n_in=3, n_out=4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = compile_etrace_graph(model, inp, include_hidden_perturb=False)

        included = graph.explain(kind=DiagnosticKind.RELATION_INCLUDED)
        assert len(included) == 1
        rec = included[0]
        assert isinstance(rec, CompilationRecord)
        assert rec.primitive is etp_rtest_mv_p

    def test_plugin_primitive_weight_and_x_vars_resolved(self):
        model = _RNNWithPluginPrimitive(n_in=3, n_out=4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = compile_etrace_graph(model, inp, include_hidden_perturb=False)

        rel = graph.hidden_param_op_relations[0]
        # Spec-driven dispatch must have picked invar[1] as the weight
        # (trainable_invars_fn -> {'weight': 1}) and invar[0] as x (x_invar_index=0).
        assert rel.weight_var is not None
        assert rel.x_var is not None
        # Weight shape follows (n_in + n_out, n_out).
        assert tuple(rel.weight_var.aval.shape) == (7, 4)
        # x is the (concat(x, h)) vector of length 7.
        assert tuple(rel.x_var.aval.shape) == (7,)


class TestLegacyRegistrationStillWorks:
    """Third-party primitives registered through the older
    :func:`register_primitive` + ``register_*`` API (no spec) continue to
    work. The compiler falls back to the historical convention when no spec
    is registered."""

    def test_legacy_primitive_falls_back_to_default_convention(self):
        from braintrace._etrace_op import (
            ETP_PRIMITIVE_SPECS,
            register_primitive,
        )

        def _impl(x, w):
            return x @ w

        prim = register_primitive(
            'etp_rtest_legacy', _impl, batched=False,
        )
        prim.register_yw_to_w(
            lambda hidden_dim, trace: trace * jnp.expand_dims(hidden_dim, 0)
        )
        prim.register_xy_to_dw(
            lambda x, hidden_dim, w: jax.vjp(
                lambda w_: u.get_mantissa(x @ w_), w
            )[1](hidden_dim)[0]
        )
        prim.register_init_drtrl(
            lambda x_var, y_var, weight_var, ns:
            jnp.zeros((*weight_var.aval.shape, ns))
        )
        prim.register_init_pp(
            lambda x_var, y_var, weight_var, ns:
            jnp.zeros((*y_var.aval.shape, ns), dtype=y_var.aval.dtype)
        )

        # Legacy API does not touch the spec registry.
        assert prim not in ETP_PRIMITIVE_SPECS
        assert get_primitive_spec(prim) is None


class TestTrainableInvarsFn:

    def test_trainable_invars_fn_is_required(self):
        import pytest
        with pytest.raises(TypeError):
            # trainable_invars_fn is now required; omitting it is a TypeError.
            ETPPrimitiveSpec(
                name='dummy',
                impl=lambda *a, **k: a[0],
                yw_to_w=lambda *a, **k: a[1],
                xy_to_dw=lambda *a, **k: a[0],
                init_drtrl=lambda *a, **k: None,
                init_pp=lambda *a, **k: None,
            )

    def test_custom_fn_is_preserved(self):
        fn = lambda params: {'weight': 1, 'bias': 2} if params.get('has_bias') else {'weight': 1}
        spec = ETPPrimitiveSpec(
            name='dummy',
            impl=lambda *a, **k: a[0],
            yw_to_w=lambda *a, **k: a[1],
            xy_to_dw=lambda *a, **k: a[0],
            init_drtrl=lambda *a, **k: None,
            init_pp=lambda *a, **k: None,
            trainable_invars_fn=fn,
        )
        assert spec.trainable_invars_fn({'has_bias': True}) == {'weight': 1, 'bias': 2}
        assert spec.trainable_invars_fn({}) == {'weight': 1}

    def test_resolve_trainable_invars_delegates_to_fn(self):
        fn = lambda params: {'lora_b': 1, 'lora_a': 2}
        spec = ETPPrimitiveSpec(
            name='dummy',
            impl=lambda *a, **k: a[0],
            yw_to_w=lambda *a, **k: a[1],
            xy_to_dw=lambda *a, **k: a[0],
            init_drtrl=lambda *a, **k: None,
            init_pp=lambda *a, **k: None,
            trainable_invars_fn=fn,
        )
        assert spec.resolve_trainable_invars({}) == {'lora_b': 1, 'lora_a': 2}
