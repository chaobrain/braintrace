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

"""Tests for :class:`ETPPrimitive` and :func:`register_primitive`.

The contract of :func:`register_primitive` is that *all* standard JAX
rules — eager impl, abstract eval, MLIR lowering, JVP, transpose,
batching — are auto-derived from a single Python ``impl_fn``. These
tests register a fresh primitive per scenario and verify that ``jit``,
``grad``, ``vmap``, ``jvp`` all work without any additional plumbing.

The four ETP rule registration helpers (`register_yw_to_w`,
`register_xy_to_dw`, `register_init_drtrl`, `register_init_pp`,
`register_etp_rules`) are exercised with both the per-rule and bulk
APIs.
"""



import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace._compatible_imports import Primitive
from braintrace._op import (
    BATCHED_PRIMITIVES,
    ETP_PRIMITIVES,
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
    GRADIENT_ENABLED_PRIMITIVES,
    ETPPrimitive,
    register_primitive,
)

# A counter to keep test-only primitives uniquely named so they don't
# collide with one another or with shipped primitives.
_PRIMITIVE_COUNTER = 0


def _fresh_name(stub):
    global _PRIMITIVE_COUNTER
    _PRIMITIVE_COUNTER += 1
    return f'etp_test_{stub}_{_PRIMITIVE_COUNTER}'


class TestETPPrimitiveType:
    """``ETPPrimitive`` must subclass ``jax.core.Primitive`` so JAX
    machinery treats it like any other primitive."""

    def test_is_jax_primitive_subclass(self):
        assert issubclass(ETPPrimitive, Primitive)

    def test_register_returns_etp_primitive(self):
        p = register_primitive(_fresh_name('subclass'), lambda x: x)
        assert isinstance(p, ETPPrimitive)
        assert isinstance(p, Primitive)


class TestRegistrationSideEffects:
    """``register_primitive`` must populate :data:`ETP_PRIMITIVES` and the
    relevant flag-sets in lockstep with the kwargs."""

    def test_default_flags_only_membership(self):
        p = register_primitive(_fresh_name('default'), lambda x: x)
        assert p in ETP_PRIMITIVES
        assert p not in BATCHED_PRIMITIVES
        assert p not in GRADIENT_ENABLED_PRIMITIVES

    def test_batched_flag(self):
        p = register_primitive(
            _fresh_name('batched'), lambda x: x, batched=True,
        )
        assert p in BATCHED_PRIMITIVES
        assert p not in GRADIENT_ENABLED_PRIMITIVES

    def test_gradient_enabled_flag(self):
        p = register_primitive(
            _fresh_name('grad'), lambda x: x, gradient_enabled=True,
        )
        assert p not in BATCHED_PRIMITIVES
        assert p in GRADIENT_ENABLED_PRIMITIVES

    def test_both_flags(self):
        p = register_primitive(
            _fresh_name('both'), lambda x: x,
            batched=True, gradient_enabled=True,
        )
        assert p in BATCHED_PRIMITIVES
        assert p in GRADIENT_ENABLED_PRIMITIVES


class TestEagerImpl:

    def test_impl_runs_via_bind(self):
        p = register_primitive(_fresh_name('impl'), lambda x, y: x + y)
        out = p.bind(jnp.asarray(2.0), jnp.asarray(3.0))
        np.testing.assert_allclose(out, 5.0)

    def test_impl_with_kwargs(self):
        def _add_const(x, *, c):
            return x + c

        p = register_primitive(_fresh_name('kwarg'), _add_const)
        out = p.bind(jnp.asarray(2.0), c=10)
        np.testing.assert_allclose(out, 12.0)


class TestAbstractEval:
    """``abstract_eval`` is auto-derived via ``jax.eval_shape(impl_fn)`` and
    drives shape inference inside ``jit``."""

    def test_abstract_eval_under_jit(self):
        p = register_primitive(_fresh_name('abs'), lambda x: x * 2.0)

        @jax.jit
        def f(x):
            return p.bind(x)

        out = f(jnp.ones((3, 4)))
        assert out.shape == (3, 4)
        np.testing.assert_allclose(out, jnp.full((3, 4), 2.0))


class TestJITAndJVP:

    def test_jit_runs(self):
        p = register_primitive(_fresh_name('jit'), lambda x, y: x * y)
        out = jax.jit(lambda x, y: p.bind(x, y))(
            jnp.asarray(3.0), jnp.asarray(4.0),
        )
        np.testing.assert_allclose(out, 12.0)

    def test_jvp_matches_impl(self):
        p = register_primitive(_fresh_name('jvp'), lambda x: x ** 2)

        primal, tangent = jax.jvp(
            lambda x: p.bind(x), (jnp.asarray(3.0),), (jnp.asarray(1.0),),
        )
        np.testing.assert_allclose(primal, 9.0)
        np.testing.assert_allclose(tangent, 6.0)  # d(x^2)/dx = 2x

    def test_grad_matches_analytic(self):
        p = register_primitive(_fresh_name('grad'), lambda x: x ** 3)

        g = jax.grad(lambda x: p.bind(x).sum())(jnp.asarray(2.0))
        np.testing.assert_allclose(g, 12.0)  # 3x^2 at x=2

    def test_jvp_handles_zero_tangent(self):
        """Symbolic ``ad.Zero`` tangents are converted to numeric zeros so
        the user-supplied ``impl_fn`` can take the JVP unconditionally."""
        p = register_primitive(_fresh_name('jvpzero'), lambda x, y: x * y)

        # vjp implicitly produces Zero tangents for non-active inputs.
        def f(x, y):
            return p.bind(x, y)

        # Differentiate only wrt x — y's tangent is symbolic zero.
        g = jax.grad(f, argnums=0)(jnp.asarray(2.0), jnp.asarray(5.0))
        np.testing.assert_allclose(g, 5.0)


class TestIntBoolPrimalZeroTangent:
    """Symbolic-zero tangents for int/bool primals must be materialized with
    JAX's ``float0`` tangent dtype, not the primal's own dtype.

    A non-inexact (int/bool) primal's *real* tangent type in JAX is the
    zero-sized ``float0`` dtype. The auto-derived ``_jvp`` in
    ``register_primitive`` used to instantiate ``ad.Zero`` tangents as
    ``jnp.zeros(pr.shape, pr.dtype)`` unconditionally, which produced an
    ``int32``/``bool`` tangent array for such primals. Passing that array
    into ``jax.jvp`` alongside an int/bool primal raises a ``TypeError``
    because JAX requires the tangent dtype to be exactly ``float0`` in that
    case. This matters in practice whenever an ETP op consumes an integer
    or boolean input (e.g. spike counts, masks) while only some *other*
    input (e.g. the weight) is being differentiated.
    """

    def test_jvp_int_primal_zero_tangent_does_not_raise(self):
        p = register_primitive(_fresh_name('int_zero'), lambda x, y: x * y)

        # Differentiate only wrt the float `y` — the int `x`'s tangent is
        # symbolic zero and must be materialized as float0, not int32.
        def f(y):
            x = jnp.asarray(3, dtype=jnp.int32)
            return p.bind(x, y)

        primal, tangent = jax.jvp(f, (jnp.asarray(2.0),), (jnp.asarray(1.0),))
        np.testing.assert_allclose(primal, 6.0)
        np.testing.assert_allclose(tangent, 3.0)  # d(x*y)/dy = x = 3

    def test_jvp_bool_primal_zero_tangent_does_not_raise(self):
        p = register_primitive(
            _fresh_name('bool_zero'),
            lambda mask, y: jnp.where(mask, y, 0.0),
        )

        def f(y):
            mask = jnp.asarray(True)
            return p.bind(mask, y)

        primal, tangent = jax.jvp(f, (jnp.asarray(2.0),), (jnp.asarray(1.0),))
        np.testing.assert_allclose(primal, 2.0)
        np.testing.assert_allclose(tangent, 1.0)

    def test_grad_int_primal_matches_float_cast(self):
        """``jax.grad`` through an ETP op with an int input must agree with
        the gradient computed from the float-cast version of that input."""
        import braintrace

        x_int = jnp.ones((2, 3), dtype=jnp.int32)
        w = jnp.ones((3, 4))

        def loss(w, x):
            return braintrace.matmul(x, w).sum()

        g_int = jax.grad(loss, argnums=0)(w, x_int)
        g_float = jax.grad(loss, argnums=0)(w, x_int.astype(jnp.float32))

        assert jnp.all(jnp.isfinite(g_int))
        np.testing.assert_allclose(g_int, g_float)

    def test_grad_bool_primal_matches_float_cast(self):
        """Same contract, for a boolean ETP input."""
        import braintrace

        x_bool = jnp.ones((2, 3), dtype=bool)
        w = jnp.ones((3, 4))

        def loss(w, x):
            return braintrace.matmul(x, w).sum()

        g_bool = jax.grad(loss, argnums=0)(w, x_bool)
        g_float = jax.grad(loss, argnums=0)(w, x_bool.astype(jnp.float32))

        assert jnp.all(jnp.isfinite(g_bool))
        np.testing.assert_allclose(g_bool, g_float)


class TestVmap:

    def test_vmap_in_axes_zero(self):
        p = register_primitive(
            _fresh_name('vmap'), lambda x: x * 2.0, batched=True,
        )

        out = jax.vmap(lambda x: p.bind(x))(jnp.arange(5.0))
        np.testing.assert_allclose(out, jnp.arange(5.0) * 2.0)

    def test_vmap_two_args(self):
        p = register_primitive(_fresh_name('vmap2'), lambda x, y: x + y)
        x = jnp.arange(4.0)
        y = jnp.arange(4.0) * 10
        out = jax.vmap(lambda a, b: p.bind(a, b))(x, y)
        np.testing.assert_allclose(out, x + y)


class TestRegisterPerRuleAPI:

    def test_register_yw_to_w(self):
        p = register_primitive(_fresh_name('yw'), lambda x: x)
        marker = object()

        def _rule(hidden_dim, trace, **params):
            return marker

        p.register_yw_to_w(_rule)
        assert ETP_RULES_YW_TO_W[p] is _rule
        assert ETP_RULES_YW_TO_W[p](None, None) is marker

    def test_register_xy_to_dw(self):
        p = register_primitive(_fresh_name('xy'), lambda x: x)

        def _rule(x, hidden_dim, w, **params):
            return 'dw'

        p.register_xy_to_dw(_rule)
        assert ETP_RULES_XY_TO_DW[p] is _rule

    def test_register_init_drtrl(self):
        p = register_primitive(_fresh_name('drtrl'), lambda x: x)

        def _init(x_var, y_var, weight_var, num_hidden_state):
            return 'drtrl'

        p.register_init_drtrl(_init)
        assert ETP_RULES_INIT_DRTRL[p] is _init

    def test_register_init_pp(self):
        p = register_primitive(_fresh_name('pp'), lambda x: x)

        def _init(x_var, y_var, weight_var, num_hidden_state):
            return 'pp'

        p.register_init_pp(_init)
        assert ETP_RULES_INIT_PP[p] is _init


class TestRegisterEtpRulesBulk:

    def test_register_all_four(self):
        p = register_primitive(_fresh_name('bulk'), lambda x: x)
        a, b, c, d = (lambda *x, **k: i for i in range(4))
        p.register_etp_rules(yw_to_w=a, xy_to_dw=b, init_drtrl=c, init_pp=d)
        assert ETP_RULES_YW_TO_W[p] is a
        assert ETP_RULES_XY_TO_DW[p] is b
        assert ETP_RULES_INIT_DRTRL[p] is c
        assert ETP_RULES_INIT_PP[p] is d

    def test_register_partial_skips_none(self):
        p = register_primitive(_fresh_name('partial'), lambda x: x)
        rule = lambda hidden_dim, trace, **params: None
        # Only yw_to_w is supplied — the other three must remain absent.
        p.register_etp_rules(yw_to_w=rule)
        assert ETP_RULES_YW_TO_W[p] is rule
        assert p not in ETP_RULES_XY_TO_DW
        assert p not in ETP_RULES_INIT_DRTRL
        assert p not in ETP_RULES_INIT_PP

    def test_overwrites_previous_rule(self):
        """Re-registering must replace the previous rule, not silently keep
        the old one."""
        p = register_primitive(_fresh_name('overwrite'), lambda x: x)
        rule_a = lambda *a, **k: 'a'
        rule_b = lambda *a, **k: 'b'
        p.register_yw_to_w(rule_a)
        assert ETP_RULES_YW_TO_W[p] is rule_a
        p.register_yw_to_w(rule_b)
        assert ETP_RULES_YW_TO_W[p] is rule_b


class TestPrimitiveNameAndRepr:

    def test_name_round_trip(self):
        name = _fresh_name('name')
        p = register_primitive(name, lambda x: x)
        assert p.name == name


class TestUnknownKwargRejected:
    """``register_primitive`` only accepts the documented keyword set; extra
    kwargs surface as ``TypeError`` so typos cannot silently misconfigure."""

    def test_unknown_kwarg_raises(self):
        with pytest.raises(TypeError):
            register_primitive(
                _fresh_name('bad'), lambda x: x, unknown=True,
            )
