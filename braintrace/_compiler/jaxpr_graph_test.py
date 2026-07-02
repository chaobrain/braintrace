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

import jax
import jax.numpy as jnp
import numpy as np

from braintrace._compatible_imports import Var, is_jit_primitive
from braintrace._compiler.jaxpr_graph import (
    build_consumer_map,
    build_producer_map,
    forward_closure,
    inline_jit_calls,
)


def _has_jit_eqn(jaxpr) -> bool:
    return any(is_jit_primitive(eqn) for eqn in jaxpr.eqns)


def _eval_closed(closed_jaxpr, *args):
    return jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)


class TestProducerConsumerMaps:

    def test_producer_and_consumer_maps(self):
        def f(a, b):
            c = a + b
            d = c * a
            return d

        jaxpr = jax.make_jaxpr(f)(1.0, 2.0).jaxpr
        producers = build_producer_map(jaxpr)
        consumers = build_consumer_map(jaxpr)

        add_eqn, mul_eqn = jaxpr.eqns
        c_var = add_eqn.outvars[0]
        assert producers[c_var] is add_eqn
        assert mul_eqn in consumers[c_var]
        # the first invar `a` feeds both equations
        a_var = jaxpr.invars[0]
        assert consumers[a_var] == [add_eqn, mul_eqn]

    def test_forward_closure_is_ordered_and_complete(self):
        def f(a):
            b = a + 1.0
            c = b * 2.0
            d = a - 1.0  # not downstream of b
            return c, d

        jaxpr = jax.make_jaxpr(f)(1.0).jaxpr
        consumers = build_consumer_map(jaxpr)
        b_var = jaxpr.eqns[0].outvars[0]
        closure = forward_closure(b_var, consumers)
        c_var = jaxpr.eqns[1].outvars[0]
        d_var = jaxpr.eqns[2].outvars[0]
        assert b_var in closure
        assert c_var in closure
        assert d_var not in closure
        # insertion order: start var first
        assert next(iter(closure)) is b_var


class TestInlineJitCalls:

    def test_no_jit_returns_same_object(self):
        closed = jax.make_jaxpr(lambda x: x * 2.0 + 1.0)(1.0)
        assert inline_jit_calls(closed) is closed

    def test_single_jit_is_inlined_and_equivalent(self):
        c = np.arange(3.0)

        @jax.jit
        def g(x):
            return x * 2.0 + c

        def f(x):
            return g(x) + 1.0

        x = jnp.ones(3)
        closed = jax.make_jaxpr(f)(x)
        assert _has_jit_eqn(closed.jaxpr)

        inlined = inline_jit_calls(closed)
        assert not _has_jit_eqn(inlined.jaxpr)
        # invars are preserved as the same Var objects
        assert list(inlined.jaxpr.invars) == list(closed.jaxpr.invars)
        np.testing.assert_allclose(_eval_closed(inlined, x)[0], f(x))

    def test_nested_jit_two_deep(self):
        @jax.jit
        def inner(x):
            return x + 1.0

        @jax.jit
        def outer(x):
            return inner(x) * 3.0

        def f(x):
            return outer(x) - 2.0

        x = jnp.arange(4.0)
        closed = jax.make_jaxpr(f)(x)
        inlined = inline_jit_calls(closed)
        assert not _has_jit_eqn(inlined.jaxpr)
        np.testing.assert_allclose(_eval_closed(inlined, x)[0], f(x))

    def test_passthrough_output(self):
        @jax.jit
        def g(x):
            return x  # identity: inner outvar == inner invar

        def f(x):
            return g(x) + g(x)

        x = jnp.float32(3.0)
        closed = jax.make_jaxpr(f)(x)
        inlined = inline_jit_calls(closed)
        assert not _has_jit_eqn(inlined.jaxpr)
        np.testing.assert_allclose(_eval_closed(inlined, x)[0], f(x))

    def test_multi_output_jit(self):
        @jax.jit
        def g(x):
            return x * 2.0, x + 1.0

        def f(x):
            a, b = g(x)
            return a - b

        x = jnp.arange(3.0)
        closed = jax.make_jaxpr(f)(x)
        inlined = inline_jit_calls(closed)
        assert not _has_jit_eqn(inlined.jaxpr)
        np.testing.assert_allclose(_eval_closed(inlined, x)[0], f(x))

    def test_jit_output_is_final_output(self):
        @jax.jit
        def g(x):
            return x * 2.0

        def f(x):
            return g(x)  # jit result is directly the jaxpr output

        x = jnp.arange(3.0)
        closed = jax.make_jaxpr(f)(x)
        inlined = inline_jit_calls(closed)
        assert not _has_jit_eqn(inlined.jaxpr)
        np.testing.assert_allclose(_eval_closed(inlined, x)[0], f(x))

    def test_consts_are_lifted(self):
        c = np.full((2,), 7.0)

        @jax.jit
        def g(x):
            return x + c

        x = jnp.ones(2)
        closed = jax.make_jaxpr(lambda x: g(x) * 1.0)(x)
        inlined = inline_jit_calls(closed)
        assert not _has_jit_eqn(inlined.jaxpr)
        assert len(inlined.jaxpr.constvars) == len(inlined.consts)
        np.testing.assert_allclose(_eval_closed(inlined, x)[0], x + c)

    def test_scan_body_jit_left_untouched(self):
        # jit inside a scan body is NOT inlined (control flow is opaque here).
        @jax.jit
        def g(x):
            return x * 2.0

        def f(x):
            def body(carry, _):
                return g(carry), None

            out, _ = jax.lax.scan(body, x, None, length=3)
            return out

        x = jnp.float32(1.0)
        closed = jax.make_jaxpr(f)(x)
        inlined = inline_jit_calls(closed)
        np.testing.assert_allclose(_eval_closed(inlined, x)[0], f(x))

    def test_same_jit_called_twice_is_freshened_per_site(self):
        # JAX's trace cache gives both call eqns the SAME inner ClosedJaxpr
        # object; splicing it twice without freshening defines the same Vars
        # twice and silently aliases the two call results.
        @jax.jit
        def g(a, b):
            return a * b

        def f(x):
            return g(x, 2.0) + g(x, 3.0)  # = 5 x

        x = jnp.arange(1.0, 4.0)
        closed = jax.make_jaxpr(f)(x)
        inlined = inline_jit_calls(closed)
        assert not _has_jit_eqn(inlined.jaxpr)
        # SSA: every var is defined at most once.
        seen = set()
        for eqn in inlined.jaxpr.eqns:
            for ov in eqn.outvars:
                assert id(ov) not in seen, 'duplicate var definition'
                seen.add(id(ov))
        np.testing.assert_allclose(_eval_closed(inlined, x)[0], f(x))

    def test_same_jit_called_twice_gradients(self):
        @jax.jit
        def g(a, w):
            return (a * w).sum()

        def f(x):
            return g(x, jnp.full(3, 2.0)) + g(x, jnp.full(3, 3.0))

        x = jnp.arange(1.0, 4.0)
        closed = jax.make_jaxpr(f)(x)
        inlined = inline_jit_calls(closed)

        def f_inlined(xi):
            return _eval_closed(inlined, xi)[0]

        np.testing.assert_allclose(jax.grad(f_inlined)(x), jax.grad(f)(x))
