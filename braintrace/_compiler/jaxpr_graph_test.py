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


class TestInlineJitInsideControlFlow:
    """Phase 4: jit bodies inside scan/while/cond are inlined so body-level
    analysis (structured scan descent) sees flat ETP equations."""

    def test_inline_jit_calls_descends_into_scan_body(self):
        from braintrace._compatible_imports import is_scan_primitive

        @jax.jit
        def inner(a):
            return a * 2.0

        def body(c, x):
            return c + inner(x), c

        def f(xs):
            return jax.lax.scan(body, jnp.zeros(()), xs)

        closed = jax.make_jaxpr(f)(jnp.arange(4.0))
        scan_eqn = next(e for e in closed.jaxpr.eqns if is_scan_primitive(e))
        assert _has_jit_eqn(scan_eqn.params['jaxpr'].jaxpr)

        cleaned = inline_jit_calls(closed)
        scan_eqn2 = next(e for e in cleaned.jaxpr.eqns if is_scan_primitive(e))
        assert not _has_jit_eqn(scan_eqn2.params['jaxpr'].jaxpr)
        # outer interface identity preserved
        assert cleaned.jaxpr.invars == closed.jaxpr.invars
        assert list(cleaned.jaxpr.outvars) == list(closed.jaxpr.outvars)
        # numerics preserved
        xs = jnp.arange(4.0)
        ref = _eval_closed(closed, xs)
        got = _eval_closed(cleaned, xs)
        for a, b in zip(ref, got):
            np.testing.assert_allclose(a, b)

    def test_inline_jit_calls_keeps_jitfree_scan_eqn_by_identity(self):
        from braintrace._compatible_imports import is_scan_primitive

        @jax.jit
        def inner(a):
            return a * 2.0

        def body(c, x):
            return c + x, c

        def f(xs):
            # a top-level jit forces the pass to actually run; the jit-free
            # scan eqn must still come through by identity
            return jax.lax.scan(body, inner(jnp.zeros(())), xs)

        closed = jax.make_jaxpr(f)(jnp.arange(4.0))
        cleaned = inline_jit_calls(closed)
        e1 = next(e for e in closed.jaxpr.eqns if is_scan_primitive(e))
        e2 = next(e for e in cleaned.jaxpr.eqns if is_scan_primitive(e))
        assert e1.params['jaxpr'] is e2.params['jaxpr']

    def test_jitfree_program_with_scan_returned_unchanged(self):
        def body(c, x):
            return c + x, c

        closed = jax.make_jaxpr(
            lambda xs: jax.lax.scan(body, jnp.zeros(()), xs))(jnp.arange(4.0))
        assert inline_jit_calls(closed) is closed

    def test_inline_jit_calls_descends_into_cond_branches(self):
        from braintrace._compatible_imports import is_cond_primitive

        @jax.jit
        def inner(a):
            return a * 3.0

        def f(x):
            return jax.lax.cond(x.sum() > 0., lambda: inner(x), lambda: x * 2.)

        closed = jax.make_jaxpr(f)(jnp.arange(3.0) - 1.0)
        cond_eqn = next(e for e in closed.jaxpr.eqns if is_cond_primitive(e))
        assert any(_has_jit_eqn(b.jaxpr) for b in cond_eqn.params['branches'])

        cleaned = inline_jit_calls(closed)
        cond_eqn2 = next(e for e in cleaned.jaxpr.eqns if is_cond_primitive(e))
        assert all(
            not _has_jit_eqn(b.jaxpr) for b in cond_eqn2.params['branches'])
        for arg in (jnp.arange(3.0) + 1.0, jnp.arange(3.0) - 5.0):
            np.testing.assert_allclose(
                _eval_closed(cleaned, arg)[0], f(arg))


_M = np.linspace(0.1, 0.9, 9).reshape(3, 3)


def _dense_tanh(a):
    """Closes over a NumPy constant, so its jit body has a constvar."""
    return jnp.tanh(a @ jnp.asarray(_M, a.dtype))


_SCALE = np.asarray([0.5, 1.0, 1.5])


def _scaled_tanh(a):
    """Closes over a NumPy constant and keeps positions apart."""
    return jnp.tanh(a * jnp.asarray(_SCALE, a.dtype))


def _remat_eqns(jaxpr):
    return [e for e in jaxpr.eqns if e.primitive.name in ('remat2', 'checkpoint')]


class TestInlineJitInsideCheckpoint:
    """A ``jit`` nested in a ``checkpoint`` (#251). A ``remat2`` body is an open
    jaxpr: JAX's ``remat_jvp`` asserts it has no constvars. Inlining the jit
    hoists its constants into operands of the checkpoint, so the body stays
    open and the walker sees one level."""

    @staticmethod
    def _check(f, x, n_remat=1):
        closed = jax.make_jaxpr(f)(x)
        inlined = inline_jit_calls(closed)
        remats = _remat_eqns(inlined.jaxpr)
        assert len(remats) == n_remat
        for eqn in remats:
            body = eqn.params['jaxpr']
            assert not body.constvars and not _has_jit_eqn(body)
            assert len(body.invars) == len(eqn.invars)
        # the same values, and the same derivatives in both modes
        run = lambda v: _eval_closed(inlined, v)[0]
        np.testing.assert_allclose(run(x), f(x), rtol=1e-6)
        t = jnp.linspace(-1., 1., x.size).reshape(x.shape)
        np.testing.assert_allclose(jax.jvp(run, (x,), (t,))[1], jax.jvp(f, (x,), (t,))[1], rtol=1e-6)
        np.testing.assert_allclose(jax.grad(lambda v: run(v).sum())(x),
                                   jax.grad(lambda v: f(v).sum())(x), rtol=1e-6)
        return inlined

    def test_a_jit_with_constants_in_a_checkpoint(self):
        inlined = self._check(jax.checkpoint(jax.jit(_dense_tanh)), jnp.ones((1, 3)) * 0.3)
        (eqn,) = _remat_eqns(inlined.jaxpr)
        assert len(eqn.invars) == 2  # the hoisted constant, then the operand

    def test_a_jit_without_constants_in_a_checkpoint(self):
        self._check(jax.checkpoint(jax.jit(jnp.tanh)), jnp.ones((1, 3)) * 0.3)

    def test_a_checkpoint_in_a_checkpoint(self):
        self._check(jax.checkpoint(lambda a: 2. * jax.checkpoint(jax.jit(_dense_tanh))(a)),
                    jnp.ones((1, 3)) * 0.3)

    def test_a_checkpoint_in_a_scan_body(self):
        step = jax.checkpoint(jax.jit(_dense_tanh))
        f = lambda a: jax.lax.scan(lambda c, _: (step(c), None), a, None, length=2)[0]
        x = jnp.ones((1, 3)) * 0.3
        inlined = inline_jit_calls(jax.make_jaxpr(f)(x))
        (scan,) = [e for e in inlined.jaxpr.eqns if e.primitive.name == 'scan']
        (eqn,) = _remat_eqns(scan.params['jaxpr'].jaxpr)
        assert not eqn.params['jaxpr'].constvars
        run = lambda v: _eval_closed(inlined, v)[0]
        np.testing.assert_allclose(jax.grad(lambda v: run(v).sum())(x),
                                   jax.grad(lambda v: f(v).sum())(x), rtol=1e-6)

    def test_online_learning_through_the_checkpoint(self):
        """The issue's reproduction: D-RTRL differentiates the rewritten
        checkpoint; it raised an AssertionError in JAX's ``remat_jvp``. With
        an elementwise body (a mixing one would make D-RTRL approximate), D-RTRL
        is exact, so BPTT is the oracle."""
        import brainstate
        import braintrace
        from braintrace._testing.oracle import (
            assert_param_gradients_close,
            bptt_param_gradients,
            online_param_gradients_singlestep_naive,
        )
        step = jax.checkpoint(jax.jit(_scaled_tanh))
        assert _remat_eqns(jax.make_jaxpr(step)(jnp.ones(3)).jaxpr)[0].params['jaxpr'].eqns[0].params['jaxpr'].consts

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.W = brainstate.ParamState(jnp.asarray(_M, jnp.float32))

            def init_state(self, batch_size=None, **kwargs):
                self.h = brainstate.HiddenState(0.3 * jnp.ones((batch_size, 3)))

            def update(self, x):
                self.h.value = 0.5 * self.h.value + step(braintrace.matmul(x, self.W.value))
                return self.h.value

        with brainstate.random.seed_context(0):
            xs = brainstate.random.uniform(-1., 1., (5, 1, 3))
        g = online_param_gradients_singlestep_naive(Cell, xs, algo_factory=braintrace.D_RTRL)
        assert_param_gradients_close(g, bptt_param_gradients(Cell, xs), atol=1e-6, rtol=1e-5)

    def test_a_jit_with_constants_in_a_shard_map_is_kept(self):
        """Another open body: its jit is kept, not inlined with unbound
        constants (on main its evaluation raised in ``eval_jaxpr``)."""
        from jax.sharding import PartitionSpec as P
        mesh = jax.make_mesh((1,), ('x',))
        f = jax.shard_map(jax.jit(_scaled_tanh), mesh=mesh, in_specs=P(), out_specs=P())
        x = jnp.asarray([0.3, -0.7, 1.1], jnp.float32)
        new = inline_jit_calls(jax.make_jaxpr(f)(x))
        (eqn,) = new.jaxpr.eqns
        assert eqn.params['jaxpr'].constvars == []
        assert [e.primitive.name for e in eqn.params['jaxpr'].eqns] == ['jit']
        run = lambda a: jax.core.eval_jaxpr(new.jaxpr, new.consts, a)[0]
        np.testing.assert_array_equal(run(x), f(x))
        np.testing.assert_array_equal(jax.jvp(run, (x,), (x,))[1], jax.jvp(f, (x,), (x,))[1])
