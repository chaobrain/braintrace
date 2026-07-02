# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace import ControlFlowPolicy
from braintrace._compatible_imports import is_scan_primitive
from braintrace._compiler.module_info import extract_module_info
from braintrace._compiler.scan_descent import (
    _descent_blockers,
    _is_etp_relevant,
)


def _scan_model_jaxpr(loops):
    """A leaky SNN-style model whose update runs ``loops`` inner sub-steps in a
    ``for_loop``; extraction keeps the scan opaque (descent off, limit 4)."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(jnp.ones((3, 3)) * 0.1)
            self.h = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            x_row = x.reshape(1, -1)

            def substep(_):
                self.h.value = 0.9 * self.h.value + jnp.tanh(
                    braintrace.matmul(x_row, self.w.value))
                return self.h.value

            return brainstate.transform.for_loop(substep, jnp.arange(loops))[-1]

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    minfo = extract_module_info(
        net, jnp.ones((3,), dtype='float32'),
        control_flow=ControlFlowPolicy(scan_unroll_limit=4, scan_descent='off'))
    eqn = next(e for e in minfo.jaxpr.eqns if is_scan_primitive(e))
    return eqn, minfo


class TestDescendabilityPredicate:
    def test_descendable_scan_has_no_blockers(self):
        eqn, minfo = _scan_model_jaxpr(loops=8)
        policy = ControlFlowPolicy(scan_unroll_limit=4, scan_descent='auto')
        assert _is_etp_relevant(
            eqn.params['jaxpr'].jaxpr, eqn, set(minfo.weight_invars))
        assert _descent_blockers(eqn, policy, set(minfo.weight_invars)) is None

    def test_policy_off_blocks_descent(self):
        eqn, minfo = _scan_model_jaxpr(loops=8)
        policy = ControlFlowPolicy(scan_unroll_limit=4, scan_descent='off')
        blocker = _descent_blockers(eqn, policy, set(minfo.weight_invars))
        assert 'scan_descent' in blocker

    def test_short_scan_left_to_unroll(self):
        eqn, minfo = _scan_model_jaxpr(loops=8)
        policy = ControlFlowPolicy(scan_unroll_limit=16, scan_descent='auto')
        blocker = _descent_blockers(eqn, policy, set(minfo.weight_invars))
        assert 'unroll' in blocker

    def test_reverse_scan_blocked(self):
        def f(xs):
            return jax.lax.scan(
                lambda c, x: (c + x, c), jnp.zeros(()), xs, reverse=True)

        closed = jax.make_jaxpr(f)(jnp.arange(8.0))
        eqn = next(e for e in closed.jaxpr.eqns if is_scan_primitive(e))
        policy = ControlFlowPolicy(scan_unroll_limit=4, scan_descent='auto')
        blocker = _descent_blockers(eqn, policy, set())
        assert 'reverse' in blocker

    def test_nested_control_flow_blocked(self):
        def body(c, x):
            c2 = jax.lax.while_loop(
                lambda v: jnp.sum(v) < 1.0, lambda v: v + x, c)
            return c2, c

        closed = jax.make_jaxpr(
            lambda xs: jax.lax.scan(body, jnp.zeros((3,)), xs)
        )(jnp.ones((8, 3)))
        eqn = next(e for e in closed.jaxpr.eqns if is_scan_primitive(e))
        policy = ControlFlowPolicy(scan_unroll_limit=4, scan_descent='auto')
        blocker = _descent_blockers(eqn, policy, set())
        assert 'nested control flow' in blocker

    def test_weight_scanned_as_xs_blocked(self):
        def f(w_stack, x):
            return jax.lax.scan(lambda c, w: (c @ w, c), x, w_stack)

        closed = jax.make_jaxpr(f)(jnp.ones((8, 3, 3)), jnp.ones((3,)))
        eqn = next(e for e in closed.jaxpr.eqns if is_scan_primitive(e))
        w_stack_var = closed.jaxpr.invars[0]
        policy = ControlFlowPolicy(scan_unroll_limit=4, scan_descent='auto')
        blocker = _descent_blockers(eqn, policy, {w_stack_var})
        assert 'xs' in blocker

    def test_etp_irrelevant_scan_not_relevant(self):
        def f(xs):
            return jax.lax.scan(lambda c, x: (c + x, c), jnp.zeros(()), xs)

        closed = jax.make_jaxpr(f)(jnp.arange(8.0))
        eqn = next(e for e in closed.jaxpr.eqns if is_scan_primitive(e))
        assert not _is_etp_relevant(eqn.params['jaxpr'].jaxpr, eqn, set())
