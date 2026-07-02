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


class TestAddScanYs:
    def test_add_scan_ys_emits_per_substep_values(self):
        from braintrace._compiler.scan_descent import add_scan_ys
        from braintrace._compatible_imports import Jaxpr

        def body(c, x):
            y = jnp.tanh(c) + x
            return y, y * 2.0

        closed = jax.make_jaxpr(
            lambda xs: jax.lax.scan(body, jnp.zeros(()), xs))(jnp.arange(4.0))
        eqn = next(e for e in closed.jaxpr.eqns if is_scan_primitive(e))
        body_jaxpr = eqn.params['jaxpr'].jaxpr
        # hoist: the tanh intermediate (a body-computed var) and the carry invar
        tanh_var = next(e.outvars[0] for e in body_jaxpr.eqns
                        if e.primitive.name == 'tanh')
        num_consts, num_carry = eqn.params['num_consts'], eqn.params['num_carry']
        carry_invar = body_jaxpr.invars[num_consts]

        new_eqn, stacked = add_scan_ys(eqn, [tanh_var, carry_invar])
        assert list(new_eqn.outvars[:len(eqn.outvars)]) == list(eqn.outvars)
        assert stacked[tanh_var].aval.shape == (4,)
        assert stacked[carry_invar].aval.shape == (4,)
        assert new_eqn.params['num_carry'] == num_carry
        assert new_eqn.params['num_consts'] == num_consts
        assert new_eqn.params['length'] == eqn.params['length']
        # body eqns preserved by identity
        assert new_eqn.params['jaxpr'].jaxpr.eqns == body_jaxpr.eqns

        # evaluate the rewritten jaxpr: replace the eqn, extend outvars, compare
        new_eqns = [new_eqn if e is eqn else e for e in closed.jaxpr.eqns]
        new_jaxpr = Jaxpr(
            constvars=closed.jaxpr.constvars, invars=closed.jaxpr.invars,
            outvars=list(closed.jaxpr.outvars) + [stacked[tanh_var],
                                                  stacked[carry_invar]],
            eqns=new_eqns, effects=closed.jaxpr.effects,
            debug_info=closed.jaxpr.debug_info)
        xs = jnp.arange(4.0)
        outs = jax.core.eval_jaxpr(new_jaxpr, closed.consts, xs)
        # reference: replay by hand
        cs, tanhs = [], []
        c = jnp.zeros(())
        for x in xs:
            cs.append(c)
            t = jnp.tanh(c)
            c = t + x
            tanhs.append(t)
        assert jnp.allclose(outs[-2], jnp.stack(tanhs))
        assert jnp.allclose(outs[-1], jnp.stack(cs))
        # original outputs unchanged
        ref = jax.core.eval_jaxpr(closed.jaxpr, closed.consts, xs)
        assert jnp.allclose(outs[0], ref[0])
        assert jnp.allclose(outs[1], ref[1])

    def test_add_scan_ys_dedups_preserving_order(self):
        from braintrace._compiler.scan_descent import add_scan_ys

        def body(c, x):
            return c + x, c

        closed = jax.make_jaxpr(
            lambda xs: jax.lax.scan(body, jnp.zeros(()), xs))(jnp.arange(4.0))
        eqn = next(e for e in closed.jaxpr.eqns if is_scan_primitive(e))
        carry_invar = eqn.params['jaxpr'].jaxpr.invars[eqn.params['num_consts']]
        new_eqn, stacked = add_scan_ys(eqn, [carry_invar, carry_invar])
        assert len(stacked) == 1
        assert len(new_eqn.outvars) == len(eqn.outvars) + 1


class TestDescentContextTypes:
    def test_descent_context_types_and_default_fields(self):
        from braintrace._compiler.scan_descent import (
            ScanDescentInfo, GroupDescent, RelationDescent)
        from braintrace._compiler.hid_param_op import HiddenParamOpRelation
        from braintrace._compiler.hidden_group import HiddenGroup
        assert HiddenParamOpRelation._field_defaults.get(
            'control_flow_context') is None
        assert 'control_flow_context' in HiddenParamOpRelation._field_defaults
        assert HiddenGroup._field_defaults.get('descent') is None
        assert 'descent' in HiddenGroup._field_defaults
        assert ScanDescentInfo._fields == (
            'length', 'num_consts', 'num_carry', 'body_jaxpr',
            'stacked_var_map', 'scan_eqn_id')
        assert GroupDescent._fields == ('scan', 'body_hidden_invars')
        assert RelationDescent._fields == ('scan',)


class TestAnalyzeAndRewriteScan:
    def _analyze(self, loops):
        eqn, minfo = _scan_model_jaxpr(loops)
        from braintrace._compiler.scan_descent import analyze_and_rewrite_scan
        return analyze_and_rewrite_scan(eqn, minfo), minfo

    def test_snn_body_yields_one_group_one_relation(self):
        bundle, minfo = self._analyze(loops=8)
        assert bundle is not None
        assert len(bundle.groups) == 1
        g = bundle.groups[0]
        assert g.descent is not None
        assert g.num_state == 1
        # outer-facing hidden vars are the scan carry vars, known to minfo
        assert g.hidden_outvars[0] in minfo.outvar_to_hidden_path
        assert g.hidden_invars[0] in minfo.invar_to_hidden_path
        # body-scoped transition: one substep, invars == descent.body_hidden_invars
        assert list(g.transition_jaxpr.invars) == list(g.descent.body_hidden_invars)

        assert len(bundle.relations) == 1
        r = bundle.relations[0]
        assert r.control_flow_context is not None
        assert r.trainable_paths['weight'] == ('w',)
        assert r.hidden_groups[0] is g
        # everything the executor needs is in the stacked map
        m = bundle.info.stacked_var_map
        assert r.x_var in m and r.y_var in m
        for j in r.y_to_hidden_group_jaxprs:
            assert all(v in m for v in j.constvars)
        assert all(v in m for v in g.descent.body_hidden_invars)
        assert all(v in m for v in g.transition_jaxpr_constvars)
        # stacked avals carry the substep axis
        L = bundle.info.length
        assert L == 8
        assert all(m[v].aval.shape[0] == L for v in m)
        # the rewritten eqn's outvars extend the original with the stacked vars
        assert list(bundle.new_eqn.outvars[-len(bundle.stacked_outer_vars):]) \
            == list(bundle.stacked_outer_vars)
        assert bundle.info.scan_eqn_id == id(bundle.new_eqn)

    def test_mixing_body_registers_both_relations(self):
        """Body ``h = tanh(x@w + h@w)`` (scan_body_rnn shape): within ONE
        substep neither matmul's output crosses another ETP op before the
        carry hidden (the tail add/tanh is non-parametric), so BOTH register
        as body-scoped relations."""

        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((3, 3)) * 0.1)
                self.h = brainstate.HiddenState(jnp.zeros((1, 3)))

            def update(self, x):
                x_row = x.reshape(1, -1)

                def substep(_):
                    self.h.value = jax.nn.tanh(
                        braintrace.matmul(x_row, self.w.value)
                        + braintrace.matmul(self.h.value, self.w.value))
                    return self.h.value

                return brainstate.transform.for_loop(
                    substep, jnp.arange(8))[-1]

        net = Net()
        brainstate.nn.init_all_states(net, batch_size=1)
        from braintrace._compiler.scan_descent import analyze_and_rewrite_scan
        minfo = extract_module_info(
            net, jnp.ones((3,), dtype='float32'),
            control_flow=ControlFlowPolicy(scan_unroll_limit=4,
                                           scan_descent='off'))
        eqn = next(e for e in minfo.jaxpr.eqns if is_scan_primitive(e))
        bundle = analyze_and_rewrite_scan(eqn, minfo)
        assert bundle is not None
        assert len(bundle.groups) == 1
        assert len(bundle.relations) == 2
        assert all(r.control_flow_context is not None
                   for r in bundle.relations)
        # tied weight: both relations route the SAME param via distinct y_vars
        assert bundle.relations[0].y_var is not bundle.relations[1].y_var
