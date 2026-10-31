# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import time
import warnings

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace import ControlFlowPolicy
from braintrace._compatible_imports import (
    is_scan_primitive,
    jaxpr_constvars,
    scan_num_consts_carry,
    split_jaxpr_invars,
)
from braintrace._compiler.canonicalize import DEFAULT_CONTROL_FLOW_POLICY
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
        assert _is_etp_relevant(eqn)
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
        assert not _is_etp_relevant(eqn)


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
        num_consts, num_carry = scan_num_consts_carry(eqn)
        carry_invar = body_jaxpr.invars[num_consts]

        new_eqn, stacked = add_scan_ys(eqn, [tanh_var, carry_invar])
        assert list(new_eqn.outvars[:len(eqn.outvars)]) == list(eqn.outvars)
        assert stacked[tanh_var].aval.shape == (4,)
        assert stacked[carry_invar].aval.shape == (4,)
        # the input structure (consts/carry) is preserved; only ys grow
        assert scan_num_consts_carry(new_eqn) == (num_consts, num_carry)
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
        num_consts, _ = scan_num_consts_carry(eqn)
        carry_invar = eqn.params['jaxpr'].jaxpr.invars[num_consts]
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
        assert RelationDescent._fields == ('scan', 'lifted')
        assert RelationDescent._field_defaults == {'lifted': False}


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
        # body-scoped transition: one substep, invars == descent.body_hidden_invars.
        # Split by invar count rather than reading ``.invars`` -- since JAX 0.11
        # that also lists the constvars for a value-less open jaxpr.
        _, transition_invars = split_jaxpr_invars(
            g.transition_jaxpr, len(g.descent.body_hidden_invars)
        )
        assert list(transition_invars) == list(g.descent.body_hidden_invars)

        assert len(bundle.relations) == 1
        r = bundle.relations[0]
        assert r.control_flow_context is not None
        assert r.trainable_paths['weight'] == ('w',)
        assert r.hidden_groups[0] is g
        # everything the executor needs is in the stacked map
        m = bundle.info.stacked_var_map
        assert r.x_var in m and r.y_var in m
        for j in r.y_to_hidden_group_jaxprs:
            assert all(v in m for v in jaxpr_constvars(j, 1))
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


DESCENT_POLICY = ControlFlowPolicy(scan_unroll_limit=4, scan_descent='auto')


def _make_snn_net(loops, n_rec=4):
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            with brainstate.random.seed_context(0):
                self.w = brainstate.ParamState(
                    0.1 * brainstate.random.randn(n_rec, n_rec))
            self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

        def update(self, x):
            x_row = x.reshape(1, -1)

            def substep(_):
                self.h.value = 0.9 * self.h.value + jnp.tanh(
                    braintrace.matmul(x_row, self.w.value))
                return self.h.value

            return brainstate.transform.for_loop(substep, jnp.arange(loops))[-1]

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


class TestApplyScanDescentPipeline:
    def test_long_scan_compiles_with_descent(self):
        from braintrace._compiler.graph import compile_etrace_graph
        from braintrace._compiler.diagnostics import DiagnosticKind
        net = _make_snn_net(loops=40)
        graph = compile_etrace_graph(
            net, jnp.ones((4,), dtype='float32'), control_flow=DESCENT_POLICY)
        rels = [r for r in graph.hidden_param_op_relations
                if r.control_flow_context is not None]
        grps = [g for g in graph.hidden_groups if g.descent is not None]
        assert len(rels) == 1 and len(grps) == 1
        assert [g.index for g in graph.hidden_groups] == list(
            range(len(graph.hidden_groups)))
        assert any(d.kind is DiagnosticKind.SCAN_DESCENT_APPLIED
                   for d in graph.diagnostics)
        # stacked temps are hoisted: every mapped outer var is a jaxpr outvar
        m = rels[0].control_flow_context.scan.stacked_var_map
        outvars = set(graph.module_info.jaxpr.outvars)
        assert all(sv in outvars for sv in m.values())

    def test_mixing_body_compiles_with_two_descended_relations(self):
        from braintrace._compiler.graph import compile_etrace_graph

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
                    substep, jnp.arange(40))[-1]

        net = Net()
        brainstate.nn.init_all_states(net, batch_size=1)
        graph = compile_etrace_graph(
            net, jnp.ones((3,), dtype='float32'), control_flow=DESCENT_POLICY)
        rels = [r for r in graph.hidden_param_op_relations
                if r.control_flow_context is not None]
        assert len(rels) == 2

    def test_policy_off_preserves_old_error(self):
        net = _make_snn_net(loops=40)
        algo = braintrace.D_RTRL(net, control_flow=ControlFlowPolicy(
            scan_unroll_limit=4, scan_descent='off'))
        with pytest.raises(NotImplementedError):
            algo.compile_graph(jnp.ones((4,), dtype='float32'))

    def test_algorithm_gate_admits_supporting_algorithm(self):
        """Part 1 pinned this call as gate-blocked; since the Part 2 substep
        fold landed, ``D_RTRL`` declares ``_supports_scan_descent`` and the
        gate admits the descended graph. The io-dim family remains blocked —
        pinned in ``scan_descent_support_test.test_io_dim_algorithm_still_gated``."""
        net = _make_snn_net(loops=40)
        algo = braintrace.D_RTRL(net, control_flow=DESCENT_POLICY)
        algo.compile_graph(jnp.ones((4,), dtype='float32'))
        assert any(r.control_flow_context is not None
                   for r in algo.graph.hidden_param_op_relations)

    def test_pre_scan_hidden_transform_blocks_descent(self):
        """A hidden state transformed between step entry and the scan
        (``h *= 0.5`` before the loop) puts part of the per-step transition
        outside the scan body; descending anyway would fold a silently
        wrong trace (review finding: chunked gradients off by O(1) vs
        BPTT). Descent must skip loudly and the pre-descent hard error
        must fire."""
        from braintrace._compiler.graph import compile_etrace_graph

        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                x_row = x.reshape(1, -1)
                self.h.value = self.h.value * 0.5

                def substep(_):
                    self.h.value = 0.9 * self.h.value + jnp.tanh(
                        braintrace.matmul(x_row, self.w.value))
                    return self.h.value

                return brainstate.transform.for_loop(
                    substep, jnp.arange(40))[-1]

        net = Net()
        brainstate.nn.init_all_states(net, batch_size=1)
        with pytest.warns(UserWarning, match='transformed value'):
            with pytest.raises(NotImplementedError, match='within a scan'):
                compile_etrace_graph(
                    net, jnp.ones((4,), dtype='float32'),
                    control_flow=DESCENT_POLICY)

    def test_plain_op_weight_in_body_warns_and_excludes(self):
        """A weight consumed by a descended scan only through plain (non-
        ETP) ops is excluded from online learning — the primitive-based
        selection principle — but the exclusion must be loud, because
        pre-descent this scan was a hard error."""
        from braintrace._compiler.graph import compile_etrace_graph
        from braintrace._compiler.diagnostics import DiagnosticKind

        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
                self.p = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                x_row = x.reshape(1, -1)

                def substep(_):
                    # ``w`` through an ETP primitive (so the scan descends),
                    # ``p`` through a plain matmul: deliberately NOT ETP
                    self.h.value = 0.9 * self.h.value + jnp.tanh(
                        braintrace.matmul(x_row, self.w.value) + x_row @ self.p.value)
                    return self.h.value

                return brainstate.transform.for_loop(
                    substep, jnp.arange(40))[-1]

        net = Net()
        brainstate.nn.init_all_states(net, batch_size=1)
        with pytest.warns(UserWarning, match='only through plain ops'):
            graph = compile_etrace_graph(
                net, jnp.ones((4,), dtype='float32'),
                control_flow=DESCENT_POLICY)
        rec, = [d for d in graph.diagnostics
                if d.kind is DiagnosticKind.SCAN_DESCENT_NO_RELATIONS]
        assert rec.context['weight_paths'] == (('p',),)
        assert len(graph.hidden_param_op_relations) == 1
        assert sum(1 for g in graph.hidden_groups
                   if g.descent is not None) == 1

    def test_blocked_etp_scan_beyond_the_limit_warns(self):
        """An ETP-relevant scan beyond the unroll limit that descent cannot
        take (here ``reverse=True``) is reported, and the weight inside the
        opaque scan then raises."""
        from braintrace._compiler.graph import compile_etrace_graph

        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                x_row = x.reshape(1, -1)

                def body(h, _):
                    h = 0.9 * h + jnp.tanh(braintrace.matmul(x_row, self.w.value))
                    return h, h

                self.h.value, _ = jax.lax.scan(
                    body, self.h.value, None, length=8, reverse=True)
                return self.h.value

        net = Net()
        brainstate.nn.init_all_states(net, batch_size=1)
        with pytest.warns(UserWarning, match='cannot be descended: .*reverse'):
            with pytest.raises(NotImplementedError, match='weight'):
                compile_etrace_graph(
                    net, jnp.ones((4,), dtype='float32'),
                    control_flow=DESCENT_POLICY)

    def test_plain_op_only_scan_is_not_descended(self):
        """A scan whose body holds no ETP primitive holds no relation, so it
        is not descended (issue #209). A weight it reads through plain ops is
        a plain use, as at top level; the scan stays one opaque node, and the
        change from the old hard error is reported (SCAN_PLAIN_WEIGHT)."""
        from braintrace._compiler.graph import compile_etrace_graph
        from braintrace._compiler.diagnostics import DiagnosticKind

        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                x_row = x.reshape(1, -1)

                def substep(_):
                    # plain matmul: deliberately NOT an ETP primitive
                    self.h.value = 0.9 * self.h.value + jnp.tanh(
                        x_row @ self.w.value)
                    return self.h.value

                return brainstate.transform.for_loop(
                    substep, jnp.arange(40))[-1]

        net = Net()
        brainstate.nn.init_all_states(net, batch_size=1)
        with pytest.warns(UserWarning, match='plain ops'):
            graph = compile_etrace_graph(
                net, jnp.ones((4,), dtype='float32'),
                control_flow=DESCENT_POLICY)
        rec, = [d for d in graph.diagnostics
                if d.kind is DiagnosticKind.SCAN_PLAIN_WEIGHT]
        assert rec.context['weight_paths'] == (('w',),)
        assert rec.context['hidden_paths'] == (('h',),)
        assert not any(d.kind is DiagnosticKind.SCAN_DESCENT_NO_RELATIONS
                       for d in graph.diagnostics)
        assert len(graph.hidden_param_op_relations) == 0
        assert all(g.descent is None for g in graph.hidden_groups)


# ---------------------------------------------------------------------------
# Issue #211: an ETP relation outside a descended scan, whose output the scan
# reads as a loop constant, is lifted into the scan's per-substep fold
# ---------------------------------------------------------------------------

UNROLL_POLICY = ControlFlowPolicy(scan_unroll_limit=64)


class _TwoState(brainstate.nn.Module):
    """Two hidden states integrated in ``k`` Euler substeps of one ``lax.scan``,
    with a trainable time constant ``tau`` (the issue #211 model family)."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__()
        self.k = k
        with brainstate.random.seed_context(0):
            self.W = brainstate.ParamState(0.5 * brainstate.random.randn(n_in, n))
        self.tau = brainstate.ParamState(jnp.full(n, 2.0))
        self.V = brainstate.HiddenState(jnp.zeros(n))
        self.w = brainstate.HiddenState(jnp.zeros(n))

    def drive(self, x):
        return braintrace.matmul(x, self.W.value)

    def scan(self, body):
        (self.V.value, self.w.value), _ = jax.lax.scan(
            body, (self.V.value, self.w.value), None, length=self.k)
        return jax.nn.sigmoid(self.V.value)

    def substep(self, c, I):
        V, w = c
        h = 1.0 / self.k
        tau = braintrace.element_wise(self.tau.value)  # an ETP primitive in the body
        return (V + h / tau * (-V + jnp.tanh(2 * V) - w + I), w + h * (0.5 * V - w)), None


class _TauInBody(_TwoState):
    """The issue model: the input projection outside the scan, ``tau`` inside."""

    def update(self, x):
        I = self.drive(x)
        return self.scan(lambda c, _: self.substep(c, I))


class _DriveInBody(_TwoState):
    """The same dynamics with the input projection inside the body."""

    def update(self, x):
        return self.scan(lambda c, _: self.substep(c, self.drive(x)))


class _SplitDrive(_TwoState):
    """The projection also drives a hidden state outside the scan."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__(k, n_in=n_in, n=n)
        self.u = brainstate.HiddenState(jnp.zeros(n))

    def update(self, x):
        I = self.drive(x)
        self.u.value = 0.8 * self.u.value + jnp.tanh(I)
        return self.scan(lambda c, _: self.substep(c, I)) + self.u.value


def _shape(d):
    return 2.0 * jnp.tanh(d) + 0.1 * d  # two paths from ``d``


class _ShapedDrive(_TwoState):
    """The scan reads a nonlinear function of the projection alone."""

    def update(self, x):
        I = _shape(self.drive(x))
        return self.scan(lambda c, _: self.substep(c, I))


class _ShapedDriveInBody(_TwoState):
    """:class:`_ShapedDrive` with the projection inside the body."""

    def update(self, x):
        return self.scan(lambda c, _: self.substep(c, _shape(self.drive(x))))


class _OuterGain(_TwoState):
    """Two outer relations: the projection and a gain the scan reads as the
    ``element_wise`` output itself (no equation in between)."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__(k, n_in=n_in, n=n)
        self.gain = brainstate.ParamState(jnp.linspace(0.5, 1.5, n))

    def update(self, x):
        I, g = self.drive(x), braintrace.element_wise(self.gain.value)
        return self.scan(lambda c, _: self.substep(c, g * I))


class _OuterGainInBody(_OuterGain):
    """:class:`_OuterGain` with both ETP operations inside the body."""

    def update(self, x):
        return self.scan(lambda c, _: self.substep(
            c, braintrace.element_wise(self.gain.value) * self.drive(x)))


class _DriveOnlyState(_TwoState):
    """A third carried state set from the constant alone: its transition
    from the constant reads no body value."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__(k, n_in=n_in, n=n)
        self.u = brainstate.HiddenState(jnp.zeros(n))

    def update(self, x):
        I = self.drive(x)

        def body(c, _):
            (V, w), _u = c
            return (self.substep((V, w), I)[0], jnp.tanh(I)), None

        ((self.V.value, self.w.value), self.u.value), _ = jax.lax.scan(
            body, ((self.V.value, self.w.value), self.u.value), None, length=self.k)
        return jax.nn.sigmoid(self.V.value) + self.u.value


class _DriveOnlyTheConstantState(_DriveOnlyState):
    """The projection reaches only the state set from the constant alone:
    the lifted relation's map reads no per-substep value at all."""

    def update(self, x):
        I = self.drive(x)

        def body(c, _):
            (V, w), _u = c
            return (self.substep((V, w), 0.5)[0], jnp.tanh(I)), None

        ((self.V.value, self.w.value), self.u.value), _ = jax.lax.scan(
            body, ((self.V.value, self.w.value), self.u.value), None, length=self.k)
        return jax.nn.sigmoid(self.V.value) + self.u.value


class _WideState(_TwoState):
    """A carried state whose shape does not match the projection's output:
    the relation leaves it out, as at top level."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__(k, n_in=n_in, n=n)
        self.z = brainstate.HiddenState(jnp.zeros(5))

    def update(self, x):
        I = self.drive(x)

        def body(c, _):
            (V, w), z = c
            return (self.substep((V, w), I)[0], 0.9 * z + jnp.mean(I)), None

        ((self.V.value, self.w.value), self.z.value), _ = jax.lax.scan(
            body, ((self.V.value, self.w.value), self.z.value), None, length=self.k)
        return jax.nn.sigmoid(self.V.value) + jnp.mean(self.z.value)


class _TwoScans(_TwoState):
    """Two descended scans; the projection drives only the first."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__(k, n_in=n_in, n=n)
        self.beta = brainstate.ParamState(jnp.full(n, 0.7))
        self.u = brainstate.HiddenState(jnp.zeros(n))

    def update(self, x):
        I = self.drive(x)
        out = self.scan(lambda c, _: self.substep(c, I))

        def leak(u, _):
            return braintrace.element_wise(self.beta.value) * u + 0.1, None

        self.u.value, _ = jax.lax.scan(leak, self.u.value, None, length=self.k)
        return out + self.u.value


class _NonParamProjection(_TwoState):
    """The scan reads an ETP output whose weight is not a ``ParamState``."""

    def update(self, x):
        I = braintrace.matmul(x, jnp.ones((3, 4)) * 0.1)
        return self.scan(lambda c, _: self.substep(c, I))


class _MixedDrive(_TwoState):
    """The scan reads the projection combined with another outer value."""

    def update(self, x):
        I = self.drive(x) + jnp.sum(x)
        return self.scan(lambda c, _: self.substep(c, I))


class _ScannedDrive(_TwoState):
    """The scan reads the projection as a scanned input."""

    def update(self, x):
        Is = jnp.broadcast_to(self.drive(x), (self.k, 4))
        (self.V.value, self.w.value), _ = jax.lax.scan(
            self.substep, (self.V.value, self.w.value), Is)
        return jax.nn.sigmoid(self.V.value)


class _TwoProjectionDrive(_TwoState):
    """The scan reads the sum of two projections."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__(k, n_in=n_in, n=n)
        self.W2 = brainstate.ParamState(jnp.ones((n_in, n)) * 0.2)

    def update(self, x):
        I = self.drive(x) + braintrace.matmul(x, self.W2.value)
        return self.scan(lambda c, _: self.substep(c, I))


class _ConstAndCarriedDrive(_TwoState):
    """The scan reads the projection as a constant and as a carry init."""

    def update(self, x):
        I = self.drive(x)

        def body(c, _):
            (V, w), z = c
            return (self.substep((V, w), I)[0], 0.5 * z), None

        ((self.V.value, self.w.value), z), _ = jax.lax.scan(
            body, ((self.V.value, self.w.value), I), None, length=self.k)
        return jax.nn.sigmoid(self.V.value) + z


class _TwiceReadDrive(_TwoState):
    """The scan reads the projection in two operands."""

    def update(self, x):
        I = self.drive(x)
        J = 0.5 * I
        return self.scan(lambda c, _: self.substep(c, I + J * c[0]))


class _DriveThroughBodyWeight(brainstate.nn.Module):
    """The outer projection reaches the carry only through a trainable ETP
    primitive in the body (the non-parametric-tail invariant excludes it)."""

    def __init__(self, k=40):
        super().__init__()
        self.k = k
        self.w_in = brainstate.ParamState(jnp.ones((3, 4)) * 0.1)
        self.w = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
        self.h = brainstate.HiddenState(jnp.zeros(4))

    def update(self, x):
        drive = braintrace.matmul(x, self.w_in.value)

        def substep(h, _):
            return 0.9 * h + jnp.tanh(braintrace.matmul(drive, self.w.value)), None

        self.h.value, _ = jax.lax.scan(substep, self.h.value, None, length=self.k)
        return self.h.value


class _MixedPaths(_TwoState):
    """The constant reaches ``V`` directly and through a trainable ETP
    primitive in the body (a partial path)."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__(k, n_in=n_in, n=n)
        self.W2 = brainstate.ParamState(0.3 * jnp.eye(n) + 0.05)

    def update(self, x):
        I = self.drive(x)
        return self.scan(lambda c, _: self.substep(c, I + jnp.tanh(braintrace.matmul(I, self.W2.value))))


class _MixedPathsInBody(_MixedPaths):
    """:class:`_MixedPaths` with the projection inside the body."""

    def update(self, x):
        def body(c, _):
            I = self.drive(x)
            return self.substep(c, I + jnp.tanh(braintrace.matmul(I, self.W2.value)))

        return self.scan(body)


class _UnreachedWideState(_WideState):
    """A carried state of another shape that the constant does not reach."""

    def update(self, x):
        I = self.drive(x)

        def body(c, _):
            (V, w), z = c
            return (self.substep((V, w), I)[0], 0.9 * z + 0.1), None

        ((self.V.value, self.w.value), self.z.value), _ = jax.lax.scan(
            body, ((self.V.value, self.w.value), self.z.value), None, length=self.k)
        return jax.nn.sigmoid(self.V.value) + jnp.mean(self.z.value)


class _PlainUseInBody(_TwoState):
    """``W`` drives the scan through the outer projection and is also read
    by a plain op in the body."""

    def update(self, x):
        I = self.drive(x)
        return self.scan(lambda c, _: self.substep(c, I + 0.01 * jnp.sum(self.W.value)))


def _carry_scan(model, body, extra_init, xs=None):
    """Run ``body`` over ``((V, w), extra)`` and store the hidden carries."""
    ((model.V.value, model.w.value), extra), _ = jax.lax.scan(
        body, ((model.V.value, model.w.value), extra_init), xs,
        length=None if xs is not None else model.k)
    return extra


class _AccumulatedDrive(_TwoState):
    """A non-hidden carry accumulates the loop constant and drives ``V``."""

    def update(self, x):
        I = self.drive(x)

        def body(c, _):
            (V, w), z = c
            z = z + I
            return (self.substep((V, w), 0.3 * z)[0], z), None

        _carry_scan(self, body, jnp.zeros(4))
        return jax.nn.sigmoid(self.V.value)


class _AccumulatedDriveInBody(_TwoState):
    """:class:`_AccumulatedDrive` with the projection inside the body."""

    def update(self, x):
        def body(c, _):
            (V, w), z = c
            z = z + self.drive(x)
            return (self.substep((V, w), 0.3 * z)[0], z), None

        _carry_scan(self, body, jnp.zeros(4))
        return jax.nn.sigmoid(self.V.value)


class _FedBackState(_TwoState):
    """A non-hidden carry filters ``V`` and feeds it back."""

    def update(self, x):
        I = self.drive(x)

        def body(c, _):
            (V, w), z = c
            z = 0.9 * z + 0.5 * V
            return (self.substep((V, w), I + z)[0], z), None

        _carry_scan(self, body, jnp.zeros(4))
        return jax.nn.sigmoid(self.V.value)


class _FedBackStateInBody(_TwoState):
    """:class:`_FedBackState` with the projection inside the body."""

    def update(self, x):
        def body(c, _):
            (V, w), z = c
            z = 0.9 * z + 0.5 * V
            return (self.substep((V, w), self.drive(x) + z)[0], z), None

        _carry_scan(self, body, jnp.zeros(4))
        return jax.nn.sigmoid(self.V.value)


class _DelayedFeedback(_TwoState):
    """``V`` comes back two substeps later through two non-hidden carries:
    neither carry reaches a hidden carry and is reached in one iteration."""

    def update(self, x):
        def body(c, _):
            (V, w), (z1, z2) = c
            return (self.substep((V, w), self.drive(x) + z2)[0], (0.5 * V, z1)), None

        _carry_scan(self, body, (jnp.zeros(4), jnp.zeros(4)))
        return jax.nn.sigmoid(self.V.value)


class _ClippedState(_TwoState):
    """``V`` is clipped after the scan, so its carry is not the hidden state."""

    def update(self, x):
        I = self.drive(x)
        (V, self.w.value), _ = jax.lax.scan(
            lambda c, _: self.substep(c, I), (self.V.value, self.w.value), None, length=self.k)
        self.V.value = jnp.clip(V, -10.0, 10.0)
        return jax.nn.sigmoid(self.V.value) + self.w.value


class _ClippedStateInBody(_TwoState):
    """:class:`_ClippedState` with the projection inside the body."""

    def update(self, x):
        (V, self.w.value), _ = jax.lax.scan(
            lambda c, _: self.substep(c, self.drive(x)), (self.V.value, self.w.value), None,
            length=self.k)
        self.V.value = jnp.clip(V, -10.0, 10.0)
        return jax.nn.sigmoid(self.V.value) + self.w.value


class _StepStartConst(_TwoState):
    """The body reads a value computed from ``w`` before the scan."""

    def update(self, x):
        g = jnp.tanh(self.w.value)
        return self.scan(lambda c, _: self.substep(c, self.drive(x) + g))


class _StepStartXs(_TwoState):
    """:class:`_StepStartConst` with the value scanned as ``xs``."""

    def update(self, x):
        gs = jnp.broadcast_to(jnp.tanh(self.w.value), (self.k, 4))
        (self.V.value, self.w.value), _ = jax.lax.scan(
            lambda c, g: self.substep(c, self.drive(x) + g), (self.V.value, self.w.value), gs)
        return jax.nn.sigmoid(self.V.value)


class _ChainedScans(_TwoState):
    """A second scan reads the state the first scan carries out, in the
    same step."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__(k, n_in=n_in, n=n)
        self.beta = brainstate.ParamState(jnp.full(n, 0.7))
        self.u = brainstate.HiddenState(jnp.zeros(n))

    def update(self, x):
        out = self.scan(lambda c, _: self.substep(c, self.drive(x)))
        V = self.V.value

        def leak(u, _):
            return braintrace.element_wise(self.beta.value) * u + 0.5 * V, None

        self.u.value, _ = jax.lax.scan(leak, self.u.value, None, length=self.k)
        return out + self.u.value


class _SubstepClock(_TwoState):
    """A non-hidden carry that counts substeps scales the drive: it depends
    on no hidden state and no ETP output."""

    def update(self, x):
        I = self.drive(x)

        def body(c, _):
            (V, w), t = c
            return (self.substep((V, w), I * (1.0 + t))[0], t + 1.0 / self.k), None

        _carry_scan(self, body, jnp.zeros(()))
        return jax.nn.sigmoid(self.V.value)


class _SpikeCount(_TwoState):
    """An integer carry counts threshold crossings of ``V`` and gates it:
    no derivative flows through an integer."""

    def update(self, x):
        I = self.drive(x)

        def body(c, _):
            (V, w), n = c
            (V, w), _ = self.substep((V, w), I)
            n = n + (V > 0.05).astype(jnp.int32)
            return ((jnp.where(n > 3, 0.9 * V, V), w), n), None

        _carry_scan(self, body, jnp.zeros(4, dtype=jnp.int32))
        return jax.nn.sigmoid(self.V.value)


class _RefractoryTimer(_TwoState):
    """A float refractory timer, set by a threshold test on ``V``, holds ``V``
    (p20 of the #217 fix-check): no derivative flows through the boolean, so
    the timer carries no credit although it is computed from ``V``."""

    def update(self, x):
        I = self.drive(x)

        def body(c, _):
            (V, w), r = c
            r = jnp.where(V > 0.2, 0.3, jnp.maximum(r - 1.0 / self.k, 0.0))
            (V2, w2), _ = self.substep((V, w), I)
            return ((jnp.where(r > 0.25, V, V2), w2), r), None

        _carry_scan(self, body, jnp.zeros(4))
        return jax.nn.sigmoid(self.V.value)


class _FloatSpikeCount(_TwoState):
    """A float count of threshold crossings of ``V``, cast from a boolean,
    adapts the drive (p20 of the #217 fix-check)."""

    def update(self, x):
        I = self.drive(x)

        def body(c, _):
            (V, w), n = c
            n = n + (V > 0.2).astype(V.dtype)
            return (self.substep((V, w), I - 0.05 * n)[0], n), None

        _carry_scan(self, body, jnp.zeros(4))
        return jax.nn.sigmoid(self.V.value)


def _reverse_delay_line(N, M):
    """p22 of the #217 fix-check: ``N`` non-hidden carries in a delay line
    from ``V`` back into ``V``, in the order that a fixpoint over the carries
    reaches one new carry per pass, every tap also feeding a shared chain of
    ``M`` operations."""

    class _ReverseDelayLine(_TwoState):
        def update(self, x):
            I = self.drive(x)

            def body(c, _):
                (V, w), zs = c
                tot = zs[0]
                for z in zs:  # trace-time loops that build a long body
                    tot = tot + 0.001 * z
                for _ in range(M):
                    tot = jnp.tanh(tot) * 0.99
                return (self.substep((V, w), I + 0.01 * tot)[0], zs[1:] + (V,)), None

            _carry_scan(self, body, tuple(jnp.zeros(4) for _ in range(N)))
            return jax.nn.sigmoid(self.V.value)

    return _ReverseDelayLine


class _RecurrentOutside(_TwoState):
    """A recurrent ETP projection of ``V`` outside the scan: the constant
    depends on the hidden state only through the ETP primitive."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__(k, n_in=n_in, n=n)
        with brainstate.random.seed_context(5):
            self.Wrec = brainstate.ParamState(0.6 * brainstate.random.randn(n, n))

    def update(self, x):
        I = self.drive(x)
        R = braintrace.matmul(self.V.value, self.Wrec.value)
        return self.scan(lambda c, _: self.substep(c, I + jnp.tanh(R)))


class _DetachedStepStartConst(_TwoState):
    """:class:`_StepStartConst` behind a ``stop_gradient``."""

    def update(self, x):
        g = jax.lax.stop_gradient(jnp.tanh(self.w.value))
        return self.scan(lambda c, _: self.substep(c, self.drive(x) + g))


_LIFT_ALGOS = {
    'd_rtrl': lambda m, p: braintrace.D_RTRL(m, control_flow=p),
    'eprop': lambda m, p: braintrace.EProp(m, control_flow=p),
}


def _lift_inputs(T=6, n_in=3):
    with brainstate.random.seed_context(1):
        return brainstate.random.rand(T, n_in)


def _online(make, algo, policy=DEFAULT_CONTROL_FLOW_POLICY):
    from braintrace._testing.oracle import online_param_gradients_singlestep_naive
    g = online_param_gradients_singlestep_naive(
        make, _lift_inputs(), algo_factory=lambda m: _LIFT_ALGOS[algo](m, policy))
    return {p[0]: np.asarray(v) for p, v in g.items()}


def _compiled_graph(model, policy=DEFAULT_CONTROL_FLOW_POLICY):
    from braintrace._compiler.graph import compile_etrace_graph
    brainstate.nn.init_all_states(model)
    return compile_etrace_graph(model, jnp.zeros(3), control_flow=policy)


def _lifted(graph):
    return [r for r in graph.hidden_param_op_relations
            if r.control_flow_context is not None and r.control_flow_context.lifted]


def _close(a, b, tol=1e-12):
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol * max(1.0, float(np.abs(b).max())))


class TestOuterRelationLift:

    @pytest.mark.parametrize('algo', list(_LIFT_ALGOS))
    @pytest.mark.parametrize('outer, inner', [(_TauInBody, _DriveInBody),
                                              (_ShapedDrive, _ShapedDriveInBody),
                                              (_OuterGain, _OuterGainInBody)],
                             ids=['output', 'function-of-output', 'two-outer-relations'])
    def test_gradients_equal_the_unrolled_path_and_the_projection_in_the_body(self, algo, outer, inner):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                lifted = _online(lambda: outer(17), algo)
                unrolled = _online(lambda: outer(17), algo, UNROLL_POLICY)
                in_body = _online(lambda: inner(17), algo)
        for p in ('W', 'tau'):
            assert np.abs(unrolled[p]).max() > 1e-3
            _close(lifted[p], unrolled[p])
            _close(lifted[p], in_body[p])

    @pytest.mark.parametrize('make', [_DriveOnlyState, _DriveOnlyTheConstantState, _WideState,
                                      _TwoScans],
                             ids=['state-from-constant-alone', 'only-that-state',
                                  'unrelatable-shape', 'two-scans'])
    def test_other_body_shapes_equal_the_unrolled_path(self, make):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                lifted = _online(lambda: make(17), 'd_rtrl')
                unrolled = _online(lambda: make(17), 'd_rtrl', UNROLL_POLICY)
            graph = _compiled_graph(make(17))
        assert [r.path for r in _lifted(graph)] == [('W',)]
        assert set(lifted) == set(unrolled)
        for p in lifted:
            _close(lifted[p], unrolled[p])
        assert np.abs(unrolled['W']).max() > 1e-3

    def test_an_etp_output_without_a_trainable_weight_is_not_lifted(self):
        from braintrace._compiler.scan_descent import apply_scan_descent
        model = _NonParamProjection(17)
        brainstate.nn.init_all_states(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, (bundle,) = apply_scan_descent(extract_module_info(model, jnp.zeros(3)))
            graph = _compiled_graph(_NonParamProjection(17))
        assert bundle.lift_templates == {}
        assert _lifted(graph) == []

    def test_chunked_d_rtrl_matches_bptt(self):
        # chunks of 2 steps: the total gradient goes through the lifted
        # eligibility trace at every chunk boundary (one whole-sequence
        # multi-step call would not use the trace at all)
        from braintrace._testing.oracle import bptt_param_gradients, chunked_online_param_gradients
        make = lambda: _TauInBody(17)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                bptt = bptt_param_gradients(make, _lift_inputs())
                chunked = chunked_online_param_gradients(
                    make, _lift_inputs(), chunk_size=2,
                    algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
        for p in bptt:
            assert np.abs(np.asarray(bptt[p])).max() > 1e-3
            _close(np.asarray(chunked[p]), np.asarray(bptt[p]))

    def test_a_relation_that_also_drives_a_flat_state_is_split(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                lifted = _online(lambda: _SplitDrive(17), 'd_rtrl')
                unrolled = _online(lambda: _SplitDrive(17), 'd_rtrl', UNROLL_POLICY)
            graph = _compiled_graph(_SplitDrive(17))
        for p in ('W', 'tau'):
            _close(lifted[p], unrolled[p])
        # one flat relation (to u) and one lifted relation (to V, w) for W
        w_rels = [r for r in graph.hidden_param_op_relations
                  if ('W',) in r.trainable_paths.values()]
        assert sorted(r.control_flow_context is None for r in w_rels) == [False, True]
        (lift,) = _lifted(graph)
        assert {p for g in lift.hidden_groups for p in g.hidden_paths} == {('V',), ('w',)}

    def test_the_lift_is_recorded_and_made_only_for_the_pattern(self):
        from braintrace._compiler.diagnostics import DiagnosticKind
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            graph = _compiled_graph(_TauInBody(17))
            in_body = _compiled_graph(_DriveInBody(17))
        (lift,) = _lifted(graph)
        assert lift.trainable_paths == {'weight': ('W',)}
        # x and y stay outer, read once per step; the scan gets no new input
        stacked = lift.control_flow_context.scan.stacked_var_map
        assert lift.x_var not in stacked and lift.y_var not in stacked
        model = _TauInBody(17)
        brainstate.nn.init_all_states(model)
        minfo = extract_module_info(model, jnp.zeros(3))
        (scan,) = [e for e in minfo.jaxpr.eqns if e.primitive.name == 'scan']
        (lifted_scan,) = [e for e in graph.module_info.jaxpr.eqns if e.primitive.name == 'scan']
        assert len(lifted_scan.invars) == len(scan.invars)
        (rec,) = [d for d in graph.diagnostics if d.kind is DiagnosticKind.SCAN_DESCENT_RELATION_LIFTED]
        assert rec.weight_path == ('W',)
        # no lift where the scan reads no outer ETP output
        assert _lifted(in_body) == []
        assert not any(d.kind is DiagnosticKind.SCAN_DESCENT_RELATION_LIFTED for d in in_body.diagnostics)

    def test_no_new_ys_without_the_pattern(self):
        # the scan of a model without an outer relation keeps the ys the
        # descent of its body relations and groups needs, and nothing more
        from braintrace._compiler.scan_descent import apply_scan_descent
        model = _DriveInBody(17)
        brainstate.nn.init_all_states(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            minfo = extract_module_info(model, jnp.zeros(3))
            _, (bundle,) = apply_scan_descent(minfo)
        assert bundle.lift_templates == {}

    @pytest.mark.parametrize('model, route', [
        (_MixedDrive, 'the loop constant at position 1 (computed from it, combined with another outer value)'),
        (_TwoProjectionDrive, 'the loop constant at position 1 (computed from it, combined with another outer value)'),
        (_TwiceReadDrive, 'the loop constant at position 0 (computed from it) and the loop constant at position 1 (computed from it)'),
        (_ScannedDrive, 'the scanned input at position 0 (computed from it)'),
        (_ConstAndCarriedDrive, 'the loop constant at position 1 (computed from it) and the carry init at position 2 (computed from it)'),
    ], ids=['mixed', 'two-projections', 'twice', 'scanned', 'const-and-carried'])
    def test_other_routes_into_the_scan_still_raise_without_the_off_hint(self, model, route):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(NotImplementedError, match='loop constant') as err:
                _compiled_graph(model(17))
        assert "scan_descent='off'" not in str(err.value)
        assert 'inside the scan body' in str(err.value)
        assert "('V',)" in str(err.value)
        assert f'it reads {route}.' in str(err.value)

    def test_a_lifted_relation_reports_its_partial_path(self):
        from braintrace._compiler.diagnostics import DiagnosticKind

        def partial(model, policy=DEFAULT_CONTROL_FLOW_POLICY):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                graph = _compiled_graph(model, policy)
            return sorted((d.weight_path, d.hidden_paths) for d in graph.diagnostics
                          if d.kind is DiagnosticKind.RELATION_PARTIAL_PATH)

        # as with the projection in the body: per substep the constant
        # reaches V both ways (unrolled, the whole step also reaches w so)
        expected = [(('W',), (('V',),))]
        assert partial(_MixedPaths(17)) == expected
        assert partial(_MixedPathsInBody(17)) == expected
        assert set(expected) <= set(partial(_MixedPaths(17), UNROLL_POLICY))

    @pytest.mark.parametrize('model, reported', [(_UnreachedWideState, False), (_WideState, True)],
                             ids=['unreached', 'reached'])
    def test_a_carry_of_another_shape_is_reported_only_when_the_constant_reaches_it(
            self, model, reported):
        from braintrace._compiler.diagnostics import DiagnosticKind
        for policy in (DEFAULT_CONTROL_FLOW_POLICY, UNROLL_POLICY):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                graph = _compiled_graph(model(17), policy)
            got = [d.hidden_paths for d in graph.diagnostics
                   if d.kind is DiagnosticKind.RELATION_EXCLUDED_SHAPE_MISMATCH]
            assert got == ([(('z',),)] if reported else []), policy

    def test_a_plain_use_in_the_body_does_not_deny_the_lifted_weight(self):
        from braintrace._compiler.diagnostics import DiagnosticKind
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            graph = _compiled_graph(_PlainUseInBody(17))
        assert [r.path for r in _lifted(graph)] == [('W',)]
        (rec,) = [d for d in graph.diagnostics if d.kind is DiagnosticKind.SCAN_DESCENT_NO_RELATIONS]
        assert 'only through plain ops' in rec.message and 'still learns' in rec.message
        assert 'NOT' not in rec.message

    def test_a_projection_through_a_body_weight_is_excluded_as_when_unrolled(self):
        with pytest.warns(UserWarning, match='only through another trainable ETP primitive'):
            graph = _compiled_graph(_DriveThroughBodyWeight(), DESCENT_POLICY)
        assert {p for r in graph.hidden_param_op_relations
                for p in r.trainable_paths.values()} == {('w',)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                descended = _online(_DriveThroughBodyWeight, 'd_rtrl', DESCENT_POLICY)
                unrolled = _online(_DriveThroughBodyWeight, 'd_rtrl', UNROLL_POLICY)
        for p in ('w', 'w_in'):
            _close(descended[p], unrolled[p])
        np.testing.assert_array_equal(descended['w_in'], 0.0)


class TestUntrackedCredit:
    """Descent folds the trace over the hidden carries only. A scan input or
    carried value other than a hidden carry that is differentiably computed
    from the hidden state, or carries an ETP output across substeps, and feeds
    a hidden carry, would be missed silently; such a scan is refused (review of
    #217, HIGH-1). A value computed through a boolean or an integer carries no
    derivative and is accepted (fix-check of #217, Q2)."""

    @pytest.mark.parametrize('model, what', [
        (_AccumulatedDrive, 'carried value at carry position 2'),
        (_AccumulatedDriveInBody, 'carried value at carry position 2'),
        (_FedBackState, 'carried value at carry position 2'),
        (_FedBackStateInBody, 'carried value at carry position 2'),
        (_DelayedFeedback, 'carried value at carry position 3'),
        (_ClippedState, 'carried value at carry position 0'),
        (_ClippedStateInBody, 'carried value at carry position 0'),
        (_StepStartConst, 'scan constant'),
        (_StepStartXs, 'scanned input'),
        (_ChainedScans, 'scan constant'),
    ], ids=['accumulated-constant', 'accumulated-projection', 'fed-back', 'fed-back-in-body',
            'delayed-two-carries', 'clipped-after-the-scan', 'clipped-in-body',
            'step-start-constant', 'step-start-xs', 'chained-scans'])
    def test_a_scan_with_untracked_credit_is_refused(self, model, what):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(NotImplementedError, match='not a hidden state carried') as err:
                _compiled_graph(model(17))
        message = str(err.value)
        assert what in message
        assert 'scan_unroll_limit=17' in message
        assert "scan_descent='off'" not in message

    def test_the_unrolling_the_message_suggests_is_exact(self):
        # the unrolled path equals BPTT through the eligibility trace
        # (chunked multi-step D-RTRL)
        from braintrace._testing.oracle import bptt_param_gradients, chunked_online_param_gradients
        model = _AccumulatedDrive
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                bptt = bptt_param_gradients(lambda: model(17), _lift_inputs(T=4))
                chunked = chunked_online_param_gradients(
                    lambda: model(17), _lift_inputs(T=4), chunk_size=2,
                    algo_factory=lambda m: braintrace.D_RTRL(
                        m, control_flow=UNROLL_POLICY, vjp_method='multi-step'))
        for p in bptt:
            assert np.abs(np.asarray(bptt[p])).max() > 1e-3
            _close(np.asarray(chunked[p]), np.asarray(bptt[p]))

    def test_the_guard_is_linear_in_the_body(self, monkeypatch):
        # p22 of the #217 fix-check: re-closing from every newly reached carry
        # walks the shared chain once per carry (6 s for this body on b1f4db4)
        import braintrace._compiler.scan_descent as sd
        passes, guard = sd._passes_credit, sd._untracked_credit
        visits, size, spent = [0], [0], [0.0]

        def counted(e):
            visits[0] += 1
            return passes(e)

        def timed(eqn, body, *args):
            size[0] = sum(len(e.invars) + len(e.outvars) for e in body.eqns)
            monkeypatch.setattr(sd, '_passes_credit', counted)
            t = time.perf_counter()
            try:
                return guard(eqn, body, *args)
            finally:
                spent[0] += time.perf_counter() - t
                monkeypatch.setattr(sd, '_passes_credit', passes)

        monkeypatch.setattr(sd, '_untracked_credit', timed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(NotImplementedError, match='not a hidden state carried'):
                _compiled_graph(_reverse_delay_line(1000, 3000)(17))
        # one forward closure, one backward closure per hidden carry (V and w)
        assert 0 < visits[0] <= 3 * size[0]
        assert spent[0] < 1.0

    @pytest.mark.parametrize('model', [_SubstepClock, _SpikeCount, _RecurrentOutside,
                                       _DetachedStepStartConst, _TauInBody,
                                       _RefractoryTimer, _FloatSpikeCount],
                             ids=['substep-clock', 'integer-carry', 'recurrent-projection-outside',
                                  'detached-step-start-constant', 'plain',
                                  'float-refractory-timer', 'float-spike-count'])
    def test_inputs_without_untracked_credit_are_accepted(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                descended = _online(lambda: model(17), 'd_rtrl')
                unrolled = _online(lambda: model(17), 'd_rtrl', UNROLL_POLICY)
        assert set(descended) == set(unrolled)
        for p in descended:
            _close(descended[p], unrolled[p])
        assert np.abs(unrolled['W']).max() > 1e-3


class _Batched(_TwoState):
    """The issue #211 family with a batch of two: hidden states of shape ``(2, n)``."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__(k, n_in=n_in, n=n)
        self.V = brainstate.HiddenState(jnp.zeros((2, n)))
        self.w = brainstate.HiddenState(jnp.zeros((2, n)))


class _SharedDrive(_Batched):
    """p05 of the #217 review: one input row shared by the batch, projected
    before the scan, so the scan reads a ``(n,)`` constant into ``(2, n)`` states."""

    def update(self, x):
        I = self.drive(x[0])
        return self.scan(lambda c, _: self.substep(c, I))


class _SharedDriveInBody(_Batched):
    def update(self, x):
        return self.scan(lambda c, _: self.substep(c, self.drive(x[0])))


class _SharedDriveBroadcastFirst(_Batched):
    """The shared projection is broadcast to the batch before the scan reads it."""

    def update(self, x):
        I = jnp.broadcast_to(self.drive(x[0]), self.V.value.shape)
        return self.scan(lambda c, _: self.substep(c, I))


class _SingletonDrive(_Batched):
    """A batch of one, ``(1, n)``, broadcast to the batch of two."""

    def update(self, x):
        I = self.drive(x[:1])
        return self.scan(lambda c, _: self.substep(c, I))


class _NarrowDriveInBody(_Batched):
    """A ``(2, 1)`` projection broadcast along the state's feature axis."""

    def __init__(self, k, n_in=3, n=4):
        super().__init__(k, n_in=n_in, n=n)
        with brainstate.random.seed_context(0):
            self.W = brainstate.ParamState(0.5 * brainstate.random.randn(n_in, 1))

    def update(self, x):
        return self.scan(lambda c, _: self.substep(c, self.drive(x)))


class _BatchedDrive(_Batched):
    def update(self, x):
        I = self.drive(x)
        return self.scan(lambda c, _: self.substep(c, I))


class _BatchedDriveInBody(_Batched):
    def update(self, x):
        return self.scan(lambda c, _: self.substep(c, self.drive(x)))


def _batched_inputs(T=6, n_in=3):
    with brainstate.random.seed_context(1):
        return brainstate.random.rand(T, 2, n_in)


def _online_batched(make, algo, policy=DEFAULT_CONTROL_FLOW_POLICY):
    from braintrace._testing.oracle import online_param_gradients_singlestep_naive
    g = online_param_gradients_singlestep_naive(
        make, _batched_inputs(), algo_factory=lambda m: _LIFT_ALGOS[algo](m, policy))
    return {p[0]: np.asarray(v) for p, v in g.items()}


class TestBroadcastIntoTheDescendedState:
    """An ETP output that reaches the hidden state it drives in a descended scan
    only by broadcasting (p05 of the #217 review: an unbatched output into a
    batched state). Its trace has the axes the broadcast adds; #224 allocates
    it at that shape, so the substep fold carries it. On main the lifted form
    was refused by the #211 message and the in-body form failed with a scan
    carry type error; #217 refused both. A broadcast along an axis of the
    weight (a ``(2, 1)`` output into ``(2, 4)`` states) is refused (#231)."""

    @pytest.mark.parametrize('algo', ['d_rtrl', 'eprop'])
    @pytest.mark.parametrize('model', [
        _SharedDrive, _SharedDriveInBody, _SharedDriveBroadcastFirst, _SingletonDrive,
    ], ids=['shared-row', 'shared-row-in-body', 'broadcast-before-the-scan', 'batch-of-one'])
    def test_an_output_broadcast_into_the_state_matches_the_unrolled_scan(self, model, algo):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                descended = _online_batched(lambda: model(17), algo)
                unrolled = _online_batched(lambda: model(17), algo, UNROLL_POLICY)
        assert set(descended) == {'W', 'tau'}
        for p in descended:
            assert np.abs(unrolled[p]).max() > 1e-3
            _close(descended[p], unrolled[p])

    @pytest.mark.parametrize('algo', ['d_rtrl', 'eprop'])
    def test_a_broadcast_along_a_weight_axis_is_refused(self, algo):
        m = _NarrowDriveInBody(17)
        brainstate.nn.init_all_states(m)
        learner = _LIFT_ALGOS[algo](m, DEFAULT_CONTROL_FLOW_POLICY)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(NotImplementedError, match='broadcasting along an axis of the weight') as err:
                learner.compile_graph(_batched_inputs()[0])
        message = str(err.value)
        assert 'shape (2, 1)' in message and 'shape (2, 4)' in message
        assert "('W',)" in message
        assert 'issues/231' in message

    @pytest.mark.parametrize('model', [_BatchedDrive, _BatchedDriveInBody],
                             ids=['lifted', 'in-body'])
    def test_an_output_with_the_state_shape_is_accepted(self, model):
        # ``tau`` reaches the batched states through ``element_wise`` with shape
        # (n,) as well: that trace is sized from the hidden group, so it is not refused
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with brainstate.environ.context(precision=64):
                descended = _online_batched(lambda: model(17), 'd_rtrl')
                unrolled = _online_batched(lambda: model(17), 'd_rtrl', UNROLL_POLICY)
        assert set(descended) == {'W', 'tau'}
        for p in descended:
            assert np.abs(unrolled[p]).max() > 1e-3
            _close(descended[p], unrolled[p])

