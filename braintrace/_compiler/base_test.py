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


import unittest
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
from jax._src.source_info_util import new_source_info

from braintrace._compatible_imports import (
    Var,
    JaxprEqn,
    Primitive,
    Literal,
    new_var,
)
from braintrace._compiler.base import (
    find_matched_vars,
    find_element_exist_in_the_set,
    check_unsupported_op,
    JaxprEvaluation,
)
from braintrace._compiler.canonicalize import ControlFlowPolicy
from braintrace._compiler.diagnostics import (
    DiagnosticKind,
    DiagnosticLevel,
    diagnostic_context,
)


# ---------------------------------------------------------------------------
# Helpers for constructing JAX IR objects used in tests
# ---------------------------------------------------------------------------

def _make_var(suffix: int = 0, shape=(3,), dtype=jnp.float32) -> Var:
    return new_var(suffix, jax.core.ShapedArray(shape, dtype))


def _make_literal(value: float = 1.0) -> Literal:
    return Literal(value, jax.core.ShapedArray((), jnp.float32))


def _make_eqn(
    invars,
    outvars,
    prim_name: str = 'add',
    params: dict | None = None,
) -> JaxprEqn:
    p = Primitive(prim_name)
    # JAX 0.9.0+ renamed 'effects' to 'effs' in JaxprEqn.__init__
    effs_kwarg = 'effs' if jax.__version_info__ >= (0, 9, 0) else 'effects'
    return JaxprEqn(
        invars=invars,
        outvars=outvars,
        primitive=p,
        params=params or {},
        source_info=new_source_info(),
        ctx=None,
        **{effs_kwarg: set()},
    )


def _make_jit_eqn(invars, outvars, name: str = 'my_fn') -> JaxprEqn:
    """Create a jit/pjit equation with the correct primitive name for this JAX version."""
    prim_name = 'jit' if jax.__version_info__ >= (0, 7, 0) else 'pjit'
    return _make_eqn(invars, outvars, prim_name=prim_name, params={'name': name})


def _make_scan_eqn(invars, outvars) -> JaxprEqn:
    return _make_eqn(invars, outvars, prim_name='scan')


def _make_while_eqn(invars, outvars) -> JaxprEqn:
    return _make_eqn(invars, outvars, prim_name='while')


def _make_cond_eqn(invars, outvars) -> JaxprEqn:
    return _make_eqn(invars, outvars, prim_name='cond')


def _make_evaluator(**overrides) -> JaxprEvaluation:
    """Create a JaxprEvaluation with empty defaults, overridable via kwargs."""
    defaults = dict(
        weight_invars=set(),
        hidden_invars=set(),
        hidden_outvars=set(),
        invar_to_hidden_path={},
        outvar_to_hidden_path={},
    )
    defaults.update(overrides)
    return JaxprEvaluation(**defaults)


# ===========================================================================
# Tests for find_matched_vars
# ===========================================================================

class TestFindMatchedVars(unittest.TestCase):

    def test_all_vars_matched(self):
        v1, v2 = _make_var(0), _make_var(1)
        result = find_matched_vars([v1, v2], {v1, v2})
        self.assertEqual(result, [v1, v2])

    def test_partial_match(self):
        v1, v2, v3 = _make_var(0), _make_var(1), _make_var(2)
        result = find_matched_vars([v1, v2, v3], {v2})
        self.assertEqual(result, [v2])

    def test_no_match(self):
        v1, v2 = _make_var(0), _make_var(1)
        result = find_matched_vars([v1], {v2})
        self.assertEqual(result, [])

    def test_empty_invars(self):
        v1 = _make_var(0)
        result = find_matched_vars([], {v1})
        self.assertEqual(result, [])

    def test_empty_needed_set(self):
        v1 = _make_var(0)
        result = find_matched_vars([v1], set())
        self.assertEqual(result, [])

    def test_both_empty(self):
        result = find_matched_vars([], set())
        self.assertEqual(result, [])

    def test_literals_are_skipped(self):
        """Literal values are not Var instances and should be filtered out."""
        v1 = _make_var(0)
        lit = _make_literal(1.0)
        result = find_matched_vars([lit, v1], {v1})
        self.assertEqual(result, [v1])

    def test_only_literals(self):
        lit = _make_literal(2.0)
        result = find_matched_vars([lit], set())
        self.assertEqual(result, [])

    def test_preserves_order(self):
        v1, v2, v3 = _make_var(0), _make_var(1), _make_var(2)
        result = find_matched_vars([v3, v1, v2], {v1, v2, v3})
        self.assertEqual(result, [v3, v1, v2])

    def test_duplicate_invars(self):
        v1 = _make_var(0)
        result = find_matched_vars([v1, v1], {v1})
        self.assertEqual(result, [v1, v1])


# ===========================================================================
# Tests for find_element_exist_in_the_set
# ===========================================================================

class TestFindElementExistInTheSet(unittest.TestCase):

    def test_first_element_found(self):
        v1, v2 = _make_var(0), _make_var(1)
        result = find_element_exist_in_the_set([v1, v2], {v1, v2})
        # Should return the first match (v1)
        self.assertIs(result, v1)

    def test_returns_first_match_not_later(self):
        v1, v2, v3 = _make_var(0), _make_var(1), _make_var(2)
        result = find_element_exist_in_the_set([v1, v2, v3], {v2, v3})
        self.assertIs(result, v2)

    def test_no_match_returns_none(self):
        v1, v2 = _make_var(0), _make_var(1)
        result = find_element_exist_in_the_set([v1], {v2})
        self.assertIsNone(result)

    def test_empty_elements(self):
        v1 = _make_var(0)
        result = find_element_exist_in_the_set([], {v1})
        self.assertIsNone(result)

    def test_empty_set(self):
        v1 = _make_var(0)
        result = find_element_exist_in_the_set([v1], set())
        self.assertIsNone(result)

    def test_both_empty(self):
        result = find_element_exist_in_the_set([], set())
        self.assertIsNone(result)

    def test_literals_are_skipped(self):
        v1 = _make_var(0)
        lit = _make_literal(1.0)
        # lit is not a Var, so it should be skipped; v1 is in the set
        result = find_element_exist_in_the_set([lit, v1], {v1})
        self.assertIs(result, v1)

    def test_only_literals_returns_none(self):
        lit = _make_literal(1.0)
        result = find_element_exist_in_the_set([lit], set())
        self.assertIsNone(result)


# ===========================================================================
# Tests for check_unsupported_op
# ===========================================================================

class TestCheckUnsupportedOp(unittest.TestCase):

    def _make_self_obj(self, weight_invars=None, hidden_outvars=None,
                       invar_to_hidden_path=None, outvar_to_hidden_path=None):
        """Create a mock 'self' object that check_unsupported_op expects."""
        return _make_evaluator(
            weight_invars=weight_invars or set(),
            hidden_outvars=hidden_outvars or set(),
            invar_to_hidden_path=invar_to_hidden_path or {},
            outvar_to_hidden_path=outvar_to_hidden_path or {},
        )

    def test_no_weight_no_hidden_passes(self):
        """No error when eqn doesn't touch weight or hidden vars."""
        v1, v2 = _make_var(0), _make_var(1)
        v3 = _make_var(2)
        eqn = _make_eqn([v1], [v2])
        obj = self._make_self_obj(weight_invars={v3}, hidden_outvars=set())
        # Should not raise
        check_unsupported_op(obj, eqn, 'scan')

    def test_weight_var_in_invars_raises(self):
        w = _make_var(0)
        v_out = _make_var(1)
        eqn = _make_eqn([w], [v_out])
        obj = self._make_self_obj(
            weight_invars={w},
            invar_to_hidden_path={w: ('layer', 'weight')},
        )
        with self.assertRaises(NotImplementedError) as ctx:
            check_unsupported_op(obj, eqn, 'scan')
        self.assertIn('scan', str(ctx.exception))
        self.assertIn('weight states', str(ctx.exception))

    def test_hidden_var_in_outvars_raises(self):
        """With ``while_hidden='error'`` (the pre-Phase-3 behaviour), a hidden
        state produced by an opaque control-flow equation still raises."""
        v_in = _make_var(0)
        h_out = _make_var(1)
        eqn = _make_eqn([v_in], [h_out])
        obj = self._make_self_obj(
            hidden_outvars={h_out},
            outvar_to_hidden_path={h_out: ('layer', 'hidden')},
        )
        obj.control_flow = ControlFlowPolicy(while_hidden='error')
        with self.assertRaises(NotImplementedError) as ctx:
            check_unsupported_op(obj, eqn, 'while')
        self.assertIn('while', str(ctx.exception))
        self.assertIn('hidden states', str(ctx.exception))

    def test_weight_check_takes_priority_over_hidden(self):
        """When both weight invars and hidden outvars match, weight error is raised first."""
        w = _make_var(0)
        h_out = _make_var(1)
        eqn = _make_eqn([w], [h_out])
        obj = self._make_self_obj(
            weight_invars={w},
            hidden_outvars={h_out},
            invar_to_hidden_path={w: ('w_path',)},
            outvar_to_hidden_path={h_out: ('h_path',)},
        )
        with self.assertRaises(NotImplementedError) as ctx:
            check_unsupported_op(obj, eqn, 'jit')
        # Weight error is raised first
        self.assertIn('weight states', str(ctx.exception))

    def test_error_message_includes_op_name(self):
        w = _make_var(0)
        eqn = _make_eqn([w], [_make_var(1)])
        obj = self._make_self_obj(
            weight_invars={w},
            invar_to_hidden_path={w: ('test_path',)},
        )
        for op_name in ['jit', 'scan', 'while', 'cond']:
            with self.assertRaises(NotImplementedError) as ctx:
                check_unsupported_op(obj, eqn, op_name)
            self.assertIn(op_name, str(ctx.exception))

    def test_error_message_identifies_weight_variable(self):
        """The weight-in-control-flow error identifies the offending weight
        variable and includes the equation. A weight var has no hidden-state
        path, so the message renders the variable directly rather than indexing
        the hidden-path map with it (the former F-SCAN-WEIGHT KeyError)."""
        w = _make_var(0)
        out = _make_var(1)
        eqn = _make_eqn([w], [out])
        obj = self._make_self_obj(weight_invars={w})
        with self.assertRaises(NotImplementedError) as ctx:
            check_unsupported_op(obj, eqn, 'scan')
        msg = str(ctx.exception)
        self.assertIn('weight states', msg)
        self.assertIn('scan', msg)
        self.assertIn(str(w), msg)

    def test_literal_invars_dont_trigger_weight_error(self):
        """Literals in invars should not be matched as weight vars."""
        lit = _make_literal(1.0)
        v_out = _make_var(1)
        eqn = _make_eqn([lit], [v_out])
        obj = self._make_self_obj(weight_invars=set())
        # Should not raise
        check_unsupported_op(obj, eqn, 'scan')


class TestCheckUnsupportedOpPhase3(unittest.TestCase):
    """Phase 3 control-flow policy: while-specific weight error, and the
    opaque-forward relaxation for weight-free hidden-producing control flow."""

    def _hidden_obj(self, h_out, **overrides):
        return _make_evaluator(
            hidden_outvars={h_out},
            outvar_to_hidden_path={h_out: ('layer', 'hidden')},
            **overrides,
        )

    def test_weight_in_while_raises_actionable_error(self):
        """A weight consumed by a ``while`` raises with guidance (mentions
        'while' and the fixed-length-scan alternative) and emits the
        dedicated ``WEIGHT_IN_WHILE`` kind, not the generic one."""
        w = _make_var(0)
        eqn = _make_while_eqn([w], [_make_var(1)])
        obj = _make_evaluator(weight_invars={w})
        with diagnostic_context() as reporter:
            with self.assertRaises(NotImplementedError) as ctx:
                check_unsupported_op(obj, eqn, 'while')
        msg = str(ctx.exception)
        self.assertIn('while', msg)
        self.assertIn('fixed-length', msg)
        kinds = [r.kind for r in reporter.records()]
        self.assertIn(DiagnosticKind.WEIGHT_IN_WHILE, kinds)
        self.assertNotIn(DiagnosticKind.WEIGHT_IN_CONTROL_FLOW, kinds)
        record = reporter.records()[0]
        self.assertIs(record.level, DiagnosticLevel.ERROR)

    def test_weight_in_scan_keeps_generic_kind(self):
        """Weight-in-scan/cond keeps the pre-existing generic diagnostic."""
        w = _make_var(0)
        for factory, op_name in [(_make_scan_eqn, 'scan'), (_make_cond_eqn, 'cond')]:
            eqn = factory([w], [_make_var(1)])
            obj = _make_evaluator(weight_invars={w})
            with diagnostic_context() as reporter:
                with self.assertRaises(NotImplementedError):
                    check_unsupported_op(obj, eqn, op_name)
            kinds = [r.kind for r in reporter.records()]
            self.assertIn(DiagnosticKind.WEIGHT_IN_CONTROL_FLOW, kinds)
            self.assertNotIn(DiagnosticKind.WEIGHT_IN_WHILE, kinds)

    def test_hidden_from_control_flow_default_policy_is_opaque_fwd(self):
        """Under the default policy, a weight-free while/scan/cond producing a
        hidden state does NOT raise; an INFO ``CONTROL_FLOW_OPAQUE_FWD``
        record is emitted instead."""
        for factory, op_name in [
            (_make_while_eqn, 'while'),
            (_make_scan_eqn, 'scan'),
            (_make_cond_eqn, 'cond'),
        ]:
            h_out = _make_var(1)
            eqn = factory([_make_var(0)], [h_out])
            obj = self._hidden_obj(h_out)
            with diagnostic_context() as reporter:
                check_unsupported_op(obj, eqn, op_name)  # must not raise
            records = [
                r for r in reporter.records()
                if r.kind is DiagnosticKind.CONTROL_FLOW_OPAQUE_FWD
            ]
            self.assertEqual(len(records), 1, f'op={op_name}')
            self.assertIs(records[0].level, DiagnosticLevel.INFO)
            self.assertEqual(records[0].context['op_name'], op_name)
            self.assertEqual(records[0].context['hidden_path'], ('layer', 'hidden'))

    def test_hidden_from_while_error_policy_raises(self):
        """``while_hidden='error'`` restores the pre-Phase-3 hard error."""
        h_out = _make_var(1)
        eqn = _make_while_eqn([_make_var(0)], [h_out])
        obj = self._hidden_obj(
            h_out, control_flow=ControlFlowPolicy(while_hidden='error')
        )
        with self.assertRaises(NotImplementedError) as ctx:
            check_unsupported_op(obj, eqn, 'while')
        self.assertIn('hidden states', str(ctx.exception))

    def test_hidden_from_jit_still_raises(self):
        """The opaque-forward relaxation applies only to scan/while/cond; a
        hidden state produced by an opaque jit region still raises."""
        h_out = _make_var(1)
        eqn = _make_jit_eqn([_make_var(0)], [h_out])
        obj = self._hidden_obj(h_out)
        with self.assertRaises(NotImplementedError):
            check_unsupported_op(obj, eqn, 'jit')

    def test_bogus_while_hidden_raises_value_error(self):
        """An invalid ``while_hidden`` value is a use-site ``ValueError``."""
        h_out = _make_var(1)
        eqn = _make_while_eqn([_make_var(0)], [h_out])
        obj = self._hidden_obj(
            h_out, control_flow=ControlFlowPolicy(while_hidden='bogus')
        )
        with self.assertRaises(ValueError) as ctx:
            check_unsupported_op(obj, eqn, 'while')
        self.assertIn('while_hidden', str(ctx.exception))

    def test_object_without_control_flow_attr_uses_default(self):
        """Callers that never set ``control_flow`` (e.g. hand-rolled mock
        objects) behave as the default policy."""

        class Bare:
            weight_invars = set()
            hidden_outvars = None
            outvar_to_hidden_path = None

        h_out = _make_var(1)
        obj = Bare()
        obj.hidden_outvars = {h_out}
        obj.outvar_to_hidden_path = {h_out: ('h',)}
        eqn = _make_while_eqn([_make_var(0)], [h_out])
        check_unsupported_op(obj, eqn, 'while')  # must not raise


# ===========================================================================
# Tests for JaxprEvaluation
# ===========================================================================

class TestJaxprEvaluationInit(unittest.TestCase):

    def test_stores_all_attributes(self):
        w = _make_var(0)
        h_in = _make_var(1)
        h_out = _make_var(2)
        in_map = {h_in: ('hidden_in',)}
        out_map = {h_out: ('hidden_out',)}

        evaluator = JaxprEvaluation(
            weight_invars={w},
            hidden_invars={h_in},
            hidden_outvars={h_out},
            invar_to_hidden_path=in_map,
            outvar_to_hidden_path=out_map,
        )
        self.assertEqual(evaluator.weight_invars, {w})
        self.assertEqual(evaluator.hidden_invars, {h_in})
        self.assertEqual(evaluator.hidden_outvars, {h_out})
        self.assertEqual(evaluator.invar_to_hidden_path, in_map)
        self.assertEqual(evaluator.outvar_to_hidden_path, out_map)

    def test_empty_init(self):
        evaluator = _make_evaluator()
        self.assertEqual(evaluator.weight_invars, set())
        self.assertEqual(evaluator.hidden_invars, set())
        self.assertEqual(evaluator.hidden_outvars, set())
        self.assertEqual(evaluator.invar_to_hidden_path, {})
        self.assertEqual(evaluator.outvar_to_hidden_path, {})

    def test_default_control_flow_policy_stored(self):
        from braintrace._compiler.canonicalize import DEFAULT_CONTROL_FLOW_POLICY
        evaluator = _make_evaluator()
        self.assertIs(evaluator.control_flow, DEFAULT_CONTROL_FLOW_POLICY)

    def test_custom_control_flow_policy_stored(self):
        policy = ControlFlowPolicy(while_hidden='error')
        evaluator = _make_evaluator(control_flow=policy)
        self.assertIs(evaluator.control_flow, policy)


class TestJaxprEvaluationEvalEqn(unittest.TestCase):

    def test_eval_eqn_raises_not_implemented(self):
        evaluator = _make_evaluator()
        eqn = _make_eqn([_make_var(0)], [_make_var(1)])
        with self.assertRaises(NotImplementedError) as ctx:
            evaluator._eval_eqn(eqn)
        self.assertIn('_eval_eqn', str(ctx.exception))


class _ConcreteJaxprEvaluation(JaxprEvaluation):
    """A concrete subclass that tracks which equations were evaluated."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluated_eqns = []

    def _eval_eqn(self, eqn):
        self.evaluated_eqns.append(eqn)


def _make_concrete_evaluator(**overrides) -> _ConcreteJaxprEvaluation:
    defaults = dict(
        weight_invars=set(),
        hidden_invars=set(),
        hidden_outvars=set(),
        invar_to_hidden_path={},
        outvar_to_hidden_path={},
    )
    defaults.update(overrides)
    return _ConcreteJaxprEvaluation(**defaults)


class TestJaxprEvaluationEvalJaxpr(unittest.TestCase):
    """Test _eval_jaxpr dispatching to the correct methods."""

    def test_dispatches_normal_eqn_to_eval_eqn(self):
        evaluator = _make_concrete_evaluator()
        v1, v2 = _make_var(0), _make_var(1)
        eqn = _make_eqn([v1], [v2], prim_name='add')

        mock_jaxpr = MagicMock()
        mock_jaxpr.eqns = [eqn]

        evaluator._eval_jaxpr(mock_jaxpr)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)
        self.assertIs(evaluator.evaluated_eqns[0], eqn)

    def test_dispatches_scan_to_eval_scan(self):
        evaluator = _make_concrete_evaluator()
        v1, v2 = _make_var(0), _make_var(1)
        eqn = _make_scan_eqn([v1], [v2])

        mock_jaxpr = MagicMock()
        mock_jaxpr.eqns = [eqn]

        # scan with no weight/hidden vars should pass through to _eval_eqn
        evaluator._eval_jaxpr(mock_jaxpr)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_dispatches_while_to_eval_while(self):
        evaluator = _make_concrete_evaluator()
        v1, v2 = _make_var(0), _make_var(1)
        eqn = _make_while_eqn([v1], [v2])

        mock_jaxpr = MagicMock()
        mock_jaxpr.eqns = [eqn]

        evaluator._eval_jaxpr(mock_jaxpr)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_dispatches_cond_to_eval_cond(self):
        evaluator = _make_concrete_evaluator()
        v1, v2 = _make_var(0), _make_var(1)
        eqn = _make_cond_eqn([v1], [v2])

        mock_jaxpr = MagicMock()
        mock_jaxpr.eqns = [eqn]

        evaluator._eval_jaxpr(mock_jaxpr)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_dispatches_jit_to_eval_pjit(self):
        evaluator = _make_concrete_evaluator()
        v1, v2 = _make_var(0), _make_var(1)
        eqn = _make_jit_eqn([v1], [v2], name='my_function')

        mock_jaxpr = MagicMock()
        mock_jaxpr.eqns = [eqn]

        # Non-etrace jit passes through to _eval_eqn
        evaluator._eval_jaxpr(mock_jaxpr)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_multiple_eqns_dispatched_in_order(self):
        evaluator = _make_concrete_evaluator()
        v1, v2, v3, v4 = [_make_var(i) for i in range(4)]
        eqn1 = _make_eqn([v1], [v2], prim_name='mul')
        eqn2 = _make_eqn([v2], [v3], prim_name='sin')
        eqn3 = _make_eqn([v3], [v4], prim_name='cos')

        mock_jaxpr = MagicMock()
        mock_jaxpr.eqns = [eqn1, eqn2, eqn3]

        evaluator._eval_jaxpr(mock_jaxpr)
        self.assertEqual(len(evaluator.evaluated_eqns), 3)
        self.assertIs(evaluator.evaluated_eqns[0], eqn1)
        self.assertIs(evaluator.evaluated_eqns[1], eqn2)
        self.assertIs(evaluator.evaluated_eqns[2], eqn3)

    def test_empty_jaxpr(self):
        evaluator = _make_concrete_evaluator()
        mock_jaxpr = MagicMock()
        mock_jaxpr.eqns = []

        evaluator._eval_jaxpr(mock_jaxpr)
        self.assertEqual(len(evaluator.evaluated_eqns), 0)


class TestJaxprEvaluationEvalPjit(unittest.TestCase):

    def test_etrace_op_with_gradient_evaluates_eqn(self):
        """An etrace op with gradient enabled should call _eval_eqn."""
        evaluator = _make_concrete_evaluator()
        v1, v2 = _make_var(0), _make_var(1)
        # Use the etrace gradient-enabled name
        eqn = _make_jit_eqn([v1], [v2], name='_etrace_operator_call_enable_grad_my_op')

        evaluator._eval_pjit(eqn)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_etrace_op_without_gradient_returns_early(self):
        """Regular pjits (non-ETP primitives) are evaluated as normal jaxpr eqns
        in the primitive-based compiler, so name-based skipping no longer applies."""
        evaluator = _make_concrete_evaluator()
        v1, v2 = _make_var(0), _make_var(1)
        # Base etrace name (no gradient suffix) — now treated as a regular pjit
        eqn = _make_jit_eqn([v1], [v2], name='_etrace_operator_call')

        evaluator._eval_pjit(eqn)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_non_etrace_jit_passes_to_eval_eqn(self):
        """A regular jit (non-etrace) should check unsupported and then evaluate."""
        evaluator = _make_concrete_evaluator()
        v1, v2 = _make_var(0), _make_var(1)
        eqn = _make_jit_eqn([v1], [v2], name='regular_function')

        evaluator._eval_pjit(eqn)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_non_etrace_jit_with_weight_var_raises(self):
        w = _make_var(0)
        v_out = _make_var(1)
        evaluator = _make_concrete_evaluator(
            weight_invars={w},
            invar_to_hidden_path={w: ('weight_path',)},
        )
        eqn = _make_jit_eqn([w], [v_out], name='some_function')

        with self.assertRaises(NotImplementedError):
            evaluator._eval_pjit(eqn)

    def test_non_etrace_jit_with_hidden_outvar_raises(self):
        v_in = _make_var(0)
        h_out = _make_var(1)
        evaluator = _make_concrete_evaluator(
            hidden_outvars={h_out},
            outvar_to_hidden_path={h_out: ('hidden_path',)},
        )
        eqn = _make_jit_eqn([v_in], [h_out], name='some_function')

        with self.assertRaises(NotImplementedError):
            evaluator._eval_pjit(eqn)


class TestJaxprEvaluationEvalScan(unittest.TestCase):

    def test_scan_no_conflict_evaluates(self):
        evaluator = _make_concrete_evaluator()
        v1, v2 = _make_var(0), _make_var(1)
        eqn = _make_scan_eqn([v1], [v2])

        evaluator._eval_scan(eqn)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_scan_with_weight_var_raises(self):
        w = _make_var(0)
        evaluator = _make_concrete_evaluator(
            weight_invars={w},
            invar_to_hidden_path={w: ('w',)},
        )
        eqn = _make_scan_eqn([w], [_make_var(1)])

        with self.assertRaises(NotImplementedError) as ctx:
            evaluator._eval_scan(eqn)
        self.assertIn('scan', str(ctx.exception))

    def test_scan_with_hidden_outvar_default_policy_evaluates(self):
        """Default policy: a weight-free hidden-producing scan is kept as an
        opaque forward node and still dispatched to ``_eval_eqn``."""
        h = _make_var(1)
        evaluator = _make_concrete_evaluator(
            hidden_outvars={h},
            outvar_to_hidden_path={h: ('h',)},
        )
        eqn = _make_scan_eqn([_make_var(0)], [h])

        evaluator._eval_scan(eqn)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_scan_with_hidden_outvar_error_policy_raises(self):
        h = _make_var(1)
        evaluator = _make_concrete_evaluator(
            hidden_outvars={h},
            outvar_to_hidden_path={h: ('h',)},
            control_flow=ControlFlowPolicy(while_hidden='error'),
        )
        eqn = _make_scan_eqn([_make_var(0)], [h])

        with self.assertRaises(NotImplementedError) as ctx:
            evaluator._eval_scan(eqn)
        self.assertIn('scan', str(ctx.exception))


class TestJaxprEvaluationEvalWhile(unittest.TestCase):

    def test_while_no_conflict_evaluates(self):
        evaluator = _make_concrete_evaluator()
        v1, v2 = _make_var(0), _make_var(1)
        eqn = _make_while_eqn([v1], [v2])

        evaluator._eval_while(eqn)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_while_with_weight_var_raises(self):
        w = _make_var(0)
        evaluator = _make_concrete_evaluator(
            weight_invars={w},
            invar_to_hidden_path={w: ('w',)},
        )
        eqn = _make_while_eqn([w], [_make_var(1)])

        with self.assertRaises(NotImplementedError) as ctx:
            evaluator._eval_while(eqn)
        self.assertIn('while', str(ctx.exception))

    def test_while_with_hidden_outvar_default_policy_evaluates(self):
        """Default policy: a weight-free hidden-producing while is kept as an
        opaque forward node and still dispatched to ``_eval_eqn``."""
        h = _make_var(1)
        evaluator = _make_concrete_evaluator(
            hidden_outvars={h},
            outvar_to_hidden_path={h: ('h',)},
        )
        eqn = _make_while_eqn([_make_var(0)], [h])

        evaluator._eval_while(eqn)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_while_with_hidden_outvar_error_policy_raises(self):
        h = _make_var(1)
        evaluator = _make_concrete_evaluator(
            hidden_outvars={h},
            outvar_to_hidden_path={h: ('h',)},
            control_flow=ControlFlowPolicy(while_hidden='error'),
        )
        eqn = _make_while_eqn([_make_var(0)], [h])

        with self.assertRaises(NotImplementedError) as ctx:
            evaluator._eval_while(eqn)
        self.assertIn('while', str(ctx.exception))


class TestJaxprEvaluationEvalCond(unittest.TestCase):

    def test_cond_no_conflict_evaluates(self):
        evaluator = _make_concrete_evaluator()
        v1, v2 = _make_var(0), _make_var(1)
        eqn = _make_cond_eqn([v1], [v2])

        evaluator._eval_cond(eqn)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_cond_with_weight_var_raises(self):
        w = _make_var(0)
        evaluator = _make_concrete_evaluator(
            weight_invars={w},
            invar_to_hidden_path={w: ('w',)},
        )
        eqn = _make_cond_eqn([w], [_make_var(1)])

        with self.assertRaises(NotImplementedError) as ctx:
            evaluator._eval_cond(eqn)
        self.assertIn('cond', str(ctx.exception))

    def test_cond_with_hidden_outvar_default_policy_evaluates(self):
        """Default policy: a weight-free hidden-producing cond is kept as an
        opaque forward node and still dispatched to ``_eval_eqn``."""
        h = _make_var(1)
        evaluator = _make_concrete_evaluator(
            hidden_outvars={h},
            outvar_to_hidden_path={h: ('h',)},
        )
        eqn = _make_cond_eqn([_make_var(0)], [h])

        evaluator._eval_cond(eqn)
        self.assertEqual(len(evaluator.evaluated_eqns), 1)

    def test_cond_with_hidden_outvar_error_policy_raises(self):
        h = _make_var(1)
        evaluator = _make_concrete_evaluator(
            hidden_outvars={h},
            outvar_to_hidden_path={h: ('h',)},
            control_flow=ControlFlowPolicy(while_hidden='error'),
        )
        eqn = _make_cond_eqn([_make_var(0)], [h])

        with self.assertRaises(NotImplementedError) as ctx:
            evaluator._eval_cond(eqn)
        self.assertIn('cond', str(ctx.exception))


class TestJaxprEvaluationModuleAttribute(unittest.TestCase):

    def test_module_name(self):
        self.assertEqual(JaxprEvaluation.__module__, 'braintrace')


class TestJaxprEvaluationIntegration(unittest.TestCase):
    """Integration tests using real jaxpr from JAX tracing."""

    def test_eval_jaxpr_from_traced_function(self):
        """Trace a simple function and evaluate its jaxpr."""
        from jax import make_jaxpr

        jaxpr_closed = make_jaxpr(lambda x, y: x * y + x)(
            jnp.ones(3), jnp.ones(3)
        )

        evaluator = _make_concrete_evaluator()
        evaluator._eval_jaxpr(jaxpr_closed.jaxpr)

        # x * y -> mul, then + x -> add: 2 equations
        self.assertEqual(len(evaluator.evaluated_eqns), 2)

    def test_eval_jaxpr_with_jit_traced(self):
        """Trace a jitted function and verify jit dispatch."""
        from jax import make_jaxpr, jit

        @jit
        def fn(x):
            return x ** 2

        jaxpr_closed = make_jaxpr(fn)(jnp.ones(3))

        evaluator = _make_concrete_evaluator()
        evaluator._eval_jaxpr(jaxpr_closed.jaxpr)

        # The jit wrapping produces a single jit/pjit equation that delegates to _eval_eqn
        self.assertGreaterEqual(len(evaluator.evaluated_eqns), 1)

    def test_eval_jaxpr_with_scan_and_weight_conflict(self):
        """Verify that a real scan jaxpr raises when weight vars are involved."""
        from jax import make_jaxpr, lax

        def scan_fn(carry, x):
            return carry + x, carry

        def f(init, xs):
            return lax.scan(scan_fn, init, xs)

        jaxpr_closed = make_jaxpr(f)(jnp.ones(3), jnp.ones((5, 3)))
        jaxpr = jaxpr_closed.jaxpr

        # Find the scan equation's invars
        scan_eqn = None
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == 'scan':
                scan_eqn = eqn
                break
        self.assertIsNotNone(scan_eqn)

        # Set one of the scan's invars as a weight var
        weight_var = None
        for v in scan_eqn.invars:
            if isinstance(v, Var):
                weight_var = v
                break
        self.assertIsNotNone(weight_var)

        evaluator = _make_concrete_evaluator(
            weight_invars={weight_var},
            invar_to_hidden_path={weight_var: ('test_weight',)},
        )

        with self.assertRaises(NotImplementedError):
            evaluator._eval_jaxpr(jaxpr)


if __name__ == '__main__':
    unittest.main()
