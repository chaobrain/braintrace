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

"""Tests for the precondition probes' verdict logic.

The probes themselves run in ``braintrace/_compile_test.py``
(``TestPreconditions``); here the verdict (:func:`satisfied`) is checked on
constructed outcomes, so every branch is pinned whatever today's compiler does.
"""

import brainstate
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._compiler.diagnostics import DiagnosticKind as K
from braintrace._compiler.diagnostics import DiagnosticLevel as L
from braintrace._testing import precondition_models as pm


def _case(check, issue=1):
    """A case that lists RELATION_PARTIAL_PATH (the credit kinds for ``CREDIT_RECORDED``)."""
    records = pm.CREDIT_KINDS if check == pm.CREDIT_RECORDED else (K.RELATION_PARTIAL_PATH.value,)
    return pm.PreconditionCase('c', pm.TailNet, check, issue, records=records)


EXACT = {'w': 1e-16}
WRONG = {'w': 0.3}
ERROR = ((K.RELATION_PARTIAL_PATH, L.ERROR),)
WARNING = ((K.RELATION_PARTIAL_PATH, L.WARNING),)
OTHER_ERROR = ((K.SCAN_PLAIN_WEIGHT, L.ERROR),)  # a kind no case lists
CREDIT = ((K.ONLINE_GRADIENT_ZERO, L.WARNING),)
REFUSED = braintrace.NotSupportedError('refused')
CRASHED = ValueError('a broken probe')


@pytest.mark.parametrize('check,outcome,expected', [
    # exact against BPTT
    (pm.EXACT_OR_ERROR, pm.Outcome(errors=EXACT), True),
    (pm.EXACT_OR_ERROR, pm.Outcome(errors=WRONG), False),
    (pm.EXACT_OR_ERROR, pm.Outcome(errors=WRONG, records=WARNING), False),  # warn-only is not enough
    (pm.EXACT_OR_ERROR, pm.Outcome(errors=WRONG, records=ERROR), True),
    (pm.EXACT_OR_ERROR, pm.Outcome(errors=WRONG, records=OTHER_ERROR), False),  # an unrelated ERROR
    (pm.EXACT_OR_ERROR, pm.Outcome(raised=REFUSED), True),
    (pm.EXACT_OR_ERROR, pm.Outcome(raised=braintrace.CompilationError('refused')), True),
    (pm.EXACT_OR_ERROR, pm.Outcome(raised=CRASHED), False),  # a crash is not a refusal
    (pm.EXACT_OR_LOUD, pm.Outcome(errors=WRONG, records=WARNING), True),
    (pm.EXACT_OR_LOUD, pm.Outcome(errors=WRONG, records=ERROR), True),
    (pm.EXACT_OR_LOUD, pm.Outcome(errors=WRONG, records=((K.SCAN_PLAIN_WEIGHT, L.WARNING),)), False),
    (pm.EXACT_OR_LOUD, pm.Outcome(errors=WRONG), False),
    (pm.EXACT, pm.Outcome(errors=EXACT), True),
    (pm.EXACT, pm.Outcome(errors=WRONG, records=ERROR), False),
    (pm.EXACT, pm.Outcome(errors={}), False),  # no gradient compared is not exact
    (pm.EXACT, pm.Outcome(raised=REFUSED), False),
    # the Jacobian D-RTRL uses
    (pm.TRANSITION, pm.Outcome(errors=WRONG, deviation=0.0, diag_sites=1), True),
    (pm.TRANSITION, pm.Outcome(errors=WRONG, deviation=0.0), False),  # nothing measured
    (pm.TRANSITION, pm.Outcome(errors=EXACT, deviation=0.5, diag_sites=1), False),
    (pm.TRANSITION, pm.Outcome(deviation=0.5, diag_sites=1, records=ERROR), True),
    (pm.TRANSITION, pm.Outcome(deviation=0.5, diag_sites=1, records=OTHER_ERROR), False),
    # online-credit record
    (pm.CREDIT_RECORDED, pm.Outcome(errors=WRONG, records=CREDIT), True),
    (pm.CREDIT_RECORDED, pm.Outcome(errors=WRONG, records=((K.ONLINE_GRADIENT_PARTIAL, L.INFO),)), True),
    (pm.CREDIT_RECORDED, pm.Outcome(errors=WRONG, records=WARNING), False),
    (pm.CREDIT_RECORDED, pm.Outcome(raised=REFUSED), True),
    (pm.CREDIT_RECORDED, pm.Outcome(raised=CRASHED), False),
])
def test_verdict_for_a_violation(check, outcome, expected):
    assert pm.satisfied(_case(check), outcome) is expected


def test_a_case_without_listed_kinds_needs_exactness_or_a_refusal():
    case = pm.PreconditionCase('c', pm.TailNet, pm.EXACT_OR_ERROR, 1)
    assert not pm.satisfied(case, pm.Outcome(errors=WRONG, records=ERROR))
    assert pm.satisfied(case, pm.Outcome(raised=REFUSED))


@pytest.mark.parametrize('check,outcome,expected', [
    (pm.EXACT, pm.Outcome(raised=REFUSED), False),
    (pm.EXACT, pm.Outcome(errors=EXACT, records=OTHER_ERROR), False),  # any ERROR fails a control
    (pm.TRANSITION, pm.Outcome(deviation=0.0, diag_sites=1, records=ERROR), False),
    (pm.TRANSITION, pm.Outcome(deviation=0.0, diag_sites=1), True),
    (pm.CREDIT_RECORDED, pm.Outcome(raised=REFUSED), False),
    (pm.CREDIT_RECORDED, pm.Outcome(records=((K.ONLINE_GRADIENT_ZERO, L.ERROR),)), True),
])
def test_a_control_must_hold_without_refusing(check, outcome, expected):
    assert pm.satisfied(_case(check, issue=None), outcome) is expected


def test_unknown_check_is_an_error():
    with pytest.raises(ValueError, match='unknown check'):
        pm.satisfied(_case('nonsense'), pm.Outcome(errors=WRONG))


class _NoEtp(brainstate.nn.Module):
    """No weight goes through an ETP op: ``braintrace.compile`` refuses it."""

    def __init__(self):
        super().__init__()
        self.w = brainstate.ParamState(jnp.ones((3, 2)))
        self.h = brainstate.HiddenState(jnp.zeros((1, 2)))

    def update(self, x):
        self.h.value = 0.9 * self.h.value + x @ self.w.value
        return self.h.value


def test_evaluate_records_a_refusal():
    case = pm.PreconditionCase('no-etp', _NoEtp, pm.EXACT_OR_ERROR, issue=1)
    outcome = pm.evaluate(case)
    assert isinstance(outcome.raised, pm.REFUSALS) and outcome.errors == {}
    assert pm.satisfied(case, outcome)
    assert pm.describe(outcome).startswith(f'raised {type(outcome.raised).__name__}')


def test_cases_are_unique_and_cover_every_issue():
    names = [c.name for c in pm.CASES]
    assert len(names) == len(set(names))
    assert {c.issue for c in pm.CASES} == {None, 233, 234, 235, 236, 237, 238, 239}
    for issue in {c.issue for c in pm.CASES} - {None}:
        checks = {c.check for c in pm.CASES if c.issue == issue}
        family = {c.model.__mro__[1] for c in pm.CASES if c.issue == issue}
        controls = [c for c in pm.CASES if c.issue is None and c.model.__mro__[1] in family]
        assert controls, f'#{issue} has no control'
        assert pm.EXACT not in checks


def test_listed_record_kinds_are_diagnostic_kinds():
    for case in pm.CASES:
        for kind in case.records:
            K(kind)  # raises on a misspelt kind
        assert (case.check == pm.CREDIT_RECORDED) == bool(case.records), case.name


def test_evaluate_counts_the_measured_diag_sites():
    case = next(c for c in pm.CASES if c.name == 'transition-none')
    outcome = pm.evaluate(case)
    assert outcome.diag_sites > 0 and outcome.deviation < 1e-12
    assert pm.satisfied(case, outcome)
