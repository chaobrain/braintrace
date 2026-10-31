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

"""Probe models for the compiler's unchecked in-step preconditions (test support).

Each :class:`PreconditionCase` is a small model that either violates one
precondition the online algorithms rely on (issues #233-#239) or is the
control that satisfies it. :func:`evaluate` compiles a case with
:func:`braintrace.compile`, runs it one step per call over a short window in
float64, and compares the summed online gradient with BPTT over the same
window (:func:`~braintrace._testing.oracle.bptt_param_gradients`, the same
sum-of-squares loss). :func:`satisfied` then states what the case promises:

``EXACT``
    The online gradient equals BPTT (a control).
``EXACT_OR_ERROR``
    Equal to BPTT, or an ERROR-level compilation record, or compiling or
    running raises. Today's WARNING-only paths do not satisfy it.
``EXACT_OR_LOUD``
    Equal to BPTT, or a WARNING- or ERROR-level record, or a raise. Used for
    the mixing tails (#234), where the fix may keep the approximation and
    report it.
``TRANSITION``
    The Jacobian the algorithm uses for a default-mode group equals the true
    per-position block (#233), or an ERROR-level record, or a raise.
``CREDIT_RECORDED``
    An online-gradient record (zero or partial) names the weight, or a raise
    (#238).

A control (``issue is None``) must satisfy its check without raising and
without an ERROR record. The tests live in ``braintrace/_compile_test.py``
(``TestPreconditions``); the stage-6 self-check reuses these cases.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

import braintrace
from braintrace._compiler.diagnostics import DiagnosticKind, DiagnosticLevel
from braintrace._testing import census
from braintrace._testing.oracle import bptt_param_gradients

__all__ = [
    'EXACT',
    'EXACT_OR_ERROR',
    'EXACT_OR_LOUD',
    'TRANSITION',
    'CREDIT_RECORDED',
    'PreconditionCase',
    'Outcome',
    'CASES',
    'evaluate',
    'satisfied',
]

EXACT = 'exact'
EXACT_OR_ERROR = 'exact_or_error'
EXACT_OR_LOUD = 'exact_or_loud'
TRANSITION = 'transition'
CREDIT_RECORDED = 'credit_recorded'

#: Relative error (f64) under which an online gradient counts as equal to BPTT.
RTOL = 1e-9

N_IN = 3


def _randn(*shape: int, seed: int) -> jax.Array:
    with brainstate.random.seed_context(seed):
        return 0.5 * brainstate.random.randn(*shape)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class TransitionNet(brainstate.nn.Module):
    """#233: the recurrence couples positions through an op that is not a matmul."""

    coupling = 'none'
    N = 5

    def __init__(self) -> None:
        super().__init__()
        self.w = brainstate.ParamState(_randn(N_IN, self.N, seed=1))
        self.h = brainstate.HiddenState(jnp.zeros((1, self.N)))

    def update(self, x: jax.Array) -> jax.Array:
        h = self.h.value
        rec = {
            'none': lambda: 0.9 * h,                                     # control
            'roll': lambda: 0.9 * jnp.roll(h, 1, axis=-1),               # h[p] reads h[p-1]
            'mean': lambda: 0.8 * h + 0.3 * h.mean(-1, keepdims=True),   # mean field
            'reverse': lambda: 0.9 * h[..., ::-1],
        }[self.coupling]()
        self.h.value = jnp.tanh(rec + braintrace.matmul(x, self.w.value))
        return self.h.value


class TailNet(brainstate.nn.Module):
    """#234: the map from the ETP output to the hidden state is not elementwise."""

    tail = 'elementwise'
    rows = 1
    N = 4

    def __init__(self) -> None:
        super().__init__()
        n = self.rows if self.tail == 'transpose' else self.N
        self.w = brainstate.ParamState(_randn(N_IN, n, seed=1))
        self.h = brainstate.HiddenState(jnp.zeros((self.rows, n)))

    def update(self, x: jax.Array) -> jax.Array:
        y = braintrace.matmul(x, self.w.value)
        y = {
            'elementwise': lambda: jnp.tanh(y),                                  # control
            'roll': lambda: jnp.roll(y, 1, axis=-1),
            'reverse': lambda: y[..., ::-1],
            'normalise': lambda: y / jnp.sqrt((y ** 2).sum(-1, keepdims=True) + 1.),
            'softmax': lambda: jax.nn.softmax(y, axis=-1),
            'transpose': lambda: y.T,                                            # square (rows, rows)
        }[self.tail]()
        self.h.value = 0.9 * self.h.value + y
        return self.h.value


_MASK = (np.arange(N_IN * 4).reshape(N_IN, 4) % 3 != 0).astype(np.float64)


class WeightTransformNet(brainstate.nn.Module):
    """#235: the weight is transformed by plain ops before it reaches the ETP op."""

    kind = 'raw'
    N = 4

    def __init__(self) -> None:
        super().__init__()
        self.p = brainstate.ParamState(_randn(N_IN, self.N, seed=1))
        self.h = brainstate.HiddenState(jnp.zeros((1, self.N)))

    def update(self, x: jax.Array) -> jax.Array:
        p = self.p.value
        mask = _MASK.astype(p.dtype)  # a concrete constant: weight_fn must not close over a tracer
        if self.kind == 'raw':                                   # control
            y = braintrace.matmul(x, p)
        elif self.kind == 'mask_as_weight_fn':                   # control: transform inside the op
            y = braintrace.matmul(x, p, weight_fn=lambda w: w * mask)
        else:
            w = {
                '2p': lambda: 2.0 * p,
                'tanh': lambda: jnp.tanh(p),
                'mask': lambda: p * mask,
                'abs': lambda: jnp.abs(p),
            }[self.kind]()
            y = braintrace.matmul(x, w)
        self.h.value = 0.9 * self.h.value + y
        return self.h.value


class TwoSourceNet(brainstate.nn.Module):
    """#236: one trainable ETP input is computed from two parameter sources."""

    kind = 'one'
    N = 4

    def __init__(self) -> None:
        super().__init__()
        if self.kind == 'two_leaves':
            self.p = brainstate.ParamState({'a': _randn(N_IN, self.N, seed=1), 'b': _randn(N_IN, self.N, seed=2)})
        else:
            self.p1 = brainstate.ParamState(_randn(N_IN, self.N, seed=1))
            if self.kind != 'one':
                self.p2 = brainstate.ParamState(_randn(N_IN, self.N, seed=2))
        self.h = brainstate.HiddenState(jnp.zeros((1, self.N)))

    def update(self, x: jax.Array) -> jax.Array:
        w = {
            'one': lambda: self.p1.value,                                   # control
            'two_leaves': lambda: self.p.value['a'] + self.p.value['b'],
            'two_params': lambda: self.p1.value + self.p2.value,
            'product': lambda: self.p1.value * jnp.tanh(self.p2.value),
        }[self.kind]()
        self.h.value = 0.9 * self.h.value + braintrace.matmul(x, w)
        return self.h.value


class DropNet(brainstate.nn.Module):
    """#237: a relation is removed by the shape filter, or cut at a second ETP op."""

    kind = 'transposed'

    def __init__(self) -> None:
        super().__init__()
        if self.kind == 'transposed':
            self.W = brainstate.ParamState(_randn(N_IN, 2, seed=1))
            # a second weight with a kept relation, so that compile does not refuse
            # the model for having nothing to train online
            self.U = brainstate.ParamState(_randn(N_IN, 4, seed=2))
            self.V = brainstate.HiddenState(jnp.zeros((2, 4)))
        elif self.kind == 'reshaped':
            self.w = brainstate.ParamState(_randn(N_IN, 4, seed=1))
            self.h1 = brainstate.HiddenState(jnp.zeros((1, 4)))
            self.h2 = brainstate.HiddenState(jnp.zeros((1, 2, 2)))
        elif self.kind == 'partial':
            self.w = brainstate.ParamState(_randn(N_IN, 4, seed=1))
            self.w2 = brainstate.ParamState(_randn(4, 4, seed=2))
            self.h1 = brainstate.HiddenState(jnp.zeros((1, 4)))
        else:  # 'direct', the control
            self.w = brainstate.ParamState(_randn(N_IN, 4, seed=1))
            self.h1 = brainstate.HiddenState(jnp.zeros((1, 4)))

    def update(self, x: jax.Array) -> jax.Array:  # x: (4, N_IN)
        if self.kind == 'transposed':  # y (4, 2) written transposed into V (2, 4)
            y = braintrace.matmul(x, self.W.value)
            self.V.value = 0.9 * self.V.value + jnp.tanh(y.T) + braintrace.matmul(x[:2], self.U.value)
            return self.V.value
        y = braintrace.matmul(x[:1], self.w.value)
        if self.kind == 'reshaped':  # y drives h1 directly and h2 through a reshape
            self.h1.value = 0.9 * self.h1.value + y
            self.h2.value = 0.8 * self.h2.value + jnp.reshape(y, (1, 2, 2))
            return jnp.concatenate([self.h1.value, self.h2.value.reshape(1, 4)], -1)
        if self.kind == 'partial':  # y reaches h1 directly and through a second ETP op
            y = y + braintrace.matmul(jnp.tanh(y), self.w2.value)
        self.h1.value = 0.9 * self.h1.value + y
        return self.h1.value


class CreditNet(brainstate.nn.Module):
    """#238: the ETP op sits in a substep scan and the output reads only last step's value.

    The loss reaches ``w`` only through an old hidden value, so the online
    gradient is zero. With 4 substeps the scan is unrolled and the
    online-credit check records it; with 20 it is descended and not checked.
    """

    substeps = 20
    N = 4

    def __init__(self) -> None:
        super().__init__()
        self.w = brainstate.ParamState(_randn(N_IN, self.N, seed=1))
        self.V = brainstate.HiddenState(jnp.zeros((1, self.N)))
        self.ring = brainstate.ShortTermState(jnp.zeros((1, self.N)))  # a one-step delay line

    def update(self, x: jax.Array) -> jax.Array:
        w = self.w.value

        def sub(V: jax.Array, _: Any) -> Tuple[jax.Array, None]:
            return 0.95 * V + 0.05 * braintrace.matmul(x, w), None

        V, _ = jax.lax.scan(sub, self.V.value, None, length=self.substeps)
        self.V.value = V
        out = jnp.tanh(self.ring.value)
        self.ring.value = V
        return out


class NewReadNet(brainstate.nn.Module):
    """#239: a hidden state's update reads another hidden state's new value."""

    order = 'old'
    N = 5

    def __init__(self) -> None:
        super().__init__()
        self.w = brainstate.ParamState(_randn(N_IN, self.N, seed=1))
        self.g = brainstate.HiddenState(0.5 * jnp.ones((1, self.N)))
        self.V = brainstate.HiddenState(jnp.zeros((1, self.N)))

    def update(self, x: jax.Array) -> jax.Array:
        g_old = self.g.value
        self.g.value = 0.9 * g_old + braintrace.matmul(x, self.w.value)
        g_read = self.g.value if self.order == 'new' else g_old  # 'old' is the control
        self.V.value = 0.95 * self.V.value + jnp.tanh(g_read)
        return self.V.value


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PreconditionCase:
    """One probe: a model, its check, and the issue it reproduces.

    Attributes
    ----------
    name : str
        Test id.
    model : type
        A ``brainstate.nn.Module`` subclass with a no-argument constructor.
    check : str
        One of ``EXACT``, ``EXACT_OR_ERROR``, ``EXACT_OR_LOUD``,
        ``TRANSITION``, ``CREDIT_RECORDED``.
    issue : int or None
        The braintrace issue; ``None`` for a control.
    rows : int
        Leading axis of each input step.
    steps : int
        Window length.
    """

    name: str
    model: type
    check: str
    issue: Optional[int] = None
    rows: int = 1
    steps: int = 8

    @property
    def control(self) -> bool:
        return self.issue is None

    def inputs(self) -> jax.Array:
        """``(steps, rows, N_IN)`` inputs in ``[0, 1)``, seeded."""
        with brainstate.random.seed_context(7):
            return brainstate.random.rand(self.steps, self.rows, N_IN)


def _variant(base: type, **attrs: Any) -> type:
    name = base.__name__ + ''.join(f'_{v}' for v in attrs.values())
    return type(name, (base,), dict(attrs))


def _cases() -> Tuple[PreconditionCase, ...]:
    c = PreconditionCase
    cases = [
        c('transition-none', _variant(TransitionNet, coupling='none'), TRANSITION),
        c('transition-roll', _variant(TransitionNet, coupling='roll'), TRANSITION, 233),
        c('transition-mean', _variant(TransitionNet, coupling='mean'), TRANSITION, 233),
        c('transition-reverse', _variant(TransitionNet, coupling='reverse'), TRANSITION, 233),
        c('tail-elementwise', _variant(TailNet, tail='elementwise'), EXACT),
        c('tail-roll', _variant(TailNet, tail='roll'), EXACT_OR_ERROR, 234),
        c('tail-reverse', _variant(TailNet, tail='reverse'), EXACT_OR_ERROR, 234),
        c('tail-transpose', _variant(TailNet, tail='transpose', rows=3), EXACT_OR_ERROR, 234, rows=3),
        c('tail-normalise', _variant(TailNet, tail='normalise'), EXACT_OR_LOUD, 234),
        c('tail-softmax', _variant(TailNet, tail='softmax'), EXACT_OR_LOUD, 234),
        c('weight-raw', _variant(WeightTransformNet, kind='raw'), EXACT),
        c('weight-mask-as-weight-fn', _variant(WeightTransformNet, kind='mask_as_weight_fn'), EXACT),
        c('weight-2p', _variant(WeightTransformNet, kind='2p'), EXACT_OR_ERROR, 235),
        c('weight-tanh', _variant(WeightTransformNet, kind='tanh'), EXACT_OR_ERROR, 235),
        c('weight-mask', _variant(WeightTransformNet, kind='mask'), EXACT_OR_ERROR, 235),
        c('weight-abs', _variant(WeightTransformNet, kind='abs'), EXACT_OR_ERROR, 235),
        c('source-one', _variant(TwoSourceNet, kind='one'), EXACT),
        c('source-two-leaves', _variant(TwoSourceNet, kind='two_leaves'), EXACT_OR_ERROR, 236),
        c('source-two-params', _variant(TwoSourceNet, kind='two_params'), EXACT_OR_ERROR, 236),
        c('source-product', _variant(TwoSourceNet, kind='product'), EXACT_OR_ERROR, 236),
        c('drop-direct', _variant(DropNet, kind='direct'), EXACT, rows=4),
        c('drop-transposed', _variant(DropNet, kind='transposed'), EXACT_OR_ERROR, 237, rows=4),
        c('drop-reshaped', _variant(DropNet, kind='reshaped'), EXACT_OR_ERROR, 237, rows=4),
        c('drop-partial', _variant(DropNet, kind='partial'), EXACT_OR_ERROR, 237, rows=4),
        c('credit-4-substeps', _variant(CreditNet, substeps=4), CREDIT_RECORDED),
        c('credit-20-substeps', _variant(CreditNet, substeps=20), CREDIT_RECORDED, 238),
        c('newread-old', _variant(NewReadNet, order='old'), EXACT),
        c('newread-new', _variant(NewReadNet, order='new'), EXACT_OR_ERROR, 239),
    ]
    return tuple(cases)


#: Every probe, controls included.
CASES: Tuple[PreconditionCase, ...] = _cases()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Outcome:
    """What one run of a case showed.

    Attributes
    ----------
    raised : BaseException or None
        The exception compiling or running raised, if any.
    errors : dict
        ``{leaf: relative error of the online gradient against BPTT}``.
    records : tuple
        ``(kind, level)`` of every compilation record, including those the
        algorithm added at run time.
    deviation : float
        Largest DIAG deviation measured while running (see
        :mod:`braintrace._testing.census`).
    """

    raised: Optional[BaseException] = None
    errors: Dict[str, float] = dataclasses.field(default_factory=dict)
    records: Tuple[Tuple[DiagnosticKind, DiagnosticLevel], ...] = ()
    deviation: float = 0.0

    @property
    def exact(self) -> bool:
        return self.raised is None and bool(self.errors) and max(self.errors.values()) <= RTOL

    def has(self, *levels: DiagnosticLevel) -> bool:
        return any(level in levels for _, level in self.records)


def _flat(tree: Any) -> Dict[str, np.ndarray]:
    return {jax.tree_util.keystr(k): np.asarray(v) for k, v in jax.tree_util.tree_flatten_with_path(tree)[0]}


def relative_errors(online: Any, reference: Any) -> Dict[str, float]:
    """``{leaf: |online - reference| / |reference|}`` (absolute when the reference is 0)."""
    got, ref = _flat(online), _flat(reference)
    out = {}
    for key, r in ref.items():
        d = float(np.linalg.norm(got[key] - r))
        n = float(np.linalg.norm(r))
        out[key] = d / n if n > 0 else d
    return out


def online_gradients(learner: Any, model: brainstate.nn.Module, xs: jax.Array) -> Any:
    """Sum over steps of ``grad((learner(x_t) ** 2).sum())``, one step per call."""
    params = model.states(brainstate.ParamState)

    def step(x: jax.Array) -> Any:
        return brainstate.transform.grad(lambda: (learner(x) ** 2).sum(), params)()

    grads = brainstate.transform.for_loop(step, xs)
    return jax.tree.map(lambda g: g.sum(0), grads)


def evaluate(case: PreconditionCase, algorithm: str = 'd_rtrl', **options: Any) -> Outcome:
    """Compile ``case`` with :func:`braintrace.compile` and compare with BPTT, in float64.

    Parameters
    ----------
    case : PreconditionCase
        The probe.
    algorithm : str
        Passed to :func:`braintrace.compile`.
    **options
        Passed to :func:`braintrace.compile` (e.g. ``vjp_method``).

    Returns
    -------
    Outcome
    """
    outcome = Outcome()
    with brainstate.environ.context(precision=64), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = case.inputs()
        reference = bptt_param_gradients(case.model, xs)
        learner = None
        with census.measuring(case.name) as measured:
            try:
                model = case.model()
                learner = braintrace.compile(model, algorithm, xs[0], **options)
                online = online_gradients(learner, model, xs)
                outcome.errors = relative_errors(online, reference)
            except Exception as e:  # noqa: BLE001 -- a raise is one of the accepted outcomes
                outcome.raised = e
        outcome.deviation = measured.max('DIAG')
        if learner is not None:
            outcome.records = tuple((r.kind, r.level) for r in learner.graph.diagnostics)
    return outcome


def satisfied(case: PreconditionCase, outcome: Outcome) -> bool:
    """Whether ``outcome`` keeps the promise ``case.check`` states.

    A control must keep it without raising and, unless the record is what
    it checks for (``CREDIT_RECORDED``), without an ERROR record.
    """
    credit = (DiagnosticKind.ONLINE_GRADIENT_ZERO, DiagnosticKind.ONLINE_GRADIENT_PARTIAL)
    if case.check == CREDIT_RECORDED:
        if outcome.raised is not None:
            return not case.control
        return any(kind in credit for kind, _ in outcome.records)
    error = outcome.has(DiagnosticLevel.ERROR)
    if outcome.raised is not None or error:
        return not case.control and case.check != EXACT
    if case.check in (EXACT, EXACT_OR_ERROR):
        return outcome.exact
    if case.check == EXACT_OR_LOUD:
        return outcome.exact or outcome.has(DiagnosticLevel.WARNING)
    if case.check == TRANSITION:
        return outcome.deviation <= RTOL
    raise ValueError(f'unknown check {case.check!r}')


def describe(outcome: Outcome) -> str:
    """One line for a test failure message."""
    if outcome.raised is not None:
        return f'raised {type(outcome.raised).__name__}: {str(outcome.raised)[:200]}'
    loud = sorted({f'{k.value}:{lv.name}' for k, lv in outcome.records if lv is not DiagnosticLevel.INFO})
    errs = ', '.join(f'{k} {v:.3g}' for k, v in outcome.errors.items())
    return f'rel. error vs BPTT: {errs}; DIAG deviation {outcome.deviation:.3g}; records >= WARNING: {loud}'
