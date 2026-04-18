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

"""Legacy (v0.1.x) parameter-state shims.

* :class:`ETraceParam` — ``ParamState`` subclass that routes
  :meth:`execute` through an ETP primitive; picked up by the compiler.
* :class:`ElemWiseParam` — element-wise variant.
* :class:`NonTempParam` — ``ParamState`` subclass that routes through
  *plain* JAX ops (no ETP primitive), so the compiler registers no
  eligibility-trace relation for this weight.
* :class:`FakeETraceParam`, :class:`FakeElemWiseParam` — plain ``object``
  wrappers (NOT ``ParamState``), invisible to the compiler.
"""



import warnings
from enum import Enum
from typing import Optional

import brainstate

from ._ops import ElemWiseOp, ETraceOp

__all__ = [
    'ETraceParam',
    'ElemWiseParam',
    'NonTempParam',
    'FakeETraceParam',
    'FakeElemWiseParam',
]

_warned: set = set()


def _deprecate(cls_name: str, replacement: str) -> None:
    if cls_name in _warned:
        return
    _warned.add(cls_name)
    warnings.warn(
        f'braintrace._legacy.{cls_name} is deprecated; use {replacement} instead.',
        DeprecationWarning,
        stacklevel=3,
    )


class ETraceGrad(str, Enum):
    """Legacy gradient-type enum. Kept for API compatibility; no effect
    on the new primitive-based compiler."""
    full = 'full'
    approx = 'approx'
    adaptive = 'adaptive'


# ---------------------------------------------------------------------------
# ETraceParam
# ---------------------------------------------------------------------------

class ETraceParam(brainstate.ParamState):
    """Legacy eligibility-trace parameter.

    Wraps a pytree weight (typically ``{'weight': ..., 'bias': ...}``)
    and an :class:`ETraceOp`. :meth:`execute` routes through the op's
    ETP-primitive path, so the compiler will register an eligibility-
    trace relation when this parameter flows into a hidden state.

    .. deprecated:: 0.2.0
        Use :class:`brainstate.ParamState` + the new ETP primitive
        functions (:func:`braintrace.matmul` etc.) directly.
    """
    __module__ = 'braintrace._legacy'

    def __init__(
        self,
        weight,
        op: ETraceOp,
        grad: Optional[object] = None,
        name: Optional[str] = None,
    ):
        super().__init__(weight, name=name)
        if not isinstance(op, ETraceOp):
            raise TypeError(f'op must be ETraceOp, got {type(op)}')
        self.op = op
        if grad is None:
            grad = ETraceGrad.adaptive
        elif isinstance(grad, str):
            grad = ETraceGrad(grad)
        self.gradient = grad
        self.is_etrace = True
        _deprecate(
            'ETraceParam',
            'brainstate.ParamState + braintrace.matmul/conv/...',
        )

    def execute(self, x):
        return self.op(x, self.value)


# ---------------------------------------------------------------------------
# ElemWiseParam
# ---------------------------------------------------------------------------

class ElemWiseParam(ETraceParam):
    """Legacy element-wise trace parameter."""
    __module__ = 'braintrace._legacy'

    def __init__(
        self,
        weight,
        op=(lambda w: w),
        name: Optional[str] = None,
    ):
        if not isinstance(op, ElemWiseOp):
            op = ElemWiseOp(op)
        super().__init__(weight, op=op, grad=ETraceGrad.full, name=name)

    def execute(self):  # type: ignore[override]
        return self.op(self.value)


# ---------------------------------------------------------------------------
# NonTempParam
# ---------------------------------------------------------------------------

class NonTempParam(brainstate.ParamState):
    """Legacy parameter — spatial gradient only, no eligibility trace.

    ``execute`` calls the op's plain-JAX path (``raw_xw_to_y``), so no
    ETP primitive appears in the jaxpr for this weight. The compiler
    therefore registers zero relations for this ``ParamState``.

    .. deprecated:: 0.2.0
        Use :class:`brainstate.ParamState` with plain JAX ops (``x @ w``)
        directly.
    """
    __module__ = 'braintrace._legacy'

    def __init__(
        self,
        value,
        op,
        name: Optional[str] = None,
        **_kwargs,
    ):
        super().__init__(value, name=name)
        if isinstance(op, ETraceOp):
            self._etrace_op = op
            self.op = op.raw_xw_to_y
        else:
            if not callable(op):
                raise TypeError(f'op must be callable, got {type(op)}')
            self._etrace_op = None
            self.op = op
        _deprecate(
            'NonTempParam',
            'brainstate.ParamState with plain JAX ops',
        )

    def execute(self, x):
        return self.op(x, self.value)


# ---------------------------------------------------------------------------
# FakeETraceParam — non-ParamState wrapper
# ---------------------------------------------------------------------------

class FakeETraceParam(object):
    """Legacy fake parameter — NOT a ``ParamState``.

    Stores a value + callable and exposes :meth:`execute`. Since this is
    not a ``ParamState``, the compiler never sees it — no gradient will
    be produced for this weight.

    Routes through the op's plain-JAX path to avoid emitting any ETP
    primitive (which would trigger
    :attr:`DiagnosticKind.RELATION_EXCLUDED_NO_PARAMSTATE` warnings).

    .. deprecated:: 0.2.0
        Use :class:`brainstate.FakeState` with plain JAX ops.
    """
    __module__ = 'braintrace._legacy'

    def __init__(self, value, op):
        self.value = value
        if isinstance(op, ETraceOp):
            self._etrace_op = op
            self.op = op.raw_xw_to_y
        else:
            if not callable(op):
                raise TypeError(f'op must be callable, got {type(op)}')
            self._etrace_op = None
            self.op = op
        _deprecate('FakeETraceParam', 'brainstate.FakeState')

    def execute(self, x):
        return self.op(x, self.value)


# ---------------------------------------------------------------------------
# FakeElemWiseParam — non-ParamState wrapper
# ---------------------------------------------------------------------------

class FakeElemWiseParam(object):
    """Legacy fake element-wise parameter — NOT a ``ParamState``.

    .. deprecated:: 0.2.0
        Use :class:`brainstate.FakeState` with plain JAX ops.
    """
    __module__ = 'braintrace._legacy'

    def __init__(
        self,
        weight,
        op=(lambda w: w),
        name: Optional[str] = None,
    ):
        self._is_etrace_op = False
        if isinstance(op, ETraceOp):
            if not isinstance(op, ElemWiseOp):
                raise TypeError(
                    f'op must be ElemWiseOp when an ETraceOp is supplied, got {type(op)}'
                )
            self._etrace_op = op
            self.op = op.raw_xw_to_y
            self._is_etrace_op = True
        else:
            if not callable(op):
                raise TypeError(f'op must be callable, got {type(op)}')
            self._etrace_op = None
            self.op = op
        self.value = weight
        self.name = name
        _deprecate('FakeElemWiseParam', 'brainstate.FakeState')

    def execute(self):
        if self._is_etrace_op:
            return self.op(None, self.value)
        return self.op(self.value)
