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
    r"""Legacy eligibility-trace parameter.

    Wraps a pytree weight (typically ``{'weight': ..., 'bias': ...}``) and
    an :class:`ETraceOp`. :meth:`execute` routes through the op's
    ETP-primitive path, so the compiler registers an eligibility-trace
    relation when this parameter flows into a hidden state.

    .. deprecated:: 0.2.0
        Use :class:`brainstate.ParamState` together with the new ETP
        primitive functions (:func:`braintrace.matmul`, etc.) directly.

    Parameters
    ----------
    weight : Any
        The pytree weight to wrap, typically a dict such as
        ``{'weight': ..., 'bias': ...}``.
    op : ETraceOp
        The eligibility-trace operator used to combine inputs with the
        weight.
    grad : object, optional
        Legacy gradient type. Defaults to the adaptive mode; kept for API
        compatibility and has no effect on the new primitive-based
        compiler.
    name : str, optional
        Optional name forwarded to :class:`brainstate.ParamState`.

    Notes
    -----
    This class is a deprecated back-compatibility shim. New code should
    use :class:`brainstate.ParamState` with the ETP primitive functions.
    """
    __module__ = 'braintrace'

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

    def execute(self, x):
        return self.op(x, self.value)


# ---------------------------------------------------------------------------
# ElemWiseParam
# ---------------------------------------------------------------------------

class ElemWiseParam(ETraceParam):
    r"""Legacy element-wise eligibility-trace parameter.

    Element-wise variant of :class:`ETraceParam`: wraps a weight together
    with an :class:`ElemWiseOp` and routes :meth:`execute` through the
    element-wise ETP-primitive path.

    .. deprecated:: 0.2.0
        Use :class:`brainstate.ParamState` together with
        :func:`braintrace.element_wise` directly.

    Parameters
    ----------
    weight : Any
        The pytree weight to wrap.
    op : ElemWiseOp or Callable, optional
        Element-wise operator (or a callable wrapped into one). Defaults to
        the identity ``lambda w: w``.
    name : str, optional
        Optional name forwarded to :class:`brainstate.ParamState`.

    Notes
    -----
    This class is a deprecated back-compatibility shim. New code should
    use :class:`brainstate.ParamState` with :func:`braintrace.element_wise`.
    """
    __module__ = 'braintrace'

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
    r"""Legacy parameter with spatial gradient only and no eligibility trace.

    :meth:`execute` calls the op's plain-JAX path (``raw_xw_to_y``), so no
    ETP primitive appears in the jaxpr for this weight. The compiler
    therefore registers zero eligibility-trace relations for this
    ``ParamState``.

    .. deprecated:: 0.2.0
        Use :class:`brainstate.ParamState` with plain JAX ops (``x @ w``)
        directly.

    Parameters
    ----------
    value : Any
        The pytree weight to wrap.
    op : ETraceOp or Callable
        Operator used in :meth:`execute`. If an :class:`ETraceOp` is given,
        its plain-JAX ``raw_xw_to_y`` path is used so no ETP primitive is
        emitted; otherwise a plain callable is used directly.
    name : str, optional
        Optional name forwarded to :class:`brainstate.ParamState`.

    Notes
    -----
    This class is a deprecated back-compatibility shim. New code should
    use :class:`brainstate.ParamState` with plain JAX ops.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        value,
        op,
        name: Optional[str] = None,
        **_kwargs,
    ):
        super().__init__(value, name=name)
        if isinstance(op, ETraceOp):
            self._etrace_op: Optional[ETraceOp] = op
            self.op = op.raw_xw_to_y
        else:
            if not callable(op):
                raise TypeError(f'op must be callable, got {type(op)}')
            self._etrace_op = None
            self.op = op

    def execute(self, x):
        return self.op(x, self.value)


# ---------------------------------------------------------------------------
# FakeETraceParam — non-ParamState wrapper
# ---------------------------------------------------------------------------

class FakeETraceParam(object):
    r"""Legacy fake parameter that is NOT a ``ParamState``.

    Stores a value and a callable and exposes :meth:`execute`. Because it
    is not a ``ParamState``, the compiler never sees it, so no gradient is
    produced for this weight. The op is routed through its plain-JAX path
    to avoid emitting any ETP primitive (which would otherwise trigger
    :attr:`DiagnosticKind.RELATION_EXCLUDED_NO_PARAMSTATE` warnings).

    .. deprecated:: 0.2.0
        Use :class:`brainstate.FakeState` with plain JAX ops.

    Parameters
    ----------
    value : Any
        The pytree weight to store.
    op : ETraceOp or Callable
        Operator used in :meth:`execute`. An :class:`ETraceOp` is routed
        through its plain-JAX ``raw_xw_to_y`` path; otherwise a plain
        callable is used directly.

    Notes
    -----
    This class is a deprecated back-compatibility shim. New code should
    use :class:`brainstate.FakeState` with plain JAX ops.
    """
    __module__ = 'braintrace'

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

    def execute(self, x):
        return self.op(x, self.value)


# ---------------------------------------------------------------------------
# FakeElemWiseParam — non-ParamState wrapper
# ---------------------------------------------------------------------------

class FakeElemWiseParam(object):
    r"""Legacy fake element-wise parameter that is NOT a ``ParamState``.

    Element-wise variant of :class:`FakeETraceParam`. Because it is not a
    ``ParamState``, the compiler never sees it and produces no gradient for
    this weight.

    .. deprecated:: 0.2.0
        Use :class:`brainstate.FakeState` with plain JAX ops.

    Parameters
    ----------
    weight : Any
        The pytree weight to store.
    op : ElemWiseOp or Callable, optional
        Element-wise operator. An :class:`ElemWiseOp` is routed through its
        plain-JAX path; otherwise a plain callable is used directly.
        Defaults to the identity ``lambda w: w``.
    name : str, optional
        Optional name stored on the instance.

    Notes
    -----
    This class is a deprecated back-compatibility shim. New code should
    use :class:`brainstate.FakeState` with plain JAX ops.
    """
    __module__ = 'braintrace'

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
            self._etrace_op: Optional[ElemWiseOp] = op
            self.op = op.raw_xw_to_y
            self._is_etrace_op = True
        else:
            if not callable(op):
                raise TypeError(f'op must be callable, got {type(op)}')
            self._etrace_op = None
            self.op = op
        self.value = weight
        self.name = name

    def execute(self):
        if self._is_etrace_op:
            return self.op(None, self.value)
        return self.op(self.value)
