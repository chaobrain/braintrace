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

r"""General linear-in-weight contraction ETP primitive.

``etp_einsum_p`` marks a two-operand ``jnp.einsum`` whose second operand is
the trainable weight. The ETP rules are derived mechanically from the
equation by classifying each weight/output axis letter:

* **diagonal** — in both ``w_spec`` and ``y_spec``: ``hidden_dim``
  broadcasts along the trace on these axes (dense-style).
* **contracted** — in ``w_spec`` only (consumed by ``x``): free trace axes.
* **shared** — in ``y_spec`` only (weight reused across them):
  ``hidden_dim`` is summed over them before broadcasting.

The batched equation form is required: the leading ``x``/``y`` letter is the
batch axis and must not appear in ``w_spec``.

Shared-axis equations are currently rejected at the user API
(:data:`_SHARED_AXES_SUPPORTED` is ``False``): the sum-then-broadcast
treatment is the scheme conv used for spatial axes *before* the ETP audit
established it was inexact and rewrote conv with a per-position kernel
trace (``ETP_RULES_INSTANT_DRTRL`` / ``ETP_RULES_SOLVE_DRTRL``). The gate
opens only if the BPTT oracle proves exactness for einsum's case — see the
phase-3 plan, Task 6.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import brainunit as u

from ._primitive import register_primitive
from braintrace._typing import ArrayLike, WeightFn

__all__ = [
    'etp_einsum_p',
    'einsum',
]

_SHARED_AXES_SUPPORTED = False


class EinsumSpec(NamedTuple):
    """Parsed, classified form of an ETP einsum equation."""
    x_spec: str
    w_spec: str
    y_spec: str
    batch: str
    diagonal: str
    contracted: str
    shared: str


def parse_etp_einsum(equation: str) -> EinsumSpec:
    """Parse and validate an ETP einsum equation (see module docstring).

    Parameters
    ----------
    equation : str
        A two-operand explicit einsum equation ``'x_spec,w_spec->y_spec'``.

    Returns
    -------
    EinsumSpec
        The normalized specs plus the batch letter and the
        diagonal/contracted/shared axis classification.

    Raises
    ------
    ValueError
        On any violation of the v1 equation restrictions.
    """
    eq = equation.replace(' ', '')
    if '->' not in eq:
        raise ValueError(f"equation must be explicit ('lhs->rhs'): {equation!r}")
    lhs, y_spec = eq.split('->', 1)
    operands = lhs.split(',')
    if len(operands) != 2:
        raise ValueError(
            f'exactly two operands (x, weight) are required: {equation!r}')
    x_spec, w_spec = operands
    for name, s in (('x', x_spec), ('weight', w_spec), ('output', y_spec)):
        if not s or not (s.isalpha() and s.islower()):
            raise ValueError(
                f'{name} spec must be non-empty lowercase letters '
                f'(no ellipsis/digits): {equation!r}')
        if len(set(s)) != len(s):
            raise ValueError(f'repeated axis letter in {name} spec: {equation!r}')
    if x_spec[0] != y_spec[0]:
        raise ValueError(
            'batched form required: x and output must share the leading '
            f'batch letter: {equation!r}')
    batch = x_spec[0]
    if batch in w_spec:
        raise ValueError(
            f'batch axis {batch!r} must not appear in the weight spec: {equation!r}')
    unknown = set(y_spec) - set(x_spec) - set(w_spec)
    if unknown:
        raise ValueError(
            f'output letters {sorted(unknown)} appear in no input: {equation!r}')
    diagonal = ''.join(c for c in w_spec if c in y_spec)
    contracted = ''.join(c for c in w_spec if c not in y_spec)
    missing = [c for c in contracted if c not in x_spec]
    if missing:
        raise ValueError(
            f'weight letters {missing} appear in neither x nor output: {equation!r}')
    shared = ''.join(c for c in y_spec if c not in w_spec and c != batch)
    missing = [c for c in shared if c not in x_spec]
    if missing:
        raise ValueError(
            f'output letters {missing} are not driven by x: {equation!r}')
    return EinsumSpec(x_spec, w_spec, y_spec, batch, diagonal, contracted, shared)


def _etp_einsum_impl(x: Any, w: Any, *, equation: str,
                     weight_fn: WeightFn | None = None) -> Any:
    ww = w if weight_fn is None else weight_fn(w)
    return jnp.einsum(equation, x, ww)


def _einsum_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    """The weight is always invar 1 (x is invar 0)."""
    return {'weight': 1}


etp_einsum_p = register_primitive(
    'etp_einsum',
    _etp_einsum_impl,
    batched=True,
    trainable_invars_fn=_einsum_trainable_invars,
    x_invar_index=0,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def einsum(
    equation: str,
    x: ArrayLike,
    weight: ArrayLike,
    *,
    weight_fn: WeightFn | None = None,
) -> ArrayLike:
    r"""ETP-aware two-operand einsum, linear in the trainable weight.

    Computes ``jnp.einsum(equation, x, weight_fn(weight))`` through an ETP
    primitive whose trace rules are derived mechanically from the equation's
    axis classification (see the module docstring): weight axes present in
    the output broadcast (*diagonal*), weight axes consumed by ``x`` are free
    trace axes (*contracted*), and output axes absent from the weight are
    *shared* (currently gated off, see Raises).

    Parameters
    ----------
    equation : str
        A two-operand explicit einsum equation ``'x_spec,w_spec->y_spec'``
        in **batched form**: the leading letter of ``x_spec`` must equal the
        leading letter of ``y_spec`` and must not appear in ``w_spec``.
        v1 restrictions: lowercase letters only, no ellipsis, no repeated
        letter within one spec, every output letter present in some input,
        every weight letter present in ``x`` or the output. Spaces are
        stripped before binding.
    x : ArrayLike
        The non-trainable operand, of rank ``len(x_spec)``.
    weight : ArrayLike
        The trainable operand, of rank ``len(w_spec)``. May be a
        :class:`brainunit.Quantity`; the units of ``x`` and ``weight``
        multiply into the result.
    weight_fn : Callable, optional
        Element-wise transform applied to the weight *inside* the primitive
        before the contraction. Its Jacobian is composed automatically in
        the weight-gradient rule.

    Returns
    -------
    ArrayLike
        The contraction result, of rank ``len(y_spec)``.

    Raises
    ------
    ValueError
        If the equation violates the v1 restrictions, or if ``x`` /
        ``weight`` rank does not match the equation.
    NotImplementedError
        If the equation has shared axes (output axes absent from the weight
        spec, e.g. ``'btk,kn->btn'``): their sum-then-broadcast trace
        propagation is not exact (the same defect the ETP audit fixed in
        conv with per-position kernel traces), so they stay gated until a
        per-position trace lands.

    See Also
    --------
    matmul : ETP-aware dense matrix multiplication (rank ≤ 2 inputs).
    grouped_matmul : ETP-aware block-diagonal (grouped) matrix multiplication.

    Notes
    -----
    There is no bias parameter: compose with a plain add (a plain-add
    parameter is deliberately excluded from ETP by the selection principle)
    or use :func:`matmul` / :func:`grouped_matmul`, which carry one.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> x = jnp.ones((5, 3))
        >>> w = jnp.ones((3, 4))
        >>> braintrace.einsum('bk,kn->bn', x, w).shape   # dense matmul
        (5, 4)
        >>>
        >>> xh = jnp.ones((5, 2, 3))
        >>> wh = jnp.ones((2, 3, 4))
        >>> braintrace.einsum('bhd,hde->bhe', xh, wh).shape  # per-head mixing
        (5, 2, 4)
    """
    spec = parse_etp_einsum(equation)
    if spec.shared and not _SHARED_AXES_SUPPORTED:
        raise NotImplementedError(
            f'einsum equations with output axes absent from the weight spec '
            f'(shared axes {spec.shared!r}) are not yet supported for ETP '
            f'online learning: {equation!r}. Supported today: equations whose '
            'non-batch output axes all appear in the weight spec.'
        )
    if getattr(x, 'ndim', None) != len(spec.x_spec):
        raise ValueError(
            f'x has rank {getattr(x, "ndim", None)} but the equation expects '
            f'{len(spec.x_spec)}: {equation!r}')
    if getattr(weight, 'ndim', None) != len(spec.w_spec):
        raise ValueError(
            f'weight has rank {getattr(weight, "ndim", None)} but the equation '
            f'expects {len(spec.w_spec)}: {equation!r}')
    normalized = f'{spec.x_spec},{spec.w_spec}->{spec.y_spec}'
    x_v, x_u = u.split_mantissa_unit(x)
    w_v, w_u = u.split_mantissa_unit(weight)
    r = etp_einsum_p.bind(x_v, w_v, equation=normalized, weight_fn=weight_fn)
    return u.maybe_decimal(r * (x_u * w_u))
