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

r"""Grouped (block-diagonal) matmul ETP primitives.

``etp_gmm_p`` is the batched primitive (``x`` shape ``(batch, groups, in)``);
``etp_gmv_p`` is the unbatched primitive (``x`` shape ``(groups, in)``).
Semantically a block-diagonal linear map with ``G`` independent ``K×N``
blocks:

.. math::

    y_{\dots g n} = \sum_k x_{\dots g k} W_{g k n} \; (+ b_{g n})

Each weight entry touches exactly one output component per example
(:math:`\partial y_{\dots gn} / \partial W_{g'kn'} = \delta_{gg'}\delta_{nn'}
x_{\dots gk}`), the same diagonal structure as dense matmul — so ``yw_to_w``
is the dense broadcast generalized by one leading group axis, and the same
closed-form fast path applies.

The D-RTRL weight trace is ``(B, G, K, N, n_state)`` — a ``G×`` reduction
versus the ``(B, GK, GN, n_state)`` trace of an equivalent dense
block-diagonal matrix.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import brainunit as u

from ._primitive import register_primitive
from ._registries import FastPathRules, register_batched_counterpart
from braintrace._typing import ArrayLike, WeightFn

__all__ = [
    'etp_gmm_p',
    'etp_gmv_p',
    'grouped_matmul',
]


def _etp_grouped_matmul_impl(*args: Any, has_bias: bool = False,
                             weight_fn: WeightFn | None = None,
                             bias_fn: WeightFn | None = None) -> Any:
    x, w = args[0], args[1]
    if weight_fn is not None:
        w = weight_fn(w)
    y = jnp.einsum('...gk,gkn->...gn', x, w)
    if has_bias:
        b = args[2]
        if bias_fn is not None:
            b = bias_fn(b)
        y = y + b
    return y


# ---------------------------------------------------------------------------
# etp_gmm_p — batched
# ---------------------------------------------------------------------------

def _gmm_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


etp_gmm_p = register_primitive(
    'etp_gmm',
    _etp_grouped_matmul_impl,
    batched=True,
    trainable_invars_fn=_gmm_trainable_invars,
    x_invar_index=0,
)


# ---------------------------------------------------------------------------
# etp_gmv_p — unbatched
# ---------------------------------------------------------------------------

def _gmv_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


etp_gmv_p = register_primitive(
    'etp_gmv',
    _etp_grouped_matmul_impl,
    batched=False,
    trainable_invars_fn=_gmv_trainable_invars,
    x_invar_index=0,
)
register_batched_counterpart(etp_gmv_p, etp_gmm_p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grouped_matmul(
    x: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike | None = None,
    *,
    weight_fn: WeightFn | None = None,
    bias_fn: WeightFn | None = None,
) -> ArrayLike:
    r"""ETP-aware grouped (block-diagonal) matrix multiplication.

    Computes ``y[..., g, :] = x[..., g, :] @ weight_fn(weight)[g]
    (+ bias_fn(bias)[g])`` — a block-diagonal linear map with ``G``
    independent ``in_features → out_features`` blocks, routed through an ETP
    primitive so ``weight`` (and the optional ``bias``) participate in
    eligibility-trace computation. Auto-dispatches to ``etp_gmm_p``
    (``x.ndim == 3``, batched) or ``etp_gmv_p`` (``x.ndim == 2``, unbatched).

    Parameters
    ----------
    x : ArrayLike
        Input of shape ``(groups, in_features)`` (unbatched) or
        ``(batch, groups, in_features)`` (batched). Any other rank raises
        ``ValueError``; fold extra leading axes into the batch axis first.
    weight : ArrayLike
        Block weights of shape ``(groups, in_features, out_features)``.
    bias : ArrayLike, optional
        Per-block bias of shape ``(groups, out_features)``. Its unit must be
        compatible with ``x.unit * weight.unit``.
    weight_fn : Callable, optional
        Elementwise transform applied to ``weight`` inside the primitive
        (e.g. a sign or non-negativity constraint).
    bias_fn : Callable, optional
        Elementwise transform applied to ``bias`` inside the primitive.

    Returns
    -------
    ArrayLike
        Output of shape ``(groups, out_features)`` or
        ``(batch, groups, out_features)``, carrying the product unit when the
        inputs are :class:`brainunit.Quantity`.

    Raises
    ------
    ValueError
        If ``weight`` is not rank-3, ``x`` is not rank-2/rank-3, or the
        trailing ``(groups, in_features)`` axes of ``x`` do not match the
        leading axes of ``weight``.

    See Also
    --------
    matmul : the dense ETP matrix multiplication.

    Notes
    -----
    A grouped map is equivalent to a dense linear map whose weight matrix is
    ``block_diag(weight[0], ..., weight[G-1])``, but its D-RTRL eligibility
    trace is ``(batch, G, K, N, n_state)`` — a factor-``G`` memory reduction
    over the dense ``(batch, G·K, G·N, n_state)`` trace.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainunit as u
        >>> import braintrace
        >>> x = jnp.ones((16, 4, 8))          # (batch, groups, in)
        >>> w = jnp.ones((4, 8, 8))           # (groups, in, out)
        >>> y = braintrace.grouped_matmul(x, w)
        >>> print(y.shape)
        (16, 4, 8)
        >>>
        >>> # physical-unit quantities are supported
        >>> yq = braintrace.grouped_matmul(x * u.mV, w * u.siemens)
        >>> print(yq.shape)
        (16, 4, 8)
        >>>
        >>> # constrain block weights to be non-negative via a transform
        >>> y2 = braintrace.grouped_matmul(x, w, weight_fn=lambda ww: ww ** 2)
        >>> print(y2.shape)
        (16, 4, 8)
    """
    if getattr(weight, 'ndim', None) != 3:
        raise ValueError(
            'grouped_matmul weight must have shape (groups, in_features, '
            f'out_features); got ndim={getattr(weight, "ndim", None)}.'
        )
    if x.ndim not in (2, 3):  # type: ignore[union-attr]
        raise ValueError(
            'grouped_matmul() supports x.ndim of 2 (groups, in_features) or 3 '
            '(batch, groups, in_features); fold extra leading axes into the '
            f'batch axis first. Got x.ndim={x.ndim}.'  # type: ignore[union-attr]
        )
    if x.shape[-2:] != weight.shape[:2]:  # type: ignore[union-attr]  # rank checks above exclude scalar ArrayLike items
        raise ValueError(
            f'x trailing axes {x.shape[-2:]} must equal weight leading axes '  # type: ignore[union-attr]
            f'{weight.shape[:2]} (groups, in_features).'  # type: ignore[union-attr]
        )
    p = etp_gmm_p if x.ndim == 3 else etp_gmv_p  # type: ignore[union-attr]
    x_v, x_u = u.split_mantissa_unit(x)
    weight_v, weight_u = u.split_mantissa_unit(weight)
    unit = x_u * weight_u
    if bias is not None:
        bias_v = u.Quantity(bias).to_decimal(unit)
        r = p.bind(x_v, weight_v, bias_v, has_bias=True,
                   weight_fn=weight_fn, bias_fn=bias_fn)
    else:
        r = p.bind(x_v, weight_v, has_bias=False,
                   weight_fn=weight_fn, bias_fn=bias_fn)
    return u.maybe_decimal(r * unit)
