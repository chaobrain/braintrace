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

r"""Embedding (trainable gather) ETP primitives.

``etp_emb_p`` is the batched primitive (``indices`` shape ``(batch,)``);
``etp_emb_v_p`` is the unbatched primitive (scalar index). The lookup

.. math::

    y_{b d} = T_{\mathrm{idx}_b, d}

has :math:`\partial y_{bd} / \partial T_{v d'} = \delta_{dd'}\,
\delta_{v,\mathrm{idx}_b}` — diagonal in the feature axis, one-hot in the
vocabulary axis. ``yw_to_w`` is therefore the dense broadcast over the
vocabulary axis, and ``xy_to_dw`` is the scatter-add VJP of the gather.

The indices are the primitive's ``x`` input and are **never
differentiated** (their tangent space is trivial).

.. warning::

    The D-RTRL weight trace has shape ``(B, V, D, n_state)`` — it scales
    with the vocabulary size ``V``. For large vocabularies prefer the
    output-shaped pp-prop trace (``(B, D, n_state)``).

No closed-form fast path is registered: the instant kernel would
materialize the same one-hot outer product as the generic rule path, so
there is nothing to gain (possible follow-up: an index-sparse trace).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import brainunit as u

from ._primitive import register_primitive
from ._registries import register_batched_counterpart
from braintrace._typing import ArrayLike, WeightFn

__all__ = [
    'etp_emb_p',
    'etp_emb_v_p',
    'embedding',
]


def _etp_embedding_impl(indices: Any, weight: Any, *,
                        weight_fn: WeightFn | None = None) -> Any:
    w = weight if weight_fn is None else weight_fn(weight)
    return jnp.take(w, indices, axis=0)


def _emb_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    """The table is the single trainable input (invar 1; indices are invar 0)."""
    return {'weight': 1}


etp_emb_p = register_primitive(
    'etp_emb',
    _etp_embedding_impl,
    batched=True,
    trainable_invars_fn=_emb_trainable_invars,
    x_invar_index=0,
)

etp_emb_v_p = register_primitive(
    'etp_emb_v',
    _etp_embedding_impl,
    batched=False,
    trainable_invars_fn=_emb_trainable_invars,
    x_invar_index=0,
)
register_batched_counterpart(etp_emb_v_p, etp_emb_p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embedding(
    indices: ArrayLike,
    weight: ArrayLike,
    *,
    weight_fn: WeightFn | None = None,
) -> ArrayLike:
    r"""ETP-aware embedding lookup (trainable gather).

    Computes ``y = weight_fn(weight)[indices]``, routed through an ETP
    primitive so the table participates in eligibility-trace computation.
    Auto-dispatches on the index rank: ``etp_emb_p`` for a ``(batch,)``
    index vector, ``etp_emb_v_p`` for a scalar index.

    Parameters
    ----------
    indices : ArrayLike
        Integer token indices, scalar ``()`` or rank-1 ``(batch,)``. The
        indices are never differentiated.
    weight : ArrayLike
        The embedding table, of shape ``(num_embeddings, features)``. May be
        a :class:`brainunit.Quantity`; the unit is split off, the lookup is
        computed on the mantissa, and the unit is reattached to the result.
    weight_fn : Callable, optional
        Element-wise transform applied to the table *inside* the primitive
        before the lookup (e.g. a mask or normalization). Its Jacobian is
        composed automatically in the weight-gradient rule.

    Returns
    -------
    ArrayLike
        The gathered rows, of shape ``(features,)`` for a scalar index or
        ``(batch, features)`` for a ``(batch,)`` index vector.

    Raises
    ------
    TypeError
        If ``indices`` is not of integer dtype.
    NotImplementedError
        If ``indices.ndim >= 2``. Traces are defined per-step,
        per-batch-element; embed one step at a time or flatten the leading
        axes outside the op.
    ValueError
        If ``weight`` is not a rank-2 ``(num_embeddings, features)`` matrix.

    See Also
    --------
    matmul : ETP-aware dense matrix multiplication.
    grouped_matmul : ETP-aware block-diagonal (grouped) matrix multiplication.

    Notes
    -----
    The D-RTRL eligibility trace for the table has shape
    ``(batch, num_embeddings, features, n_state)`` — it scales linearly with
    the vocabulary size. For large vocabularies prefer :func:`braintrace.pp_prop`
    (ES-D-RTRL), whose trace is output-shaped
    (``(batch, features, n_state)``) and independent of the vocabulary size.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> table = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        >>> tokens = jnp.array([0, 2], dtype=jnp.int32)
        >>> braintrace.embedding(tokens, table)
        Array([[0., 1., 2.],
               [6., 7., 8.]], dtype=float32)
        >>>
        >>> import brainunit as u
        >>> y = braintrace.embedding(tokens, table * u.mV)
        >>> y.unit
        mvolt
    """
    indices = jnp.asarray(indices)
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise TypeError(
            f'embedding indices must be integers; got dtype {indices.dtype}.'
        )
    if indices.ndim > 1:
        raise NotImplementedError(
            'braintrace.embedding supports scalar or rank-1 (batch,) indices; '
            f'got indices.ndim={indices.ndim}. Embed one step at a time or '
            'flatten the leading axes outside the op.'
        )
    if getattr(weight, 'ndim', None) != 2:
        raise ValueError(
            'embedding weight must be a (num_embeddings, features) matrix; '
            f'got ndim={getattr(weight, "ndim", None)}.'
        )
    w_v, w_u = u.split_mantissa_unit(weight)
    p = etp_emb_p if indices.ndim == 1 else etp_emb_v_p
    r = p.bind(indices, w_v, weight_fn=weight_fn)
    return u.maybe_decimal(r * w_u)
