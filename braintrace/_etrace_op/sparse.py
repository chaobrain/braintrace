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

r"""Sparse-matmul ETP primitives.

``etp_sp_mm_p`` (batched) and ``etp_sp_mv_p`` (unbatched). The sparse
structure is supplied as a static parameter (``sparse_mat``); only the
non-zero values flow through the primitive as the ``weight_data`` invar.
The structure object must implement ``with_data`` (substitute new data
into the structure) and ``yw_to_w_transposed`` (apply the transposed
sparse pattern to a trace).

**Dict rule API (N-trainable-input refactor)**

Both primitives declare ``trainable_invars_fn``, which returns
``{'weight': 1}`` when ``has_bias=False`` and ``{'weight': 1, 'bias': 2}``
when ``has_bias=True``. The four ETP rules accept / return
``Dict[str, Array]`` instead of bare arrays so the executor can route
gradients to *both* weight and bias ``ParamState`` objects in one pass.

When ``has_bias=False`` the ``'bias'`` key is simply absent from every
dict, so the legacy (no-bias) code path is unchanged in behaviour.
"""

import jax
import jax.numpy as jnp
import saiunit as u

from ._spec import ETPPrimitiveSpec, register_primitive_spec

__all__ = [
    'etp_sp_mm_p',
    'etp_sp_mv_p',
    'sparse_matmul',
]


def _etp_sp_matmul_impl(*args, sparse_mat=None, has_bias=False):
    x, weight_data = args[0], args[1]
    w = sparse_mat.with_data(weight_data)
    y = x @ w
    if has_bias:
        y = y + args[2]
    return y


# ---------------------------------------------------------------------------
# trainable_invars_fn — shared by both mm and mv
# ---------------------------------------------------------------------------

def _sp_trainable_invars(params):
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


# ---------------------------------------------------------------------------
# etp_sp_mm_p — batched
# ---------------------------------------------------------------------------

def _sp_mm_yw_to_w(hidden_dim, trace, *, sparse_mat=None, has_bias=False):
    r"""Batched: propagate hidden-state Jacobian through the sparse trace.

    ``trace['weight']`` has shape ``(batch, nnz, n_state)`` (parameter-dim)
    or ``(nnz, n_state)`` (after vmap removes the batch dim in the gradient
    computation path).  ``sparse_mat.yw_to_w_transposed`` applies the
    transposed sparse pattern.

    ``trace['bias']`` has shape ``(batch, out, n_state)`` and the bias
    gradient is simply the elementwise product with ``hidden_dim``.
    """
    out = {'weight': sparse_mat.yw_to_w_transposed(hidden_dim, trace['weight'])}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _sp_xy_to_dw(x, hidden_dim, weights, *, sparse_mat=None, has_bias=False):
    r"""VJP of ``y = x @ sparse_mat.with_data(w) (+ b)`` in one pass.

    Returns a ``Dict[str, Array]`` with keys ``'weight'`` (and ``'bias'``
    when ``has_bias=True``).
    """
    def _fwd(w_dict):
        y = x @ sparse_mat.with_data(w_dict['weight'])
        if has_bias:
            y = y + w_dict['bias']
        return u.get_mantissa(y)

    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _sp_mm_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    """Return zero-filled ``Dict[str, Array]`` for the D-RTRL parameter-dim trace.

    The bias trace has shape ``(batch, out, n_state)`` where ``out`` is taken
    from ``y_var.aval.shape[1]`` (the output feature dimension, stripping
    the batch axis).
    """
    batch = x_var.aval.shape[0]
    nnz = weight_vars['weight'].aval.shape[0]
    out = {'weight': jnp.zeros((batch, nnz, num_hidden_state))}
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (batch, *weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _sp_mm_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


# ---------------------------------------------------------------------------
# etp_sp_mv_p — unbatched
# ---------------------------------------------------------------------------

def _sp_mv_yw_to_w(hidden_dim, trace, *, sparse_mat=None, has_bias=False):
    r"""Unbatched: propagate hidden-state Jacobian through the sparse trace."""
    out = {'weight': sparse_mat.yw_to_w_transposed(hidden_dim, trace['weight'])}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _sp_mv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    """Return zero-filled ``Dict[str, Array]`` for the D-RTRL parameter-dim trace."""
    nnz = weight_vars['weight'].aval.shape[0]
    out = {'weight': jnp.zeros((nnz, num_hidden_state))}
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (*weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _sp_mv_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


# ---------------------------------------------------------------------------
# Primitive registration
# ---------------------------------------------------------------------------

etp_sp_mm_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_sp_mm',
        impl=_etp_sp_matmul_impl,
        yw_to_w=_sp_mm_yw_to_w,
        xy_to_dw=_sp_xy_to_dw,
        init_drtrl=_sp_mm_init_drtrl,
        init_pp=_sp_mm_init_pp,
        trainable_invars_fn=_sp_trainable_invars,
        x_invar_index=0,
        batched=True,
    )
)

etp_sp_mv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_sp_mv',
        impl=_etp_sp_matmul_impl,
        yw_to_w=_sp_mv_yw_to_w,
        xy_to_dw=_sp_xy_to_dw,
        init_drtrl=_sp_mv_init_drtrl,
        init_pp=_sp_mv_init_pp,
        trainable_invars_fn=_sp_trainable_invars,
        x_invar_index=0,
        batched=False,
    )
)


def sparse_matmul(x, weight_data, *, sparse_mat, bias=None):
    r"""ETP-aware sparse matrix multiplication.

    Computes :math:`y = x \mathbin{@} \mathrm{sparse}(w) \; (+ b)`.

    Auto-dispatches batched/unbatched based on ``x.ndim``.

    Args:
        x: Input array.
        weight_data: Sparse-matrix data (non-zero values).
        sparse_mat: The sparse-matrix structure (e.g. a
            ``saiunit.sparse`` matrix object) — must expose ``with_data``
            and ``yw_to_w_transposed``.
        bias: Optional bias.

    Returns:
        Output array.
    """
    p = etp_sp_mm_p if x.ndim >= 2 else etp_sp_mv_p
    x_v, x_u = u.split_mantissa_unit(x)
    w_v, w_u = u.split_mantissa_unit(weight_data)
    unit = x_u * w_u
    if bias is not None:
        bias_v = u.Quantity(bias).to_decimal(unit)
        r = p.bind(x_v, w_v, bias_v, sparse_mat=sparse_mat, has_bias=True)
    else:
        r = p.bind(x_v, w_v, sparse_mat=sparse_mat, has_bias=False)
    return u.maybe_decimal(r * x_u * w_u)
