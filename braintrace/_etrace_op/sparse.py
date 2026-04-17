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


def _sp_mm_yw_to_w(hidden_dim, trace, *, sparse_mat=None, has_bias=False):
    return sparse_mat.yw_to_w_transposed(hidden_dim, trace)


def _sp_mv_yw_to_w(hidden_dim, trace, *, sparse_mat=None, has_bias=False):
    return sparse_mat.yw_to_w_transposed(hidden_dim, trace)


def _sp_xy_to_dw(x, hidden_dim, w, *, sparse_mat=None, has_bias=False):
    _, vjp_fn = jax.vjp(lambda w_: u.get_mantissa(x @ sparse_mat.with_data(w_)), w)
    return u.get_mantissa(vjp_fn(hidden_dim)[0])


def _sp_mm_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    batch = x_var.aval.shape[0]
    nnz = weight_var.aval.shape[0]
    return jnp.zeros((batch, nnz, num_hidden_state))


def _sp_mm_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _sp_mv_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    nnz = weight_var.aval.shape[0]
    return jnp.zeros((nnz, num_hidden_state))


def _sp_mv_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_sp_mm_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_sp_mm',
        impl=_etp_sp_matmul_impl,
        yw_to_w=_sp_mm_yw_to_w,
        xy_to_dw=_sp_xy_to_dw,
        init_drtrl=_sp_mm_init_drtrl,
        init_pp=_sp_mm_init_pp,
        weight_invar_index=1,
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
        weight_invar_index=1,
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
