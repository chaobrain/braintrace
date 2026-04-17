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

r"""LoRA (Low-Rank Adaptation) ETP primitives.

``etp_lora_mm_p`` (batched) and ``etp_lora_mv_p`` (unbatched) compute
:math:`y = \alpha \cdot x \mathbin{@} B \mathbin{@} A` plus an optional
bias. The trace and gradient state are pytrees with ``B`` and ``A``
leaves; the originating ``ParamState`` holds both factors as a pytree.
"""

import jax
import jax.numpy as jnp
import saiunit as u

from ._spec import ETPPrimitiveSpec, register_primitive_spec

__all__ = [
    'etp_lora_mm_p',
    'etp_lora_mv_p',
    'lora_matmul',
]


def _etp_lora_impl(*args, alpha=1.0, has_bias=False):
    x, B, A = args[0], args[1], args[2]
    y = alpha * (x @ B @ A)
    if has_bias:
        y = y + args[3]
    return y


def _lora_mm_yw_to_w(hidden_dim, trace, *, alpha=1.0, has_bias=False):
    r"""LoRA batched: trace is ``{B: (batch, in, rank, ns), A: (batch, rank, out, ns)}``.

    Only propagate through A (B frozen during trace propagation).
    """
    trace_A = trace['A']
    trace_A = trace_A * jnp.expand_dims(hidden_dim, axis=1)
    return {'B': trace['B'], 'A': trace_A}


def _lora_mv_yw_to_w(hidden_dim, trace, *, alpha=1.0, has_bias=False):
    r"""LoRA unbatched: trace is ``{B: (in, rank, ns), A: (rank, out, ns)}``."""
    trace_A = trace['A']
    trace_A = trace_A * jnp.expand_dims(hidden_dim, axis=0)
    return {'B': trace['B'], 'A': trace_A}


def _lora_xy_to_dw(x, hidden_dim, w_B, w_A, *, alpha=1.0, has_bias=False):
    r"""VJP of ``y = alpha * x @ B @ A``."""

    def _fwd(B, A):
        return u.get_mantissa(alpha * (x @ B @ A))

    _, vjp_fn = jax.vjp(_fwd, w_B, w_A)
    dB, dA = vjp_fn(hidden_dim)
    return {'B': u.get_mantissa(dB), 'A': u.get_mantissa(dA)}


def _lora_mm_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    # weight_var is the B invar of the primitive; A's shape is (rank, out).
    batch = x_var.aval.shape[0]
    B_shape = weight_var.aval.shape
    rank = B_shape[1]
    out = y_var.aval.shape[-1]
    A_shape = (rank, out)
    return {
        'B': jnp.zeros((batch, *B_shape, num_hidden_state)),
        'A': jnp.zeros((batch, *A_shape, num_hidden_state)),
    }


def _lora_mm_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _lora_mv_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    B_shape = weight_var.aval.shape
    rank = B_shape[1]
    out = y_var.aval.shape[-1]
    A_shape = (rank, out)
    return {
        'B': jnp.zeros((*B_shape, num_hidden_state)),
        'A': jnp.zeros((*A_shape, num_hidden_state)),
    }


def _lora_mv_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_lora_mm_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_lora_mm',
        impl=_etp_lora_impl,
        yw_to_w=_lora_mm_yw_to_w,
        xy_to_dw=_lora_xy_to_dw,
        init_drtrl=_lora_mm_init_drtrl,
        init_pp=_lora_mm_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=True,
    )
)

etp_lora_mv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_lora_mv',
        impl=_etp_lora_impl,
        yw_to_w=_lora_mv_yw_to_w,
        xy_to_dw=_lora_xy_to_dw,
        init_drtrl=_lora_mv_init_drtrl,
        init_pp=_lora_mv_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=False,
    )
)


def lora_matmul(x, B, A, *, alpha=1.0, bias=None):
    r"""ETP-aware LoRA (Low-Rank Adaptation) matrix multiplication.

    Computes :math:`y = \alpha \cdot x \mathbin{@} B \mathbin{@} A \; (+ b)`.

    Auto-dispatches batched/unbatched based on ``x.ndim``.

    Args:
        x: Input array.
        B: Low-rank matrix B, shape ``(in_features, rank)``.
        A: Low-rank matrix A, shape ``(rank, out_features)``.
        alpha: Scaling factor.
        bias: Optional bias.

    Returns:
        Output array.
    """
    p = etp_lora_mm_p if x.ndim >= 2 else etp_lora_mv_p
    x_v, x_u = u.split_mantissa_unit(x)
    B_v, B_u = u.split_mantissa_unit(B)
    A_v, A_u = u.split_mantissa_unit(A)
    unit = x_u * B_u * A_u
    if bias is not None:
        bias_v = u.Quantity(bias).to_decimal(unit)
        r = p.bind(x_v, B_v, A_v, bias_v, alpha=alpha, has_bias=True)
    else:
        r = p.bind(x_v, B_v, A_v, alpha=alpha, has_bias=False)
    return u.maybe_decimal(r * x_u * B_u * A_u)
