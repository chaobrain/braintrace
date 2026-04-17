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

r"""Convolution ETP primitive (``etp_conv_p``).

Always expects a batch dimension on the input. The full keyword surface
of ``jax.lax.conv_general_dilated`` is preserved; the wrapper splits and
recombines saiunit quantities for the input and kernel.
"""

from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import saiunit as u

from ._spec import ETPPrimitiveSpec, register_primitive_spec

__all__ = [
    'etp_conv_p',
    'conv',
]


def _etp_conv_impl(
    *args,
    has_bias=False,
    strides=(1,),
    padding='SAME',
    lhs_dilation=None,
    rhs_dilation=None,
    feature_group_count=1,
    batch_group_count=1,
    dimension_numbers=None,
):
    x, kernel = args[0], args[1]
    y = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        dimension_numbers=dimension_numbers,
    )
    if has_bias:
        y = y + args[2]
    return y


def _conv_trainable_invars(params):
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


def _conv_yw_to_w(hidden_dim, trace, **params):
    r"""Propagate the hidden-state Jacobian through the weight-shaped trace.

    This function is called in two different vmap contexts:

    1. **scan / etrace-update path** (no outer batch vmap):
       ``hidden_dim``      = ``(batch, *spatial_out, out_ch)``,
       ``trace['weight']`` = ``(batch, *kernel_dims, out_ch)``,
       ``trace['bias']``   = ``(batch, *spatial_out, out_ch)`` if present.

    2. **gradient-computation path** (outer batch vmap applied):
       ``hidden_dim``      = ``(*spatial_out, out_ch)``,
       ``trace['weight']`` = ``(*kernel_dims, out_ch)``,
       ``trace['bias']``   = ``(*spatial_out, out_ch)`` if present.

    The bias shares parameters across spatial positions, so the bias gradient
    requires summing over spatial dims. The bias trace has the **same shape as
    the output y** (per element ∂h/∂b = ∂h/∂y), and ``yw_to_w`` sums the
    elementwise product ``hidden_dim * trace['bias']`` over the spatial axes.

    For scalar ``hidden_dim`` (0-D) the function multiplies elementwise and
    returns immediately.
    """
    has_bias = params.get('has_bias', False)
    w_trace = trace['weight']

    if hidden_dim.ndim == 0:
        # Scalar (degenerate) case: multiply all trace entries elementwise.
        out = {'weight': w_trace * hidden_dim}
        if has_bias:
            out['bias'] = jnp.sum(trace['bias'] * hidden_dim)
        return out

    # ── Determine batch prefix length ─────────────────────────────────────────
    # hidden_dim = (*batch, *spatial, out_ch),  w_trace = (*batch, *kernel, out_ch).
    # ``sum_start`` is the first axis that is spatial (not batch, not out_ch).
    #
    # Context 1 (scan): batch=1, hidden=(1,H,C), w=(1,Hk,Cin,C)
    #   sum_start = max(0, min(4,3)-2) = 1
    # Context 2 (grad): batch=0, hidden=(H,C),   w=(Hk,Cin,C)
    #   sum_start = max(0, min(3,2)-2) = 0
    sum_start = max(0, min(w_trace.ndim, hidden_dim.ndim) - 2)

    # ── Weight: reduce hidden_dim to (*batch, out_ch) then broadcast ──────────
    w_sum_axes = tuple(range(sum_start, hidden_dim.ndim - 1))
    hd_reduced = jnp.sum(hidden_dim, axis=w_sum_axes) if w_sum_axes else hidden_dim

    n_expand = w_trace.ndim - hd_reduced.ndim
    hd_for_weight = hd_reduced
    for _ in range(n_expand):
        hd_for_weight = jnp.expand_dims(hd_for_weight, axis=-2)

    out = {'weight': w_trace * hd_for_weight}

    # ── Bias: trace['bias'] has y-output shape (*batch, *spatial, out_ch). ────
    # Multiply elementwise with hidden_dim and sum over the spatial dims
    # (axes sum_start … ndim-2) to obtain (*batch, out_ch).
    if has_bias:
        b_trace = trace['bias']
        b_sum_axes = tuple(range(sum_start, b_trace.ndim - 1))
        out['bias'] = jnp.sum(b_trace * hidden_dim, axis=b_sum_axes) if b_sum_axes else b_trace * hidden_dim
    return out


def _conv_xy_to_dw(x, hidden_dim, weights, **params):
    r"""Direct-trace initial contribution: ``∂h/∂w`` and ``∂h/∂b``.

    For the **kernel** ``w``: uses VJP of ``y = conv(x, w)`` — this is a true
    function of ``x`` and requires the full conv VJP.

    For the **bias** ``b``: ``y_{nhk} = conv(x)_{nhk} + b_k``, so
    ``∂y_{nhk}/∂b_k = 1``.  Therefore ``∂h_{nhk}/∂b_k = hidden_dim_{nhk}``
    (the cotangent ∂h/∂y at that position), with **no spatial summation**.
    The bias trace stores per-position values (same shape as ``y``); the
    summation over spatial positions is deferred to ``_conv_yw_to_w`` during
    the gradient-computation pass.
    """
    has_bias = params.get('has_bias', False)
    # Build conv_general_dilated kwargs; remap 'strides' -> 'window_strides'.
    conv_kw = {}
    for k, v in params.items():
        if k == 'has_bias':
            continue
        if k == 'strides':
            conv_kw['window_strides'] = v
        else:
            conv_kw[k] = v

    # The batched D-RTRL executor vmaps over the batch dimension, so x may
    # arrive here without a leading batch axis (e.g. shape (H, C) instead of
    # (N, H, C)).  conv_general_dilated always requires a batch dim, so we
    # temporarily add one when needed.
    unbatched = x.ndim < 3  # heuristic: 1-D conv needs at least (N, H, C)
    if unbatched:
        x_in = x[None]          # (1, H, C)
        hd_in = hidden_dim[None]  # (1, H, C) or similar
    else:
        x_in = x
        hd_in = hidden_dim

    # Kernel gradient via VJP (needs x).
    def _fwd_w(w):
        return u.get_mantissa(
            jax.lax.conv_general_dilated(x_in, w, **conv_kw)
        )

    _, vjp_fn = jax.vjp(_fwd_w, weights['weight'])
    dw = u.get_mantissa(vjp_fn(hd_in)[0])
    out = {'weight': dw}

    if has_bias:
        # Bias gradient = hidden_dim (cotangent at each output position).
        # No spatial summation — the trace stores per-position ∂h/∂b.
        out['bias'] = hidden_dim  # shape (H, C) or (N, H, C)

    return out


def _conv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    """Return zero-filled Dict[str, Array] for the D-RTRL parameter-dim trace.

    The bias trace has shape ``(batch, *y_spatial_and_ch, n_state)`` where
    ``y_spatial_and_ch = y_var.aval.shape[1:]`` (all output dims except batch).
    This stores the per-position Jacobian ``∂h_{nhk}/∂b_k``; the spatial
    summation happens in ``_conv_yw_to_w`` during gradient computation.
    """
    batch = x_var.aval.shape[0]
    out = {
        'weight': jnp.zeros(
            (batch, *weight_vars['weight'].aval.shape, num_hidden_state)
        )
    }
    if 'bias' in weight_vars:
        # y_var.aval.shape = (batch, *spatial, out_ch); strip the batch dim.
        out['bias'] = jnp.zeros(
            (batch, *y_var.aval.shape[1:], num_hidden_state)
        )
    return out


def _conv_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_conv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_conv',
        impl=_etp_conv_impl,
        yw_to_w=_conv_yw_to_w,
        xy_to_dw=_conv_xy_to_dw,
        init_drtrl=_conv_init_drtrl,
        init_pp=_conv_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=True,
        trainable_invars_fn=_conv_trainable_invars,
    )
)


def conv(
    x,
    kernel,
    bias=None,
    *,
    strides: Sequence[int] = (1,),
    padding: str = 'SAME',
    lhs_dilation: Optional[Sequence[int]] = None,
    rhs_dilation: Optional[Sequence[int]] = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    dimension_numbers: Any = None,
):
    r"""ETP-aware convolution.

    Computes :math:`y = \mathrm{conv}(x, kernel) \; (+ b)`.
    Always expects a batch dimension on ``x``.

    Args:
        x: Input tensor with batch dimension.
        kernel: Convolution kernel.
        bias: Optional bias.
        strides: Window strides.
        padding: Padding mode.
        lhs_dilation: Left-hand-side dilation.
        rhs_dilation: Right-hand-side dilation.
        feature_group_count: Feature group count.
        batch_group_count: Batch group count.
        dimension_numbers: Convolution dimension numbers.

    Returns:
        Convolution output.
    """
    conv_kwargs = dict(
        strides=tuple(strides),
        padding=padding,
        lhs_dilation=tuple(lhs_dilation) if lhs_dilation is not None else None,
        rhs_dilation=tuple(rhs_dilation) if rhs_dilation is not None else None,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        dimension_numbers=dimension_numbers,
    )
    x_v, x_u = u.split_mantissa_unit(x)
    kernel_v, kernel_u = u.split_mantissa_unit(kernel)
    unit = x_u * kernel_u
    if bias is not None:
        bias_v = u.Quantity(bias).to_decimal(unit)
        r = etp_conv_p.bind(x_v, kernel_v, bias_v, has_bias=True, **conv_kwargs)
    else:
        r = etp_conv_p.bind(x_v, kernel_v, has_bias=False, **conv_kwargs)
    return u.maybe_decimal(r * x_u * kernel_u)
