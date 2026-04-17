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


def _conv_layout(params, y_ndim):
    """Return ``(n_spatial, channel_axis, batch_axis)`` for the conv output layout.

    ``n_spatial``:    spatial rank (1, 2, or 3).
    ``channel_axis``: position of the output-channel axis in an output array
                      of rank ``y_ndim``.
    ``batch_axis``:   position of the batch axis in an output array of rank
                      ``y_ndim``.

    Sources used (in priority order):

    1. ``params['dimension_numbers']`` — when a ``ConvDimensionNumbers``
       namedtuple is present, ``out_spec[0]`` is the batch position and
       ``out_spec[1]`` is the channel position in the output.
    2. ``params['strides']`` — ``len(strides)`` gives ``n_spatial``.
    3. When ``dimension_numbers`` is ``None`` JAX defaults to ``iota``
       (``(0,1,2,...)``) which maps to the NCHW / NCH convention:
       batch at axis 0, channel at axis 1.

    Notes on ``ConvDimensionNumbers.out_spec``::

        ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)

    ``out_spec`` is a tuple of length ``n_spatial + 2`` where each element
    is the *physical* axis index in the output array corresponding to the
    logical dimension in the order ``(N, C, spatial...)``.  Concretely::

        out_spec[0]  → position of N (batch)   in the output
        out_spec[1]  → position of C (channel) in the output
        out_spec[2:] → positions of spatial dims in the output

    Example: NHWC gives ``out_spec = (0, 3, 1, 2)`` so
    ``batch_axis=0``, ``channel_axis=3``.
    """
    n_spatial = len(params.get('strides', (1,)))
    dn = params.get('dimension_numbers', None)
    if dn is None:
        # JAX default: iota = (0,1,2,...) → NCHW / NCH convention.
        batch_axis = 0
        channel_axis = 1
    elif isinstance(dn, tuple) and len(dn) == 3 and isinstance(dn[2], str):
        # String-tuple form e.g. ('NHWC', 'HWIO', 'NHWC').
        out_spec_str = dn[2]
        batch_axis = out_spec_str.index('N')
        channel_axis = out_spec_str.index('C')
    else:
        # ConvDimensionNumbers namedtuple: out_spec[0]=batch_pos, out_spec[1]=channel_pos.
        out_spec = dn.out_spec
        batch_axis = out_spec[0]
        channel_axis = out_spec[1]
    return n_spatial, channel_axis, batch_axis


def _conv_yw_to_w(hidden_dim, trace, **params):
    r"""Propagate the hidden-state Jacobian through the weight-shaped trace.

    This function is called in two different vmap contexts:

    1. **scan / etrace-update path** (no outer batch vmap):
       ``hidden_dim``      = ``(batch, *spatial_out, out_ch)`` (or permuted),
       ``trace['weight']`` = ``(batch, *kernel_dims, out_ch)``,
       ``trace['bias']``   = ``(batch, *spatial_out, out_ch)`` if present.

    2. **gradient-computation path** (outer batch vmap applied):
       ``hidden_dim``      = ``(*spatial_out, out_ch)`` (batch axis stripped),
       ``trace['weight']`` = ``(*kernel_dims, out_ch)`` (batch axis stripped),
       ``trace['bias']``   = ``(*spatial_out, out_ch)`` if present.

    The bias shares parameters across spatial positions, so the bias gradient
    requires summing over spatial dims.  The bias trace has the **same shape
    as the output y** (per element ∂h/∂b = ∂h/∂y), and ``yw_to_w`` sums the
    elementwise product ``hidden_dim * trace['bias']`` over the spatial axes.

    Layout awareness: spatial axes are derived from the actual
    ``dimension_numbers`` / ``strides`` params rather than assuming any
    fixed channel-last convention.

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

    # ── Determine layout from params ──────────────────────────────────────────
    # y_ndim for the *batched* output (scan context) is hidden_dim.ndim when
    # a batch prefix is present, or hidden_dim.ndim + 1 when it is stripped
    # (grad context).  We infer which context we are in from whether
    # hidden_dim has a batch prefix:
    #   scan context:  hidden_dim.ndim == n_spatial + 2  (batch + spatial + ch)
    #   grad context:  hidden_dim.ndim == n_spatial + 1  (spatial + ch only)
    n_spatial, channel_axis_batched, batch_axis_batched = _conv_layout(
        params, y_ndim=None  # y_ndim not needed; we use n_spatial directly
    )
    # hidden_dim.ndim in scan context = n_spatial + 2 (batch + spatial + channel)
    # hidden_dim.ndim in grad context = n_spatial + 1 (spatial + channel, no batch)
    has_batch_prefix = (hidden_dim.ndim == n_spatial + 2)

    # Compute spatial axes in hidden_dim (same permutation as y output).
    if has_batch_prefix:
        # Axes in full output: {batch_axis_batched, channel_axis_batched} excluded.
        spatial_axes_hd = tuple(
            sorted(set(range(hidden_dim.ndim)) - {batch_axis_batched, channel_axis_batched})
        )
        # channel_axis in hidden_dim (same as in y).
        ch_axis_hd = channel_axis_batched
    else:
        # Batch axis is stripped.  Remaining axes: spatial + channel.
        # The original channel_axis_batched and batch_axis_batched are for the
        # batched layout.  After stripping the batch axis, remaining axes are
        # renumbered: remove batch_axis_batched from the set.
        all_axes = set(range(n_spatial + 2))
        remaining = sorted(all_axes - {batch_axis_batched})
        # remaining[i] is the original axis index; map to new (shifted) index.
        ch_axis_hd = remaining.index(channel_axis_batched)
        spatial_axes_hd = tuple(i for i in range(len(remaining)) if i != ch_axis_hd)

    # ── Weight: reduce hidden_dim over spatial axes, then broadcast ───────────
    # Target shape after reduction: axes {ch_axis_hd} remain, spatial removed.
    hd_reduced = jnp.sum(hidden_dim, axis=spatial_axes_hd) if spatial_axes_hd else hidden_dim
    # hd_reduced shape: (*batch_prefix, out_ch)  [only batch and ch axes survive]

    n_expand = w_trace.ndim - hd_reduced.ndim
    hd_for_weight = hd_reduced
    for _ in range(n_expand):
        hd_for_weight = jnp.expand_dims(hd_for_weight, axis=-2)

    out = {'weight': w_trace * hd_for_weight}

    # ── Bias: trace['bias'] has y-output shape; sum over spatial axes ─────────
    if has_bias:
        b_trace = trace['bias']
        # b_trace has same ndim as hidden_dim (same layout).
        b_sum_axes = spatial_axes_hd
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
    # arrive here without a leading batch axis.
    # Unbatched detection: a batched input has ndim == n_spatial + 2
    # (batch + spatial + channel / or the permuted equivalent), while an
    # unbatched input has ndim == n_spatial + 1.
    n_spatial = len(params.get('strides', (1,)))
    unbatched = (x.ndim == n_spatial + 1)
    if unbatched:
        x_in = x[None]
        hd_in = hidden_dim[None]
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
        out['bias'] = hidden_dim

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
        trainable_invars_fn=_conv_trainable_invars,
        x_invar_index=0,
        batched=True,
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
