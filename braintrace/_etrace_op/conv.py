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


def _conv_yw_to_w(hidden_dim, trace, **params):
    r"""Broadcast ``hidden_dim`` across all dims except output channel."""
    n_expand = trace.ndim - hidden_dim.ndim
    for _ in range(n_expand):
        hidden_dim = jnp.expand_dims(hidden_dim, axis=0)
    return trace * hidden_dim


def _conv_xy_to_dw(x, hidden_dim, w, **params):
    r"""VJP of ``y = conv(x, w)`` w.r.t. ``w``."""
    conv_kw = {k: v for k, v in params.items() if k != 'has_bias'}

    def _fwd(w_):
        return u.get_mantissa(jax.lax.conv_general_dilated(x, w_, **conv_kw))

    _, vjp_fn = jax.vjp(_fwd, w)
    return u.get_mantissa(vjp_fn(hidden_dim)[0])


def _conv_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    batch = x_var.aval.shape[0]
    kernel_shape = weight_var.aval.shape
    return jnp.zeros((batch, *kernel_shape, num_hidden_state))


def _conv_init_pp(x_var, y_var, weight_var, num_hidden_state):
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
