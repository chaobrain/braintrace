# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""
User-facing functions for ETP operations.

These functions are the primary API for marking parameter operations
in the computation graph. During normal execution they compute the
forward pass; under the ``etp()`` transformation they are recognised
as ETP operations and Jacobian information is extracted automatically.
"""

from typing import Callable, Optional, Sequence, Any

import jax

from ._primitives import etp_matmul_p, etp_elemwise_p, etp_conv_p

__all__ = [
    'matmul',
    'element_wise',
    'conv',
]


def matmul(x, weight, bias=None):
    r"""
    ETP-aware matrix multiplication.

    Computes :math:`y = x \mathbin{@} w \; (+ b)`.

    During normal execution this is a standard matrix multiplication.
    Under the ``etp()`` transformation it is recognised as a parameter
    operation so that eligibility-trace Jacobians are extracted.

    Args:
        x: Input array, shape ``(..., in_features)``.
        weight: Weight matrix, shape ``(in_features, out_features)``.
        bias: Optional bias vector, shape ``(out_features,)``.

    Returns:
        Output array, shape ``(..., out_features)``.

    Example::

        >>> y = bt.matmul(x, params['weight'], params['bias'])
    """
    if bias is not None:
        return etp_matmul_p.bind(x, weight, bias, has_bias=True)
    return etp_matmul_p.bind(x, weight, has_bias=False)


def element_wise(weight, fn=lambda w: w):
    r"""
    ETP-aware element-wise operation.

    Applies ``fn`` to ``weight`` and passes the result through a marker
    primitive.  The operation is treated as *diagonal* in the hidden-state
    space, enabling more efficient eligibility-trace computation.

    ``fn`` is executed as normal JAX operations (so JAX's autodiff handles
    the chain rule through ``fn``).  The primitive only serves as a marker
    in the Jaxpr.

    Args:
        weight: Weight parameter (same shape as the hidden state it will
            be multiplied with).
        fn: Element-wise function applied to *weight*.  Must be a pure
            function.  Defaults to identity.

    Returns:
        ``fn(weight)`` — same shape and dtype as ``fn(weight)`` would
        normally produce.

    Example::

        >>> tau_h = bt.element_wise(params['tau'], fn=jax.nn.softplus)
    """
    y = fn(weight)
    return etp_elemwise_p.bind(y)


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
    r"""
    ETP-aware convolution.

    Computes :math:`y = \mathrm{conv}(x, w) \; (+ b)`.

    Args:
        x: Input tensor.
        kernel: Convolution kernel.
        bias: Optional bias.
        strides: Window strides.
        padding: Padding mode (``'SAME'`` or ``'VALID'``).
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
    if bias is not None:
        return etp_conv_p.bind(x, kernel, bias, has_bias=True, **conv_kwargs)
    return etp_conv_p.bind(x, kernel, has_bias=False, **conv_kwargs)
