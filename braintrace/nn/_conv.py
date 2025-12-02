# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

import collections.abc
from typing import Callable, Tuple, Union, Sequence, Optional, TypeVar

import brainstate
import jax
from braintools import init

from braintrace._etrace_concepts import ETraceParam
from braintrace._etrace_operators import ConvOp
from braintrace._typing import ArrayLike

__all__ = [
    'Conv1d',
    'Conv2d',
    'Conv3d',
]

T = TypeVar('T')


def to_dimension_numbers(
    num_spatial_dims: int,
    channels_last: bool,
    transpose: bool
) -> jax.lax.ConvDimensionNumbers:
    """Create a `lax.ConvDimensionNumbers` for the given inputs."""
    num_dims = num_spatial_dims + 2
    if channels_last:
        spatial_dims = tuple(range(1, num_dims - 1))
        image_dn = (0, num_dims - 1) + spatial_dims
    else:
        spatial_dims = tuple(range(2, num_dims))
        image_dn = (0, 1) + spatial_dims
    if transpose:
        kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
    else:
        kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))
    return jax.lax.ConvDimensionNumbers(
        lhs_spec=image_dn,
        rhs_spec=kernel_dn,
        out_spec=image_dn
    )


def replicate(
    element: Union[T, Sequence[T]],
    num_replicate: int,
    name: str,
) -> Tuple[T, ...]:
    """Replicates entry in `element` `num_replicate` if needed."""
    if isinstance(element, (str, bytes)) or not isinstance(element, collections.abc.Sequence):
        return (element,) * num_replicate
    elif len(element) == 1:
        return tuple(list(element) * num_replicate)
    elif len(element) == num_replicate:
        return tuple(element)
    else:
        raise TypeError(
            f"{name} must be a scalar or sequence of length 1 or "
            f"sequence of length {num_replicate}."
        )


class _Conv(brainstate.nn.Module):
    # the number of spatial dimensions
    num_spatial_dims: int

    def __init__(
        self,
        in_size: Sequence[int],
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
        lhs_dilation: Union[int, Tuple[int, ...]] = 1,
        rhs_dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        w_init: Union[Callable, ArrayLike] = init.XavierNormal(),
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        w_mask: Optional[Union[ArrayLike, Callable]] = None,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # general parameters
        assert self.num_spatial_dims + 1 == len(in_size)
        self.in_size = tuple(in_size)
        self.in_channels = in_size[-1]
        self.out_channels = out_channels
        self.stride = replicate(stride, self.num_spatial_dims, 'stride')
        self.kernel_size = replicate(kernel_size, self.num_spatial_dims, 'kernel_size')
        self.lhs_dilation = replicate(lhs_dilation, self.num_spatial_dims, 'lhs_dilation')
        self.rhs_dilation = replicate(rhs_dilation, self.num_spatial_dims, 'rhs_dilation')
        self.groups = groups
        self.dimension_numbers = to_dimension_numbers(self.num_spatial_dims, channels_last=True, transpose=False)

        # the padding parameter
        if isinstance(padding, str):
            assert padding in ['SAME', 'VALID']
        elif isinstance(padding, int):
            padding = tuple((padding, padding) for _ in range(self.num_spatial_dims))
        elif isinstance(padding, (tuple, list)):
            if isinstance(padding[0], int):
                padding = (padding,) * self.num_spatial_dims
            elif isinstance(padding[0], (tuple, list)):
                if len(padding) == 1:
                    padding = tuple(padding) * self.num_spatial_dims
                else:
                    if len(padding) != self.num_spatial_dims:
                        raise ValueError(
                            f"Padding {padding} must be a Tuple[int, int], "
                            f"or sequence of Tuple[int, int] with length 1, "
                            f"or sequence of Tuple[int, int] with length {self.num_spatial_dims}."
                        )
                    padding = tuple(padding)
        else:
            raise ValueError
        self.padding = padding

        # the number of in-/out-channels
        assert self.out_channels % self.groups == 0, '"out_channels" should be divisible by groups'
        assert self.in_channels % self.groups == 0, '"in_channels" should be divisible by groups'

        # kernel shape and w_mask
        kernel_shape = tuple(self.kernel_size) + (self.in_channels // self.groups, self.out_channels)
        self.kernel_shape = kernel_shape
        self.w_mask = init.param(w_mask, kernel_shape, allow_none=True)

        # --- initializers --- #
        self.w_initializer = w_init
        self.b_initializer = b_init

        # --- weights --- #
        params = {}
        params['weight'] = init.param(self.w_initializer, self.kernel_shape, allow_none=False)
        if self.b_initializer is not None:
            bias_shape = (1,) * len(self.kernel_size) + (self.out_channels,)
            bias = init.param(self.b_initializer, bias_shape, allow_none=True)
            params['bias'] = bias

        # --- operation --- #
        xinfo = jax.ShapeDtypeStruct(self.in_size, params['weight'].dtype)

        op = ConvOp(
            xinfo=xinfo,
            window_strides=self.stride,
            padding=self.padding,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.rhs_dilation,
            feature_group_count=self.groups,
            dimension_numbers=self.dimension_numbers,
            weight_mask=self.w_mask,
        )

        # --- Evaluate the output shape --- #
        abstract_y = jax.eval_shape(op, xinfo, params)
        self.out_size = abstract_y.shape

        # --- parameters --- #
        self.weight_op = param_type(params, op=op)

    def _check_input_dim(self, x):
        if x.ndim == self.num_spatial_dims + 1:
            x_shape = x.shape
        elif x.ndim == self.num_spatial_dims + 2:
            x_shape = x.shape[1:]
        else:
            raise ValueError(
                f"expected {self.num_spatial_dims + 2}D (with batch) or "
                f"{self.num_spatial_dims + 1}D (without batch) "
                f"input (got {x.ndim}D input, {x.shape})"
            )
        if self.in_size != x_shape:
            raise ValueError(
                f"The expected input shape is {self.in_size}, "
                f"while we got {x_shape}."
            )

    def update(self, x):
        self._check_input_dim(x)
        return self.weight_op.execute(x)


class Conv1d(_Conv):
    """One-dimensional convolution.

    The input should be a 3d array with the shape of ``[B, H, C]``.

    Parameters
    ----------
    in_size : tuple of int
        The input shape, without the batch size. This argument is important, since it is
        used to evaluate the shape of the output.
    out_channels : int
        The number of output channels.
    kernel_size : int or sequence of int
        The shape of the convolutional kernel.
        For 1D convolution, the kernel size can be passed as an integer.
        For all other cases, it must be a sequence of integers.
    stride : int or sequence of int, optional
        An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
    padding : str or int or sequence of int or sequence of tuple, optional
        Either the string `'SAME'`, the string `'VALID'`, or a sequence of n `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. Default is 'SAME'.
    lhs_dilation : int or sequence of int, optional
        An integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
    rhs_dilation : int or sequence of int, optional
        An integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as 'atrous convolution'.
    groups : int, optional
        If specified, divides the input features into groups. Default is 1.
    w_init : Callable or ArrayLike, optional
        The initializer for the convolutional kernel. Default is XavierNormal().
    b_init : Callable or ArrayLike or None, optional
        The initializer for the bias. Default is None.
    w_mask : ArrayLike or Callable or None, optional
        The optional mask of the weights. Default is None.
    name : str or None, optional
        The name of the object. Default is None.
    param_type : type, optional
        The parameter type. Default is ETraceParam.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a 1D convolution layer
        >>> conv1d = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3)
        >>>
        >>> # Input with batch size 4
        >>> x = brainstate.random.randn(4, 10, 3)
        >>> y = conv1d(x)
        >>> print(y.shape)
        (4, 10, 16)
    """
    __module__ = 'braintrace.nn'
    num_spatial_dims: int = 1


class Conv2d(_Conv):
    """Two-dimensional convolution.

    The input should be a 4d array with the shape of ``[B, H, W, C]``.

    Parameters
    ----------
    in_size : tuple of int
        The input shape, without the batch size. This argument is important, since it is
        used to evaluate the shape of the output.
    out_channels : int
        The number of output channels.
    kernel_size : int or sequence of int
        The shape of the convolutional kernel.
        For 1D convolution, the kernel size can be passed as an integer.
        For all other cases, it must be a sequence of integers.
    stride : int or sequence of int, optional
        An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
    padding : str or int or sequence of int or sequence of tuple, optional
        Either the string `'SAME'`, the string `'VALID'`, or a sequence of n `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. Default is 'SAME'.
    lhs_dilation : int or sequence of int, optional
        An integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
    rhs_dilation : int or sequence of int, optional
        An integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as 'atrous convolution'.
    groups : int, optional
        If specified, divides the input features into groups. Default is 1.
    w_init : Callable or ArrayLike, optional
        The initializer for the convolutional kernel. Default is XavierNormal().
    b_init : Callable or ArrayLike or None, optional
        The initializer for the bias. Default is None.
    w_mask : ArrayLike or Callable or None, optional
        The optional mask of the weights. Default is None.
    name : str or None, optional
        The name of the object. Default is None.
    param_type : type, optional
        The parameter type. Default is ETraceParam.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a 2D convolution layer
        >>> conv2d = braintrace.nn.Conv2d(in_size=(28, 28, 1), out_channels=32, kernel_size=3, stride=1)
        >>>
        >>> # Input with batch size 8
        >>> x = brainstate.random.randn(8, 28, 28, 1)
        >>> y = conv2d(x)
        >>> print(y.shape)
        (8, 28, 28, 32)
    """
    __module__ = 'braintrace.nn'
    num_spatial_dims: int = 2


class Conv3d(_Conv):
    """Three-dimensional convolution.

    The input should be a 5d array with the shape of ``[B, H, W, D, C]``.

    Parameters
    ----------
    in_size : tuple of int
        The input shape, without the batch size. This argument is important, since it is
        used to evaluate the shape of the output.
    out_channels : int
        The number of output channels.
    kernel_size : int or sequence of int
        The shape of the convolutional kernel.
        For 1D convolution, the kernel size can be passed as an integer.
        For all other cases, it must be a sequence of integers.
    stride : int or sequence of int, optional
        An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
    padding : str or int or sequence of int or sequence of tuple, optional
        Either the string `'SAME'`, the string `'VALID'`, or a sequence of n `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. Default is 'SAME'.
    lhs_dilation : int or sequence of int, optional
        An integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
    rhs_dilation : int or sequence of int, optional
        An integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as 'atrous convolution'.
    groups : int, optional
        If specified, divides the input features into groups. Default is 1.
    w_init : Callable or ArrayLike, optional
        The initializer for the convolutional kernel. Default is XavierNormal().
    b_init : Callable or ArrayLike or None, optional
        The initializer for the bias. Default is None.
    w_mask : ArrayLike or Callable or None, optional
        The optional mask of the weights. Default is None.
    name : str or None, optional
        The name of the object. Default is None.
    param_type : type, optional
        The parameter type. Default is ETraceParam.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a 3D convolution layer
        >>> conv3d = braintrace.nn.Conv3d(in_size=(16, 16, 16, 3), out_channels=64, kernel_size=3, stride=2)
        >>>
        >>> # Input with batch size 2
        >>> x = brainstate.random.randn(2, 16, 16, 16, 3)
        >>> y = conv3d(x)
        >>> print(y.shape)
        (2, 8, 8, 8, 64)
    """
    __module__ = 'braintrace.nn'
    num_spatial_dims: int = 3

