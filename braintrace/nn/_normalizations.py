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

from functools import partial
from typing import Callable, Union, Sequence, Optional, Any

import brainstate
import braintools
import brainunit as u
import jax
from brainstate import BatchState
from brainstate.nn._normalizations import _BatchNorm

from braintrace._etrace_concepts import ETraceParam
from braintrace._etrace_operators import ETraceOp, Y, W, general_y2w
from braintrace._typing import ArrayLike, Size, Axes

__all__ = [
    'BatchNorm0d',
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'LayerNorm',
    'RMSNorm',
    'GroupNorm'
]


class ScaleOp(ETraceOp):
    def xw_to_y(self, x, param):
        if 'scale' in param:
            x = x * param['scale']
        if 'bias' in param:
            x = x + param['bias']
        return x

    def yw_to_w(
        self,
        hidden_dim_arr: Y,
        weight_dim_tree: W,
    ) -> W:
        w_like = general_y2w(self.xw_to_y, hidden_dim_arr, hidden_dim_arr, weight_dim_tree)
        return jax.tree.map(u.math.multiply, weight_dim_tree, w_like)


class _BatchNormETrace(_BatchNorm):
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: Size,
        feature_axis: Axes = -1,
        track_running_stats: bool = True,
        epsilon: float = 1e-5,
        momentum: float = 0.99,
        affine: bool = True,
        bias_initializer: Union[ArrayLike, Callable] = braintools.init.Constant(0.),
        scale_initializer: Union[ArrayLike, Callable] = braintools.init.Constant(1.),
        axis_name: Optional[Union[str, Sequence[str]]] = None,
        axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
        name: Optional[str] = None,
        dtype: Any = None,
        mean_type: type = BatchState,
        param_type: type = ETraceParam,
    ):
        weight_type = partial(
            param_type,
            op=ScaleOp(is_diagonal=True),
            grad='full',
        )

        super().__init__(
            in_size=in_size,
            feature_axis=feature_axis,
            track_running_stats=track_running_stats,
            epsilon=epsilon,
            momentum=momentum,
            affine=affine,
            bias_initializer=bias_initializer,
            scale_initializer=scale_initializer,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups,
            dtype=dtype,
            mean_type=mean_type,
            param_type=weight_type,
            name=name
        )


class BatchNorm0d(_BatchNormETrace):
    """Batch normalization for 0D inputs (no spatial dimensions).

    Applies batch normalization over a 2D input (batch_size, features) as described
    in the paper "Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift".

    Parameters
    ----------
    in_size : int or sequence of int
        The input size.
    feature_axis : int or sequence of int, optional
        The axis or axes that should be normalized (typically the features axis).
        Default is -1.
    track_running_stats : bool, optional
        Whether to track running mean and variance during training. Default is True.
    epsilon : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    momentum : float, optional
        The value used for the running mean and variance computation. Default is 0.99.
    affine : bool, optional
        Whether to use learnable affine parameters. Default is True.
    bias_initializer : ArrayLike or Callable, optional
        The initializer for the bias parameter. Default is Constant(0.).
    scale_initializer : ArrayLike or Callable, optional
        The initializer for the scale parameter. Default is Constant(1.).
    axis_name : str or sequence of str or None, optional
        The axis name for distributed training. Default is None.
    axis_index_groups : sequence of sequence of int or None, optional
        The axis index groups for distributed training. Default is None.
    name : str or None, optional
        The name of the layer. Default is None.
    dtype : Any, optional
        The dtype of the layer parameters. Default is None.
    mean_type : type, optional
        The type for storing running statistics. Default is BatchState.
    param_type : type, optional
        The type of the parameter. Default is :class:`ETraceParam`.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a BatchNorm0d layer
        >>> bn = braintrace.nn.BatchNorm0d(in_size=128)
        >>>
        >>> # Input with batch size 32
        >>> x = brainstate.random.randn(32, 128)
        >>> y = bn(x)
        >>> print(y.shape)
        (32, 128)
    """
    __module__ = 'braintrace.nn'
    num_spatial_dims: int = 0


class BatchNorm1d(_BatchNormETrace):
    """Batch normalization for 1D inputs (one spatial dimension).

    Applies batch normalization over a 3D input (batch_size, length, features)
    as described in the paper "Batch Normalization: Accelerating Deep Network
    Training by Reducing Internal Covariate Shift".

    Parameters
    ----------
    in_size : int or sequence of int
        The input size.
    feature_axis : int or sequence of int, optional
        The axis or axes that should be normalized (typically the features axis).
        Default is -1.
    track_running_stats : bool, optional
        Whether to track running mean and variance during training. Default is True.
    epsilon : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    momentum : float, optional
        The value used for the running mean and variance computation. Default is 0.99.
    affine : bool, optional
        Whether to use learnable affine parameters. Default is True.
    bias_initializer : ArrayLike or Callable, optional
        The initializer for the bias parameter. Default is Constant(0.).
    scale_initializer : ArrayLike or Callable, optional
        The initializer for the scale parameter. Default is Constant(1.).
    axis_name : str or sequence of str or None, optional
        The axis name for distributed training. Default is None.
    axis_index_groups : sequence of sequence of int or None, optional
        The axis index groups for distributed training. Default is None.
    name : str or None, optional
        The name of the layer. Default is None.
    dtype : Any, optional
        The dtype of the layer parameters. Default is None.
    mean_type : type, optional
        The type for storing running statistics. Default is BatchState.
    param_type : type, optional
        The type of the parameter. Default is :class:`ETraceParam`.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a BatchNorm1d layer
        >>> bn = braintrace.nn.BatchNorm1d(in_size=(100, 64))
        >>>
        >>> # Input with batch size 16
        >>> x = brainstate.random.randn(16, 100, 64)
        >>> y = bn(x)
        >>> print(y.shape)
        (16, 100, 64)
    """
    __module__ = 'braintrace.nn'
    num_spatial_dims: int = 1


class BatchNorm2d(_BatchNormETrace):
    """Batch normalization for 2D inputs (two spatial dimensions).

    Applies batch normalization over a 4D input (batch_size, height, width, features)
    as described in the paper "Batch Normalization: Accelerating Deep Network
    Training by Reducing Internal Covariate Shift".

    Parameters
    ----------
    in_size : int or sequence of int
        The input size.
    feature_axis : int or sequence of int, optional
        The axis or axes that should be normalized (typically the features axis).
        Default is -1.
    track_running_stats : bool, optional
        Whether to track running mean and variance during training. Default is True.
    epsilon : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    momentum : float, optional
        The value used for the running mean and variance computation. Default is 0.99.
    affine : bool, optional
        Whether to use learnable affine parameters. Default is True.
    bias_initializer : ArrayLike or Callable, optional
        The initializer for the bias parameter. Default is Constant(0.).
    scale_initializer : ArrayLike or Callable, optional
        The initializer for the scale parameter. Default is Constant(1.).
    axis_name : str or sequence of str or None, optional
        The axis name for distributed training. Default is None.
    axis_index_groups : sequence of sequence of int or None, optional
        The axis index groups for distributed training. Default is None.
    name : str or None, optional
        The name of the layer. Default is None.
    dtype : Any, optional
        The dtype of the layer parameters. Default is None.
    mean_type : type, optional
        The type for storing running statistics. Default is BatchState.
    param_type : type, optional
        The type of the parameter. Default is :class:`ETraceParam`.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a BatchNorm2d layer
        >>> bn = braintrace.nn.BatchNorm2d(in_size=(28, 28, 32))
        >>>
        >>> # Input with batch size 8
        >>> x = brainstate.random.randn(8, 28, 28, 32)
        >>> y = bn(x)
        >>> print(y.shape)
        (8, 28, 28, 32)
    """
    __module__ = 'braintrace.nn'
    num_spatial_dims: int = 2


class BatchNorm3d(_BatchNormETrace):
    """Batch normalization for 3D inputs (three spatial dimensions).

    Applies batch normalization over a 5D input (batch_size, depth, height, width, features)
    as described in the paper "Batch Normalization: Accelerating Deep Network
    Training by Reducing Internal Covariate Shift".

    Parameters
    ----------
    in_size : int or sequence of int
        The input size.
    feature_axis : int or sequence of int, optional
        The axis or axes that should be normalized (typically the features axis).
        Default is -1.
    track_running_stats : bool, optional
        Whether to track running mean and variance during training. Default is True.
    epsilon : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    momentum : float, optional
        The value used for the running mean and variance computation. Default is 0.99.
    affine : bool, optional
        Whether to use learnable affine parameters. Default is True.
    bias_initializer : ArrayLike or Callable, optional
        The initializer for the bias parameter. Default is Constant(0.).
    scale_initializer : ArrayLike or Callable, optional
        The initializer for the scale parameter. Default is Constant(1.).
    axis_name : str or sequence of str or None, optional
        The axis name for distributed training. Default is None.
    axis_index_groups : sequence of sequence of int or None, optional
        The axis index groups for distributed training. Default is None.
    name : str or None, optional
        The name of the layer. Default is None.
    dtype : Any, optional
        The dtype of the layer parameters. Default is None.
    mean_type : type, optional
        The type for storing running statistics. Default is BatchState.
    param_type : type, optional
        The type of the parameter. Default is :class:`ETraceParam`.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a BatchNorm3d layer
        >>> bn = braintrace.nn.BatchNorm3d(in_size=(16, 16, 16, 64))
        >>>
        >>> # Input with batch size 4
        >>> x = brainstate.random.randn(4, 16, 16, 16, 64)
        >>> y = bn(x)
        >>> print(y.shape)
        (4, 16, 16, 16, 64)
    """
    __module__ = 'braintrace.nn'
    num_spatial_dims: int = 3


class LayerNorm(brainstate.nn.LayerNorm):
    """Layer normalization.

    Applies layer normalization over the input as described in the paper
    "Layer Normalization". Unlike batch normalization, layer normalization
    normalizes across the features dimension instead of the batch dimension.

    Parameters
    ----------
    in_size : int or sequence of int
        The shape of the input to be normalized. Can be a single integer (for 1D)
        or a tuple of integers (for multi-dimensional inputs).
    epsilon : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    elementwise_affine : bool, optional
        Whether to use learnable affine parameters (scale and bias). Default is True.
    bias_initializer : ArrayLike or Callable, optional
        The initializer for the bias parameter. Default is Constant(0.).
    scale_initializer : ArrayLike or Callable, optional
        The initializer for the scale parameter. Default is Constant(1.).
    name : str or None, optional
        The name of the layer. Default is None.
    param_type : type, optional
        The type of the parameter. Default is :class:`ETraceParam`.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a LayerNorm layer
        >>> ln = braintrace.nn.LayerNorm(in_size=512)
        >>>
        >>> # Input with batch size 10 and sequence length 20
        >>> x = brainstate.random.randn(10, 20, 512)
        >>> y = ln(x)
        >>> print(y.shape)
        (10, 20, 512)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        *args,
        param_type: type = ETraceParam,
        **kwargs,
    ):
        weight_type = partial(
            param_type,
            op=ScaleOp(is_diagonal=True),
            grad='full',
        )
        super().__init__(*args, param_type=weight_type, **kwargs)


class RMSNorm(brainstate.nn.RMSNorm):
    """Root Mean Square Layer Normalization.

    Applies RMS normalization over the input as described in the paper
    "Root Mean Square Layer Normalization". RMSNorm is a simplified version
    of layer normalization that only rescales the input using the root mean
    square statistic, without subtracting the mean.

    Parameters
    ----------
    in_size : int or sequence of int
        The shape of the input to be normalized. Can be a single integer (for 1D)
        or a tuple of integers (for multi-dimensional inputs).
    epsilon : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    elementwise_affine : bool, optional
        Whether to use learnable affine parameters (scale only). Default is True.
    scale_initializer : ArrayLike or Callable, optional
        The initializer for the scale parameter. Default is Constant(1.).
    name : str or None, optional
        The name of the layer. Default is None.
    param_type : type, optional
        The type of the parameter. Default is :class:`ETraceParam`.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create an RMSNorm layer
        >>> rms = braintrace.nn.RMSNorm(in_size=768)
        >>>
        >>> # Input with batch size 8 and sequence length 128
        >>> x = brainstate.random.randn(8, 128, 768)
        >>> y = rms(x)
        >>> print(y.shape)
        (8, 128, 768)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        *args,
        param_type: type = ETraceParam,
        **kwargs,
    ):
        weight_type = partial(
            param_type,
            op=ScaleOp(is_diagonal=True),
            grad='full',
        )
        super().__init__(*args, param_type=weight_type, **kwargs)


class GroupNorm(brainstate.nn.GroupNorm):
    """Group normalization.

    Applies group normalization over the input as described in the paper
    "Group Normalization". Group normalization divides the channels into groups
    and normalizes the features within each group independently.

    Parameters
    ----------
    num_groups : int
        The number of groups to divide the channels into.
    num_channels : int
        The number of channels in the input.
    epsilon : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    affine : bool, optional
        Whether to use learnable affine parameters (scale and bias). Default is True.
    bias_initializer : ArrayLike or Callable, optional
        The initializer for the bias parameter. Default is Constant(0.).
    scale_initializer : ArrayLike or Callable, optional
        The initializer for the scale parameter. Default is Constant(1.).
    name : str or None, optional
        The name of the layer. Default is None.
    param_type : type, optional
        The type of the parameter. Default is :class:`ETraceParam`.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a GroupNorm layer with 8 groups for 64 channels
        >>> gn = braintrace.nn.GroupNorm(num_groups=8, num_channels=64)
        >>>
        >>> # Input with batch size 4 and spatial dimensions
        >>> x = brainstate.random.randn(4, 32, 32, 64)
        >>> y = gn(x)
        >>> print(y.shape)
        (4, 32, 32, 64)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        *args,
        param_type: type = ETraceParam,
        **kwargs,
    ):
        weight_type = partial(
            param_type,
            op=ScaleOp(is_diagonal=True),
            grad='full',
        )
        super().__init__(*args, param_type=weight_type, **kwargs)
