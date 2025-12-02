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

from typing import Callable, Union, Sequence, Optional

import brainstate
import brainunit as u
from braintools import init

from braintrace._etrace_concepts import ETraceParam
from braintrace._etrace_operators import MatMulOp, LoraOp, SpMatMulOp
from braintrace._typing import ArrayLike

__all__ = [
    'Linear',
    'SignedWLinear',
    'SparseLinear',
    'LoRA',
]


class Linear(brainstate.nn.Module):
    """A Linear layer that performs a linear transformation on the input data.

    This class represents a fully connected linear layer, which applies a linear
    transformation to the incoming data: `y = xW + b`, where `x` is the input,
    `W` is the weight matrix, and `b` is the bias vector.

    Parameters
    ----------
    in_size : int or sequence of int
        The size of the input features.
    out_size : int or sequence of int
        The size of the output features.
    w_init : Callable or ArrayLike, optional
        The initializer for the weights. Default is KaimingNormal().
    b_init : Callable or ArrayLike or None, optional
        The initializer for the bias. Default is ZeroInit().
    w_mask : ArrayLike or Callable or None, optional
        An optional mask for the weights. Default is None.
    name : str or None, optional
        The name of the layer. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Attributes
    ----------
    in_size : int or sequence of int
        The size of the input features.
    out_size : int or sequence of int
        The size of the output features.
    w_mask : ArrayLike or Callable or None
        An optional mask for the weights.
    weight_op : ETraceParam
        The parameter object that holds the weights and the operation to be
        performed on them.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a linear layer
        >>> brainstate.environ.set(precision=64)
        >>> linear = braintrace.nn.Linear(in_size=128, out_size=64)
        >>>
        >>> # Input with batch size 10
        >>> x = brainstate.random.randn(10, 128)
        >>> y = linear(x)
        >>> print(y.shape)
        (10, 64)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: Union[int, Sequence[int]],
        out_size: Union[int, Sequence[int]],
        w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
        b_init: Optional[Union[Callable, ArrayLike]] = init.ZeroInit(),
        w_mask: Optional[Union[ArrayLike, Callable]] = None,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size

        # w_mask
        w_shape = (self.in_size[-1], self.out_size[-1])
        b_shape = (self.out_size[-1],)
        self.w_mask = init.param(w_mask, w_shape)

        # weights
        params = dict(weight=init.param(w_init, w_shape, allow_none=False))
        if b_init is not None:
            params['bias'] = init.param(b_init, b_shape, allow_none=False)

        # weight + op
        self.weight_op = param_type(params, op=MatMulOp(self.w_mask))

    def update(self, x):
        return self.weight_op.execute(x)


class SignedWLinear(brainstate.nn.Module):
    """A Linear layer with signed weights.

    This class represents a linear layer where the weights can be constrained
    to have specific signs. It applies a linear transformation to the input
    data, with the option to mask the weights with a sign matrix.

    Parameters
    ----------
    in_size : int or sequence of int
        The size of the input features.
    out_size : int or sequence of int
        The size of the output features.
    w_init : Callable or ArrayLike, optional
        The initializer for the weights. Default is KaimingNormal().
    w_sign : ArrayLike or None, optional
        The sign matrix to constrain the weights. Default is None.
    name : str or None, optional
        The name of the layer. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Attributes
    ----------
    in_size : int or sequence of int
        The size of the input features.
    out_size : int or sequence of int
        The size of the output features.
    weight_op : ETraceParam
        The parameter object that holds the weights and the operation to be
        performed on them.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a signed weight linear layer
        >>> brainstate.environ.set(precision=64)
        >>> w_sign = brainstate.random.choice([-1, 1], size=(64, 32))
        >>> linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32, w_sign=w_sign)
        >>>
        >>> # Input with batch size 5
        >>> x = brainstate.random.randn(5, 64)
        >>> y = linear(x)
        >>> print(y.shape)
        (5, 32)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: Union[int, Sequence[int]],
        out_size: Union[int, Sequence[int]],
        w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
        w_sign: Optional[ArrayLike] = None,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size

        # weights
        w_shape = (self.in_size[-1], self.out_size[-1])
        weight = init.param(w_init, w_shape, allow_none=False)
        op = MatMulOp(
            weight_mask=w_sign,
            weight_fn=u.math.abs,
        )
        self.weight_op = param_type({'weight': weight}, op=op)

    def update(self, x):
        return self.weight_op.execute(x)


class ScaledWSLinear(brainstate.nn.Module):
    """Linear layer with Weight Standardization.

    Applies weight standardization to the weights of the linear layer.

    Parameters
    ----------
    in_size : int or sequence of int
        The input size.
    out_size : int or sequence of int
        The output size.
    w_init : Callable or ArrayLike, optional
        The initializer for the weights. Default is KaimingNormal().
    b_init : Callable or ArrayLike, optional
        The initializer for the bias. Default is ZeroInit().
    w_mask : ArrayLike or Callable or None, optional
        The optional mask of the weights. Default is None.
    ws_gain : bool, optional
        Whether to use gain for the weights. Default is True.
    eps : float, optional
        The epsilon value for the weight standardization. Default is 1e-4.
    name : str or None, optional
        The name of the object. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Attributes
    ----------
    in_size : int or sequence of int
        The input size.
    out_size : int or sequence of int
        The output size.
    w_mask : ArrayLike or Callable or None
        The optional mask of the weights.
    eps : float
        The epsilon value for the weight standardization.
    weight_op : ETraceParam
        The parameter object that holds the weights and the operation.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a weight standardization linear layer
        >>> brainstate.environ.set(precision=64)
        >>> linear = braintrace.nn.ScaledWSLinear(in_size=256, out_size=128, ws_gain=True, eps=1e-4)
        >>>
        >>> # Input with batch size 16
        >>> x = brainstate.random.randn(16, 256)
        >>> y = linear(x)
        >>> print(y.shape)
        (16, 128)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: Union[int, Sequence[int]],
        out_size: Union[int, Sequence[int]],
        w_init: Callable = init.KaimingNormal(),
        b_init: Callable = init.ZeroInit(),
        w_mask: Optional[Union[ArrayLike, Callable]] = None,
        ws_gain: bool = True,
        eps: float = 1e-4,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size

        # w_mask
        self.w_mask = init.param(w_mask, (self.in_size[0], 1))

        # parameters
        self.eps = eps

        # weights
        w_shape = (self.in_size[-1], self.out_size[-1])
        b_shape = (self.out_size[-1],)
        params = dict(weight=init.param(w_init, w_shape, allow_none=False))
        if b_init is not None:
            params['bias'] = init.param(b_init, b_shape, allow_none=False)
        # gain
        if ws_gain:
            params['gain'] = u.math.ones((1, w_shape[-1]), dtype=params['weight'].dtype)

        # operation
        op = MatMulOp(
            weight_mask=self.w_mask,
            weight_fn=lambda w: brainstate.nn.weight_standardization(
                w['weight'], self.eps, w.get('gain', None)),
            apply_weight_fn_before_mask=True
        )

        # weight + op
        self.weight_op = param_type(params, op=op)

    def update(self, x):
        return self.weight_op.execute(x)


class SparseLinear(brainstate.nn.Module):
    """A Linear layer that utilizes a sparse matrix for efficient computation.

    This class represents a linear transformation layer where the weight matrix
    is sparse, allowing for efficient storage and computation. It supports various
    sparse matrix formats such as CSR, CSC, and COO, provided by the `brainunit.sparse`
    module.

    Parameters
    ----------
    sparse_mat : brainunit.sparse.SparseMatrix
        The sparse weight matrix to be used in the linear transformation.
        Can be ``brainunit.sparse.CSR``, ``brainunit.sparse.CSC``,
        ``brainunit.sparse.COO``, or any other sparse matrix format.
    b_init : Callable or ArrayLike or None, optional
        The initializer for the bias. If None, no bias is used. Default is None.
    in_size : brainstate.typing.Size or None, optional
        The size of the input features. If provided, it must match the first n-1
        dimensions of the output size. Default is None.
    name : str or None, optional
        The name of the layer. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Attributes
    ----------
    in_size : brainstate.typing.Size or None
        The size of the input features. If provided, it must match the first n-1
        dimensions of the output size.
    out_size : int
        The size of the output features, determined by the last dimension of the
        sparse matrix.
    weight_op : ETraceParam
        The parameter object that holds the sparse weights and the operation to
        be performed on them.

    Raises
    ------
    AssertionError
        If the first n-1 dimensions of "in_size" and "out_size" do not match.
        If "sparse_mat" is not an instance of u.sparse.SparseMatrix.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>> import brainunit as u
        >>>
        >>> # Create a sparse weight matrix
        >>> brainstate.environ.set(precision=64)
        >>> indices = brainstate.random.randint(0, 512, size=(2, 1000))
        >>> values = brainstate.random.randn(1000)
        >>> sparse_mat = u.sparse.COO((values, indices), shape=(512, 256))
        >>>
        >>> # Create a sparse linear layer
        >>> linear = braintrace.nn.SparseLinear(sparse_mat, b_init=None)
        >>>
        >>> # Input with batch size 8
        >>> x = brainstate.random.randn(8, 512)
        >>> y = linear(x)
        >>> print(y.shape)
        (8, 256)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        sparse_mat: u.sparse.SparseMatrix,
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        in_size: brainstate.typing.Size = None,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # input and output shape
        if in_size is not None:
            self.in_size = in_size
        self.out_size = sparse_mat.shape[-1]
        if in_size is not None:
            assert self.in_size[:-1] == self.out_size[:-1], (
                'The first n-1 dimensions of "in_size" '
                'and "out_size" must be the same.'
            )

        # weights
        assert isinstance(sparse_mat, u.sparse.SparseMatrix), '"weight" must be a brainunit.sparse.SparseMatrix.'
        params = dict(weight=sparse_mat.data)
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size[-1], allow_none=False)
        op = SpMatMulOp(sparse_mat=sparse_mat)  # x @ sparse matrix
        self.weight_op = param_type(params, op=op)

    def update(self, x):
        return self.weight_op.execute(x)


class LoRA(brainstate.nn.Module):
    r"""A standalone LoRA layer.

    LoRA (Low-Rank Adaptation) is a technique used to adapt pre-trained models
    by introducing low-rank matrices into the model's weight matrices. This
    allows for efficient fine-tuning of large models with a reduced number of
    parameters.

    The LoRA layer modifies the original weight matrix :math:`W` by adding a
    low-rank component :math:`\frac{\alpha}{r} B A`, where :math:`B` and :math:`A`
    are learnable matrices of rank :math:`r`, and :math:`\alpha` is a scaling factor.

    .. math::

        W_{\mathrm{LoRA}} = W_{\text{orig}} + \frac{\alpha}{r} B A

    Parameters
    ----------
    in_features : brainstate.typing.Size
        The number of input features.
    lora_rank : int
        The rank of the LoRA dimension.
    out_features : brainstate.typing.Size
        The number of output features.
    alpha : float, optional
        A scaling factor for the LoRA operation. Default is 1.
    base_module : Callable or None, optional
        A base module to call and substitute, if possible. Default is None.
    B_init : Callable or ArrayLike, optional
        Initializer function for the weight matrix B. Default is ZeroInit().
    A_init : Callable or ArrayLike, optional
        Initializer function for the weight matrix A. Default is LecunNormal().
    param_type : type, optional
        The type of the LoRA parameters. Default is ETraceParam.

    Attributes
    ----------
    in_features : brainstate.typing.Size
        The number of input features.
    lora_rank : int
        The rank of the LoRA dimension.
    out_features : brainstate.typing.Size
        The number of output features.
    alpha : float
        A scaling factor for the LoRA operation.
    base_module : Callable or None
        A base module to call and substitute, if possible.
    weight_op : ETraceParam
        The parameter object that holds the LoRA weights and the operation to
        be performed on them.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> # Create a standalone LoRA layer
        >>> brainstate.environ.set(precision=64)
        >>> layer = braintrace.nn.LoRA(3, 2, 4)
        >>> x = brainstate.random.randn(16, 3)
        >>> y = layer(x)
        >>> print(y.shape)
        (16, 4)
        >>>
        >>> # Wrap around existing linear layer
        >>> linear = brainstate.nn.Linear(3, 4)
        >>> wrapper = braintrace.nn.LoRA(3, 2, 4, base_module=linear)
        >>> assert wrapper.base_module == linear
        >>> y = wrapper(x)
        >>> print(y.shape)
        (16, 4)
    """

    def __init__(
        self,
        in_features: brainstate.typing.Size,
        lora_rank: int,
        out_features: brainstate.typing.Size,
        *,
        alpha: float = 1.,
        base_module: Optional[Callable] = None,
        B_init: Union[Callable, ArrayLike] = init.ZeroInit(),
        A_init: Union[Callable, ArrayLike] = init.LecunNormal(),
        param_type: type = ETraceParam,
    ):
        super().__init__()

        # input and output shape
        self.in_size = in_features
        self.out_size = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.alpha = alpha

        # others
        if base_module is not None:
            assert callable(base_module), '"base_module" must be callable.'
        self.base_module = base_module

        # weights
        param = dict(
            B=B_init((self.in_size[-1], lora_rank)),
            A=A_init((lora_rank, self.out_size[-1]))
        )
        op = LoraOp(alpha=self.alpha / self.lora_rank)
        self.weight_op = param_type(param, op=op)

    def update(self, x: ArrayLike):
        out = self.weight_op.execute(x)
        if self.base_module is not None:
            out += self.base_module(x)
        return out
