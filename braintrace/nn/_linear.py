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

import brainstate
import saiunit as u

from braintrace._etrace_operators import matmul, sparse_matmul, lora_matmul
from braintrace._typing import ArrayLike

__all__ = [
    'Linear',
    'SignedWLinear',
    'SparseLinear',
    'LoRA',
]


class Linear(brainstate.nn.Linear):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.Linear.__doc__.replace('brainstate', 'braintrace')

    def update(self, x):
        w = self.weight.value['weight']
        if self.w_mask is not None:
            w = w * self.w_mask
        b = self.weight.value.get('bias')
        return matmul(x, w, b)


class SignedWLinear(brainstate.nn.SignedWLinear):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.SignedWLinear.__doc__.replace('brainstate', 'braintrace')

    def update(self, x):
        w = u.math.abs(self.weight.value)
        if self.w_sign is not None:
            w = w * self.w_sign
        return matmul(x, w)


class ScaledWSLinear(brainstate.nn.ScaledWSLinear):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.ScaledWSLinear.__doc__.replace('brainstate', 'braintrace')

    def update(self, x):
        params = self.weight.value
        w = brainstate.nn.weight_standardization(params['weight'], self.eps, params.get('gain', None))
        if self.w_mask is not None:
            w = w * self.w_mask
        b = params.get('bias', None)
        return matmul(x, w, b)


class SparseLinear(brainstate.nn.SparseLinear):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.SparseLinear.__doc__.replace('brainstate', 'braintrace')

    def update(self, x):
        weight = self.weight.value['weight']
        bias = self.weight.value.get('bias', None)
        return sparse_matmul(x, weight, sparse_mat=self.spar_mat, bias=bias)


class LoRA(brainstate.nn.LoRA):
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
    weight_op : ParamState
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
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.LoRA.__doc__.replace('brainstate', 'braintrace')

    def update(self, x: ArrayLike):
        param = self.weight.value
        alpha = 1.
        lora_rank = param['lora_b'].shape[0]
        out = lora_matmul(x, param['lora_b'], param['lora_a'], alpha=alpha / lora_rank)
        if self.base_module is not None:
            out += self.base_module(x)
        return out
