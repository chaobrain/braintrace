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
import brainunit as u

from braintrace._etrace_op import matmul, sparse_matmul, lora_matmul
from braintrace._typing import ArrayLike

__all__ = [
    'Linear',
    'SignedWLinear',
    'SparseLinear',
    'LoRA',
]


class Linear(brainstate.nn.Linear):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.Linear.__doc__ or '').replace('brainstate', 'braintrace')

    def update(self, x):
        """Apply the linear transform through the ETP ``matmul`` primitive.

        Routing the matrix multiplication through :func:`braintrace.matmul`
        (instead of a plain JAX dot) is what makes ``weight`` eligible for
        online-learning trace computation.

        Parameters
        ----------
        x : ArrayLike
            Input array, of shape ``(..., in_size)``.

        Returns
        -------
        ArrayLike
            The transformed output, of shape ``(..., out_size)``.
        """
        w = self.weight.value['weight']
        if self.w_mask is not None:
            w = w * self.w_mask
        b = self.weight.value.get('bias')
        return matmul(x, w, b)


class SignedWLinear(brainstate.nn.SignedWLinear):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.SignedWLinear.__doc__ or '').replace('brainstate', 'braintrace')

    def update(self, x):
        """Apply the sign-constrained linear transform through ETP ``matmul``.

        The stored weight magnitudes are made non-negative and then given a
        fixed sign before being routed through :func:`braintrace.matmul`, so
        the weight participates in online-learning trace computation.

        Parameters
        ----------
        x : ArrayLike
            Input array, of shape ``(..., in_size)``.

        Returns
        -------
        ArrayLike
            The transformed output, of shape ``(..., out_size)``.
        """
        w = u.math.abs(self.weight.value)
        if self.w_sign is not None:
            w = w * self.w_sign
        return matmul(x, w)


class ScaledWSLinear(brainstate.nn.ScaledWSLinear):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.ScaledWSLinear.__doc__ or '').replace('brainstate', 'braintrace')

    def update(self, x):
        """Apply the weight-standardized linear transform through ETP ``matmul``.

        The weight is standardized (zero-mean, unit-variance per output unit)
        before being routed through :func:`braintrace.matmul`, so it
        participates in online-learning trace computation.

        Parameters
        ----------
        x : ArrayLike
            Input array, of shape ``(..., in_size)``.

        Returns
        -------
        ArrayLike
            The transformed output, of shape ``(..., out_size)``.
        """
        params = self.weight.value
        w = brainstate.nn.weight_standardization(params['weight'], self.eps, params.get('gain', None))
        if self.w_mask is not None:
            w = w * self.w_mask
        b = params.get('bias', None)
        return matmul(x, w, b)


class SparseLinear(brainstate.nn.SparseLinear):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.SparseLinear.__doc__ or '').replace('brainstate', 'braintrace')

    def update(self, x):
        """Apply the sparse linear transform through the ETP ``sparse_matmul``.

        The dense data of the sparse weight is routed through
        :func:`braintrace.sparse_matmul`, so it participates in
        online-learning trace computation.

        Parameters
        ----------
        x : ArrayLike
            Input array, of shape ``(..., in_size)``.

        Returns
        -------
        ArrayLike
            The transformed output, of shape ``(..., out_size)``.
        """
        weight = self.weight.value['weight']
        bias = self.weight.value.get('bias', None)
        return sparse_matmul(x, weight, sparse_mat=self.spar_mat, bias=bias)


class LoRA(brainstate.nn.LoRA):
    r"""A standalone LoRA layer.

    LoRA (Low-Rank Adaptation) injects two low-rank factors into a layer
    so a large pre-trained model can be fine-tuned with far fewer
    parameters. This subclass preserves the upstream
    :class:`brainstate.nn.LoRA` constructor and replaces only the forward
    pass so that the multiplication is routed through
    :func:`braintrace.lora_matmul` and therefore participates in
    eligibility-trace computation.

    The layer adds a low-rank component :math:`\frac{1}{r} B A` to the
    base weight, where :math:`B` and :math:`A` are learnable factors of
    rank :math:`r`:

    .. math::

        W_{\mathrm{LoRA}} = W_{\text{orig}} + \frac{1}{r} B A

    The scaling factor is fixed to ``1 / lora_rank``.

    Parameters
    ----------
    in_features : int
        Number of input features.
    lora_rank : int
        Rank of the LoRA decomposition.
    out_features : int
        Number of output features.
    base_module : brainstate.nn.Module or None, optional
        Optional base layer that is called on ``x`` and added to the LoRA
        branch. Default ``None``.
    kernel_init : Callable or ArrayLike, optional
        Initializer used for **both** ``lora_a`` (in×rank) and ``lora_b``
        (rank×out). Default is ``LecunNormal()``. To get the classic
        "LoRA-zero" initialisation use ``init.ZeroInit()``.
    param_type : type, optional
        ``ParamState`` subclass used to wrap the weights. Default is
        ``brainstate.ParamState``.
    in_size : int or Sequence[int], optional
        Optional explicit input size override. Default ``None``.

    Attributes
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    base_module : brainstate.nn.Module or None
        The optional base layer added to the LoRA branch.
    weight : ParamState
        ``ParamState`` whose value is a dict with two keys: ``'lora_a'``
        of shape ``(in_features, lora_rank)`` and ``'lora_b'`` of shape
        ``(lora_rank, out_features)``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> # Create a standalone LoRA layer
        >>> brainstate.environ.set(precision=64)
        >>> layer = braintrace.nn.LoRA(in_features=3, lora_rank=2, out_features=4)
        >>> x = brainstate.random.randn(16, 3)
        >>> y = layer(x)
        >>> print(y.shape)
        (16, 4)
        >>>
        >>> # Wrap around an existing linear layer
        >>> linear = brainstate.nn.Linear(3, 4)
        >>> wrapper = braintrace.nn.LoRA(3, 2, 4, base_module=linear)
        >>> assert wrapper.base_module is linear
        >>> y = wrapper(x)
        >>> print(y.shape)
        (16, 4)
    """
    __module__ = 'braintrace.nn'

    def update(self, x: ArrayLike):
        r"""Apply the low-rank adaptation through the ETP ``lora_matmul``.

        Computes :math:`\frac{1}{r} B A\, x` via :func:`braintrace.lora_matmul`
        (so the LoRA factors participate in online-learning trace
        computation) and adds the optional base-module output.

        Parameters
        ----------
        x : ArrayLike
            Input array, of shape ``(..., in_features)``.

        Returns
        -------
        ArrayLike
            The adapted output, of shape ``(..., out_features)``.
        """
        param = self.weight.value
        lora_rank = param['lora_b'].shape[0]
        out = lora_matmul(x, param['lora_b'], param['lora_a'], alpha=1.0 / lora_rank)
        if self.base_module is not None:
            out += self.base_module(x)
        return out
