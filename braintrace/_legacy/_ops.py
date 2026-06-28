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

"""Legacy (v0.1.x and 0.0.x) ETraceOp shims.

Each op's ``xw_to_y`` routes through the new ETP primitive user-API
(``braintrace.matmul``, ``braintrace.conv``, ...). Each op also exposes
a ``raw_xw_to_y`` that performs the same computation with plain JAX
ops, used by :class:`NonTempParam` / :class:`FakeETraceParam` to keep
the weight *out* of the ETP graph.
"""



import contextlib
import threading
from typing import Any, Callable, Optional, Sequence

import jax
import numpy as np
import brainunit as u

from .._op import (
    conv as _new_conv,
    element_wise as _new_element_wise,
    lora_matmul as _new_lora_matmul,
    matmul as _new_matmul,
    sparse_matmul as _new_sparse_matmul,
)

__all__ = [
    'ETraceOp',
    'MatMulOp',
    'ElemWiseOp',
    'ConvOp',
    'SpMatMulOp',
    'LoraOp',
    'general_y2w',
    'stop_param_gradients',
]

# ---------------------------------------------------------------------------
# stop_param_gradients context (kept for API compat; no effect on shim path)
# ---------------------------------------------------------------------------

class _OpContext(threading.local):
    def __init__(self):
        super().__init__()
        self.stop = [False]


_ctx = _OpContext()


@contextlib.contextmanager
def stop_param_gradients(stop_or_not: bool = True):
    """Context manager — legacy compat shim. Pushes a flag onto a stack;
    has no effect in the new ETP primitive path (gradients flow through
    JAX autodiff on the primitive directly).
    """
    _ctx.stop.append(stop_or_not)
    try:
        yield
    finally:
        _ctx.stop.pop()


def general_y2w(xw2y: Callable, x, y, w):
    """Legacy helper: VJP-based y→w pullback."""
    x = u.math.ones_like(x)
    primals, f_vjp = jax.vjp(lambda w_: u.get_mantissa(xw2y(x, w_)), w)
    assert y.shape == primals.shape, (
        f'shape mismatch: {y.shape} vs {primals.shape}'
    )
    return f_vjp(u.get_mantissa(y))[0]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ETraceOp:
    r"""Legacy base class for eligibility-trace operators.

    Defines the operator interface used by the legacy parameter shims: a
    forward map from inputs and weights to outputs (``xw_to_y``), a
    plain-JAX variant (``raw_xw_to_y``), and gradient helpers.

    .. deprecated:: 0.2.0
        Prefer calling the new ETP primitive user-API functions directly
        (:func:`braintrace.matmul`, :func:`braintrace.conv`, etc.).

    Parameters
    ----------
    is_diagonal : bool, optional
        Whether the operator is element-wise (diagonal). Default ``False``.
    name : str, optional
        Optional operator name.

    Notes
    -----
    This class is a deprecated back-compatibility shim. New code should
    call the ETP primitive functions directly.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        is_diagonal: Optional[bool] = False,
        name: Optional[str] = None,
    ):
        self.is_diagonal = is_diagonal
        self.name = name

    def __call__(self, inputs, weights):
        return self.xw_to_y(inputs, weights)

    def xw_to_y(self, inputs, weights):
        raise NotImplementedError

    def raw_xw_to_y(self, inputs, weights):
        """Compute the forward map without emitting an ETP primitive.

        Plain-JAX variant of :meth:`xw_to_y`, used by
        :class:`NonTempParam` and :class:`FakeETraceParam` so those
        wrappers do not register an eligibility-trace relation. The base
        implementation falls through to :meth:`xw_to_y`; subclasses that
        route to an ETP primitive must override this.

        Parameters
        ----------
        inputs : Any
            The input array(s) to the operator.
        weights : Any
            The weight pytree.

        Returns
        -------
        Any
            The operator output computed with plain JAX ops.
        """
        return self.xw_to_y(inputs, weights)

    def yw_to_w(self, hidden_dim_arr, weight_dim_tree):
        raise NotImplementedError

    def xy_to_dw(self, input_dim_arr, hidden_dim_arr, weights):
        primals, f_vjp = jax.vjp(
            lambda w: u.get_mantissa(self.xw_to_y(input_dim_arr, w)),
            weights,
        )
        assert hidden_dim_arr.shape == primals.shape, (
            f'shape mismatch: {hidden_dim_arr.shape} vs {primals.shape}'
        )
        return f_vjp(u.get_mantissa(hidden_dim_arr))[0]


# ---------------------------------------------------------------------------
# MatMulOp
# ---------------------------------------------------------------------------

class MatMulOp(ETraceOp):
    r"""Legacy dense matrix-multiplication operator.

    Routes :meth:`xw_to_y` to :func:`braintrace.matmul`. The weight is
    supplied as a dict ``{'weight': ..., 'bias': <optional>}``.

    .. deprecated:: 0.2.0
        Use :func:`braintrace.matmul` directly.

    Parameters
    ----------
    weight_mask : array_like, optional
        Optional multiplicative mask applied to the weight. Default
        ``None``.
    weight_fn : Callable, optional
        Function applied to the (possibly masked) weight before the
        matmul. Default is the identity ``lambda w: w``.
    apply_weight_fn_before_mask : bool, optional
        If ``True``, ``weight_fn`` is applied before the mask; otherwise
        after. Default ``False``.

    Notes
    -----
    This class is a deprecated back-compatibility shim.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        weight_mask: Optional[Any] = None,
        weight_fn: Callable = lambda w: w,
        apply_weight_fn_before_mask: bool = False,
    ):
        super().__init__(is_diagonal=False)
        if weight_mask is None:
            pass
        elif isinstance(weight_mask, (np.ndarray, jax.Array, u.Quantity)):
            weight_mask = u.math.asarray(weight_mask)
        else:
            raise TypeError(
                f'weight_mask must be array-like, got {type(weight_mask)}'
            )
        self.weight_mask = weight_mask
        assert callable(weight_fn), f'weight_fn must be callable, got {type(weight_fn)}'
        self.weight_fn = weight_fn
        assert isinstance(apply_weight_fn_before_mask, bool)
        self.apply_weight_fn_before_mask = apply_weight_fn_before_mask

    def _check(self, w):
        if not isinstance(w, dict):
            raise TypeError(f'MatMulOp weight must be dict, got {type(w)}')
        if 'weight' not in w:
            raise ValueError("MatMulOp weight dict must contain 'weight'")

    def _process_weight(self, w):
        if self.apply_weight_fn_before_mask:
            W = self.weight_fn(w['weight'])
            if self.weight_mask is not None:
                W = W * self.weight_mask
        else:
            W = w['weight']
            if self.weight_mask is not None:
                W = W * self.weight_mask
            W = self.weight_fn(W)
        return W

    def xw_to_y(self, x, w):
        self._check(w)
        return _new_matmul(x, self._process_weight(w), bias=w.get('bias'))

    def raw_xw_to_y(self, x, w):
        self._check(w)
        y = u.math.matmul(x, self._process_weight(w))
        if 'bias' in w:
            y = y + w['bias']
        return y


# ---------------------------------------------------------------------------
# ElemWiseOp
# ---------------------------------------------------------------------------

class ElemWiseOp(ETraceOp):
    r"""Legacy element-wise operator.

    Routes :meth:`xw_to_y` to :func:`braintrace.element_wise`, applying a
    callable element-wise to the weight.

    .. deprecated:: 0.2.0
        Use :func:`braintrace.element_wise` directly.

    Parameters
    ----------
    fn : Callable, optional
        The element-wise function applied to the weight. Default is the
        identity ``lambda w: w``.

    Notes
    -----
    This class is a deprecated back-compatibility shim.
    """
    __module__ = 'braintrace'

    def __init__(self, fn: Callable = lambda w: w):
        super().__init__(is_diagonal=True)
        self._raw_fn = fn

    def __call__(self, weights):
        return self.xw_to_y(None, weights)

    def xw_to_y(self, inputs, weights):
        return _new_element_wise(weights, self._raw_fn)

    def raw_xw_to_y(self, inputs, weights):
        return self._raw_fn(weights)


# ---------------------------------------------------------------------------
# ConvOp
# ---------------------------------------------------------------------------

class ConvOp(ETraceOp):
    r"""Legacy convolution operator.

    Routes :meth:`xw_to_y` to :func:`braintrace.conv`. The weight is
    supplied as a dict ``{'weight': ..., 'bias': <optional>}``.

    .. deprecated:: 0.2.0
        Use :func:`braintrace.conv` directly.

    Parameters
    ----------
    xinfo : jax.ShapeDtypeStruct
        Shape/dtype information describing the convolution input.
    window_strides : Sequence[int]
        Strides of the convolution window.
    padding : Any
        Padding specification passed to the convolution.
    lhs_dilation : Sequence[int], optional
        Dilation factor for the input. Default ``None``.
    rhs_dilation : Sequence[int], optional
        Dilation factor for the kernel. Default ``None``.
    feature_group_count : int, optional
        Number of feature groups. Default ``1``.
    batch_group_count : int, optional
        Number of batch groups. Default ``1``.
    dimension_numbers : Any, optional
        Convolution dimension-numbers specification. Default ``None``.
    weight_mask : array_like, optional
        Optional multiplicative mask applied to the weight. Default
        ``None``.
    weight_fn : Callable, optional
        Function applied to the (possibly masked) weight. Default is the
        identity ``lambda w: w``.

    Notes
    -----
    This class is a deprecated back-compatibility shim.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        xinfo: jax.ShapeDtypeStruct,
        window_strides: Sequence[int],
        padding,
        lhs_dilation: Optional[Sequence[int]] = None,
        rhs_dilation: Optional[Sequence[int]] = None,
        feature_group_count: int = 1,
        batch_group_count: int = 1,
        dimension_numbers: Any = None,
        weight_mask: Optional[Any] = None,
        weight_fn: Callable = lambda w: w,
    ):
        super().__init__(is_diagonal=False)
        self.xinfo = xinfo
        self.window_strides = window_strides
        self.padding = padding
        self.lhs_dilation = lhs_dilation
        self.rhs_dilation = rhs_dilation
        self.feature_group_count = feature_group_count
        self.batch_group_count = batch_group_count
        self.dimension_numbers = dimension_numbers
        if weight_mask is None:
            pass
        elif isinstance(weight_mask, (np.ndarray, jax.Array, u.Quantity)):
            weight_mask = u.math.asarray(weight_mask)
        else:
            raise TypeError(
                f'weight_mask must be array-like, got {type(weight_mask)}'
            )
        self.weight_mask = weight_mask
        assert callable(weight_fn)
        self.weight_fn = weight_fn

    def _check(self, w):
        if not isinstance(w, dict):
            raise TypeError(f'ConvOp weight must be dict, got {type(w)}')
        if 'weight' not in w:
            raise ValueError("ConvOp weight dict must contain 'weight'")

    def _process_weight(self, w):
        W = w['weight']
        if self.weight_mask is not None:
            W = W * self.weight_mask
        return self.weight_fn(W)

    def _conv_kwargs(self):
        return dict(
            strides=self.window_strides,
            padding=self.padding,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.rhs_dilation,
            feature_group_count=self.feature_group_count,
            batch_group_count=self.batch_group_count,
            dimension_numbers=self.dimension_numbers,
        )

    def xw_to_y(self, x, w):
        self._check(w)
        return _new_conv(
            x, self._process_weight(w), bias=w.get('bias'), **self._conv_kwargs()
        )

    def raw_xw_to_y(self, x, w):
        self._check(w)
        W = self._process_weight(w)
        y = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=W,
            window_strides=self.window_strides,
            padding=self.padding,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.rhs_dilation,
            feature_group_count=self.feature_group_count,
            batch_group_count=self.batch_group_count,
            dimension_numbers=self.dimension_numbers,
        )
        if 'bias' in w:
            y = y + w['bias']
        return y


# ---------------------------------------------------------------------------
# SpMatMulOp
# ---------------------------------------------------------------------------

class SpMatMulOp(ETraceOp):
    r"""Legacy sparse matrix-multiplication operator.

    Routes :meth:`xw_to_y` to :func:`braintrace.sparse_matmul`. The weight
    is supplied as a dict ``{'weight': data, 'bias': <optional>}``, where
    ``data`` holds the values of the sparse matrix.

    .. deprecated:: 0.2.0
        Use :func:`braintrace.sparse_matmul` directly.

    Parameters
    ----------
    sparse_mat : brainunit.sparse.SparseMatrix
        The sparse matrix whose structure is reused; its data is replaced
        by the weight values.
    weight_fn : Callable, optional
        Function applied to the weight data before the matmul. Default is
        the identity ``lambda w: w``.

    Notes
    -----
    This class is a deprecated back-compatibility shim.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        sparse_mat,
        weight_fn: Callable = lambda w: w,
    ):
        super().__init__(is_diagonal=False)
        if not isinstance(sparse_mat, u.sparse.SparseMatrix):
            raise TypeError(
                f'sparse_mat must be a brainunit SparseMatrix, got {type(sparse_mat)}'
            )
        self.sparse_mat = sparse_mat
        assert callable(weight_fn)
        self.weight_fn = weight_fn

    def _check(self, w):
        if not isinstance(w, dict):
            raise TypeError(f'SpMatMulOp weight must be dict, got {type(w)}')
        if 'weight' not in w:
            raise ValueError("SpMatMulOp weight dict must contain 'weight'")

    def xw_to_y(self, x, w):
        self._check(w)
        data = self.weight_fn(w['weight'])
        return _new_sparse_matmul(
            x, data, sparse_mat=self.sparse_mat, bias=w.get('bias')
        )

    def raw_xw_to_y(self, x, w):
        self._check(w)
        data = self.weight_fn(w['weight'])
        mat = self.sparse_mat.with_data(data)
        y = x @ mat
        if 'bias' in w:
            y = y + w['bias']
        return y


# ---------------------------------------------------------------------------
# LoraOp
# ---------------------------------------------------------------------------

class LoraOp(ETraceOp):
    r"""Legacy LoRA (low-rank adaptation) operator.

    Routes :meth:`xw_to_y` to :func:`braintrace.lora_matmul`. The weight is
    supplied as a dict ``{'B': ..., 'A': ..., 'bias': <optional>}`` holding
    the two low-rank factors.

    .. deprecated:: 0.2.0
        Use :func:`braintrace.lora_matmul` directly.

    Parameters
    ----------
    alpha : array_like, optional
        Scaling factor applied to the low-rank product. Defaults to
        ``1.0`` when ``None``.

    Notes
    -----
    This class is a deprecated back-compatibility shim.
    """
    __module__ = 'braintrace'

    def __init__(self, alpha: Optional[Any] = None):
        super().__init__(is_diagonal=False)
        if alpha is not None:
            alpha = u.math.asarray(alpha)
        self.alpha = alpha

    def _check(self, w):
        if not isinstance(w, dict):
            raise TypeError(f'LoraOp weight must be dict, got {type(w)}')
        if 'B' not in w or 'A' not in w:
            raise ValueError("LoraOp weight dict must contain 'B' and 'A'")

    def xw_to_y(self, x, w):
        self._check(w)
        alpha = 1.0 if self.alpha is None else self.alpha
        return _new_lora_matmul(x, w['B'], w['A'], alpha=alpha, bias=w.get('bias'))

    def raw_xw_to_y(self, x, w):
        self._check(w)
        if self.alpha is not None:
            x = self.alpha * x
        y = x @ w['B'] @ w['A']
        if 'bias' in w:
            y = y + w['bias']
        return y
