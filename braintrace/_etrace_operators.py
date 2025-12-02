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

import contextlib
import threading
from typing import Callable, Optional, Dict, Sequence, Any

import brainstate
import brainunit as u
import jax
import numpy as np

__all__ = [
    'ETraceOp',  # base class
    'MatMulOp',  # x @ f(w * m) + b
    'ElemWiseOp',  # element-wise operation
    'ConvOp',  # x [convolution] f(w * m) + bias
    'SpMatMulOp',  # x @ f(sparse_weight) + b
    'LoraOp',  # low-rank approximation

    'general_y2w',
    'stop_param_gradients',  # stop weight gradients
]

_etrace_op_name = '_etrace_operator_call'
_etrace_op_name_enable_grad = f'{_etrace_op_name}_enable_grad_'
_etrace_op_name_elemwise = f'{_etrace_op_name}_enable_grad_elemwise_'

X = brainstate.typing.ArrayLike
W = brainstate.typing.PyTree
Y = brainstate.typing.ArrayLike


class OperatorContext(threading.local):
    """
    The context for the eligibility trace operator.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a new instance of the OperatorContext class.

        This constructor initializes the context for the eligibility trace operator,
        setting up the initial state for stopping parameter gradients.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.stop_param_gradient = [False]


context = OperatorContext()


@contextlib.contextmanager
def stop_param_gradients(stop_or_not: bool = True):
    """
    Stop the weight gradients for the ETrace weight operator.

    Example::

      >>> import braintrace
      >>> with braintrace.stop_param_gradients():
      >>>    # do something

    Args:
        stop_or_not: Whether to stop the weight gradients.
    """
    try:
        context.stop_param_gradient.append(stop_or_not)
        yield
    finally:
        context.stop_param_gradient.pop()


def wrap_etrace_fun(fun, name: str = _etrace_op_name):
    """
    Wraps a function by assigning it a new name.

    This utility function is used to rename a given function, which can be useful
    for tracking or identifying functions during debugging or logging.

    Args:
        fun: The function to be wrapped and renamed.
        name: The new name to assign to the function. Defaults to '_etrace_operator_call'.

    Returns:
        The original function with its name attribute set to the specified name.
    """
    fun.__name__ = name
    return fun


def is_etrace_op(jit_param_name: str):
    """
    Determines if a given jitted parameter name corresponds to an eligibility trace operator.

    This function checks if the provided parameter name starts with a predefined prefix
    that identifies it as an eligibility trace operator.

    Args:
        jit_param_name (str): The name of the jitted parameter to check.

    Returns:
        bool: True if the parameter name indicates an eligibility trace operator, False otherwise.
    """
    return jit_param_name.startswith(_etrace_op_name)


def is_etrace_op_enable_gradient(jit_param_name: str) -> bool:
    """
    Determines if a given jitted parameter name corresponds to an eligibility trace operator
    with the gradient enabled.

    This function checks if the provided parameter name starts with a predefined prefix
    that identifies it as an eligibility trace operator with gradient capabilities.

    Args:
        jit_param_name (str): The name of the jitted parameter to check.

    Returns:
        bool: True if the parameter name indicates an eligibility trace operator with
        gradient enabled, False otherwise.
    """
    return jit_param_name.startswith(_etrace_op_name_enable_grad)


def is_etrace_op_elemwise(jit_param_name: str):
    """
    Determines if a given jitted parameter name corresponds to an element-wise eligibility trace operator.

    This function checks if the provided parameter name starts with a predefined prefix
    that identifies it as an element-wise eligibility trace operator.

    Args:
        jit_param_name (str): The name of the jitted parameter to check.

    Returns:
        bool: True if the parameter name indicates an element-wise eligibility trace operator, False otherwise.
    """
    return jit_param_name.startswith(_etrace_op_name_elemwise)


def general_y2w(
    xw2y: Callable[[X, W], Y],
    x: X,
    y: Y,
    w: W,
):
    """
    General function to compute the weight from the hidden dimensional array.

    Args:
        xw2y: The function to compute the output from the input and weight.
        x: The input data.
        y: The hidden dimensional array.
        w: The weight dimensional array.

    Returns:
        The updated weight dimensional array.
    """
    x = u.math.ones_like(x)
    primals, f_vjp = jax.vjp(
        # dimensionless processing
        lambda w: u.get_mantissa(xw2y(x, w)),
        w
    )
    assert y.shape == primals.shape, (
        f'The shape of the hidden_dim_arr must be the same as the primals. '
        f'Got {y.shape} and {primals.shape}'
    )
    w_like = f_vjp(
        # dimensionless processing
        u.get_mantissa(y)
    )[0]
    return w_like


class ETraceOp(brainstate.util.PrettyObject):
    """
    The Eligibility Trace Operator.

    The function must have the signature: ``(x: jax.Array, weight: PyTree) -> jax.Array``.

    Attributes:
        fun: The operator function.
        is_diagonal: bool. Whether the operator is in the hidden diagonal or not.

    Args:
        is_diagonal: bool. Whether the operator is in the hidden diagonal or not.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        is_diagonal: Optional[bool] = False,
        name: Optional[str] = None,
    ):
        super().__init__()

        # whether the operator is in the hidden diagonal
        self.is_diagonal = is_diagonal

        # function JIT name
        if name is None:
            name = (
                _etrace_op_name_enable_grad
                if is_diagonal else
                _etrace_op_name
            )

        # JIT the operator function
        # This is important during compilation of eligibility trace graph
        self._jitted_call = jax.jit(wrap_etrace_fun(self._define_call(), name))

    def _define_call(self):
        return lambda x, weights: self.xw_to_y(x, weights)

    def __pretty_repr_item__(self, k, v):
        if k == '_jitted_call':
            return None, None
        return k, v

    def __call__(
        self,
        inputs: X,
        weights: W,
    ) -> Y:
        y = self._jitted_call(inputs, weights)
        if context.stop_param_gradient[-1] and not self.is_diagonal:
            y = jax.lax.stop_gradient(y)
        return y

    def xw_to_y(
        self,
        inputs: X,
        weights: W,
    ) -> Y:
        """
        This function is used to compute the output of the operator.

        It computes:

        $$
        y = f(x, w)
        $$

        Args:
            inputs: The input data.
            weights: The weight parameters.

        Returns:
            The output of the operator.
        """
        raise NotImplementedError

    def yw_to_w(
        self,
        hidden_dim_arr: Y,
        weight_dim_tree: W,
    ) -> W:
        """
        This function is used to compute the weight from the hidden dimensional array.

        It computes:

        $$
        w = f(y, w)
        $$

        This function is mainly used when computing eligibility trace updates based on
        :py:class:`ParamDimVjpAlgorithm`.
        """
        raise NotImplementedError

    def xy_to_dw(
        self,
        input_dim_arr: X,
        hidden_dim_arr: Y,
        weights: W,
    ) -> W:
        r"""
        Computes the gradient of the weights (dw) with respect to the loss.

        This function is primarily used for computing eligibility trace updates.
        It calculates the weight gradients by performing a vector-Jacobian
        product (VJP) operation. The core idea is to apply the chain rule:

        $$
        dw = \frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial w}
        $$

        where $L$ is the loss, $y$ is the output of the operator ($y = f(x, w)$),
        and $w$ are the weights.

        Args:
            input_dim_arr (X): The input dimensional array, representing the input data ($x$).
            hidden_dim_arr (Y): The hidden dimensional array, representing the gradient of the loss
                with respect to the operator's output ($\frac{\partial L}{\partial y}$).
            weights (W): The current weight parameters of the operator ($w$).

        Returns:
            W: The computed gradient of the weights ($\frac{\partial L}{\partial w}$).
        """
        primals, f_vjp = jax.vjp(lambda w: u.get_mantissa(self.xw_to_y(input_dim_arr, w)), weights)
        assert hidden_dim_arr.shape == primals.shape, (
            f'The shape of the hidden_dim_arr must be the same as the primals. '
            f'Got {hidden_dim_arr.shape} and {primals.shape}'
        )
        w_like = f_vjp(u.get_mantissa(hidden_dim_arr))[0]
        return w_like


class MatMulOp(ETraceOp):
    """
    The matrix multiplication operator for eligibility trace-based gradient learning.

    This operator is used to compute the output of the operator, mathematically:

    If ``apply_weight_fn_before_mask`` is ``False``:

    $$
    y = x @ f(w * m) + b
    $$

    If ``apply_weight_fn_before_mask`` is ``True``:

    $$
    y = x @ (f(w) * m) + b
    $$

    $b$ is the bias term, which can be optional, $m$ is the weight mask,
    and $f$ is the weight function.

    By default, the weight function is the identity function, and
    the weight mask is None.
    """

    def __init__(
        self,
        weight_mask: Optional[jax.Array] = None,
        weight_fn: Callable[[X], X] = lambda w: w,
        apply_weight_fn_before_mask: bool = False,
    ):
        super().__init__(is_diagonal=False)

        # weight mask
        if weight_mask is None:
            pass
        elif isinstance(weight_mask, (np.ndarray, jax.Array, u.Quantity)):
            weight_mask = u.math.asarray(weight_mask)
        else:
            raise TypeError(f'The weight_mask must be an array-like. But got {type(weight_mask)}')
        self.weight_mask = weight_mask

        # weight function
        assert callable(weight_fn), f'The weight_fn must be callable. But got {type(weight_fn)}'
        self.weight_fn = weight_fn

        # apply weight function before mask
        assert isinstance(apply_weight_fn_before_mask, bool), 'apply_weight_fn_before_mask must be a boolean.'
        self.apply_weight_fn_before_mask = apply_weight_fn_before_mask

    def _vjp_weight_fn(self, w, dw):
        _, f_vjp = jax.vjp(self.weight_fn, w)
        dw, = f_vjp(dw)
        return dw

    def _check_weight(self, w: Dict[str, brainstate.typing.ArrayLike]):
        if not isinstance(w, dict):
            raise TypeError(f'{self.__class__.__name__} only supports '
                            f'the dictionary weight. But got {type(w)}')
        if 'weight' not in w:
            raise ValueError(f'The weight must contain the key "weight".')

    def xw_to_y(
        self,
        x: brainstate.typing.ArrayLike,
        w: Dict[str, brainstate.typing.ArrayLike]
    ):
        r"""
        This function is used to compute the output of the operator, mathematically:

        $$
        y = x @ f(w * m) + b
        $$

        if the bias is provided.

        $$
        y = x @ f(w * m)
        $$

        if the bias is not provided.

        """
        self._check_weight(w)

        if self.apply_weight_fn_before_mask:

            # Case 1: apply the weight function before the mask
            weight = self.weight_fn(w['weight'])
            if self.weight_mask is not None:
                weight = weight * self.weight_mask
            y = u.math.matmul(x, weight)

        else:

            # Case 2: apply the weight function after the mask
            weight = w['weight']
            if self.weight_mask is not None:
                weight = weight * self.weight_mask
            weight = self.weight_fn(weight)
            y = u.math.matmul(x, self.weight_fn(weight))

        # add bias
        if 'bias' in w:
            y = y + w['bias']
        return y

    def yw_to_w(
        self,
        hidden_dim_arr: brainstate.typing.ArrayLike,
        weight_dim_tree: Dict[str, brainstate.typing.ArrayLike],
    ) -> Dict[str, brainstate.typing.ArrayLike]:
        """
        This function is used to compute the weight from the hidden dimensional array.

        $$
        w = f(y, w)
        $$

        Args:
            hidden_dim_arr: The hidden dimensional array.
            weight_dim_tree: The weight dimensional tree.

        Returns:
            The updated weight dimensional tree.
        """
        if not isinstance(hidden_dim_arr, (np.ndarray, jax.Array, u.Quantity)):
            raise TypeError(f'The hidden_dim_arr must be an array-like. But got {type(hidden_dim_arr)}')
        self._check_weight(weight_dim_tree)

        weight_like = weight_dim_tree['weight']
        old_weight = weight_like
        if 'bias' in weight_dim_tree:
            bias_like = weight_dim_tree['bias']
        else:
            bias_like = None
        if hidden_dim_arr.ndim == 1:
            assert weight_like.ndim == 2, (
                f'The weight must be a 2D array when hidden_dim_arr is 1D. '
                f'But got the shape {weight_like.shape}'
            )
            if self.weight_mask is None:
                weight_like = weight_like * u.math.expand_dims(hidden_dim_arr, axis=0)
            else:
                raise NotImplementedError(
                    'please apply weight_mask using weight_fn. For example: \n\n'
                    'weight_fn = lambda w: w * mask'
                )
                weight_like = (
                    weight_like *
                    self.weight_mask *
                    u.math.expand_dims(hidden_dim_arr, axis=0)
                )
            if bias_like is not None:
                assert bias_like.ndim == 1, (
                    f'The bias must be a 1D array when hidden_dim_arr is 1D. '
                    f'But got the shape {bias_like.shape}'
                )
                bias_like = bias_like * hidden_dim_arr
        elif hidden_dim_arr.ndim == 2:
            assert weight_like.ndim == 3, (
                f'The weight must be a 3D array when hidden_dim_arr is 2D. '
                f'But got the shape {weight_like.shape}'
            )
            # assume batch size is the first dimension
            if self.weight_mask is None:
                weight_like = weight_like * u.math.expand_dims(hidden_dim_arr, axis=1)
            else:
                raise NotImplementedError(
                    'please apply weight_mask using weight_fn. For example: \n\n'
                    'weight_fn = lambda w: w * mask'
                )
                weight_like = (
                    weight_like *
                    u.math.expand_dims(self.weight_mask, axis=0) *
                    u.math.expand_dims(hidden_dim_arr, axis=1)
                )
            if bias_like is not None:
                assert bias_like.ndim == 2, (
                    f'The bias must be a 2D array when hidden_dim_arr is 2D. '
                    f'But got the shape {bias_like.shape}'
                )
                bias_like = bias_like * hidden_dim_arr
        else:
            raise ValueError(f'The hidden_dim_arr must be a 1D or 2D array. But got the shape {hidden_dim_arr.shape}')

        weight_like = self._vjp_weight_fn(old_weight, weight_like)
        if bias_like is None:
            return {'weight': weight_like}
        else:
            return {'weight': weight_like, 'bias': bias_like}


class ConvOp(ETraceOp):
    r"""
    The convolution operator for eligibility trace-based gradient learning.

    This operator is used to compute the output of the operator, mathematically:

    $$
    y = x \mathrm{[convolution]} f(w * m) + \mathrm{bias}
    $$

    $bias$ is the bias term, which can be optional, $m$ is the weight mask,
    and $f$ is the weight function.
    """

    def __init__(
        self,
        xinfo: jax.ShapeDtypeStruct,
        window_strides: Sequence[int],
        padding: str | Sequence[tuple[int, int]],
        lhs_dilation: Sequence[int] | None = None,
        rhs_dilation: Sequence[int] | None = None,
        feature_group_count: int = 1,
        batch_group_count: int = 1,
        dimension_numbers: Any = None,
        weight_mask: Optional[jax.Array] = None,
        weight_fn: Callable[[X], X] = lambda w: w,
    ):
        super().__init__(is_diagonal=False)

        self.window_strides = window_strides
        self.padding = padding
        self.lhs_dilation = lhs_dilation
        self.rhs_dilation = rhs_dilation
        self.feature_group_count = feature_group_count
        self.batch_group_count = batch_group_count
        self.dimension_numbers = dimension_numbers

        # weight mask
        if weight_mask is None:
            pass
        elif isinstance(weight_mask, (np.ndarray, jax.Array, u.Quantity)):
            weight_mask = u.math.asarray(weight_mask)
        else:
            raise TypeError(f'The weight_mask must be an array-like. But got {type(weight_mask)}')
        self.weight_mask = weight_mask

        # weight function
        assert callable(weight_fn), f'The weight_fn must be callable. But got {type(weight_fn)}'
        self.weight_fn = weight_fn

        # input info
        self.xinfo = xinfo

    def _check_weight(self, w: Dict[str, brainstate.typing.ArrayLike]):
        if not isinstance(w, dict):
            raise TypeError(f'{self.__class__.__name__} only supports '
                            f'the dictionary weight. But got {type(w)}')
        if 'weight' not in w:
            raise ValueError(f'The weight must contain the key "weight".')

    def _vjp_weight_fn(self, w, dw):
        _, f_vjp = jax.vjp(self.weight_fn, w)
        dw, = f_vjp(dw)
        return dw

    def _pure_convolution_without_batch(
        self,
        inputs: X,
        weights: W,
    ) -> Y:
        self._check_weight(weights)
        if inputs.ndim == self.xinfo.ndim:
            inputs = u.math.expand_dims(inputs, axis=0)
        elif inputs.ndim == self.xinfo.ndim + 1:
            inputs = inputs  # already has batch dimension
        else:
            raise ValueError(
                f'The inputs must have the same number of dimensions as xinfo. '
                f'Got {inputs.ndim} and {self.xinfo.ndim}'
            )
        inputs, input_unit = u.split_mantissa_unit(inputs)
        weight, weight_unit = u.split_mantissa_unit(weights['weight'])

        # convolution
        y = jax.lax.conv_general_dilated(
            lhs=inputs,
            rhs=weight,
            window_strides=self.window_strides,
            padding=self.padding,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.rhs_dilation,
            feature_group_count=self.feature_group_count,
            batch_group_count=self.batch_group_count,
            dimension_numbers=self.dimension_numbers
        )
        y = u.maybe_decimal(y * input_unit * weight_unit)

        # bias
        if 'bias' in weights:
            y = y + weights['bias']
        return u.math.squeeze(y, axis=0)

    def xw_to_y(
        self,
        inputs: X,
        weights: W,
    ) -> Y:
        # weight processing
        weights = {k: v for k, v in weights.items()}
        if self.weight_mask is not None:
            weights['weight'] = weights['weight'] * self.weight_mask
        weights['weight'] = self.weight_fn(weights['weight'])
        # convolution
        if inputs.ndim == self.xinfo.ndim:
            return self._pure_convolution_without_batch(inputs, weights)
        elif inputs.ndim == self.xinfo.ndim + 1:
            return jax.vmap(self._pure_convolution_without_batch, in_axes=(0, None))(inputs, weights)
        else:
            raise ValueError(
                f'The inputs must have the same number of dimensions as xinfo or xinfo + 1. '
                f'Got {inputs.ndim} and {self.xinfo.ndim}'
            )

    def yw_to_w(
        self,
        hidden_dim_arr: brainstate.typing.ArrayLike,
        weight_dim_tree: Dict[str, brainstate.typing.ArrayLike],
    ) -> Dict[str, brainstate.typing.ArrayLike]:
        """
        This function is used to compute the weight from the hidden dimensional array.

        $$
        w = f(y, w)
        $$

        Args:
            hidden_dim_arr: The hidden dimensional array.
            weight_dim_tree: The weight dimensional tree.

        Returns:
            The updated weight dimensional tree.
        """
        self._check_weight(weight_dim_tree)
        w_like = general_y2w(
            self._pure_convolution_without_batch,
            self.xinfo,
            hidden_dim_arr,
            weight_dim_tree,
        )
        old_weight = weight_dim_tree['weight']
        new_weight = jax.tree.map(u.math.multiply, weight_dim_tree, w_like)
        new_weight['weight'] = self._vjp_weight_fn(old_weight, new_weight['weight'])
        return new_weight


class SpMatMulOp(ETraceOp):
    """
    The sparse matrix-vector multiplication operator for eligibility trace-based gradient learning.

    This operator is used to compute the output of the operator, mathematically:

    $$
    y = v @ f(w) + b,
    $$

    $b$ is the bias term, which can be optional, $f$ is the weight function, and $v$ is the input vector.

    By default, the weight function is the identity function.

    .. note::

       The sparse matrix must be the instance of ``brainunit.sparse.SparseMatrix``,
       which implements the protocol method ``.yw_to_w()`` that we need.

    """

    def __init__(
        self,
        sparse_mat: u.sparse.SparseMatrix,
        weight_fn: Callable[[X], X] = lambda w: w,
    ):
        super().__init__(is_diagonal=False)

        # sparse matrix
        assert isinstance(sparse_mat, u.sparse.SparseMatrix), (
            f'The sparse_mat must be a SparseMatrix. But we got {type(sparse_mat)}'
        )
        self.sparse_mat = sparse_mat

        # weight function
        assert callable(weight_fn), f'The weight_fn must be callable. But got {type(weight_fn)}'
        self.weight_fn = weight_fn

    def _check_weight(self, w: W, check_shape: bool = True):
        if not isinstance(w, dict):
            raise TypeError(f'{self.__class__.__name__} only supports '
                            f'the dictionary weight. But got {type(w)}')
        if 'weight' not in w:
            raise ValueError(f'The weight must contain the key "weight".')
        if check_shape:
            if w['weight'].shape != self.sparse_mat.data.shape:
                raise ValueError(f'The shape of the weight must be the same as the sparse matrix data. '
                                 f'Got {w["weight"].shape} and {self.sparse_mat.data.shape}.')
        if w['weight'].dtype != self.sparse_mat.data.dtype:
            raise ValueError(f'The dtype of the weight must be the same as the sparse matrix data. '
                             f'Got {w["weight"].dtype} and {self.sparse_mat.data.dtype}.')
        if u.get_unit(w['weight']) != u.get_unit(self.sparse_mat.data):
            raise ValueError(f'The unit of the weight must be the same as the sparse matrix data. '
                             f'Got {u.get_unit(w["weight"])} and {u.get_unit(self.sparse_mat.data)}.')

    def _vjp_weight_fn(self, w, dw):
        _, f_vjp = jax.vjp(self.weight_fn, w)
        dw, = f_vjp(dw)
        return dw

    def xw_to_y(
        self,
        x: brainstate.typing.ArrayLike,
        w: Dict[str, brainstate.typing.ArrayLike]
    ):
        r"""
        This function is used to compute the output of the operator, mathematically:

        $$
        y = x @ f(w) + b
        $$
        """
        self._check_weight(w)
        weight = self.weight_fn(w['weight'])
        sparse_mat = self.sparse_mat.with_data(weight)
        y = x @ sparse_mat
        if 'bias' in w:
            y = y + w['bias']
        return y

    def yw_to_w(
        self,
        hidden_dim_arr: brainstate.typing.ArrayLike,
        weight_dim_tree: Dict[str, brainstate.typing.ArrayLike],
    ) -> Dict[str, brainstate.typing.ArrayLike]:
        """
        This function is used to compute the weight from the hidden dimensional array.

        $$
        w = f(y, w)
        $$

        Args:
            hidden_dim_arr: The hidden dimensional array.
            weight_dim_tree: The weight dimensional tree.

        Returns:
            The updated weight dimensional tree.
        """
        self._check_weight(weight_dim_tree, check_shape=False)
        weight_like: brainstate.typing.ArrayLike = weight_dim_tree['weight']
        old_weight = weight_like
        if 'bias' in weight_dim_tree:
            bias_like = weight_dim_tree['bias']
        else:
            bias_like = None
        assert hidden_dim_arr.ndim == 1, (
            f'The hidden_dim_arr must be a 1D array. But got the shape {hidden_dim_arr.shape}'
        )
        weight_like = self.sparse_mat.yw_to_w_transposed(hidden_dim_arr, weight_like)
        if bias_like is not None:
            assert bias_like.ndim == 1, (
                f'The bias must be a 1D array when hidden_dim_arr is 1D. '
                f'But got the shape {bias_like.shape}'
            )
            bias_like = bias_like * hidden_dim_arr
        weight_like = self._vjp_weight_fn(old_weight, weight_like)
        if bias_like is None:
            return {'weight': weight_like}
        else:
            return {'weight': weight_like, 'bias': bias_like}


class LoraOp(ETraceOp):
    r"""
    The low-rank approximation operator for eligibility trace-based gradient learning.

    This operator is used to compute the output of the operator, mathematically:

    $$
    y = \alpha x B A + b
    $$

    $b$ is the bias term, which can be optional, $\alpha$ is the scaling factor,
    $A$ is the weight matrix, $B$ is the low-rank matrix, and $x$ is the input data.

    """

    def __init__(
        self,
        alpha: Optional[brainstate.typing.ArrayLike] = None,
    ):
        super().__init__(is_diagonal=False)

        # weight mask
        if alpha is not None:
            alpha = u.math.asarray(alpha)
        self.alpha = alpha

    def _check_weight(self, w: W):
        if not isinstance(w, dict):
            raise TypeError(f'{self.__class__.__name__} only supports '
                            f'the dictionary weight. But got {type(w)}')
        if 'B' not in w:
            raise ValueError(f'The weight must contain the key "B".')
        if 'A' not in w:
            raise ValueError(f'The weight must contain the key "A".')

    def xw_to_y(
        self,
        x: brainstate.typing.ArrayLike,
        w: Dict[str, brainstate.typing.ArrayLike]
    ):
        r"""
        This function is used to compute the output of the operator, mathematically:

        $$
        y = \alpha * x @ B @ A + b
        $$

        Args:
            x: The input data.
            w: The weight parameters.

        Returns:
            The output of the operator.
        """
        self._check_weight(w)
        if self.alpha is not None:
            x = self.alpha * x
        y = x @ w['B'] @ w['A']
        if 'bias' in w:
            y = y + w['bias']
        return y

    def yw_to_w(
        self,
        hidden_dim_arr: brainstate.typing.ArrayLike,
        weight_dim_tree: Dict[str, brainstate.typing.ArrayLike],
    ) -> Dict[str, brainstate.typing.ArrayLike]:
        """
        This function is used to compute the weight from the hidden dimensional array.

        $$
        w = f(y, w)
        $$

        Args:
            hidden_dim_arr: The hidden dimensional array.
            weight_dim_tree: The weight dimensional tree.

        Returns:
            The updated weight dimensional tree.
        """
        if not isinstance(hidden_dim_arr, (np.ndarray, jax.Array, u.Quantity)):
            raise TypeError(f'The hidden_dim_arr must be an array-like. But got {type(hidden_dim_arr)}')
        self._check_weight(weight_dim_tree)

        B_like = weight_dim_tree['B']
        A_like = weight_dim_tree['A']
        if 'bias' in weight_dim_tree:
            bias_like = weight_dim_tree['bias']
        else:
            bias_like = None
        if hidden_dim_arr.ndim == 1:
            assert B_like.ndim == 2 and A_like.ndim == 2, (
                f'The weight must be a 2D array when hidden_dim_arr is 1D. '
                f'But got the shape of B = {B_like.shape}, A = {A_like.shape}.'
            )
            A_like = (
                A_like *
                u.math.expand_dims(hidden_dim_arr, axis=0)
            )
            if bias_like is not None:
                assert bias_like.ndim == 1, (
                    f'The bias must be a 1D array when hidden_dim_arr is 1D. '
                    f'But got the shape {bias_like.shape}'
                )
                bias_like = bias_like * hidden_dim_arr
        elif hidden_dim_arr.ndim == 2:
            assert B_like.ndim == 3 and A_like.ndim == 3, (
                f'The weight must be a 3D array when hidden_dim_arr is 2D. '
                f'But got the shape B = {B_like.shape}, A = {A_like.shape}.'
            )
            # assume batch size is the first dimension
            A_like = (
                A_like *
                u.math.expand_dims(hidden_dim_arr, axis=1)
            )
            if bias_like is not None:
                assert bias_like.ndim == 2, (
                    f'The bias must be a 2D array when hidden_dim_arr is 2D. '
                    f'But got the shape {bias_like.shape}'
                )
                bias_like = bias_like * hidden_dim_arr
        else:
            raise ValueError(f'The hidden_dim_arr must be a 1D or 2D array. But got the shape {hidden_dim_arr.shape}')
        if bias_like is None:
            return {'B': B_like, 'A': A_like}
        else:
            return {'B': B_like, 'A': A_like, 'bias': bias_like}


class ElemWiseOp(ETraceOp):
    """
    The element-wise operator for the eligibility trace-based gradient learning.
    
    This interface can be used to define any element-wise operation between weight parameters and hidden states. 

    .. note::

        Different from the :py:class:`StandardOp`, the element-wise operator does not require the input data.
        Its function signature is ``(w: PyTree) -> ndarray``.

        The most important thing is that the element-wise operator must generate the output with
        the same shape as the hidden states.

    Args:
        fn: the element-wise function, which must have the signature: ``(w: PyTree) -> ndarray``.
    """

    def __init__(
        self,
        fn: Callable = lambda w: w,
    ):
        self._raw_fn = fn
        super().__init__(is_diagonal=True, name=_etrace_op_name_elemwise)

    def __pretty_repr_item__(self, k, v):
        if k == '_jitted_call':
            return None
        if k == '_raw_fn':
            return 'fn', v
        return k, v

    def _define_call(self):
        return lambda weights: self._raw_fn(weights) * 1.0

    def __call__(self, weights: W) -> Y:
        return self._jitted_call(weights)

    def xw_to_y(
        self,
        inputs: Optional[X],
        weights: W
    ) -> Y:
        r"""
        This function is used to compute the output of the element-wise operator.

        It computes:

        $$
        y = f(w)
        $$

        Args:
            inputs: The input data. It is None.
            weights: The weight parameters.

        Returns:
            The output of the operator.
        """
        return self._raw_fn(weights)

    def yw_to_w(
        self,
        hidden_dim_arr: Y,
        weight_dim_tree: W,
    ) -> W:
        r"""
        This function is used to compute the weight from the hidden dimensional array.

        It computes:

        $$
        w = f(y, w)
        $$

        Args:
            hidden_dim_arr: The hidden dimensional array.
            weight_dim_tree: The weight dimensional tree.

        Returns:
            The updated weight dimensional tree.
        """
        prim, f_vjp = jax.vjp(
            # dimensionless processing
            lambda w: u.get_mantissa(self._raw_fn(w)),
            weight_dim_tree
        )
        assert hidden_dim_arr.shape == prim.shape, (
            f'The shape of the hidden_dim_arr must be the same as the weight_dim_tree. '
            f'Got {hidden_dim_arr.shape} and {prim.shape}'
        )
        y_to_w = f_vjp(
            # dimensionless processing
            u.get_mantissa(hidden_dim_arr)
        )[0]
        # new_w = y_to_w * old_w
        new_w = jax.tree.map(
            lambda w1, w2: w1 * w2,
            weight_dim_tree,
            y_to_w,
        )
        return new_w

    def xy_to_dw(
        self,
        input_dim_arr: Optional[X],
        hidden_dim_arr: Y,
        weights: W,
    ) -> W:
        r"""
        Computes the gradient of the weights (dw) based on the hidden dimensional array.

        For element-wise operations, the computation is typically:

        $$
        dw = \frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial w}
        $$

        where $y = f(w)$. This method computes $\frac{\partial y}{\partial w}$ and multiplies
        it by the incoming gradient $\frac{\partial L}{\partial y}$ (represented by `hidden_dim_arr`).

        Args:
            input_dim_arr: The input dimensional array. This is not used for ElemWiseOp.
            hidden_dim_arr: The hidden dimensional array, representing the gradient of the loss
                            with respect to the operator's output ($dL/dy$).
            weights: The current weight parameters of the operator.

        Returns:
            The computed gradient of the weights (dw).
        """

        primals, f_vjp = jax.vjp(
            # dimensionless processing
            lambda w: u.get_mantissa(self._raw_fn(w)),
            weights
        )
        assert hidden_dim_arr.shape == primals.shape, (
            f'The shape of the hidden_dim_arr must be the same as the primals. '
            f'Got {hidden_dim_arr.shape} and {primals.shape}'
        )
        return f_vjp(
            # dimensionless processing
            u.get_mantissa(hidden_dim_arr)
        )[0]
