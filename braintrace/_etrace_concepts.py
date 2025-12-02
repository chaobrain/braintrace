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
#
# ==============================================================================
#
# Author: Chaoming Wang <chao.brain@qq.com>
# Date: 2024-04-03
# Copyright: 2024, Chaoming Wang
#
# Refinement History:
# [2025-02-06]
#   - Add the `ETraceTreeState` and `ETraceGroupState` for the multiple hidden states.
#   - Add the `ElemWiseParam` for the element-wise eligibility trace parameters.
#   - Remove previous `ETraceParam` and `ETraceParamOp`
#   - Unify the `ETraceParam` and `ETraceParamOp` into the `ETraceParam`
#   - Add the `FakeETraceParam` and `FakeElemWiseParam` for the fake parameter states.
#
# ==============================================================================

# -*- coding: utf-8 -*-

from enum import Enum
from typing import Callable, Optional

import brainstate
import jax

from ._etrace_operators import ETraceOp, ElemWiseOp
from ._misc import BaseEnum

__all__ = [

    # eligibility trace parameters and operations
    'ETraceParam',  # the parameter and operator for the etrace-based learning, combining ETraceParam and ETraceOp
    'NonTempParam',  # the parameter state with an associated operator without temporal dependent gradient learning

    # element-wise eligibility trace parameters
    'ElemWiseParam',  # the element-wise weight parameter for the etrace-based learning

    # fake parameter state
    'FakeETraceParam',  # the fake parameter state with an associated operator
    'FakeElemWiseParam',  # the fake element-wise parameter state with an associated operator
]

X = brainstate.typing.ArrayLike
W = brainstate.typing.PyTree
Y = brainstate.typing.ArrayLike


class ETraceGrad(BaseEnum):
    """
    The Gradient Type for the Eligibility Trace.

    This defines how the weight gradient is computed in the eligibility trace-based learning.

    - `full`: The full eligibility trace gradient is computed.
    - `approx`: The approximated eligibility trace gradient is computed.
    - `adaptive`: The adaptive eligibility trace gradient is computed.

    """
    full = 'full'
    approx = 'approx'
    adaptive = 'adaptive'


class ETraceParam(brainstate.ParamState):
    """
    The Eligibility Trace Weight and its Associated Operator.

    .. note::

        Although one weight is defined as :py:class:`ETraceParam`,
        whether eligibility traces are used for training with temporal
        dependencies depends on the final compilation result of the
        compiler regarding this parameter. If no hidden states are
        found to associate this parameter, the training based on
        eligibility traces will not be performed.
        Then, this parameter will perform the same as :py:class:`NonTempParam`.

    Args:
        weight: The weight of the ETrace.
        op: The operator for the ETrace. See :py:class:`ETraceOp`.
        grad: The gradient type for the ETrace. Default is `adaptive`.
        name: The name of the weight-operator.
    """
    __module__ = 'braintrace'

    value: brainstate.typing.PyTree  # weight
    op: ETraceOp  # operator
    is_etrace: bool  # whether the operator is a true eligibility trace

    def __init__(
        self,
        weight: brainstate.typing.PyTree,
        op: ETraceOp,
        grad: Optional[str | Enum] = None,
        name: Optional[str] = None
    ):
        # weight value
        super().__init__(weight, name=name)

        # gradient
        if grad is None:
            grad = 'adaptive'
        self.gradient = ETraceGrad.get(grad)

        # operation
        assert isinstance(op, ETraceOp), (
            f'op should be ETraceOp. '
            f'But we got {type(op)}.'
        )
        self.op = op

        # check if the operator is not an eligibility trace
        self.is_etrace = True

    def execute(self, x: X) -> Y:
        """
        Execute the operator with the input.

        This method applies the associated operator to the input data and the stored
        parameter value, performing the defined operation.

        Args:
            x (X): The input data on which the operator will be executed.

        Returns:
            Y: The result of applying the operator to the input data and the parameter value.
        """
        return self.op(x, self.value)


class ElemWiseParam(ETraceParam):
    r"""
    The Element-wise Eligibility Trace Weight and its Associated Operator.

    .. note::

        The ``element-wise`` is called with the correspondence to the hidden state.
        That means the operator performs element-wise operations with the hidden state.

    It supports all element-wise operations for the eligibility trace-based learning.
    For example, if the parameter weight has the shape with the same as the hidden state,

    $$
    I = \theta_1 * h_1
    $$

    where $\theta_1 \in \mathbb{R}^H$ is the weight and $h_1 \in \mathbb{R}^H$ is the
    hidden state. The element-wise operation is defined as:

    .. code-block:: python

       op = ElemWiseParam(weight, op=lambda w: w)

    If the parameter weight is a scalar,

    $$
    I = \theta * h
    $$

    where $\theta \in \mathbb{R}$ is the weight and $h \in \mathbb{R}^H$ is the hidden state.
    Then the element-wise operation can be defined as:

    .. code-block:: python

         h = 100  # hidden size
         op = ElemWiseParam(weight, op=lambda w: w * jax.numpy.ones(h))

    Other element-wise operations can be defined in the same way.

    Moreover, :py:class:`ElemWiseParam` support a pytree of element-wise parameters. For example,
    if the mathematical operation is defined as:

    $$
    I = \theta_1 * h_1 + \theta_2 * h_2
    $$

    where $\theta_1 \in \mathbb{R}^H$ and $\theta_2 \in \mathbb{R}^H$ are the weights and
    $h_1 \in \mathbb{R}^H$ and $h_2 \in \mathbb{R}^H$ are the hidden states. The element-wise
    operation can be defined as:

    .. code-block:: python

        op = ElemWiseParam(
            weight={'w1': weight1, 'w2': weight2},
            op=lambda w: w['w1'] * h1 + w['w2'] * h2
        )

    Args:
        weight: The weight of the ETrace.
        op: The operator for the ETrace. See :py:class:`ElemWiseOp`.
        name: The name of the weight-operator.
    """
    __module__ = 'braintrace'
    value: brainstate.typing.PyTree  # weight
    op: ElemWiseOp  # operator

    def __init__(
        self,
        weight: brainstate.typing.PyTree,
        op: ElemWiseOp | Callable[[W], Y] = (lambda w: w),
        name: Optional[str] = None,
    ):
        if not isinstance(op, ElemWiseOp):
            op = ElemWiseOp(op)
        assert isinstance(op, ElemWiseOp), (
            f'op should be ElemWiseOp. '
            f'But we got {type(op)}.'
        )
        super().__init__(
            weight,
            op=op,
            grad=ETraceGrad.full,
            name=name
        )

    def execute(self) -> Y:
        """
        Executes the associated operator on the stored weight.

        This method applies the operator to the weight of the element-wise parameter
        state, performing the defined element-wise operation.

        Returns:
            Y: The result of applying the operator to the weight.
        """
        return self.op(self.value)


class NonTempParam(brainstate.ParamState):
    r"""
    The Parameter State with an Associated Operator with no temporal dependent gradient learning.

    This class behaves the same as :py:class:`ETraceParam`, but will not build the
    eligibility trace graph when using online learning. Therefore, in a sequence
    learning task, the weight gradient can only be computed with the spatial gradients.
    That is,

    $$
    \nabla \theta = \sum_t \partial L^t / \partial \theta^t
    $$

    Instead, the gradient of the weight $\theta$ which is labeled as :py:class:`ETraceParam` is
    computed as:

    $$
    \nabla \theta = \sum_t \partial L^t / \partial \theta = \sum_t \sum_i^t \partial L^t / \partial \theta^i
    $$

    Args:
      value: The value of the parameter.
      op: The operator for the parameter. See `ETraceOp`.
    """
    __module__ = 'braintrace'
    op: Callable[[X, W], Y]  # operator
    value: brainstate.typing.PyTree  # weight

    def __init__(
        self,
        value: brainstate.typing.PyTree,
        op: Callable,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(value, name=name)

        # operation
        if isinstance(op, ETraceOp):
            op = op.xw_to_y
        self.op = op

    def execute(self, x: jax.Array) -> jax.Array:
        """
        Executes the associated operator on the input data and the stored parameter value.

        Args:
            x (jax.Array): The input data on which the operator will be executed.

        Returns:
            jax.Array: The result of applying the operator to the input data and the parameter value.
        """
        return self.op(x, self.value)


class FakeETraceParam(object):
    """
    The Parameter State with an Associated Operator that does not require to compute gradients.

    This class corresponds to the :py:class:`NonTempParam` and :py:class:`ETraceParam` but does
    not require to compute gradients. It has the same usage interface with :py:class:`NonTempParam`
    and :py:class:`ETraceParam`.

    Args:
      value: The value of the parameter.
      op: The operator for the parameter.
    """
    __module__ = 'braintrace'
    op: Callable[[X, W], Y]  # operator
    value: brainstate.typing.PyTree  # weight

    def __init__(
        self,
        value: brainstate.typing.PyTree,
        op: Callable
    ):
        super().__init__()

        self.value = value
        if isinstance(op, ETraceOp):
            op = op.xw_to_y
        self.op = op

    def execute(self, x: brainstate.typing.ArrayLike) -> brainstate.typing.ArrayLike:
        """
        Executes the associated operator on the input data and the stored parameter value.

        Args:
            x (brainstate.typing.ArrayLike): The input data on which the operator will be executed.

        Returns:
            brainstate.typing.ArrayLike: The result of applying the operator to the input data and the parameter value.
        """
        return self.op(x, self.value)


class FakeElemWiseParam(object):
    """
    The fake element-wise parameter state with an associated operator.

    This class corresponds to the :py:class:`ElemWiseParam` but does not require to compute gradients.
    It has the same usage interface with :py:class:`ElemWiseParam`. For usage, please see
    :py:class:`ElemWiseParam`.

    Args:
        weight: The weight of the ETrace.
        op: The operator for the ETrace. See :py:class:`ElemWiseOp`.
        name: The name of the weight-operator.
    """
    __module__ = 'braintrace'
    op: Callable[[W], Y]  # operator
    value: brainstate.typing.PyTree  # weight

    def __init__(
        self,
        weight: brainstate.typing.PyTree,
        op: ElemWiseOp | Callable[[W], Y] = (lambda w: w),
        name: Optional[str] = None,
    ):
        super().__init__()
        if isinstance(op, ETraceOp):
            assert isinstance(op, ElemWiseOp), (
                f'op should be ElemWiseOp. '
                f'But we got {type(op)}.'
            )
            op = op.xw_to_y
        self.op = op
        self.value = weight
        self.name = name

    def execute(self) -> brainstate.typing.ArrayLike:
        """
        Executes the associated operator on the stored weight.

        This method applies the operator to the weight of the fake element-wise parameter
        state, simulating the behavior of an element-wise operation without computing gradients.

        Returns:
            brainstate.typing.ArrayLike: The result of applying the operator to the weight.
        """
        return self.op(None, self.value)
