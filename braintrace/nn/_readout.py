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

import numbers
from typing import Callable, Optional

import brainstate
import braintools
import brainunit as u

from braintrace._etrace_concepts import ETraceParam
from braintrace._etrace_operators import MatMulOp
from braintrace._typing import Size, ArrayLike

__all__ = [
    'LeakyRateReadout',
]


class LeakyRateReadout(brainstate.nn.Module):
    """Leaky dynamics for the read-out module used in Real-Time Recurrent Learning.

    The LeakyRateReadout class implements a leaky integration mechanism
    for processing continuous input signals in neural networks. It is
    designed to simulate the dynamics of rate-based neurons, applying
    leaky integration to the input and producing a continuous output
    signal.

    This class is part of the BrainTrace project and integrates with
    the Brain Dynamics Programming ecosystem, providing a biologically
    inspired approach to neural computation.

    Parameters
    ----------
    in_size : Size
        The size of the input to the readout module.
    out_size : Size
        The size of the output from the readout module.
    tau : ArrayLike, optional
        The time constant for the leaky integration dynamics. Default is 5 ms.
    w_init : Callable, optional
        A callable for initializing the weights of the readout module.
        Default is KaimingNormal().
    r_init : Callable, optional
        A callable for initializing the state of the readout module.
        Default is ZeroInit().
    name : str or None, optional
        An optional name for the module. Default is None.

    Attributes
    ----------
    in_size : tuple of int
        The size of the input.
    out_size : tuple of int
        The size of the output.
    tau : ArrayLike
        The time constant for leaky integration.
    decay : ArrayLike
        The decay factor computed from tau.
    r : HiddenState
        The readout state variable.
    weight_op : ETraceParam
        The parameter object that holds the weights and operations.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>> import brainunit as u
        >>>
        >>> # Create a leaky rate readout layer
        >>> readout = braintrace.nn.LeakyRateReadout(
        ...     in_size=256,
        ...     out_size=10,
        ...     tau=5.0 * u.ms
        ... )
        >>> readout.init_state(batch_size=32)
        >>>
        >>> # Process input through the readout layer
        >>> x = brainstate.random.randn(32, 256)
        >>> output = readout(x)
        >>> print(output.shape)
        (32, 10)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        tau: ArrayLike = 5. * u.ms,
        w_init: Callable = braintools.init.KaimingNormal(),
        r_init: Callable = braintools.init.ZeroInit(),
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # parameters
        self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
        self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)
        self.tau = braintools.init.param(tau, self.in_size)
        # Compute decay handling units properly
        tau_normalized = u.maybe_decimal(self.tau / brainstate.environ.get_dt())
        self.decay = u.math.exp(-1.0 / tau_normalized)
        self.r_init = r_init

        # weights
        weight = braintools.init.param(w_init, (self.in_size[0], self.out_size[0]))
        self.weight_op = ETraceParam({'weight': weight}, op=MatMulOp())

    def init_state(self, batch_size=None, **kwargs):
        self.r = brainstate.HiddenState(
            braintools.init.param(self.r_init, self.out_size, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        self.r.value = braintools.init.param(self.r_init, self.out_size, batch_size)

    def update(self, x):
        r = self.decay * self.r.value + self.weight_op.execute(x)
        self.r.value = r
        return r
