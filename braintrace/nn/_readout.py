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
import jax

import brainpy.state
from braintrace._etrace_concepts import ETraceParam
from braintrace._etrace_operators import MatMulOp
from braintrace._typing import Size, ArrayLike, Spike

__all__ = [
    'LeakyRateReadout',
    'LeakySpikeReadout',
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


class LeakySpikeReadout(brainpy.state.Neuron):
    """Integrate-and-fire neuron model for spike-based readout.

    The LeakySpikeReadout class implements a leaky integrate-and-fire
    neuron model used for spike-based readout in neural networks. It
    simulates the dynamics of membrane potential and spike generation
    based on input spikes, using specified parameters such as time
    constant, threshold voltage, and spike function.

    This class is part of the BrainTrace project and is designed to
    integrate with the Brain Dynamics Programming ecosystem, providing
    a biologically inspired approach to neural computation.

    Parameters
    ----------
    in_size : Size
        The size of the input to the readout module.
    out_size : Size
        The size of the output from the readout module.
    tau : ArrayLike, optional
        The time constant for the leaky integration dynamics. Default is 5 ms.
    V_th : ArrayLike, optional
        The threshold voltage for spike generation. Default is 1 mV.
    w_init : Callable, optional
        A callable for initializing the weights of the readout module.
        Default is KaimingNormal(unit=u.mV).
    V_init : Callable, optional
        A callable for initializing the membrane potential. Default is ZeroInit(unit=u.mV).
    spk_fun : Callable, optional
        A callable representing the spike function (surrogate gradient).
        Default is ReluGrad().
    spk_reset : str, optional
        The method for resetting spikes after firing. Can be 'soft' or 'hard'.
        Default is 'soft'.
    name : str or None, optional
        An optional name for the module. Default is None.

    Attributes
    ----------
    in_size : tuple of int
        The size of the input.
    out_size : tuple of int
        The size of the output.
    tau : ArrayLike
        The time constant for membrane dynamics.
    V_th : ArrayLike
        The threshold voltage.
    V : HiddenState
        The membrane potential state variable.
    weight_op : ETraceParam
        The parameter object that holds the weights and operations.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>> import brainunit as u
        >>>
        >>> # Create a leaky spike readout layer
        >>> readout = braintrace.nn.LeakySpikeReadout(
        ...     in_size=512,
        ...     out_size=10,
        ...     tau=10.0 * u.ms,
        ...     V_th=1.0 * u.mV
        ... )
        >>> readout.init_state(batch_size=64)
        >>>
        >>> # Process input spikes through the readout layer
        >>> spike_input = brainstate.random.randn(64, 512) > 0.5
        >>> spike_output = readout(spike_input)
        >>> print(spike_output.shape)
        (64, 10)
    """

    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        tau: ArrayLike = 5. * u.ms,
        V_th: ArrayLike = 1. * u.mV,
        w_init: Callable = braintools.init.KaimingNormal(unit=u.mV),
        V_init: Callable = braintools.init.ZeroInit(unit=u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        # For readout layer, the state size should be out_size, not in_size
        super().__init__(out_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)
        # Store in_size separately
        self.in_size = in_size
        self.out_size = out_size  # varshape is already set to out_size by parent

        # parameters
        self.tau = braintools.init.param(tau, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_init = V_init

        # weights
        weight = braintools.init.param(w_init, (self.in_size[0], self.out_size[0]))
        self.weight_op = ETraceParam({'weight': weight}, op=MatMulOp())

    @property
    def varshape(self):
        return self.out_size

    def init_state(self, batch_size, **kwargs):
        self.V = brainstate.HiddenState(
            braintools.init.param(self.V_init, self.out_size, batch_size))

    def reset_state(self, batch_size, **kwargs):
        self.V.value = braintools.init.param(self.V_init, self.out_size, batch_size)

    @property
    def spike(self):
        return self.get_spike(self.V.value)

    def get_spike(self, V):
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, spike: Spike) -> Spike:
        # reset
        last_V = self.V.value
        last_spike = self.get_spike(last_V)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_V)
        V = last_V - V_th * last_spike
        # membrane potential
        I_syn = self.weight_op.execute(spike)
        dv = lambda v, x: (-v + x) / self.tau
        V = brainstate.nn.exp_euler_step(dv, V, I_syn)
        self.V.value = V
        return self.get_spike(V)
