# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


from typing import Callable

import brainpy
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

import braintrace
import braintools


class ALIF(brainpy.state.Neuron):
    """
    Adaptive Leaky Integrate-and-Fire (LIF) neuron model.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        R: brainstate.typing.ArrayLike = 1. * u.ohm,
        tau: brainstate.typing.ArrayLike = 5. * u.ms,
        tau_a: brainstate.typing.ArrayLike = 100. * u.ms,
        V_th: brainstate.typing.ArrayLike = 1. * u.mV,
        V_reset: brainstate.typing.ArrayLike = 0. * u.mV,
        V_rest: brainstate.typing.ArrayLike = 0. * u.mV,
        beta: brainstate.typing.ArrayLike = 0.1 * u.mV,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        V_initializer: Callable = braintools.init.Constant(0. * u.mV),
        a_initializer: Callable = braintools.init.Constant(0.),
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.tau_a = braintools.init.param(tau_a, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.beta = braintools.init.param(beta, self.varshape)

        # functions
        self.V_initializer = V_initializer
        self.a_initializer = a_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.st = braintrace.ETraceTreeState(
            {
                'V': braintools.init.param(self.V_initializer, self.varshape, batch_size),
                'a': braintools.init.param(self.a_initializer, self.varshape, batch_size),
            }
        )

    def reset_state(self, batch_size: int = None, **kwargs):
        self.st.set_value(
            {
                'V': braintools.init.param(self.V_initializer, self.varshape, batch_size),
                'a': braintools.init.param(self.a_initializer, self.varshape, batch_size),
            }
        )

    def get_spike(self, V=None, a=None):
        V = V if V is not None else self.st.get_value('V')
        a = a if a is not None else self.st.get_value('a')
        v_scaled = (V - self.V_th - self.beta * a) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        last_v = self.st.get_value('V')
        last_a = self.st.get_value('a')
        lst_spk = self.get_spike(last_v, last_a)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        V = last_v - (V_th - self.V_reset) * lst_spk
        a = last_a + lst_spk
        # membrane potential
        dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
        da = lambda a: -a / self.tau_a
        V = brainstate.nn.exp_euler_step(dv, V)
        a = brainstate.nn.exp_euler_step(da, a)
        V = self.sum_delta_inputs(V)
        self.st.set_value({'V': V})
        self.st.set_value({'a': a})
        return self.get_spike(V, a)


class ALIF_ExpCu_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with LIF neurons and dense connected exponential current synapses.
    """

    def __init__(
        self, n_in, n_rec,
        tau_mem=5. * u.ms, tau_syn=10. * u.ms,
        V_th=1. * u.mV, tau_a=100. * u.ms, beta=0.1 * u.mV,
        spk_fun=braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init=braintools.init.KaimingNormal(),
        ff_init=braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = ALIF(
            n_rec, tau=tau_mem, tau_a=tau_a, beta=beta,
            spk_fun=spk_fun, spk_reset=spk_reset,
            V_th=V_th
        )
        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec,
                jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0) * u.mS,
                b_init=braintools.init.ZeroInit(unit=u.mS)
            ),
            syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn),
            out=brainpy.state.CUBA.desc(),
            post=self.neu
        )

    def update(self, spk):
        self.syn(jnp.concat([spk, self.neu.get_spike()], axis=-1))
        self.neu()
        return self.neu.get_spike()


class ALIF_Delta_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with LIF neurons and dense connected delta synapses.
    """

    def __init__(
        self,
        n_in, n_rec, tau_mem=5. * u.ms, tau_a=100. * u.ms, V_th=1. * u.mV, beta=0.1 * u.mV,
        spk_fun=braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init=braintools.init.KaimingNormal(),
        ff_init=braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = ALIF(
            n_rec,
            tau=tau_mem,
            spk_fun=spk_fun,
            spk_reset=spk_reset,
            V_th=V_th,
            tau_a=tau_a,
            beta=beta
        )
        w_init = jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
        self.syn = brainpy.state.DeltaProj(
            comm=braintrace.nn.Linear(n_in + n_rec, n_rec, w_init=w_init * u.mV,
                                      b_init=braintools.init.ZeroInit(unit=u.mV)),
            post=self.neu
        )

    def update(self, spk):
        inp = jnp.concat([spk, self.neu.get_spike()], axis=-1)
        self.syn(inp)
        self.neu()
        return self.neu.get_spike()


class ALIF_STDExpCu_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with LIF neurons and dense connected STD-based exponential current synapses.
    """

    def __init__(
        self, n_in, n_rec, inp_std=False,
        tau_mem=5. * u.ms, tau_syn=10. * u.ms,
        V_th=1. * u.mV, tau_std=500. * u.ms, beta=0.1 * u.mV,
        spk_fun=braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init=braintools.init.KaimingNormal(),
        ff_init=braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = ALIF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th, beta=beta)
        self.std = brainpy.state.STD(n_rec, tau=tau_std, U=0.1)
        if inp_std:
            self.std_inp = brainpy.state.STD(n_in, tau=tau_std, U=0.1)
        else:
            self.std_inp = None

        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec,
                jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0) * u.mS,
                b_init=braintools.init.ZeroInit(unit=u.mS)
            ),
            syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn),
            out=brainpy.state.CUBA.desc(),
            post=self.neu
        )

    def update(self, inp_spk):
        if self.std_inp is not None:
            inp_spk = self.std_inp(inp_spk)
        last_spk = self.neu.get_spike()
        inp = jnp.concat([inp_spk, self.std(last_spk)], axis=-1)
        self.syn(inp)
        self.neu()
        return self.neu.get_spike()


class ALIF_STPExpCu_Dense_Layer(brainstate.nn.Module):
    def __init__(
        self,
        n_in, n_rec, inp_stp=False,
        tau_mem=5. * u.ms, tau_syn=10. * u.ms, V_th=1. * u.mV, beta=0.1 * u.mV,
        tau_f=500. * u.ms, tau_d=100. * u.ms, tau_a=100. * u.ms,
        spk_fun=braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init=braintools.init.KaimingNormal(),
        ff_init=braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.inp_stp = inp_stp
        self.neu = ALIF(
            n_rec,
            tau=tau_mem,
            spk_fun=spk_fun,
            spk_reset=spk_reset,
            V_th=V_th,
            tau_a=tau_a,
            beta=beta,
        )
        self.stp = brainpy.state.STP(n_rec, tau_f=tau_f, tau_d=tau_d)
        if inp_stp:
            self.stp_inp = brainpy.state.STP(n_in, tau_f=tau_f, tau_d=tau_d)

        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec,
                jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])]) * u.mS,
                b_init=braintools.init.ZeroInit(unit=u.mS)
            ),
            syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn),
            out=brainpy.state.CUBA.desc(),
            post=self.neu
        )

    def update(self, inp_spk):
        if self.inp_stp:
            inp_spk = self.stp_inp(inp_spk)
        last_spk = self.neu.get_spike()
        inp = jnp.concat([inp_spk, self.stp(last_spk)], axis=-1)
        self.syn(inp)
        self.neu()
        return self.neu.get_spike()
