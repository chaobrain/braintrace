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


import brainpy
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp

import braintrace


class IF_Delta_Dense_Layer(brainstate.nn.Module):
    def __init__(
        self, n_in, n_rec, tau_mem=5. * u.ms, V_th=1. * u.mV, spk_reset: str = 'soft',
        rec_init=braintools.init.KaimingNormal(),
        ff_init=braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = brainpy.state.IF(n_rec, tau=tau_mem, spk_reset=spk_reset, V_th=V_th)
        w_init = u.math.concatenate([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
        self.syn = braintrace.nn.Linear(
            n_in + n_rec,
            n_rec,
            w_init=w_init * u.mA,
            b_init=braintools.init.ZeroInit(unit=u.mA)
        )

    def update(self, spk):
        spk = u.math.concatenate([spk, self.neu.get_spike()], axis=-1)
        return self.neu(self.syn(spk))


class _ExpCo_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with dense connected exponential conductance-based synapses.
    """

    def __init__(
        self,
        neu,
        n_in: int,
        n_rec: int,
        input_ei_sep=False,
        tau_syn=10. * u.ms,
        rec_init=braintools.init.KaimingNormal(),
        ff_init=braintools.init.KaimingNormal()
    ):
        super().__init__()

        self.input_ei_sep = input_ei_sep
        self.n_exc_rec = int(n_rec * 0.8)
        self.n_inh_rec = n_rec - self.n_exc_rec

        self.neu = neu

        if input_ei_sep:
            self.n_exc_in = int(n_in * 0.8)
            self.n_inh_in = n_in - self.n_exc_in

            weight = jnp.concat([ff_init([self.n_exc_in, n_rec]), rec_init([self.n_exc_rec, n_rec])], axis=0)
            weight = weight * u.mS
            self.exe_syn = brainpy.state.AlignPostProj(
                comm=braintrace.nn.SignedWLinear(self.n_exc_in + self.n_exc_rec, n_rec, w_init=weight),
                syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn),
                out=brainpy.state.COBA.desc(E=3.5 * u.volt),
                post=self.neu
            )

            weight = jnp.concat([4 * ff_init([self.n_inh_in, n_rec]), 4 * rec_init([self.n_inh_rec, n_rec])], axis=0)
            weight = weight * u.mS
            self.inh_syn = brainpy.state.AlignPostProj(
                comm=braintrace.nn.SignedWLinear(self.n_inh_in + self.n_inh_rec, n_rec, w_init=weight),
                syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn),
                out=brainpy.state.COBA.desc(E=-0.5 * u.volt),
                post=self.neu
            )
        else:
            b_init = braintools.init.ZeroInit(unit=u.mS)

            self.inp_syn = brainpy.state.AlignPostProj(
                comm=braintrace.nn.Linear(n_in, n_rec, w_init=ff_init([n_in, n_rec]) * u.mS, b_init=b_init),
                syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn),
                out=brainpy.state.CUBA.desc(),
                post=self.neu
            )

            self.exe_syn = brainpy.state.AlignPostProj(
                comm=braintrace.nn.SignedWLinear(
                    self.n_exc_rec, n_rec, w_init=rec_init([self.n_exc_rec, n_rec]) * u.mS),
                syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn),
                out=brainpy.state.COBA.desc(E=1.5 * u.volt),
                post=self.neu
            )

            self.inh_syn = brainpy.state.AlignPostProj(
                comm=braintrace.nn.SignedWLinear(
                    self.n_inh_rec, n_rec, w_init=4 * rec_init([self.n_inh_rec, n_rec]) * u.mS),
                syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn),
                out=brainpy.state.COBA.desc(E=-0.5 * u.volt),
                post=self.neu
            )

    def update(self, spk):
        rec_exe_spk, rec_inh_spk = jnp.split(self.neu.get_spike(), [self.n_exc_rec], axis=-1)
        if self.input_ei_sep:
            in_exe_spk, in_inh_spk = jnp.split(spk, [self.n_exc_in], axis=-1)
            self.exe_syn(jnp.concat([in_exe_spk, rec_exe_spk], axis=-1))
            self.inh_syn(jnp.concat([in_inh_spk, rec_inh_spk], axis=-1))
            self.neu()
        else:
            self.inp_syn(spk)
            self.exe_syn(rec_exe_spk)
            self.inh_syn(rec_inh_spk)
            self.neu()
        # only output excitatory spikes
        # return self.neu.spike[..., :self.n_exc]
        return self.neu.get_spike()


class LIF_ExpCo_Dense_Layer(_ExpCo_Dense_Layer):
    """
    The RTRL layer with LIF neurons and dense connected exponential conductance-based synapses.
    """

    def __init__(
        self, n_in, n_rec, input_ei_sep=False, tau_mem=5. * u.ms, tau_syn=10. * u.ms, V_th=1. * u.mV,
        spk_fun=braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init=braintools.init.KaimingNormal(),
        ff_init=braintools.init.KaimingNormal()
    ):
        neu = brainpy.state.LIF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
        super().__init__(n_in=n_in, n_rec=n_rec, input_ei_sep=input_ei_sep, tau_syn=tau_syn,
                         rec_init=rec_init, ff_init=ff_init, neu=neu)


class ALIF_ExpCo_Dense_Layer(_ExpCo_Dense_Layer):
    """
    The RTRL layer with ALIF neurons and dense connected exponential conductance-based synapses.
    """

    def __init__(
        self,
        n_in,
        n_rec,
        input_ei_sep=False,
        tau_a=100. * u.ms,
        beta=0.1 * u.mV,
        tau_mem=5. * u.ms,
        tau_syn=10. * u.ms,
        V_th=1. * u.mV,
        spk_fun=braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init=braintools.init.KaimingNormal(),
        ff_init=braintools.init.KaimingNormal()
    ):
        neu = brainpy.state.ALIF(
            n_rec, tau=tau_mem, tau_a=tau_a, beta=beta,
            spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th
        )
        super().__init__(
            neu=neu, n_in=n_in, n_rec=n_rec,
            input_ei_sep=input_ei_sep, tau_syn=tau_syn,
            rec_init=rec_init, ff_init=ff_init
        )


class LIF_ExpCu_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with LIF neurons and dense connected exponential current synapses.
    """

    def __init__(
        self, n_in, n_rec, tau_mem=5. * u.ms, tau_syn=10. * u.ms, V_th=1. * u.mV,
        spk_fun=braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init=braintools.init.KaimingNormal(),
        ff_init=braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = brainpy.state.LIF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
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


class LIF_STDExpCu_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with LIF neurons and dense connected STD-based exponential current synapses.
    """

    def __init__(
        self, n_in, n_rec, inp_std=False,
        tau_mem=5. * u.ms, tau_syn=10. * u.ms,
        V_th=1. * u.mV, tau_std=500. * u.ms,
        spk_fun=braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init=braintools.init.KaimingNormal(),
        ff_init=braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = brainpy.state.LIF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
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
            inp_spk = self.std_inp(inp_spk) * inp_spk
        last_spk = self.neu.get_spike()
        inp = jnp.concat([inp_spk, self.std(last_spk)], axis=-1)
        self.syn(inp)
        self.neu()
        return self.neu.get_spike()


class LIF_STPExpCu_Dense_Layer(brainstate.nn.Module):
    def __init__(
        self,
        n_in, n_rec, inp_stp=False,
        tau_mem=5. * u.ms, tau_syn=10. * u.ms, V_th=1. * u.mV, tau_f=500. * u.ms, tau_d=100. * u.ms,
        spk_fun=braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init=braintools.init.KaimingNormal(),
        ff_init=braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.inp_stp = inp_stp
        self.neu = brainpy.state.LIF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
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
        self.neu = brainpy.state.ALIF(
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
        self.neu = brainpy.state.ALIF(
            n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th, tau_a=tau_a,
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
        self.neu = brainpy.state.ALIF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th, beta=beta)
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
        self.neu = brainpy.state.ALIF(
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
