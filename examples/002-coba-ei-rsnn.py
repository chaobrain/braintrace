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

import time
from typing import Callable, Iterable

import brainpy.state
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import braintrace


class EvidenceAccumulation:
    metadata = {
        'paper_link': 'https://doi.org/10.1038/nn.4403',
        'paper_name': 'History-dependent variability in population dynamics during evidence accumulation in cortex',
    }

    def __init__(
        self,
        t_interval=50. * u.ms,
        t_cue=100. * u.ms,
        t_delay=1000. * u.ms,
        t_recall=150. * u.ms,
        prob: float = 0.3,
        num_cue: int = 7,
        batch_size: int = 128,
        # number of neurons:
        #         left, right, recall, noise
        n_neurons=(25, 25, 25, 25),
        firing_rates=(40., 40., 40., 10.) * u.Hz,
    ):

        # input / output information
        self.batch_size = batch_size

        # time
        self.t_interval = t_interval
        self.t_cue = t_cue
        self.t_delay = t_delay
        self.t_recall = t_recall

        # features
        self.n_neurons = n_neurons
        self.feat_neurons = {
            'left': slice(0, n_neurons[0]),
            'right': slice(n_neurons[0],
                           n_neurons[0] + n_neurons[1]),
            'recall': slice(n_neurons[0] + n_neurons[1],
                            n_neurons[0] + n_neurons[1] + n_neurons[2]),
            'noise': slice(n_neurons[0] + n_neurons[1] + n_neurons[2],
                           n_neurons[0] + n_neurons[1] + n_neurons[2] + n_neurons[3]),
        }
        self.feat_fr = {
            'left': firing_rates[0],
            'right': firing_rates[1],
            'recall': firing_rates[2],
            'noise': firing_rates[3],
        }

        self.firing_rates = firing_rates
        self.prob = prob
        self.num_cue = num_cue

        # input / output information
        dt = brainstate.environ.get_dt()
        t_interval = int(self.t_interval / dt)
        t_cue = int(self.t_cue / dt)
        t_delay = int(self.t_delay / dt)
        t_recall = int(self.t_recall / dt)

        _time_periods = dict()
        for i in range(self.num_cue):
            _time_periods[f'interval {i}'] = t_interval
            _time_periods[f'cue {i}'] = t_cue
        _time_periods['delay'] = t_delay
        _time_periods['recall'] = t_recall
        self.periods = _time_periods
        t_total = sum(_time_periods.values())
        self.n_sim = t_total - t_recall

        def sample_a_trial(key):
            rng = brainstate.random.RandomState(key)

            # assign input spike probability
            ground_truth = rng.rand() < 0.5
            prob = u.math.where(ground_truth, self.prob, 1 - self.prob)

            # for each example in batch, draw which cues are going to be active (left or right)
            cue_assignments = u.math.asarray(rng.random(self.num_cue) > prob, dtype=int)

            X = jnp.zeros((t_total, self.num_inputs))
            # generate input spikes
            for k in range(self.num_cue):
                # input channels only fire when they are selected (left or right)
                i_start = u.math.where(cue_assignments[k],
                                       self.feat_neurons['left'].start,
                                       self.feat_neurons['right'].start)
                fr = u.math.where(cue_assignments[k], self.feat_fr['left'], self.feat_fr['right']) * dt
                update = jnp.ones((t_cue, 25)) * fr

                # reverse order of cues
                i_seq = t_interval + k * (t_interval + t_cue)
                # X[i_seq:i_seq + t_cue, i_start: i_start + 25] = fr
                X = jax.lax.dynamic_update_slice(X, update, (i_seq, i_start))

            X = u.Quantity(X)
            # recall cue
            X[-t_recall:, self.feat_neurons['recall']] = self.feat_fr['recall'] * dt

            # background noise
            X[:, self.feat_neurons['noise']] = self.feat_fr['noise'] * dt

            # generate inputs and targets
            # X = u.math.asarray(rng.rand(*X.shape) < X, dtype=float)
            X = rng.rand(*X.shape) < X
            Y = u.math.asarray(u.math.sum(cue_assignments) > (self.num_cue / 2), dtype=int)
            return X, Y

        self.sampling = jax.jit(jax.vmap(sample_a_trial))

    @property
    def num_inputs(self) -> int:
        return sum(self.n_neurons)

    @property
    def num_outputs(self) -> int:
        return 2

    def __iter__(self):
        while True:
            yield self.sampling(brainstate.random.split_key(self.batch_size))


class GIF(brainpy.state.Neuron):
    def __init__(
        self,
        size,
        V_rest=0. * u.mV,
        V_th_inf=1. * u.mV,
        R=1. * u.ohm,
        tau=20. * u.ms,
        tau_I2=50. * u.ms,
        A2=0. * u.mA,
        V_initializer: Callable = braintools.init.ZeroInit(unit=u.mV),
        I2_initializer: Callable = braintools.init.ZeroInit(unit=u.mA),
        spike_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(size, name=name, spk_fun=spike_fun, spk_reset=spk_reset)

        # parameters
        self.V_rest = braintools.init.param(V_rest, self.varshape, allow_none=False)
        self.V_th_inf = braintools.init.param(V_th_inf, self.varshape, allow_none=False)
        self.R = braintools.init.param(R, self.varshape, allow_none=False)
        self.tau = braintools.init.param(tau, self.varshape, allow_none=False)
        self.tau_I2 = braintools.init.param(tau_I2, self.varshape, allow_none=False)
        self.A2 = braintools.init.param(A2, self.varshape, allow_none=False)

        # initializers
        self._V_initializer = V_initializer
        self._I2_initializer = I2_initializer

    def init_state(self):
        # 将模型用于在线学习，需要初始化状态变量
        self.V = brainstate.HiddenState(braintools.init.param(self._V_initializer, self.varshape))
        self.I2 = brainstate.HiddenState(braintools.init.param(self._I2_initializer, self.varshape))

    def update(self, x=0. * u.mA):
        # 如果前一时刻发放了脉冲，则将膜电位和适应性电流进行重置
        last_spk = self.get_spike()
        # last_spk = jax.lax.stop_gradient(last_spk)
        last_V = self.V.value - self.V_th_inf * last_spk
        last_I2 = self.I2.value - self.A2 * last_spk
        # 更新状态
        I2 = brainstate.nn.exp_euler_step(lambda i2: - i2 / self.tau_I2, last_I2)
        V = brainstate.nn.exp_euler_step(
            lambda v, Iext: (- v + self.V_rest + self.R * self.sum_current_inputs(Iext, v)) / self.tau,
            last_V, x + I2
        )
        self.I2.value = I2
        self.V.value = self.sum_delta_inputs(V)
        # 输出
        inp = self.V.value - self.V_th_inf
        inp = jax.nn.standardize(u.get_magnitude(inp))
        return inp

    def get_spike(self, V=None):
        V = self.V.value if V is None else V
        spk = self.spk_fun((V - self.V_th_inf) / self.V_th_inf)
        return spk


class _SNNEINet(brainstate.nn.Module):
    def __init__(
        self,
        n_in, n_rec, n_out,
        tau_neu=10. * u.ms,
        tau_a=100. * u.ms,
        beta=1. * u.mA,
        tau_syn=10. * u.ms,
        tau_out=10. * u.ms,
        ff_scale=1.,
        rec_scale=1.,
        E_exc=None,
        E_inh=None,
        w_ei_ratio: float = 10.,
    ):
        super().__init__()

        self.n_exc = int(n_rec * 0.8)
        self.n_inh = n_rec - self.n_exc

        # neurons
        tau_a = brainstate.random.uniform(100. * u.ms, tau_a * 2., n_rec)
        self.pop = GIF(n_rec, tau=tau_neu, tau_I2=tau_a, A2=beta)
        # feedforward
        self.ff2r = brainpy.state.AlignPostProj(
            comm=braintrace.nn.SignedWLinear(
                n_in,
                n_rec,
                w_init=braintools.init.KaimingNormal(ff_scale, unit=u.siemens)
            ),
            syn=brainpy.state.Expon.desc(
                n_rec,
                tau=tau_syn,
                g_initializer=braintools.init.ZeroInit(unit=u.siemens)
            ),
            out=(brainpy.state.CUBA.desc() if E_exc is None else brainpy.state.COBA.desc(E=E_exc)),
            post=self.pop
        )
        # recurrent
        inh_init = braintools.init.KaimingNormal(scale=rec_scale * w_ei_ratio, unit=u.siemens)
        inh2r_conn = braintrace.nn.SignedWLinear(
            self.n_inh,
            n_rec,
            w_init=inh_init,
            w_sign=-1. if E_inh is None else None
        )
        self.inh2r = brainpy.state.AlignPostProj(
            comm=inh2r_conn,
            syn=brainpy.state.Expon.desc(
                n_rec,
                tau=tau_syn,
                g_initializer=braintools.init.ZeroInit(unit=u.siemens)
            ),
            out=(brainpy.state.CUBA.desc() if E_inh is None else brainpy.state.COBA.desc(E=E_inh)),
            post=self.pop
        )
        exc_init = braintools.init.KaimingNormal(scale=rec_scale, unit=u.siemens)
        exc2r_conn = braintrace.nn.SignedWLinear(self.n_exc, n_rec, w_init=exc_init)
        self.exc2r = brainpy.state.AlignPostProj(
            comm=exc2r_conn,
            syn=brainpy.state.Expon.desc(
                n_rec,
                tau=tau_syn,
                g_initializer=braintools.init.ZeroInit(unit=u.siemens)
            ),
            out=(brainpy.state.CUBA.desc() if E_exc is None else brainpy.state.COBA.desc(E=E_exc)),
            post=self.pop
        )
        # output
        self.out = braintrace.nn.LeakyRateReadout(n_rec, n_out, tau=tau_out)

    def update(self, spk):
        e_sps, i_sps = jnp.split(self.pop.get_spike(), [self.n_exc], axis=-1)
        self.ff2r(spk)
        self.exc2r(e_sps)
        self.inh2r(i_sps)
        return self.out(self.pop())

    def visualize(self, inputs, n2show: int = 5):
        inputs = np.transpose(inputs, (1, 0, 2))  # [B, T, N] -> [T, B, N]

        n_seq = inputs.shape[0]
        batch_size = inputs.shape[1]

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=batch_size)
        def init():
            brainstate.nn.init_all_states(self)

        init()
        model = brainstate.nn.Vmap(self, vmap_states='new')

        def step(inp):
            out = model(inp)
            n_rec = self.pop.varshape[0]
            rec_spk = self.pop.get_spike()
            rec_mem = self.pop.V.value[:, np.arange(0, n_rec, n_rec // 50)]
            return rec_spk, rec_mem.to_decimal(u.mV), out

        rec_spks, rec_mems, outs = brainstate.transform.for_loop(step, inputs, pbar=brainstate.transform.ProgressBar(10))

        fig, gs = braintools.visualize.get_figure(4, n2show, 3., 4.5)
        for i in range(n2show):
            # input spikes
            ax = fig.add_subplot(gs[0, i])
            t_indices, n_indices = np.where(inputs[:, i])
            plt.scatter(t_indices, n_indices, s=1)
            plt.xlim(0, n_seq)

            # recurrent spikes
            ax = fig.add_subplot(gs[1, i])
            t_indices, n_indices = np.where(rec_spks[:, i])
            plt.scatter(t_indices, n_indices, s=1)
            plt.xlim(0, n_seq)

            # recurrent membrane potentials
            ax = fig.add_subplot(gs[2, i])
            ax.plot(rec_mems[:, i])
            plt.xlim(0, n_seq)

            # output potentials
            ax = fig.add_subplot(gs[3, i])
            ax.plot(outs[:, i])
            plt.xlim(0, n_seq)

        plt.show()
        plt.close()


class SNNCubaNet(_SNNEINet):
    def __init__(
        self, n_in, n_rec, n_out,
        tau_neu=10. * u.ms, tau_a=100. * u.ms,
        beta=1. * u.mV, tau_syn=10. * u.ms, tau_out=10. * u.ms,
        ff_scale=1., rec_scale=1., w_ei_ratio=4.,
    ):
        super().__init__(
            n_in=n_in,
            n_rec=n_rec,
            n_out=n_out,
            tau_neu=tau_neu,
            tau_a=tau_a,
            beta=beta,
            tau_syn=tau_syn,
            tau_out=tau_out,
            ff_scale=ff_scale,
            rec_scale=rec_scale,
            E_exc=None,
            E_inh=None,
            w_ei_ratio=w_ei_ratio,
        )


class SNNCobaNet(_SNNEINet):
    def __init__(
        self, n_in, n_rec, n_out,
        tau_neu=10. * u.ms, tau_a=100. * u.ms, beta=1. * u.mV,
        tau_syn=10. * u.ms, tau_out=10. * u.ms,
        ff_scale=1., rec_scale=1., w_ei_ratio=4.,
    ):
        super().__init__(
            n_in=n_in,
            n_rec=n_rec,
            n_out=n_out,
            tau_neu=tau_neu,
            tau_a=tau_a,
            beta=beta,
            tau_syn=tau_syn,
            tau_out=tau_out,
            ff_scale=ff_scale,
            rec_scale=rec_scale,
            E_exc=5. * u.mV,
            E_inh=-10. * u.mV,
            w_ei_ratio=w_ei_ratio,
        )


class Trainer:
    def __init__(
        self,
        target_net: _SNNEINet,
        optimizer: braintools.optim.Optimizer,
        loader: Iterable,
        n_sim: int,
        n_epochs: int = 1000,
        method: str = 'expsm_diag',
        acc_threshold: float = 0.90,
    ):
        # the network
        self.target = target_net

        # the dataset
        self.loader = loader
        self.n_sim = n_sim

        # parameters
        self.n_epochs = n_epochs

        # optimizer
        weights = self.target.states().subset(brainstate.ParamState)
        self.optimizer = optimizer
        self.optimizer.register_trainable_weights(weights)

        # traning method
        assert method in ['expsm_diag', 'diag', 'hybrid'], 'Unknown online learning methods.'
        self.method = method
        self.acc_threshold = acc_threshold

    def _acc(self, outs, target):
        pred = jnp.argmax(jnp.sum(outs, 0), 1)  # [T, B, N] -> [B, N] -> [B]
        # pred = jnp.argmax(jnp.max(outs, 0), 1)  # [T, B, N] -> [B, N] -> [B]
        acc = jnp.asarray(pred == target, dtype=brainstate.environ.dftype()).mean()
        return acc

    @brainstate.transform.jit(static_argnums=(0,))
    def etrace_train(self, inputs, targets):
        inputs = jnp.asarray(inputs, dtype=brainstate.environ.dftype())  # [T, B, N]
        weights = self.target.states().subset(brainstate.ParamState)

        # initialize the online learning model
        if self.method == 'expsm_diag':
            model = braintrace.IODimVjpAlgorithm(self.target, decay_or_rank=0.99)
        elif self.method == 'diag':
            model = braintrace.ParamDimVjpAlgorithm(self.target)
        elif self.method == 'hybrid':
            model = braintrace.HybridDimVjpAlgorithm(self.target, decay_or_rank=0.99)
        else:
            raise ValueError(f'Unknown online learning methods: {self.method}.')

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(self.target)
            model.compile_graph(inputs[0, 0])

        init()
        model = brainstate.nn.Vmap(model, vmap_states='new')

        def _etrace_grad(inp):
            # call the model
            out = model(inp)
            # calculate the loss
            me = braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean()
            return me, out

        def _etrace_step(prev_grads, x):
            # no need to return weights and states, since they are generated then no longer needed
            f_grad = brainstate.transform.grad(_etrace_grad, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(x)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        def _etrace_train(inputs_):
            # forward propagation
            grads = jax.tree.map(jnp.zeros_like, weights.to_dict_values())
            grads, (outs, mse_ls) = brainstate.transform.scan(_etrace_step, grads, inputs_)
            acc = self._acc(outs, targets)

            grads = brainstate.nn.clip_grad_norm(grads, 1.)
            self.optimizer.update(grads)
            # accuracy
            return mse_ls.mean(), acc

        # running indices
        if self.n_sim > 0:
            brainstate.transform.for_loop(model, inputs[:self.n_sim])
            r = _etrace_train(inputs[self.n_sim:])
        else:
            r = _etrace_train(inputs)
        return r

    def f_train(self):
        acc_max = 0.
        t0 = time.time()
        for bar_idx, (inputs, outputs) in enumerate(self.loader):
            if bar_idx > self.n_epochs:
                break

            # self.target.visualize(inputs)

            inputs = jnp.asarray(inputs, dtype=brainstate.environ.dftype()).transpose(1, 0, 2)
            outputs = jnp.asarray(outputs, dtype=brainstate.environ.ditype())
            mse_ls, acc = self.etrace_train(inputs, outputs)
            if (bar_idx + 1) % 100 == 0:
                self.optimizer.lr.step_epoch()
            desc = (
                f'Batch {bar_idx:2d}, '
                f'CE={float(mse_ls):.8f}, '
                f'acc={float(acc):.6f}, '
                f'time={time.time() - t0:.2f} s'
            )
            print(desc)
            if acc > acc_max:
                acc_max = acc

            t0 = time.time()
            if acc_max > self.acc_threshold:
                print(f'Accuracy reaches {self.acc_threshold * 100.}% at {bar_idx}th epoch. Stop training.')
                break


def training(
    lr=1e-3,  # Learning rate
    batch_size=128,  # Batch size
    net='coba',  # Network type, 'coba' or 'cuba'
    n_rec=200,  # Number of recurrent neurons
    w_ei_ratio=4.,  # Ratio of inhibitory to excitatory weights
    ff_scale=1.,  # Feedforward weight scale
    rec_scale=1.,  # Recurrent weight scale
    beta=1.0 * u.mA,  # Beta of the neuron
    tau_a=1000. * u.ms,  # Time constant of the adaptation
    tau_neu=100. * u.ms,  # Time constant of the neuron
    tau_syn=10. * u.ms,  # Time constant of the synapse
    tau_out=10. * u.ms,  # Time constant of the output
    method='expsm_diag',  # Online learning method
):
    # data
    loader = EvidenceAccumulation(batch_size=batch_size)

    # network
    assert net in ['coba', 'cuba'], 'Unknown network type.'
    cls = SNNCobaNet if net == 'coba' else SNNCubaNet
    net = cls(
        loader.num_inputs,
        n_rec,
        loader.num_outputs,
        beta=beta,
        tau_a=tau_a,
        tau_neu=tau_neu,
        tau_syn=tau_syn,
        tau_out=tau_out,
        ff_scale=ff_scale,
        rec_scale=rec_scale,
        w_ei_ratio=w_ei_ratio,
    )
    net.visualize(loader.sampling(brainstate.random.split_key(5))[0], n2show=5)

    # trainer
    trainer = Trainer(
        net,
        braintools.optim.Adam(lr=lr),
        loader,
        loader.n_sim,
        n_epochs=1000,
        method=method,
        acc_threshold=0.90,
    )
    trainer.f_train()


if __name__ == '__main__':
    with brainstate.environ.context(dt=1.0 * u.ms):
        training(
            rec_scale=0.5,
            ff_scale=1.0,
            n_rec=400,
            w_ei_ratio=4.,
            lr=1e-3,
            net='coba',
            tau_a=1500.0 * u.ms,
            tau_syn=5. * u.ms,
            tau_neu=400. * u.ms
        )
