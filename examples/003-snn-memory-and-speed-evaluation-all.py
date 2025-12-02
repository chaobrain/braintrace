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


import argparse
import time
from functools import reduce
from typing import Callable, Union

import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braintrace

default_setting = brainstate.util.DotDict(
    method='bptt',
    lr=0.001,
    batch_size=128,
    dt=1.0,
    loss='cel',
    n_data_worker=0,
    data_length=1000,
    drop_last=1,
    exp_name='',
    spk_fun='relu',
    warmup_ratio=0.0,
    optimizer='adam',
    filepath='',
    spk_reg_factor=0.0,
    spk_reg_rate=10.0,
    v_reg_factor=0.0,
    v_reg_low=-20.0,
    v_reg_high=1.4,
    weight_L1=0.0,
    weight_L2=0.0,
    model='lif-delta',
    n_rec=512,
    n_layer=3,
    V_th=1.0,
    tau_mem_sigma=1.0,
    tau_mem=10.0,
    tau_syn=10.0,
    tau_o=10.0,
    ff_scale=10.0,
    rec_scale=2.0,
    spk_reset='soft'
)


def _format_sim_epoch(sim: Union[int, float], length: int):
    if 0. <= sim < 1.:
        return int(length * sim)
    else:
        return int(sim)


@jax.jit
def get_mem_usage() -> float:
    def fn():
        if jax.default_backend() != 'cpu':
            return jax.devices()[0].memory_stats()['bytes_in_use'] / (1024 ** 3)
        else:
            import psutil
            memory = psutil.virtual_memory()
            return memory.used / (1024 ** 3)

    return jax.pure_callback(fn, jax.ShapeDtypeStruct((), brainstate.environ.dftype()))


class _LIF_Delta_Dense_Layer(brainstate.nn.Module):
    """
    LIF neurons and dense connected delta synapses.
    """

    def __init__(
        self,
        n_in,
        n_rec,
        tau_mem=5.,
        V_th=1.,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_scale: float = 1.,
        ff_scale: float = 1.,
    ):
        super().__init__()
        self.neu = brainpy.state.LIF(
            n_rec,
            R=1.,
            tau=tau_mem,
            spk_fun=spk_fun,
            spk_reset=spk_reset,
            V_th=V_th,
            V_reset=0.,
            V_rest=0.,
            V_initializer=braintools.init.ZeroInit(),
        )
        rec_init: Callable = braintools.init.KaimingNormal(rec_scale)
        ff_init: Callable = braintools.init.KaimingNormal(ff_scale)
        w_init = jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
        self.syn = brainpy.state.DeltaProj(
            comm=braintrace.nn.Linear(n_in + n_rec, n_rec, w_init=w_init),
            post=self.neu
        )

    def update(self, spk):
        inp = jnp.concat([spk, self.neu.get_spike()], axis=-1)
        self.syn(inp)
        self.neu(0.)
        return self.neu.get_spike()


class _LIF_ExpCu_Dense_Layer(brainstate.nn.Module):
    """
    LIF neurons and dense connected exponential current synapses.
    """

    def __init__(
        self,
        n_in,
        n_rec,
        tau_mem=5.,
        tau_syn=10.,
        V_th=1.,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_scale: float = 1.,
        ff_scale: float = 1.,
    ):
        super().__init__()
        self.neu = brainpy.state.LIF(
            n_rec,
            R=1.,
            tau=tau_mem,
            spk_fun=spk_fun,
            spk_reset=spk_reset,
            V_th=V_th,
            V_reset=0.,
            V_rest=0.,
            V_initializer=braintools.init.ZeroInit(),
        )
        rec_init: Callable = braintools.init.KaimingNormal(rec_scale)
        ff_init: Callable = braintools.init.KaimingNormal(ff_scale)
        w_init = jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(n_in + n_rec, n_rec, w_init),
            syn=brainpy.state.Expon(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
            out=brainpy.state.CUBA(scale=1.),
            post=self.neu
        )

    def update(self, spk):
        self.syn(jnp.concat([spk, self.neu.get_spike()], axis=-1))
        self.neu(0.)
        return self.neu.get_spike()


class ETraceNet(brainstate.nn.Module):
    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        n_layer: int,
        args: argparse.ArgumentParser,
    ):
        super().__init__()

        # arguments
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.n_layer = n_layer

        if args.spk_fun == 's2nn':
            spk_fun = braintools.surrogate.S2NN()
        elif args.spk_fun == 'relu':
            spk_fun = braintools.surrogate.ReluGrad()
        elif args.spk_fun == 'multi_gaussian':
            spk_fun = braintools.surrogate.MultiGaussianGrad()
        else:
            raise ValueError('Unknown spiking surrogate gradient function.')

        # recurrent layers
        self.rec_layers = []
        for layer_idx in range(n_layer):
            tau_mem = (
                brainstate.random.normal(args.tau_mem, args.tau_mem_sigma, [n_rec])
                if args.tau_mem_sigma > 0. else args.tau_mem
            )
            if args.model == 'lif-exp-cu':
                rec = _LIF_ExpCu_Dense_Layer(
                    n_rec=n_rec,
                    n_in=n_in,
                    tau_mem=tau_mem,
                    tau_syn=args.tau_syn,
                    V_th=args.V_th,
                    spk_fun=spk_fun,
                    spk_reset=args.spk_reset,
                    rec_scale=args.rec_scale,
                    ff_scale=args.ff_scale,
                )
                n_in = n_rec

            elif args.model == 'lif-delta':
                rec = _LIF_Delta_Dense_Layer(
                    n_rec=n_rec,
                    n_in=n_in,
                    tau_mem=tau_mem,
                    V_th=args.V_th,
                    spk_fun=spk_fun,
                    spk_reset=args.spk_reset,
                    rec_scale=args.rec_scale,
                    ff_scale=args.ff_scale,
                )
                n_in = n_rec

            else:
                raise ValueError('Unknown neuron model.')

            self.rec_layers.append(rec)

        # output layer
        self.out = braintrace.nn.LeakyRateReadout(
            in_size=n_rec,
            out_size=n_out,
            tau=args.tau_o,
            w_init=braintools.init.KaimingNormal()
        )

    def update(self, x):
        for i in range(self.n_layer):
            x = self.rec_layers[i](x)
        return self.out(x)

    def membrane_reg(self, mem_low: float, mem_high: float, factor: float = 0.):
        loss = 0.
        if factor > 0.:
            # extract all Neuron models
            neurons = self.nodes().subset(brainpy.state.Neuron).unique().values()
            # evaluate the membrane potential
            for l in neurons:
                loss += jnp.square(
                    jnp.mean(
                        jax.nn.relu(l.V.value - mem_high) ** 2 +
                        jax.nn.relu(mem_low - l.V.value) ** 2
                    )
                )
            loss = loss * factor
        return loss

    def spike_reg(self, target_fr: float, factor: float = 0.):
        # target_fr: Hz
        loss = 0.
        if factor > 0.:
            # extract all Neuron models
            neurons = self.nodes().subset(brainpy.state.Neuron).unique().values()
            # evaluate the spiking dynamics
            for l in neurons:
                loss += (jnp.mean(l.get_spike()) - target_fr / 1e3 * brainstate.environ.get_dt()) ** 2
            loss = loss * factor
        return loss


class Trainer(object):
    def __init__(
        self,
        target: ETraceNet,
        opt: braintools.optim.Optimizer,
        args: argparse.Namespace,
    ):
        super().__init__()

        # target network
        self.target = target

        # parameters
        self.args = args

        # loss function
        if self.args.loss == 'mse':
            self.loss_fn = braintools.metric.squared_error
        elif self.args.loss == 'cel':
            self.loss_fn = braintools.metric.softmax_cross_entropy_with_integer_labels
        else:
            raise ValueError

        # optimizer
        self.opt = opt
        opt.register_trainable_weights(self.target.states().subset(brainstate.ParamState))

        # define etrace functions
        if self.args.method != 'bptt':
            self._compile_etrace_function(
                jax.ShapeDtypeStruct((self.args.batch_size, 32768,), brainstate.environ.dftype()))

    def _acc(self, out, target):
        return jnp.mean(jnp.equal(target, jnp.argmax(jnp.mean(out, axis=0), axis=1)))

    def _loss(self, out, target):
        loss = self.loss_fn(out, target).mean()

        # L1 regularization loss
        if self.args.weight_L1 != 0.:
            leaves = self.target.states().subset(brainstate.ParamState).to_dict_values()
            loss += self.args.weight_L1 * reduce(jnp.add, jax.tree.map(lambda x: jnp.sum(jnp.abs(x)), leaves))

        # membrane potential regularization loss
        if self.args.v_reg_factor != 0.:
            mem_low = self.args.v_reg_low
            mem_high = self.args.v_reg_high
            loss += self.target.membrane_reg(mem_low, mem_high, self.args.v_reg_factor)

        # spike regularization loss
        if self.args.spk_reg_factor != 0.:
            fr = self.args.spk_reg_rate
            loss += self.target.spike_reg(fr, self.args.spk_reg_factor)

        return loss

    @brainstate.transform.jit(static_argnums=0)
    def predict(self, inputs, targets):
        inputs = u.math.flatten(inputs, start_axis=2)
        brainstate.nn.vmap_init_all_states(self.target, axis_size=inputs.shape[1], state_tag='new')
        model = brainstate.nn.Vmap(self.target, vmap_states='new')

        def _step(i, inp):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                out = model(inp)
                loss = self._loss(out, targets)
                return loss, out

        losses, outs = brainstate.transform.for_loop(_step, np.arange(inputs.shape[0]), inputs)
        acc = self._acc(outs, targets)
        return losses.mean(), acc

    def _compile_etrace_function(self, input_info):
        if self.args.method == 'expsm_diag':
            model = braintrace.ES_D_RTRL(self.target, self.args.etrace_decay, mode=brainstate.mixin.Batching())
        elif self.args.method == 'diag':
            model = braintrace.D_RTRL(self.target, mode=brainstate.mixin.Batching())
        elif self.args.method == 'hybrid':
            model = braintrace.HybridDimVjpAlgorithm(self.target, self.args.etrace_decay,
                                                     mode=brainstate.mixin.Batching())
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        # initialize the states
        brainstate.nn.vmap_init_all_states(self.target, axis_size=input_info.shape[0], state_tag='new')
        model.compile_graph(input_info)

        @brainstate.transform.jit
        @brainstate.transform.vmap(in_states=model.states('new'))
        def reset_state():
            brainstate.nn.reset_all_states(model)

        @brainstate.transform.jit
        def _etrace_single_run(i, batch_inp):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                model(batch_inp)

        def _etrace_grad(i, batch_inp, targets):
            # call the model
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                out = model(batch_inp)
            # calculate the loss
            loss = self._loss(out, targets)
            return loss, out

        @brainstate.transform.jit
        def _etrace_step(prev_grads, inputs, targets):
            i, inp = inputs
            # no need to return weights and states, since they are generated then no longer needed
            f_grad = brainstate.transform.grad(
                _etrace_grad, grad_states=self.opt.param_states, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(i, inp, targets)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        @brainstate.transform.jit
        def _update_and_acc(grads, outs, targets):
            # gradient updates
            grads = brainstate.functional.clip_grad_norm(grads, 1.)
            self.opt.update(grads)

            # accuracy
            acc = self._acc(jnp.asarray(outs), targets)
            return acc

        self._etrace_reset_fun = reset_state
        self._etrace_pred_fun = _etrace_single_run
        self._update_and_acc = _update_and_acc
        self._etrace_train_fun = _etrace_step

    # @brainstate.transform.jit(static_argnums=0)
    def etrace_train(self, inputs, targets):
        inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], -1))  # [n_steps, n_samples, n_in]
        self._etrace_reset_fun()

        # initial gradients
        grads = jax.tree.map(lambda a: jnp.zeros_like(a), self.opt.param_states.to_dict_values())

        # training
        indices = np.arange(inputs.shape[0])
        n_sim = _format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
        outs, losses = [], []
        for i in indices:
            if i < n_sim:
                self._etrace_pred_fun(i, jnp.asarray(inputs[i]))
            else:
                grads, (out, loss) = self._etrace_train_fun(grads, (i, jnp.asarray(inputs[i])), targets)
                outs.append(out)
                losses.append(loss)

        acc = self._update_and_acc(grads, jnp.asarray(outs), targets)
        mem_after = get_mem_usage()
        return jnp.asarray(losses).mean(), acc, mem_after

    @brainstate.transform.jit(static_argnums=(0,))
    def bptt_train(self, inputs, targets):
        inputs = u.math.flatten(inputs, start_axis=2)
        indices = np.arange(inputs.shape[0])

        # initialize the states
        brainstate.nn.vmap_init_all_states(self.target, axis_size=inputs.shape[1], state_tag='new')
        model = brainstate.nn.Vmap(self.target, vmap_states='new')

        # the model for a single step
        def _single_step(i, inp, fit: bool = True):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt(), fit=fit):
                model(inp)

        def _run_step_train(i, inp):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                out = model(inp)
                return out, self._loss(out, targets)

        def _grad_step():
            if self.args.warmup_ratio > 0:
                n_sim = _format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
                brainstate.transform.for_loop(_single_step, indices[:n_sim], inputs[:n_sim])
                outs, losses = brainstate.transform.for_loop(_run_step_train, indices[n_sim:], inputs[n_sim:])
            else:
                outs, losses = brainstate.transform.for_loop(_run_step_train, indices, inputs)
            return losses.mean(), outs

        # gradients
        weights = self.target.states().subset(brainstate.ParamState)
        grads, loss, outs = brainstate.transform.grad(_grad_step, weights, has_aux=True, return_value=True)()

        # optimization
        grads = brainstate.nn.clip_grad_norm(grads, 1.)
        self.opt.update(grads)

        # accuracy
        acc = self._acc(outs, targets)

        mem_after = get_mem_usage()
        return loss, acc, mem_after

    def f_train(self):
        x_local = np.random.rand(self.args.data_length, self.args.batch_size, 128 * 128 * 2) < 0.1
        x_local = np.asarray(x_local, dtype=np.float32)
        y_local = np.random.randint(0, 11, size=(self.args.batch_size,))

        mem_before = get_mem_usage()

        # training
        losses, acces, mems, times = [], [], [], []
        for i_batch in range(10):
            t0 = time.time()
            if self.args.method == 'bptt':
                loss, acc, mem = jax.block_until_ready(self.bptt_train(x_local, y_local))
            else:
                loss, acc, mem = jax.block_until_ready(self.etrace_train(x_local, y_local))
            t = time.time() - t0
            print(
                f'Batch = {i_batch}, '
                f'training loss = {float(loss):.8f}, '
                f'training acc = {float(acc):.6f}, '
                f'time = {t:.5f} s, '
                f'memory before = {mem_before:.2f} GB, '
                f'memory after = {mem:.2f} GB'
            )
            times.append(t)
            losses.append(loss)
            acces.append(acc)
            mems.append(mem)
        print(
            f'training loss = {float(np.mean(losses[2:])):.8f}, '
            f'training acc = {float(np.mean(acces[2:])):.6f}, '
            f'time = {np.mean(times[2:]):.5f} s, '
            f'memory before = {mem_before:.2f} GB, '
            f'memory after = {np.mean(mems[2:]):.2f} GB'
        )


def network_training(args):
    brainstate.util.clear_buffer_memory()
    brainstate.random.seed()
    print(args)
    # environment setting
    brainstate.environ.set(dt=args.dt)

    # net: inputs correspond to 128x128 pixels with 2 channels (e.g., RGB), DVS Gesture dataset
    net = ETraceNet((128 * 128 * 2), args.n_rec, 11, args.n_layer, args=args)

    # optimizer
    if args.optimizer == 'adam':
        opt_cls = braintools.optim.Adam
    elif args.optimizer == 'momentum':
        opt_cls = braintools.optim.Momentum
    elif args.optimizer == 'sgd':
        opt_cls = braintools.optim.SGD
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')
    opt = opt_cls(lr=args.lr, weight_decay=args.weight_L2)

    # creating the trainer
    trainer = Trainer(net, opt, args)
    trainer.f_train()


if __name__ == '__main__':

    # BPTT
    for length in [50, 100, 200, 300, 400, 600, 800, 1000]:
        setting = default_setting.copy()
        setting.data_length = length
        setting.method = 'bptt'
        try:
            network_training(setting)
        except Exception as e:
            break

    # ES-D-RTRL
    for length in [50, 100, 200, 300, 400, 600, 800, 1000]:
        setting = default_setting.copy()
        setting.method = 'expsm_diag'
        setting.vjp_method = 'single-step'
        setting.data_length = length
        setting.etrace_decay = 0.9
        network_training(setting)

    # D-RTRL
    for length in [50, 100, 200, 300, 400, 600, 800, 1000]:
        setting = default_setting.copy()
        setting.method = 'diag'
        setting.vjp_method = 'single-step'
        setting.data_length = length
        network_training(setting)