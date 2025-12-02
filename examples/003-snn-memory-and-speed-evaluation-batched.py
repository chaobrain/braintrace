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

import argparse
import os
import time
from functools import reduce
from typing import Callable, Union

import brainpy
import matplotlib.pyplot as plt
import numpy as np

from utils import MyArgumentParser
# import builtins
# import traceback
#
# _old_print = builtins.print
#
# def debug_print(*args, **kwargs):
#     # 打印原信息
#     _old_print(*args, **kwargs)
#     # if args[0] == 6 and args[1] == 6:
#     traceback.print_stack(limit=5)
#
#
# builtins.print = debug_print


parser = MyArgumentParser()

# Learning parameters
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
parser.add_argument("--dt", type=float, default=1., help="The simulation time step.")
parser.add_argument("--loss", type=str, default='cel', choices=['cel', 'mse'], help="Loss function.")

# dataset
parser.add_argument("--n_data_worker", type=int, default=0, help="Number of data loading workers")
parser.add_argument("--data_length", type=int, default=200, help="")
parser.add_argument("--drop_last", type=int, default=0, help="")

# training parameters
parser.add_argument("--exp_name", type=str, default='', help="")
parser.add_argument("--spk_fun", type=str, default='relu', help="spike surrogate gradient function.")
parser.add_argument("--warmup_ratio", type=float, default=0.0, help="The ratio for network simulation.")
parser.add_argument("--optimizer", type=str, default='adam', help="")
parser.add_argument("--filepath", type=str, default='', help="The name for the current experiment.")

# regularization parameters
parser.add_argument("--spk_reg_factor", type=float, default=0.0, help="Spike regularization factor.")
parser.add_argument("--spk_reg_rate", type=float, default=10., help="Target firing rate.")
parser.add_argument("--v_reg_factor", type=float, default=0.0, help="Voltage regularization factor.")
parser.add_argument("--v_reg_low", type=float, default=-20., help="The lowest voltage for regularization.")
parser.add_argument("--v_reg_high", type=float, default=1.4, help="The highest voltage for regularization.")
parser.add_argument("--weight_L1", type=float, default=0.0, help="The weight L1 regularization.")
parser.add_argument("--weight_L2", type=float, default=0.0, help="The weight L2 regularization.")

# model parameters
parser.add_argument("--model", type=str, default='lif-delta', help="The model architecture.")
parser.add_argument("--n_rec", type=int, default=200, help="Number of recurrent neurons.")
parser.add_argument("--n_layer", type=int, default=2, help="Number of recurrent layers.")
parser.add_argument("--V_th", type=float, default=1.)
parser.add_argument("--tau_mem_sigma", type=float, default=1.)
parser.add_argument("--tau_mem", type=float, default=10.)
parser.add_argument("--tau_syn", type=float, default=10.)
parser.add_argument("--tau_o", type=float, default=10.)
parser.add_argument("--ff_scale", type=float, default=10.)
parser.add_argument("--rec_scale", type=float, default=2.)
parser.add_argument("--spk_reset", type=str, default='soft')

global_args = parser.parse_args()

import braintrace
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import tonic
from tonic.collation import PadTensors
from tonic.datasets import DVSGesture
from torch.utils.data import DataLoader


def _format_sim_epoch(sim: Union[int, float], length: int):
    if 0. <= sim < 1.:
        return int(length * sim)
    else:
        return int(sim)


def _raster_plot(sp_matrix, times):
    """Get spike raster plot which displays the spiking activity
    of a group of neurons over time.
  
    Parameters
    ----------
    sp_matrix : bnp.ndarray
        The matrix which record spiking activities.
    times : bnp.ndarray
        The time steps.
  
    Returns
    -------
    raster_plot : tuple
        Include (neuron index, spike time).
    """
    sp_matrix = np.asarray(sp_matrix)
    times = np.asarray(times)
    elements = np.where(sp_matrix > 0.)
    index = elements[1]
    times = times[elements[0]]
    return index, times


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
        args,
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

    def verify(self, dataloader, num_show=5, filepath=None):
        def _step(index, x):
            with brainstate.environ.context(i=index, t=index * brainstate.environ.get_dt()):
                out = self.update(x)
            return out, [r.neu.get_spike() for r in self.rec_layers], [r.neu.V.value for r in self.rec_layers]

        dataloader = iter(dataloader)
        xs, ys = next(dataloader)  # xs: [n_steps, n_samples, n_in]
        xs = jnp.asarray(xs)
        print(xs.shape, ys.shape)
        brainstate.nn.init_all_states(self, xs.shape[1])

        time_indices = np.arange(0, xs.shape[0])
        outs, sps, vs = brainstate.transform.for_loop(_step, time_indices, xs)
        outs = u.math.as_numpy(outs)
        sps = [u.math.as_numpy(out) for out in sps]
        vs = [u.math.as_numpy(out) for out in vs]
        # vs = [np.where(sp, v + sps_inc, v) for sp, v in zip(sps, vs)]

        ts = time_indices * brainstate.environ.get_dt()
        max_t = xs.shape[0] * brainstate.environ.get_dt()

        for i in range(min(num_show, xs.shape[1])):
            fig, gs = braintools.visualize.get_figure(2, len(self.rec_layers) + 1, 3., 3.)

            # input spiking
            ax_inp = fig.add_subplot(gs[0, 0])
            indices, times = _raster_plot(xs[:, i], ts)
            ax_inp.plot(times, indices, 'k,')
            ax_inp.set_xlim(0., max_t)
            ax_inp.set_ylabel('Input Spiking')

            # recurrent spiking
            for j in range(len(self.rec_layers)):
                ax_rec = fig.add_subplot(gs[0, j + 1])
                indices, times = _raster_plot(sps[j][:, i], ts)
                ax_rec.plot(times, indices, 'k,')
                ax_rec.set_xlim(0., max_t)
                ax_rec.set_ylabel(f'Recurrent Spiking L{j}')

            # decision activity
            ax_out = fig.add_subplot(gs[1, 0])
            ax_out.plot(ts, outs[:, i], alpha=0.7)
            ax_out.set_ylabel('Output Activity')
            ax_out.set_xlabel('Time [ms]')
            ax_out.set_xlim(0., max_t)

            # recurrent potential
            for j in range(len(self.rec_layers)):
                ax = fig.add_subplot(gs[1, j + 1])
                plt.plot(ts, vs[j][:, i])
                ax.set_xlim(0., max_t)
                ax.set_ylabel(f'Recurrent Potential L{j}')

            if filepath:
                plt.savefig(f'{filepath}/{i}.png')

            if filepath is None:
                plt.show()
        plt.close()


class Trainer(object):
    """
    The training class with only loss.
    """

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
        @brainstate.transform.vmap(in_states=model.states('new'), axis_size=input_info.shape[0])
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
                _etrace_grad, grad_states=self.opt.param_states.to_pytree(), has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(i, inp, targets)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        self._etrace_reset_fun = reset_state
        self._etrace_pred_fun = _etrace_single_run
        self._etrace_train_fun = _etrace_step

    # @brainstate.transform.jit(static_argnums=0)
    def etrace_train(self, inputs, targets):
        inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], -1))  # [n_steps, n_samples, n_in]
        self._etrace_reset_fun()

        # initial gradients
        grads = jax.tree.map(lambda a: jnp.zeros_like(a), self.opt.param_states.to_pytree_value())

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

        # gradient updates
        grads = brainstate.nn.clip_grad_norm(grads, 1.)
        self.opt.update(grads)

        # accuracy
        acc = self._acc(jnp.asarray(outs), targets)

        # memory
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
        grads = brainstate.functional.clip_grad_norm(grads, 1.)
        self.opt.update(grads)

        # accuracy
        acc = self._acc(outs, targets)

        mem_after = get_mem_usage()
        return loss, acc, mem_after

    def f_train(self, train_loader, test_loader):
        print(self.args)

        mem_before = get_mem_usage()

        max_acc = 0.
        for epoch in range(self.args.epochs):
            epoch_acc, epoch_loss, epoch_time, epoch_mem = [], [], [], []
            for batch, (x_local, y_local) in enumerate(train_loader):
                # inputs and targets
                x_local = np.asarray(x_local)
                y_local = np.asarray(y_local)
                # training
                t0 = time.time()
                if self.args.method == 'bptt':
                    loss, acc, mem = self.bptt_train(x_local, y_local)
                else:
                    loss, acc, mem = self.etrace_train(x_local, y_local)
                t = time.time() - t0
                print(
                    f'Epoch {epoch:4d}, '
                    f'training batch {batch:4d}, '
                    f'training loss = {float(loss):.8f}, '
                    f'training acc = {float(acc):.6f}, '
                    f'time = {t:.5f} s, '
                    f'memory before = {mem_before:.2f} GB, '
                    f'memory after = {mem:.2f} GB'
                )
                epoch_acc.append(acc)
                epoch_loss.append(loss)
                epoch_time.append(t)
                epoch_mem.append(mem)
            mean_loss = np.mean(epoch_loss)
            mean_acc = np.mean(epoch_acc)
            mean_time = np.mean(epoch_time[1:-1])
            mean_mem = np.mean(epoch_mem[1:-1])
            print(
                f'Epoch {epoch:4d}, '
                f'training loss = {mean_loss:.8f}, '
                f'training acc = {mean_acc:.6f}, '
                f'time = {mean_time:.5f} s, '
                f'memory before = {mem_before:.2f} GB, '
                f'memory = {mean_mem:.2f} GB'
            )
            self.opt.lr.step_epoch()

            # training accuracy
            if mean_acc > max_acc:
                max_acc = mean_acc
                #   self.target.save(epoch)
                print(f'Save the model at epoch {epoch} with accuracy {max_acc:.6f}')

            # testing accuracy
            epoch_acc, epoch_loss, epoch_time, epoch_mem = [], [], [], []
            for batch, (x_local, y_local) in enumerate(test_loader):
                x_local = np.asarray(x_local)
                y_local = jnp.asarray(y_local)
                loss, acc = self.predict(x_local, y_local)
                epoch_acc.append(acc)
                epoch_loss.append(loss)
            mean_loss = np.mean(epoch_loss)
            mean_acc = np.mean(epoch_acc)
            print(
                f'Epoch {epoch:4d}, testing loss = {mean_loss:.8f}, '
                f'testing acc = {mean_acc:.6f}'
            )
            print('')


def _get_gesture_data(args, cache_dir=os.path.expanduser("./data")):
    # The Dynamic Vision Sensor (DVS) Gesture (DVSGesture) dataset consists of 11 classes of hand gestures recorded
    # by a DVS sensor. The DVSGesture dataset is a spiking version of the MNIST dataset. The dataset consists of
    # 60k training and 10k test samples.

    n_step = args.data_length
    batch_size = args.batch_size
    num_workers = args.n_data_worker

    in_shape = DVSGesture.sensor_size
    transform = tonic.transforms.Compose(
        [
            tonic.transforms.ToFrame(sensor_size=in_shape, n_time_bins=n_step),
            # transforms.Downsample(time_factor=0.5),
            # transforms.DropEvent(p=0.001),
        ]
    )
    train_set = DVSGesture(save_to=cache_dir, train=True, transform=transform)
    test_set = DVSGesture(save_to=cache_dir, train=False, transform=transform)
    train_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=False),
        num_workers=num_workers,
        drop_last=True
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=False),
        num_workers=num_workers
    )

    return train_loader, test_loader


def network_training():
    # environment setting
    brainstate.environ.set(dt=global_args.dt)

    # loading the data
    train_loader, test_loader = _get_gesture_data(global_args, )

    # net
    net = ETraceNet(
        int(np.prod(DVSGesture.sensor_size)),
        global_args.n_rec,
        11,
        global_args.n_layer,
        args=global_args,
    )

    # optimizer
    if global_args.optimizer == 'adam':
        opt_cls = braintools.optim.Adam
    elif global_args.optimizer == 'momentum':
        opt_cls = braintools.optim.Momentum
    elif global_args.optimizer == 'sgd':
        opt_cls = braintools.optim.SGD
    else:
        raise ValueError(f'Unknown optimizer: {global_args.optimizer}')
    opt = opt_cls(lr=global_args.lr, weight_decay=global_args.weight_L2)

    # creating the trainer
    trainer = Trainer(net, opt, global_args)
    trainer.f_train(train_loader, test_loader)


if __name__ == '__main__':
    network_training()
