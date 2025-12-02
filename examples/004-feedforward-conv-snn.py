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


from typing import Callable, Iterable

import sys
sys.path.append('../')

import brainstate
import brainpy
import braintools
import brainunit as u
import jax
import numpy as np
import tonic
from tonic.collation import PadTensors
from tonic.datasets import SHD, NMNIST
from torch.utils.data import DataLoader
from tqdm import tqdm

import braintrace


class ConvSNN(brainstate.nn.Module):
    """
    Convolutional SNN example.

    The model architecture is:

    1. Conv2d -> LayerNorm -> IF -> MaxPool2d
    2. Conv2d -> LayerNorm -> IF
    3. MaxPool2d -> Flatten
    4. Linear -> IF
    5. LeakyRateReadout
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        out_sze: brainstate.typing.Size,
        tau_v: float = 2.0,
        tau_o: float = 10.,
        v_th: float = 1.0,
        n_channel: int = 32,
        ff_wscale: float = 40.0,
    ):
        super().__init__()

        conv_inits = dict(w_init=braintools.init.XavierNormal(scale=ff_wscale), b_init=None)
        linear_inits = dict(w_init=braintools.init.KaimingNormal(scale=ff_wscale), b_init=None)
        if_param = dict(
            V_th=v_th,
            tau=tau_v,
            spk_fun=braintools.surrogate.Arctan(),
            V_initializer=braintools.init.ZeroInit(),
            R=1.
        )

        self.layer1 = brainstate.nn.Sequential(
            braintrace.nn.Conv2d(in_size, n_channel, kernel_size=3, padding=1, **conv_inits),
            braintrace.nn.LayerNorm.desc(),
            brainpy.state.IF.desc(**if_param),
            brainstate.nn.MaxPool2d.desc(kernel_size=2, stride=2)  # 14 * 14
        )

        self.layer2 = brainstate.nn.Sequential(
            braintrace.nn.Conv2d(self.layer1.out_size, n_channel, kernel_size=3, padding=1, **conv_inits),
            braintrace.nn.LayerNorm.desc(),
            brainpy.state.IF.desc(**if_param),
        )
        self.layer3 = brainstate.nn.Sequential(
            brainstate.nn.MaxPool2d(kernel_size=2, stride=2, in_size=self.layer2.out_size),  # 7 * 7
            brainstate.nn.Flatten.desc()
        )
        self.layer4 = brainstate.nn.Sequential(
            braintrace.nn.Linear(self.layer3.out_size, n_channel * 4 * 4, **linear_inits),
            brainpy.state.IF.desc(**if_param),
        )
        self.layer5 = braintrace.nn.LeakyRateReadout(self.layer4.out_size, out_sze, tau=tau_o)

    def update(self, x):
        # x.shape = [B, H, W, C]
        return x >> self.layer1 >> self.layer2 >> self.layer3 >> self.layer4 >> self.layer5


class Trainer(object):
    """
    Base class for training spiking neural network models.

    This class provides the core training and evaluation loop logic for SNN models,
    including accuracy calculation, batch evaluation, and epoch-wise training/testing.
    Subclasses should implement the `batch_train` method for specific training algorithms.

    Args:
        target (brainstate.nn.Module): The neural network model to be trained.
        opt (braintools.optim.Optimizer): Optimizer for updating model parameters.
        train_loader (Iterable): DataLoader for training data.
        test_loader (Iterable): DataLoader for test data.
        x_fun (Callable): Function to preprocess input data batches.
        n_epoch (int, optional): Number of training epochs. Default is 30.

    Attributes:
        train_loader (Iterable): Training data loader.
        test_loader (Iterable): Test data loader.
        x_fun (Callable): Input preprocessing function.
        target (brainstate.nn.Module): The model being trained.
        opt (braintools.optim.Optimizer): Optimizer instance.
        n_epoch (int): Number of epochs to train.
    """
    def __init__(
        self,
        target: brainstate.nn.Module,
        opt: braintools.optim.Optimizer,
        train_loader: Iterable,
        test_loader: Iterable,
        x_fun: Callable,
        n_epoch: int = 30,
    ):
        super().__init__()

        # dataset
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.x_fun = x_fun

        # target network
        self.target = target

        # optimizer
        self.opt = opt
        weights = self.target.states().subset(brainstate.ParamState)
        opt.register_trainable_weights(weights)

        # training parameters
        self.n_epoch = n_epoch

    def _acc(self, out, target):
        return jax.numpy.mean(jax.numpy.equal(target, jax.numpy.argmax(jax.numpy.mean(out, axis=0), axis=1)))

    @brainstate.transform.jit(static_argnums=0)
    def batch_eval(self, xs, ys):
        brainstate.nn.vmap_init_all_states(self.target, axis_size=xs.shape[1], state_tag='new')
        model = brainstate.nn.Vmap(self.target, vmap_states='new')

        def _step(inp):
            with brainstate.environ.context(fit=False):
                # call the model
                out = model(inp)
                # calculate the loss
                loss = braintools.metric.softmax_cross_entropy_with_integer_labels(out, ys).mean()
                return loss, out

        losses, outs = brainstate.transform.for_loop(_step, xs)
        return losses.mean(), self._acc(outs, ys)

    def batch_train(self, xs, ys):
        raise NotImplementedError

    def f_train(self):
        train_losses, train_accs = [], []
        test_losses, test_accs = [], []
        for i_epoch in range(self.n_epoch):
            # training
            losses, accs = [], []
            bar = tqdm(self.train_loader)
            for x_local, y_local in bar:
                x_local = self.x_fun(x_local)  # [n_steps, n_samples, n_in]
                y_local = jax.numpy.asarray(y_local)  # [n_samples]
                loss, acc = self.batch_train(x_local, y_local)
                bar.set_description(f'Training loss = {loss:.5f}, acc={acc:.5f}', refresh=True)
                losses.append(loss)
                accs.append(acc)
            train_loss = np.mean(losses)
            train_acc = np.mean(accs)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # testing
            test_losses, test_accs = [], []
            bar = tqdm(self.test_loader)
            for x_local, y_local in bar:
                x_local = self.x_fun(x_local)  # [n_steps, n_samples, n_in]
                y_local = jax.numpy.asarray(y_local)  # [n_samples]
                loss, acc = self.batch_eval(x_local, y_local)
                bar.set_description(f'Testing loss = {loss:.5f}, acc={acc:.5f}', refresh=True)
                test_losses.append(loss)
                test_accs.append(acc)
            test_loss = np.mean(test_losses)
            test_acc = np.mean(test_accs)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            print(f'Epoch {i_epoch + 1}/{self.n_epoch}: '
                  f'train loss={train_loss:.5f}, acc={train_acc:.5f}, '
                  f'test loss={test_loss:.5f}, acc={test_acc:.5f}')
        return (np.asarray(train_losses),
                np.asarray(train_accs),
                np.asarray(test_losses),
                np.asarray(test_accs))


class OnlineVmapTrainer(Trainer):
    def __init__(self, *args, decay_or_rank=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_or_rank = decay_or_rank

    @brainstate.transform.jit(static_argnums=(0,))
    def batch_train(self, inputs, targets):
        # inputs: [n_step, n_batch, ...]
        # targets: [n_batch, n_out]

        # model = braintrace.ES_D_RTRL(self.target, self.decay_or_rank)
        model = braintrace.D_RTRL(self.target)

        @brainstate.transform.vmap_new_states(
            state_tag='new',
            axis_size=inputs.shape[1],
        )
        def init():
            brainstate.nn.init_all_states(self.target)
            # initialize the online learning model
            with brainstate.environ.context(fit=True):
                model.compile_graph(inputs[0, 0])
                model.show_graph()

        init()
        model = brainstate.nn.Vmap(model, vmap_states='new')

        # weights
        weights = self.target.states().subset(brainstate.ParamState)

        def _etrace_grad(inp):
            with brainstate.environ.context(fit=True):
                # call the model
                out = model(inp)
                # calculate the loss
                loss = braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean()
                return loss, out

        def _etrace_step(prev_grads, x):
            # no need to return weights and states, since they are generated then no longer needed
            f_grad = brainstate.transform.grad(_etrace_grad, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(x)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        # forward propagation
        grads = jax.tree.map(u.math.zeros_like, weights.to_dict_values())
        grads, (outs, losses) = brainstate.transform.scan(_etrace_step, grads, inputs)

        # gradient updates
        # grads = brainstate.functional.clip_grad_norm(grads, 1.)
        self.opt.update(grads)
        loss, outs = losses.mean(), outs
        return loss, self._acc(outs, targets)


class OnlineBatchTrainer(Trainer):
    def __init__(self, *args, decay_or_rank=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_or_rank = decay_or_rank

    @brainstate.transform.jit(static_argnums=(0,))
    def batch_train(self, inputs, targets):
        # inputs: [n_step, n_batch, ...]
        # targets: [n_batch, n_out]

        # model = braintrace.ES_D_RTRL(self.target, self.decay_or_rank, model=brainstate.mixin.Batching())
        model = braintrace.D_RTRL(self.target, self.decay_or_rank, model=brainstate.mixin.Batching())

        # initialize the online learning model
        brainstate.nn.init_all_states(self.target, batch_size=inputs.shape[1])
        with brainstate.environ.context(fit=True):
            model.compile_graph(inputs[0])
            model.show_graph()

        # weights
        weights = self.target.states().subset(brainstate.ParamState)

        def _etrace_grad(inp):
            with brainstate.environ.context(fit=True):
                # call the model
                out = model(inp)
                # calculate the loss
                loss = braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean()
                return loss, out

        def _etrace_step(prev_grads, x):
            # no need to return weights and states, since they are generated then no longer needed
            f_grad = brainstate.transform.grad(_etrace_grad, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(x)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        # forward propagation
        grads = jax.tree.map(u.math.zeros_like, weights.to_dict_values())
        grads, (outs, losses) = brainstate.transform.scan(_etrace_step, grads, inputs)

        # gradient updates
        # grads = brainstate.functional.clip_grad_norm(grads, 1.)
        self.opt.update(grads)
        loss, outs = losses.mean(), outs
        return loss, self._acc(outs, targets)


class BPTTTrainer(Trainer):
    @brainstate.transform.jit(static_argnums=(0,))
    def batch_train(self, inputs, targets):
        # inputs: [n_step, n_batch, ...]

        brainstate.nn.vmap_init_all_states(self.target, axis_size=inputs.shape[1], state_tag='new')
        model = brainstate.nn.Vmap(self.target, vmap_states='new')

        # the model for a single step
        def _run_step_train(inp):
            with brainstate.environ.context(fit=True):
                out = model(inp)
                loss = braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean()
                return out, loss

        def _bptt_grad_step():
            outs, losses = brainstate.transform.for_loop(_run_step_train, inputs)
            return losses.mean(), outs

        # gradients
        weights = self.target.states().subset(brainstate.ParamState)
        grads, loss, outs = brainstate.transform.grad(_bptt_grad_step, weights, has_aux=True, return_value=True)()

        # optimization
        # grads = brainstate.functional.clip_grad_norm(grads, 1.)
        self.opt.update(grads)

        return loss, self._acc(outs, targets)


def get_shd_data(
    batch_size: int,
    n_data_worker: int = 8,
    cache_dir='/mnt/d/data/shd/'
):
    # The Spiking Heidelberg Digits (SHD) dataset consists of 20 classes of spoken digits (0-9) spoken by 50 speakers.
    # The SHD dataset is an audio-based classification dataset of 1k spoken digits ranging from zero to nine in
    # the English and German languages. The audio waveforms have been converted into spike trains using an
    # artificial model of the inner ear and parts of the ascending auditory pathway. The SHD dataset has 8,156
    # training and 2,264 test samples. A full description of the dataset and how it was created can be found
    # in the paper below. Please cite this paper if you make use of the dataset.

    in_shape = SHD.sensor_size
    out_shape = 20
    transform = tonic.transforms.ToFrame(sensor_size=SHD.sensor_size, n_time_bins=300)
    train_set = SHD(save_to=cache_dir, train=True, transform=transform)
    test_set = SHD(save_to=cache_dir, train=False, transform=transform)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=False),
        num_workers=n_data_worker,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=False),
        num_workers=n_data_worker
    )

    return brainstate.util.DotDict(
        {'train_loader': train_loader,
         'test_loader': test_loader,
         'in_shape': in_shape,
         'out_shape': out_shape}
    )


def get_nmnist_data(
    batch_size: int,
    n_data_worker: int = 8,
    cache_dir='/mnt/d/data/mnist/'
    # cache_dir='D:/data/mnist/'
):
    # The Neuromorphic-MNIST (N-MNIST) dataset consists of 10 classes of handwritten digits (0-9) recorded by a
    # Dynamic Vision Sensor (DVS) sensor. The N-MNIST dataset is a spiking version of the MNIST dataset. The
    # dataset consists of 60k training and 10k test samples.

    in_shape = NMNIST.sensor_size
    out_shape = 10
    transform = tonic.transforms.ToFrame(sensor_size=in_shape, n_time_bins=200)
    train_set = NMNIST(save_to=cache_dir, train=True, transform=transform, first_saccade_only=True)
    test_set = NMNIST(save_to=cache_dir, train=False, transform=transform, first_saccade_only=True)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=False),
        num_workers=n_data_worker,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=False),
        num_workers=n_data_worker
    )

    return brainstate.util.DotDict(
        {'train_loader': train_loader,
         'test_loader': test_loader,
         'in_shape': in_shape,
         'out_shape': out_shape}
    )


def data_processing(x_local):
    assert x_local.ndim == 5  # (sequence, batch, channel, height, width)
    x_local = x_local.permute(0, 1, 3, 4, 2)  # (sequence, batch, height, width, channel)
    return u.math.asarray(x_local, dtype=brainstate.environ.dftype())


if __name__ == '__main__':
    with brainstate.environ.context(dt=1.0):
        # n-mnist data
        data = get_nmnist_data(batch_size=256, cache_dir='./data/')

        # SHD data
        # data = get_shd_data(batch_size=256)

        # model
        net = ConvSNN(data.in_shape, data.out_shape)

        # Online Trainer
        r = OnlineVmapTrainer(
            target=net,
            opt=braintools.optim.Adam(lr=1e-3),
            train_loader=data.train_loader,
            test_loader=data.test_loader,
            x_fun=data_processing,
            n_epoch=1
        ).f_train()

        # Online Trainer
        r = OnlineVmapTrainer(
            target=net,
            opt=braintools.optim.Adam(lr=1e-3),
            train_loader=data.train_loader,
            test_loader=data.test_loader,
            x_fun=data_processing,
            n_epoch=1
        ).f_train()

        # Offline Trainer
        r = BPTTTrainer(
            target=net,
            opt=braintools.optim.Adam(lr=1e-3),
            train_loader=data.train_loader,
            test_loader=data.test_loader,
            x_fun=data_processing,
            n_epoch=30
        ).f_train()
