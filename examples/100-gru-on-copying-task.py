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

# See braintrace documentation for more details:

import brainstate
import braintools
import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import braintrace


class CopyDataset:
    def __init__(self, time_lag: int, batch_size: int):
        super().__init__()
        self.seq_length = time_lag + 20
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            ids = np.zeros([self.batch_size, self.seq_length], dtype=int)
            # 随机生成10个数字
            ids[..., :10] = np.random.randint(1, 9, (self.batch_size, 10))
            # 在输入序列最后10位中添加10个占位符
            ids[..., -10:] = np.ones([self.batch_size, 10]) * 9
            # 输入序列
            x = np.zeros([self.batch_size, self.seq_length, 10])
            for i in range(self.batch_size):
                x[i, range(self.seq_length), ids[i]] = 1
            yield x, ids[..., :10]


class GRUNet(brainstate.nn.Module):
    def __init__(self, n_in, n_rec, n_out, n_layer):
        super().__init__()

        # 构建GRU多层网络
        layers = []
        for _ in range(n_layer):
            layers.append(braintrace.nn.GRUCell(n_in, n_rec))
            n_in = n_rec
        self.layer = brainstate.nn.Sequential(*layers)
        # 构建输出层
        self.readout = braintrace.nn.Linear(n_rec, n_out)

    def update(self, x):
        return self.readout(self.layer(x))


class Trainer(object):
    def __init__(
        self,
        target: brainstate.nn.Module,
        opt: braintools.optim.Optimizer,
        n_epochs: int,
        n_seq: int,
        batch_size: int = 128,
    ):
        super().__init__()

        # target network
        self.target = target

        # optimizer
        self.opt = opt
        weights = self.target.states().subset(brainstate.ParamState)
        opt.register_trainable_weights(weights)

        # training parameters
        self.n_epochs = n_epochs
        self.n_seq = n_seq
        self.batch_size = batch_size

    def batch_train(self, xs, ys):
        raise NotImplementedError

    def f_train(self):
        dataloader = CopyDataset(self.n_seq, self.batch_size)
        bar = tqdm(enumerate(dataloader), total=self.n_epochs)
        losses = []
        for i, (x_local, y_local) in bar:
            if i == self.n_epochs:
                break
            # training
            x_local = jax.numpy.asarray(np.transpose(x_local, (1, 0, 2)))
            y_local = jax.numpy.asarray(np.transpose(y_local, (1, 0)))
            r = self.batch_train(x_local, y_local)
            bar.set_description(f'Training {i:5d}, loss = {float(r):.5f}', refresh=True)
        return np.asarray(losses)


class OnlineTrainer(Trainer):
    def __init__(self, *args, vjp_method='single-step', batch_train='vmap', **kwargs):
        super().__init__(*args, **kwargs)

        self.vjp_method = vjp_method
        self.batch_train_method = batch_train
        assert batch_train in ['vmap', 'batch']

    @brainstate.transform.jit(static_argnums=(0,))
    def batch_train(self, inputs, target):
        weights = self.target.states(brainstate.ParamState)

        if self.batch_train_method == 'vmap':
            # 初始化在线学习模型
            # 此处，我们需要使用 mode 来指定使用数据集是具有 batch 维度的
            model = braintrace.ParamDimVjpAlgorithm(self.target, vjp_method=self.vjp_method)

            @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
            def init():
                # 对于每一个batch的数据，重新初始化模型状态
                brainstate.nn.init_all_states(self.target)
                # 使用一个样例数据编译在线学习eligibility trace
                model.compile_graph(inputs[0, 0])

            init()
            model = brainstate.nn.Vmap(model, vmap_states='new')

        elif self.batch_train_method == 'batch':
            model = braintrace.ParamDimVjpAlgorithm(
                self.target, vjp_method=self.vjp_method, mode=brainstate.mixin.Batching())
            brainstate.nn.init_all_states(self.target, batch_size=inputs.shape[1])
            model.compile_graph(inputs[0])

        else:
            raise ValueError

        def _etrace_loss(inp, tar):
            # call the model
            out = model(inp)

            # calculate the loss
            loss = braintools.metric.softmax_cross_entropy_with_integer_labels(out, tar).mean()
            return loss, out

        def _etrace_grad(prev_grads, x):
            inp, tar = x
            # 计算当前时刻的梯度
            f_grad = brainstate.transform.grad(_etrace_loss, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(inp, tar)
            # 累计梯度
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            # 返回累计后的梯度和损失函数值
            return next_grads, (out, local_loss)

        def _etrace_train(inputs_):
            # 初始化梯度
            grads = jax.tree.map(lambda a: jax.numpy.zeros_like(a), {k: v.value for k, v in weights.items()})
            # 沿着时间轴计算和累积梯度
            grads, (outs, losses) = brainstate.transform.scan(_etrace_grad, grads, (inputs_, target))
            # 更新梯度
            self.opt.update(grads)
            return losses.mean()

        # 在T时刻之前，模型更新其状态和eligibility trace
        n_sim = self.n_seq + 10
        brainstate.transform.for_loop(lambda inp: model(inp), inputs[:n_sim])

        # 在T时刻之后，模型开始在线学习
        r = _etrace_train(inputs[n_sim:])
        return r


class BPTTTrainer(Trainer):
    @brainstate.transform.jit(static_argnums=(0,))
    def batch_train(self, inputs, targets):
        # 需要求解梯度的参数
        weights = self.target.states(brainstate.ParamState)

        # initialize the states
        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(self.target)

        init()
        model = brainstate.nn.Vmap(self.target, vmap_states='new')

        def _run_step_train(inp, tar):
            out = model(inp)
            loss = braintools.metric.softmax_cross_entropy_with_integer_labels(out, tar).mean()
            return out, loss

        def _bptt_grad_step():
            # 在T时刻之前，模型更新其状态及其eligibility trace
            n_sim = self.n_seq + 10
            _ = brainstate.transform.for_loop(model, inputs[:n_sim])
            # 在T时刻之后，模型开始在线学习
            outs, losses = brainstate.transform.for_loop(_run_step_train, inputs[n_sim:], targets)
            return losses.mean(), outs

        # gradients
        grads, loss, outs = brainstate.transform.grad(_bptt_grad_step, weights, has_aux=True, return_value=True)()

        # optimization
        self.opt.update(grads)

        return loss


online = OnlineTrainer(
    target=GRUNet(10, 200, 10, 1),
    opt=braintools.optim.Adam(0.001),
    n_epochs=1000,
    n_seq=200,
    batch_size=128,
    # batch_train='batch',
    vjp_method='multi-step',
)
online_losses = online.f_train()

bptt = BPTTTrainer(
    target=GRUNet(10, 200, 10, 1),
    opt=braintools.optim.Adam(0.001),
    n_epochs=1000,
    n_seq=200,
    batch_size=128,
)
bptt_losses = bptt.f_train()

plt.plot(online_losses, label='Online Learning')
plt.plot(bptt_losses, label='BPTT')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
