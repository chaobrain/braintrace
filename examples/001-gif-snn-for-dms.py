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

# see braintrace documentations for more details.

import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from snn_models import DMSDataset, GifNet, OnlineTrainer, BPTTTrainer

if __name__ == '__main__':
    OnlineTrainer, BPTTTrainer


    with brainstate.environ.context(dt=1. * u.ms):
        data = DMSDataset(
            bg_fr=1. * u.Hz,
            t_fixation=10. * u.ms,
            t_sample=500. * u.ms,
            t_delay=1000. * u.ms,
            t_test=500. * u.ms,
            n_input=100,
            firing_rate=100. * u.Hz,
            batch_size=128,
            num_batch=100,
        )

        net = GifNet(
            n_in=data.num_inputs,
            n_rec=200,
            n_out=data.num_outputs,
            tau_neu=100. * u.ms,
            tau_syn=100. * u.ms,
            tau_I2=1500. * u.ms,
            A2=1. * u.mA,
        )
        # net.verify(next(iter(data))[0], num_show=2)

        onliner = BPTTTrainer(
            target=net,
            opt=braintools.optim.Adam(lr=1e-3),
            dataset=data,
            n_sim=data.n_sim,
            x_fun=lambda x_local: np.transpose(x_local, (1, 0, 2))
        )
        losses, accs = onliner.f_train()

    fig, gs = braintools.visualize.get_figure(1, 2, 4., 5.)
    fig.add_subplot(gs[0, 0])
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.add_subplot(gs[0, 1])
    plt.plot(accs)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
