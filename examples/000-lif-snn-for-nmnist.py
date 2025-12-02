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
import tonic
from tonic.datasets import NMNIST
from torch.utils.data import DataLoader

from snn_models import LIF_Delta_Net, OnlineTrainer


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


if __name__ == '__main__':
    with brainstate.environ.context(dt=1. * u.ms):
        in_shape = NMNIST.sensor_size
        out_shape = 10
        data = NMNIST(
            save_to='D:/data/mnist',
            # save_to='/mnt/d/data/mnist',
            train=True,
            first_saccade_only=True,
            transform=tonic.transforms.ToFrame(sensor_size=in_shape, n_time_bins=200)
        )
        data = DataLoader(
            data,
            shuffle=True,
            batch_size=256,
            collate_fn=numpy_collate,
            num_workers=4,
            drop_last=True,
        )

        net = LIF_Delta_Net(
            n_in=int(np.prod(in_shape)),
            n_rec=200,
            n_out=out_shape,
            tau_mem=20. * u.ms,
            tau_o=20. * u.ms,
            V_th=1. * u.mV,
            rec_scale=2.,
            ff_scale=6.,
        )
        # net.verify(next(iter(data))[0], num_show=5)

        onliner = OnlineTrainer(
            target=net,
            opt=braintools.optim.Adam(lr=1e-3),
            dataset=data,
            x_fun=lambda x: np.transpose(x.reshape(*x.shape[:2], -1), (1, 0, 2)),
            acc_th=0.90,
        )
        losses, accs = onliner.f_train()

    fig, gs = braintools.visualize.get_figure(1, 2, 4., 5.)
    fig.add_subplot(gs[0, 0])
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    fig.add_subplot(gs[0, 1])
    plt.plot(accs)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.show()
